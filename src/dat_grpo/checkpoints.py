"""Keep model adapters for comparison independently of resume-state rotation."""

import os
import shutil
import tempfile
from pathlib import Path

from .common import utc_now, write_json

ADAPTER_FILES = (
    "adapter_model.safetensors", "adapter_config.json", "tokenizer.json",
    "tokenizer_config.json", "processor_config.json", "chat_template.jinja", "README.md",
)


def archive_adapter(checkpoint, archive_root, step):
    checkpoint, archive_root = Path(checkpoint), Path(archive_root)
    destination = archive_root / f"step-{step}"
    required = ADAPTER_FILES[:2]
    if destination.exists():
        if not all((destination / name).is_file() for name in required):
            raise ValueError(f"Incomplete adapter snapshot: {destination}")
        return destination
    if not all((checkpoint / name).is_file() for name in required):
        raise ValueError(f"Checkpoint has no complete adapter: {checkpoint}")
    archive_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".step-{step}-", dir=archive_root))
    try:
        copied = []
        for name in ADAPTER_FILES:
            source = checkpoint / name
            if source.is_file():
                try:
                    os.link(source, temporary / name)
                except OSError:
                    shutil.copy2(source, temporary / name)
                copied.append(name)
        write_json(temporary / "snapshot.json", {
            "optimizer_step": step, "created_at": utc_now(),
            "source_checkpoint": str(checkpoint.resolve()), "files": copied,
        })
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def archive_full_model(checkpoint, archive_root, step):
    """Preserve full model shards by hard link, independently of optimizer rotation."""
    import json
    checkpoint, archive_root = Path(checkpoint), Path(archive_root)
    index = checkpoint / 'model.safetensors.index.json'
    if index.is_file():
        shards = set(json.loads(index.read_text())['weight_map'].values())
        required = {'config.json', index.name, *shards}
    else:
        required = {'config.json', 'model.safetensors'}
    if not all((checkpoint/name).is_file() for name in required):
        raise ValueError(f'Incomplete full-model checkpoint: {checkpoint}')
    destination = archive_root/f'step-{step}'
    if destination.exists():
        if not all((destination/name).is_file() for name in required):
            raise ValueError(f'Incomplete full-model archive: {destination}')
        return destination
    archive_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.step-{step}-',dir=archive_root))
    try:
        files = required | {path.name for path in checkpoint.iterdir() if path.is_file() and
                            (path.suffix=='.json' or path.name in ['chat_template.jinja','README.md']) and path.name!='trainer_state.json'}
        for name in files:
            try:
                os.link(checkpoint/name,temporary/name)
            except OSError:
                shutil.copy2(checkpoint/name,temporary/name)
        write_json(temporary/'snapshot.json',{'optimizer_step':step,'created_at':utc_now(),
            'source_checkpoint':str(checkpoint.resolve()),'format':'full model, no adapter','files':sorted(files)})
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise
    return destination


def publish_adapter_final(checkpoint, destination, step):
    """Publish the selected saved adapter after an externally requested stop."""
    destination = Path(destination)
    snapshot = archive_adapter(checkpoint,destination.parent/'adapter-snapshots',step)
    destination.mkdir(parents=True,exist_ok=True)
    for source in snapshot.iterdir():
        if source.is_file():
            temporary = destination/('.publish-'+source.name)
            if temporary.exists():
                temporary.unlink()
            os.link(source,temporary)
            os.replace(temporary,destination/source.name)
    return destination


def publish_full_final(checkpoint, destination, step):
    """Reuse validated checkpoint shards instead of serializing the final model twice."""
    destination = Path(destination)
    snapshot = archive_full_model(checkpoint,destination.parent/'model-snapshots',step)
    destination.mkdir(parents=True,exist_ok=True)
    for source in snapshot.iterdir():
        if not source.is_file():
            continue
        temporary = destination/('.publish-'+source.name)
        if temporary.exists():
            temporary.unlink()
        os.link(source,temporary)
        os.replace(temporary,destination/source.name)
    return destination


def full_snapshot_due(step, settings):
    """Archive selected early weights as well as the regular interval."""
    return step > 0 and (step % settings['archive_steps'] == 0
        or step in settings.get('archive_at_steps', []))


def preserve_available_early_snapshots(run_dir, steps, regular_interval):
    """Preserve completed extra checkpoints while an older trainer is running.

    Regular snapshots belong to the trainer callback, avoiding concurrent writes
    with an already-running process that cannot reload its callback code.
    """
    import json
    run_dir = Path(run_dir)
    preserved = []
    for step in steps:
        destination = run_dir/'model-snapshots'/f'step-{step}'
        if destination.exists():
            metadata = json.loads((destination/'snapshot.json').read_text())
            if metadata['optimizer_step'] != step or not all((destination/name).is_file() for name in metadata['files']):
                raise ValueError(f'Incomplete archived snapshot: {destination}')
            preserved.append(step)
            continue
        if step % regular_interval == 0:
            continue
        checkpoint = run_dir/'checkpoints'/f'checkpoint-{step}'
        if not all((checkpoint/name).is_file() for name in
                   ['trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth']):
            continue
        try:
            state = json.loads((checkpoint/'trainer_state.json').read_text())
        except json.JSONDecodeError:
            continue  # TrainerState may still be in the process of being written.
        if state['global_step'] != step:
            raise ValueError(f'Checkpoint counter mismatch: {checkpoint}')
        archive_full_model(checkpoint,run_dir/'model-snapshots',step)
        preserved.append(step)
    return preserved
