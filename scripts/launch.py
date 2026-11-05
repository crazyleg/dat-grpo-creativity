"""Launch a durable local training process; never contains W&B credentials."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora1500.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument('--restart-initialization', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.resume_from and not args.run_dir:
        parser.error("--resume-from requires the original --run-dir")
    if args.restart_initialization and not args.run_dir:
        parser.error('--restart-initialization requires the original --run-dir')
    config = json.loads(Path(args.config).read_text())
    from dat_grpo.train import validate_config
    validate_config(config)
    # A serving checkpoint and a training worker must never compete for Spark RAM.
    states = list((ROOT/'runs').glob('*/status.json'))
    launch_path = ROOT/'inference/launch.json'
    if launch_path.exists():
        states.append(launch_path)
    for state_path in states:
        state = json.loads(state_path.read_text())
        pid = state.get('pid')
        proc = Path(f'/proc/{pid}/cmdline')
        command = proc.read_bytes() if proc.exists() else b''
        if str(ROOT).encode() in command and any(token in command for token in
                [b'dat_grpo.train',b'vllm.entrypoints.openai.api_server']):
            raise SystemExit(f'Pause the existing owned GPU workload before training: PID {pid}')
    prefix = 'qwen38-27b-dat-full-' if config.get('adaptation') == 'full' else 'qwen38-27b-dat-'
    name = prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.run_dir or ROOT / "runs" / name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pid_path = run_dir / "pid"
    if pid_path.exists():
        try:
            os.kill(int(pid_path.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            parser.error("A process for this run is already alive")
    command = [sys.executable, "-u", "-m", "dat_grpo.train", "--config", args.config, "--run-dir", str(run_dir)]
    if args.resume_from:
        command.extend(["--resume-from", str(args.resume_from.resolve())])
    if args.restart_initialization:
        command.append('--restart-initialization')
    from dat_grpo.common import independent_wandb_environment
    environment = independent_wandb_environment()
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("MKL_NUM_THREADS", "8")
    if config.get('adaptation') == 'full' and config['full_finetuning'].get('optimizer') in ('chunked_adafactor','disk_adamw'):
        settings = config['full_finetuning']
        environment['UNSLOTH_COMPILE_DISABLE'] = '1'
        environment['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
        environment['UNSLOTH_GRPO_PREFIX_GROUPER'] = '0'
        environment['UNSLOTH_GRPO_SEQ_PACKING'] = '0'
        environment['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
        environment['MALLOC_ARENA_MAX'] = '2'
        environment['UNSLOTH_SAFETENSORS_PREAD'] = '1'
        command = [sys.executable, '-u', 'scripts/guarded_process.py', '--run-dir', str(run_dir),
            '--min-available-gib', str(settings['min_available_gib']),
            '--timeout-seconds', str(settings.get('watchdog_timeout_seconds',1800)), '--', *command]
    with (run_dir / "training.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(process.pid) + "\n")
    (ROOT / "runs" / "latest.txt").write_text(str(run_dir) + "\n")
    print(json.dumps({"pid": process.pid, "run_directory": str(run_dir), "log": str(run_dir / "training.log")}, indent=2))


if __name__ == "__main__":
    main()
