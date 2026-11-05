"""Record allocator and host memory while a model is being materialized."""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time


@contextmanager
def loading_trace(run_dir,torch):
    import psutil
    stop = threading.Event()
    def sample():
        with (Path(run_dir)/'loading-memory.jsonl').open('a') as stream:
            while not stop.is_set():
                record = dict(time=time.time(),allocated_gib=torch.cuda.memory_allocated()/2**30,
                    reserved_gib=torch.cuda.memory_reserved()/2**30,
                    rss_gib=psutil.Process().memory_info().rss/2**30,
                    available_gib=psutil.virtual_memory().available/2**30)
                stream.write(json.dumps(record)+'\n');stream.flush()
                stop.wait(2)
    worker = threading.Thread(target=sample,daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set();worker.join(timeout=3)


def restore_full_weights(model, checkpoint):
    """Restore an exact full checkpoint through bounded pread buffers.

    Validate all keys/shapes before copying, and preserve Parameter identities
    so the streaming optimizer's hooks continue to refer to the loaded weights.
    """
    import torch
    from safetensors import safe_open
    checkpoint=Path(checkpoint)
    index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    state=model.state_dict()
    if set(index)!=set(state):
        raise ValueError('Full checkpoint keys do not match the policy')
    shards=sorted(set(index.values()))
    for shard in shards:
        if Path(shard).name!=shard:
            raise ValueError('Invalid full checkpoint shard name')
        with safe_open(str(checkpoint/shard),framework='pt',device='cpu',backend='pread') as stream:
            for name in stream.keys():
                if name not in index or index[name]!=shard or list(state[name].shape)!=stream.get_slice(name).get_shape():
                    raise ValueError('Full checkpoint tensor mapping or shape mismatch')
    with torch.no_grad():
        for number,shard in enumerate(shards,1):
            with safe_open(str(checkpoint/shard),framework='pt',device='cpu',backend='pread') as stream:
                for name in stream.keys():
                    tensor=stream.get_tensor(name)
                    state[name].copy_(tensor)
                    del tensor
            if number%5==0 or number==len(shards):
                print(f'FULL_RESTORE_SHARDS {number}/{len(shards)}',flush=True)
