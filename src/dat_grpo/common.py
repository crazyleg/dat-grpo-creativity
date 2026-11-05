import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def independent_wandb_environment(environment=None):
    """Give a detached process its own W&B service and run identity."""
    result = dict(os.environ if environment is None else environment)
    for name in ('WANDB_SERVICE', '_WANDB_SERVICE', 'WANDB_RUN_ID',
                 'WANDB_RUN_NAME', 'WANDB_RESUME', 'WANDB_RUN_GROUP'):
        result.pop(name, None)
    return result
