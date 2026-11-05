"""Unpack the frozen NF4 backbone to BF16 for native vLLM adapter serving."""

import json
import os
from pathlib import Path

os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from dat_grpo.common import utc_now, write_json

root = Path(__file__).resolve().parents[1]
source = json.loads((root / "data/model.json").read_text())
destination = root / "data/inference-bf16"
if (destination / "export-manifest.json").exists():
    print(f"Existing export: {destination}", flush=True)
    raise SystemExit(0)
print("Loading frozen NF4 backbone", flush=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    source["snapshot"], dtype=torch.bfloat16, device_map={"":0}, local_files_only=True,
)
print("Unpacking NF4 weights to BF16; adapters are loaded separately by vLLM", flush=True)
model.dequantize()
model.to(dtype=torch.bfloat16)
if hasattr(model.config, "quantization_config"):
    delattr(model.config, "quantization_config")
destination.mkdir(parents=True, exist_ok=True)
model.save_pretrained(destination, max_shard_size="4GB", safe_serialization=True, save_original_format=False)
AutoProcessor.from_pretrained(source["snapshot"], local_files_only=True).save_pretrained(destination)
write_json(destination / "export-manifest.json", {
    "created_at":utc_now(), "source":source,
    "operation":"dequantize frozen NF4 backbone to BF16; no LoRA merged",
    "dtype":"bfloat16", "all_evaluation_variants_share_this_backbone":True,
    "shards":{p.name:p.stat().st_size for p in destination.glob('*.safetensors')},
})
print(f"Export ready: {destination}", flush=True)
