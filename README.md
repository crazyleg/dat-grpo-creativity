# DAT reward GRPO and creativity transfer

Training code and results for a small experiment: optimize word-list semantic
distance, then test whether the change transfers to writing. Qwen3.8-27B was
trained with noun/verb DAT rewards using QLoRA and full-weight GRPO on one
128 GB NVIDIA DGX Spark. The full checkpoint increased judged creativity and
imagery on a fixed story panel, with lower coherence and instruction following.

- [Read the illustrated article](article/post.md).
- [Try Full250 on Hugging Face ZeroGPU](https://huggingface.co/spaces/crazyleg11/full250-creativity).
- [Download canonical safetensors weights](https://huggingface.co/crazyleg11/qwen3.8-27b-dat-grpo-full250).
- [Download Q4_K_M or Q8_0 GGUF](https://huggingface.co/crazyleg11/qwen3.8-27b-dat-grpo-full250-GGUF).
- [Inspect results, raw response bundles and protocols](results/README.md).

## Critical reward example

For a format-valid ten-word list, the reward is

```python
reward = dat_score / 100.0 - 0.2 * wrong_class_word_count
```

Malformed or unscoreable responses receive -2.0. The original DAT scorer uses
the first seven valid unique words; the noun/verb check covers all ten.
Adjectives are excluded from training. Any WordNet sense of the requested class
is accepted, so ambiguous words can pass both noun and verb checks.

The actual implementation is
[`score_class_completion`](src/dat_grpo/reward.py), with
[`ConditionalDATReward`](src/dat_grpo/reward.py) passing group-aligned rewards to
TRL GRPO. Seven prompt templates are in [`prompts/`](prompts/dat_noun_verb.json).
[`examples/reward.py`](examples/reward.py) calls the same scorer as training.

## Reproduce the training

The recorded environment is Linux aarch64, Python 3.12, CUDA 13.0,
PyTorch 2.9.1, Transformers 5.17.0, TRL 1.15.0 and Unsloth 2026.10.11.
[`requirements-spark.lock.txt`](requirements-spark.lock.txt) records the original
environment; [`requirements-spark.in`](requirements-spark.in) pins the main
packages. GPU and architecture compatibility of these packages matters.

From this repository root, install [uv](https://docs.astral.sh/uv/) and run:

```bash
bash scripts/bootstrap.sh
uv pip install --python .venv/bin/python nltk==3.9.2
.venv/bin/python -m dat_grpo.prepare --config configs/lora1500.json --component all
.venv/bin/python scripts/prepare_wordnet.py
```

Preparation downloads the pinned NF4 backbone and the 2.18 GB GloVe archive,
builds the official DAT cache, and verifies the WordNet source and lexicon.
Keep enough disk space for the extracted GloVe text, model cache and snapshots.

To inspect a real reward without loading the language model:

```bash
.venv/bin/python examples/reward.py --class nouns \
  --words '["cat","book","ocean","piano","justice","volcano","silk","clock","forest","electron"]'
```

Set the W&B entity in your config, or use `WANDB_ENTITY`. Authenticate W&B using
`wandb login` or your own `WANDB_API_KEY`. Credentials are never stored here.
The training launcher creates a durable worker and logs its run URL, metrics,
rollouts, environment, task manifests and checkpoints under `runs/`.

QLoRA with a frozen NF4 backbone:

```bash
.venv/bin/python scripts/launch.py --config configs/lora1500.json --run-dir runs/lora1500
```

Full-weight GRPO first needs the same NF4 initialization unpacked into BF16:

```bash
.venv/bin/python scripts/export_inference_base.py
.venv/bin/python scripts/launch.py --config configs/full250.json --run-dir runs/full250
```

The full path updates all 26.90B text parameters, leaving the vision path
unchanged. It uses gradient checkpointing, checked two-pass gradient replay,
streamed updates, FP32 factored Adafactor statistics, stochastic BF16 rounding,
an NF4 reference model and an independent host-memory watchdog. Launch it
through `scripts/launch.py`; directly invoking its training worker is rejected.
The recipe is designed for the 128 GB Spark and is not a generic 24 GB GPU
full-tuning recipe. The early full run had a period of invalid responses; this
is an experimental training recipe, not a monotonic improvement guarantee.

The compared QLoRA checkpoint is step 1500. The recorded original config
scheduled 2500 steps, but this release's convenience config stops at the
compared 1500. Both original configurations and the exact environment records
are preserved under [`results/protocols/`](results/protocols/).

Full checkpoints preserve steps 50, 75, 100, 150, 200 and 250. QLoRA preserves
500-step snapshots. Resume into the original run directory, for example:

```bash
.venv/bin/python scripts/launch.py --config configs/full250.json \
  --run-dir runs/full250 --resume-from runs/full250/checkpoints/checkpoint-100
```

Check the actual checkpoint directory in your run before resuming. Snapshots
for inference and resumable optimizer checkpoints serve different purposes.

## Checks and figures

The CPU tests exercise reward gates, word classes, grouping, gradient replay,
streaming optimizer behavior, checkpoint handling and memory guards:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

To regenerate the publication figures and tables without a GPU:

```bash
uv pip install --python .venv/bin/python -r requirements-analysis.txt
.venv/bin/python scripts/build_figures.py
.venv/bin/python scripts/recompute_story_scores.py
```

The article and result manifest describe sampling, judge uncertainty,
truncation, lexical-validity failures and capability regressions. The LoRA/full
comparison uses different optimizers, group sizes and training exposure; it
does not establish a general limit of adapter tuning.

## License and attribution

Original code is MIT licensed. The original DAT code retains its MIT license
and provenance. See [third party sources](THIRD_PARTY.md) for data and benchmark
attribution. Model licensing is specified in the Hugging Face model card.
