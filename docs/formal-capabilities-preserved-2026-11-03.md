# Formal capability results preserved before writing evaluation

The resumed battery completed all 21 comparisons on 4 November. The
[completed report](formal-capabilities-final-2026-11-04.md) contains the full
results and preservation audit. This report records the earlier frozen state.

Preserved 2026-11-03T18:44:36.746646+00:00. The user stopped the capability battery to prioritize creative writing and style. 16/21 comparisons have complete scores. Partial generations are retained and excluded from completed metrics.

| Benchmark | Mode | Metric | Baseline | LoRA 1500 | Full 250 | N |
|---|---|---|---|---|---|---|
| MMLU-Pro | no_think | accuracy_percent | 59.29 | 59.29 | 56.07 | 280 |
| MMLU-Pro | medium_think | accuracy_percent | 77.86 | 77.14 | 78.57 | 280 |
| HumanEval+ | no_think | humaneval/plus_pass_at_1 | 93.9 | 94.51 | 92.07 | 164 |
| HumanEval+ | medium_think | humaneval/plus_pass_at_1 | 93.29 | 93.9 | 93.9 | 164 |
| GSM8K | no_think | gsm8k/accuracy | 98.05 | 98.44 | 98.44 | 256 |
| GSM8K | medium_think | gsm8k/accuracy | 98.44 | Unmeasured | Unmeasured | 256 |
| IFEval | no_think | ifeval/strict_prompt_accuracy | Unmeasured | Unmeasured | Unmeasured | Unmeasured |

Accuracy and pass rates are percentages on the existing fixed panels. Paired intervals are retained in `frozen-aggregate.json`. Missing cells are not zero.

Partial generations:
- gsm8k, medium_think, step1500: 153/256 responses retained.

[Formal W&B run](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/qduhht2v).

Frozen directory: `./benchmarks/results/formal-results-preserved-20261103T184436Z`. All 138 copied files were SHA-256 checked against the originals. The manifest records every file and response count. The original directories and model snapshots remain retained.
