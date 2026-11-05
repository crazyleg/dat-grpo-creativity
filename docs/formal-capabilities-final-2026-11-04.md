# Completed generic capability battery, 4 November 2026

All **21/21 comparisons** are complete, containing **5,823 first-attempt responses**. The campaign finished at 14:39 UTC on 4 November. This comparison uses the conditional noun/verb LoRA1500 and conditional noun/verb Full250 alongside their common original backbone.

[Combined W&B results](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/qduhht2v) contain the seven-row `results/all_benchmarks` table and the 14-row `results/changes_from_original` table. The remote run is finished and its execution status is completed.

## Primary scores

All values are percentages. HumanEval+ uses native extended-test pass@1. IFEval uses strict prompt accuracy.

| Benchmark | Mode | N | Baseline | LoRA1500 | Full250 |
| --- | --- | ---: | ---: | ---: | ---: |
| MMLU-Pro | no_think | 280 | 59.29 | 59.29 | 56.07 |
| MMLU-Pro | medium_think | 280 | 77.86 | 77.14 | 78.57 |
| HumanEval+ | no_think | 164 | 93.90 | 94.51 | 92.07 |
| HumanEval+ | medium_think | 164 | 93.29 | 93.90 | 93.90 |
| GSM8K | no_think | 256 | 98.05 | 98.44 | 98.44 |
| GSM8K | medium_think | 256 | 98.44 | 98.44 | 98.05 |
| IFEval | no_think | 541 | 83.18 | 80.78 | 79.48 |

## Paired changes from baseline

Differences and intervals are percentage points. Intervals use 5,000 paired question-level bootstrap resamples with seed 20261031. They are exploratory, unadjusted 95% intervals. No equivalence margins were prespecified.

| Benchmark | Mode | Variant | Delta | 95% interval |
| --- | --- | --- | ---: | --- |
| MMLU-Pro | no_think | LoRA1500 | +0.00 | [-1.07, +1.07] |
| MMLU-Pro | no_think | Full250 | -3.21 | [-7.50, +1.07] |
| MMLU-Pro | medium_think | LoRA1500 | -0.71 | [-3.57, +2.14] |
| MMLU-Pro | medium_think | Full250 | +0.71 | [-2.15, +3.57] |
| HumanEval+ | no_think | LoRA1500 | +0.61 | [+0.00, +1.83] |
| HumanEval+ | no_think | Full250 | -1.83 | [-4.88, +1.22] |
| HumanEval+ | medium_think | LoRA1500 | +0.61 | [-1.22, +3.05] |
| HumanEval+ | medium_think | Full250 | +0.61 | [-1.22, +3.05] |
| GSM8K | no_think | LoRA1500 | +0.39 | [+0.00, +1.17] |
| GSM8K | no_think | Full250 | +0.39 | [+0.00, +1.17] |
| GSM8K | medium_think | LoRA1500 | +0.00 | [+0.00, +0.00] |
| GSM8K | medium_think | Full250 | -0.39 | [-1.17, +0.00] |
| IFEval | no_think | LoRA1500 | -2.40 | [-4.62, -0.18] |
| IFEval | no_think | Full250 | -3.70 | [-7.21, -0.37] |

## Instruction-following regression

IFEval strict successes are **450/541 baseline, 437/541 LoRA1500, and 430/541 Full250**. LoRA loses 26 previously successful cases and gains 13. Full250 loses 54 and gains 34. The resulting paired drops are 2.40 and 3.70 percentage points. Both unadjusted bootstrap intervals exclude zero.

| IFEval metric | Baseline | LoRA1500 | Full250 |
| --- | ---: | ---: | ---: |
| Strict prompt accuracy | 83.18% | 80.78% | 79.48% |
| Loose prompt accuracy | 86.69% | 83.73% | 82.07% |
| Truncated responses | 6/541 | 4/541 | 4/541 |

The regression persists in the loose metric and accompanies fewer truncated responses. Truncation counts alone do not explain the decline. The other panels have smaller point changes, with no consistent improvement across tasks and modes. Their limited sizes leave substantial uncertainty about broader capability changes.

## Generation diagnostics

Counts below include every response whose serving finish reason was `length`. They remain in the native scores.

| Benchmark | Mode | Baseline | LoRA1500 | Full250 |
| --- | --- | ---: | ---: | ---: |
| MMLU-Pro | no_think | 2 | 1 | 0 |
| MMLU-Pro | medium_think | 12 | 9 | 8 |
| HumanEval+ | no_think | 0 | 0 | 0 |
| HumanEval+ | medium_think | 0 | 1 | 0 |
| GSM8K | no_think | 0 | 2 | 0 |
| GSM8K | medium_think | 0 | 0 | 0 |
| IFEval | no_think | 6 | 4 | 4 |

## Matched protocol and scope

MMLU-Pro uses the existing fixed 280-question panel, with 20 questions per subject. GSM8K uses the existing fixed 256-question panel. HumanEval+ includes all 164 tasks and IFEval includes all 541 prompts. Each model receives a single sampled response per problem and mode. IFEval is evaluated in no-thinking mode.

| Parameter | No-thinking | Medium-thinking |
| --- | ---: | ---: |
| Temperature | 0.7 | 1.0 |
| top-p | 0.8 | 0.95 |
| top-k | 20 | 20 |
| Presence penalty | 1.5 | 0 |
| Repetition penalty | 1 | 1 |
| Thinking enabled | false | true |
| Reasoning effort | unset | medium |

Medium-thinking output caps are 8,192 total completion tokens. No-thinking caps are 32 for MMLU-Pro, 2,048 for HumanEval+ and IFEval, and 1,536 for GSM8K. Seeds are 20261031 plus the numeric case suffix. Each saved record retains its response, separate reasoning field, usage, finish reason, model alias, seed, and timestamp. All 2,100 medium-thinking records have nonempty reasoning and native reasoning-token counters.

The baseline is the shared BF16 unpacking of the pinned NF4 checkpoint. LoRA1500 uses that backbone and the conditional noun/verb adapter. Full250 uses the separately trained full-weight checkpoint. vLLM serves BF16 with 16,384-token context, 24 GiB KV cache and up to 64 sequences. Serving casts the archived FP32 adapter tensors and full-model normalization tensors to BF16.

This is a matched local screen. It does not reproduce an official leaderboard setup. EvalPlus uses its unchanged native test predicates, including the documented numerical reference self-check failure for HumanEval/32. Different optimizer, training duration and group settings in the two training runs also limit causal claims about LoRA versus full-weight training.

[The full protocol](noun-verb-final-capability-protocol-2026-11-03.md) records model paths, routing, inference settings and the pinned cases.

## Preservation and verification

Completion checks verified exact unique case coverage, model and mode metadata, deterministic seeds and agreement between per-item outcomes and all saved aggregate scores. The aggregate was recomputed and matched exactly. All 3,841 responses cached before resumption retain byte-exact file prefixes. All 138 files in the frozen 3 November archive retain their original SHA-256 hashes, and its 16 completed score cells retain their values.

[Completion audit](../results/protocols/capabilities-completion-audit.json) and [remote W&B verification](../results/protocols/capabilities-completion-audit.json) record the checks. The [preserved intermediate report](formal-capabilities-preserved-2026-11-03.md) remains available.

Raw responses and native item scores are under `benchmarks/results/noun-verb-final-capabilities-20261103T115750Z/`. The complete aggregate is [available locally](../results/summary/capabilities-aggregate.json).
