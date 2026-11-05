# Final conditional LoRA and full-model capability battery

Completed all 21 comparisons and 5,823 responses on 4 November at 14:39 UTC.
The [completed report](formal-capabilities-final-2026-11-04.md) includes the full
score table, paired intervals, generation diagnostics and retention audit.

## Continuation on 4 November

The user requested finishing the remaining generic evaluations before writing
the blog or paper. The original campaign resumed at 12:53 UTC with 16/21
comparisons complete and 3,841 responses cached. The remaining generation scope
is 103 LoRA medium-thinking GSM8K responses, 256 Full250 medium-thinking GSM8K
responses, and all 541 no-thinking IFEval cases for each of the three models,
giving 1,982 remaining responses.

The current configs match the preserved archive. The resumption uses the
original frozen questions, sampling settings, seeds, token limits, native
scorers, and W&B run IDs. `resume-20261104.json` records each prior response
file's length and hash so completion can verify exact retention of the cached
response prefixes. It also retains the previous campaign and server launch
metadata.

The idle writing-judge server was stopped to make the Spark available for the
capability models. The controller loads the original backbone with the
conditional LoRA1500 adapter, the separate Full250 checkpoint, and subsequent
IFEval variants as scheduled.

The explicit resume path now clears stopped-stage state once at startup. A
later stop remains effective. Thirteen handoff/resume tests and four focused
benchmark tests passed, and the native IFEval scorer and tokenizer were probed
successfully before launch.

```sh
.venv-inference/bin/python scripts/launch_overnight.py \
  --resume benchmarks/results/noun-verb-final-capabilities-20261103T115750Z
```

Launched 3 November 2026 at 11:58 UTC. This is the requested matched capability
screen for the final noun/verb-trained LoRA1500 and full250, plus a freshly
generated original baseline. The DAT and class-validity screens have already
completed and are linked separately below.

## Live results

- [Combined W&B comparison](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/qduhht2v) — open `results/all_benchmarks` for named
  Baseline / LoRA 1500 / Full 250 columns and `results/changes_from_original`
  for paired differences and bootstrap intervals as they complete.
- [MMLU-Pro W&B](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/iel8xk02) — first active benchmark.
- [Local live comparison](../benchmarks/results/noun-verb-final-capabilities-20261103T115750Z/comparison.md).

Empty cells mean pending; completed scores appear after each model/mode and
are aggregated without waiting for the whole battery. Progress counters count
generated responses, not training updates. The campaign runs detached from
this chat and resumes saved outputs after recoverable worker failures.

## Exact scope

| Benchmark | Cases per model/mode | Modes | Primary metric | No-thinking output limit | Medium output limit |
| --- | ---: | --- | --- | ---: | ---: |
| MMLU-Pro | 280, 20 per subject | no-thinking, medium-thinking | Final-answer accuracy | 32 | 8192 |
| HumanEval+ | 164, full dataset | no-thinking, medium-thinking | Extended-test pass@1 | 2048 | 8192 |
| GSM8K | 256, frozen test sample | no-thinking, medium-thinking | Numeric-answer accuracy | 1536 | 8192 |
| IFEval | 541, full dataset | no-thinking | Strict prompt accuracy | 2048 | Not scheduled |

Three variants give **21 comparisons and 5,823 generated responses**. IFEval
keeps the no-thinking-only scope of the previously completed LoRA battery;
its previously cancelled medium-thinking evaluation is not silently reinstated.
No additional DAT generation is scheduled.

Order: MMLU-Pro, HumanEval+, GSM8K, IFEval. Within each benchmark, all three
variants run without thinking, then all three with medium thinking where
scheduled. Baseline and LoRA share one original-backbone server; the full
checkpoint is loaded separately. This preserves the earlier preference to
finish comparisons across models before moving to the next benchmark.

## Frozen protocol

The existing `data/benchmark-basic/cases.jsonl` is unchanged. SHA-256:
`a405b3decedaaa06a142636d216e0eb17819bdc71f4ebc2995e3558c365a0226`. Prompts, selected questions, expected answers,
scorers, seed 20261031 plus case index, output limits, and capability sampling
match the previous completed battery. Scores use newly generated responses;
old baseline or model outputs are not copied into this campaign.

| Parameter | No-thinking | Medium-thinking |
| --- | ---: | ---: |
| Temperature | 0.7 | 1.0 |
| top-p | 0.8 | 0.95 |
| top-k | 20 | 20 |
| min-p | 0 | 0 |
| Presence penalty | 1.5 | 0 |
| Repetition penalty | 1 | 1 |
| enable_thinking | false | true |
| reasoning_effort | unset | medium |

Only the final answer is graded. The initial baseline mode preflight passed:
both modes return the correct final answer, only medium returns a separate
reasoning field, and neither simple probe is truncated. Raw outputs, finish
reasons, token usage, reasoning, and per-item scores are retained. Missing
answers and truncation failures are visible rather than excluded.

The existing native EvalPlus and IFEval scorers are retained. HumanEval/32 has
the previously documented numerical reference self-check failure; its native
test predicates remain unchanged. Paired intervals use 5,000 bootstrap
resamples, seed 20261031, without multiple-comparison correction. Small panels
and single sampled responses do not prove capability equivalence.

## Model identity and serving

| Comparison label | Request alias | Weights |
| --- | --- | --- |
| Baseline | original | Common BF16 unpacking of the pinned original NF4 checkpoint, no adapter |
| LoRA 1500 | step1500 | Same backbone plus conditional noun/verb LoRA snapshot 1500 |
| Full 250 | step250 | Conditional noun/verb full-weight snapshot 250, no adapter |

LoRA: `runs/qwen38-27b-dat-noun-verb-lora-20261102T170042Z/adapter-snapshots/step-1500`.
Full: `runs/qwen38-27b-dat-noun-verb-full-b6-20261102T214703Z/model-snapshots/step-250`.
These are the newer conditional noun/verb runs, not the earlier noun-only
adapters or full checkpoints. Snapshot identity and full shard availability
were checked before launch. Both training workers were confirmed stopped.

vLLM uses BF16, 16,384-token context, 24 GiB KV cache, 64 concurrent sequences,
eager execution, prefix caching disabled, and the existing qwen3 reasoning
parser. Serving casts FP32 adapter archives and full-model normalization
tensors to BF16; saved archives are unchanged. This is a matched serving
screen rather than an exact replay of the training arithmetic.

The completed LoRA DAT server was checked and adopted for the initial baseline
and LoRA requests, avoiding another initial base-model load. Its extra old
adapter aliases are outside the capability serving spec. Before any full250
request the controller must replace that server with the separate full
checkpoint. Every model/mode response directory saves `server-launch.json`
with the actual server identity. The routing tests cover base/adapter reuse,
full weights without adapters, and rejection of adapters on a trained full
backbone. Twenty relevant routing, queue, scoring, payload, and resume tests
passed before launch.

## Artifacts

Campaign directory: `benchmarks/results/noun-verb-final-capabilities-20261103T115750Z/`.
Configuration: `configs/benchmark_noun_verb_final_capabilities.json`.
The campaign contains `plan.json`, stage-specific configs, `aggregate.json`,
`status.json`, launch validation, inference adoption evidence, W&B run records,
and per-benchmark generations/scores. The supervisor verifies complete case
coverage and metric counts before advancing, prevents simultaneous workers,
and retries an exited failed worker at most three times. It does not replace a
live worker just because inference is slow.

Completed DAT reports:

- [LoRA1500 and earlier adapters](noun-verb-lora1500-dat-results-2026-11-03.md).
- [Full250 and earlier full snapshots](noun-verb-full250-dat-results-2026-11-03.md).
