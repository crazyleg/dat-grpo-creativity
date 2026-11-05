# Noun/verb GRPO: completed full-weight run

Verified on 3 November 2026. LoRA stopped at **1,500 updates**, and the fresh
full-weight run completed **250 updates with six completions of one prompt per
group**. Both W&B runs are finished and their workers have stopped.

- [Full-weight run](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/s1rlnaat)
- [LoRA run](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/iolv9hr3)
- [Detailed protocol and incident history](noun-verb-full-protocol.md)

The full run fits the Spark, produces real full-weight updates, and preserves
all requested model snapshots. Instruction following was unstable during
training. At training completion, adjective transfer and general capability
preservation were unmeasured. The subsequent [three-class DAT screen](noun-verb-full250-dat-results-2026-11-03.md)
now compares the baseline and final model; earlier snapshots are still running.

## Recipe and fidelity

The starting policy is the original pinned NF4 backbone unpacked to BF16,
without a trained adapter or earlier full-run weights. The source is
`unsloth/Qwen3.8-27B-unsloth-bnb-4bit`, revision
`8aa5f05d26b7205477066e1449e0af13f762a299`.

The full run uses chunked Adafactor with FP32 factored statistics, stochastic
BF16 rounding, constant learning rate `1e-6`, global gradient clipping `0.5`,
activation checkpointing, and checked two-pass streamed gradients. Its frozen
KL reference is NF4. GRPO uses beta `0.04`, epsilon `0.1`, group reward scaling,
one iteration, batch six, and accumulation one. Sampling is temperature `1`,
top-p `1`, top-k `0`, and at most 192 completion tokens, with the closed-thinking
assistant prefix.

The backbone, task, and reward configurations match LoRA. Seven wordings cover
nouns and verbs, with optional class-only reminders: 28 distinct prompts and
1,120 training rows. Adjectives remain a held-out request class. The rendered
dataset SHA-256 is
`cd8072241c799b052ae41b86e22c5e30ff294a569dc79e8c09018ba021cf8e21`.

Reward remains official DAT / 100 minus `0.2` per wrong or unverified word,
checking all ten requested words. Malformed or DAT-unscoreable responses
receive `-2`. Official DAT uses the first seven valid unique words. Reward
and prompt source files are unchanged from the recorded sequence plan.

This is a shared-reward experiment with different optimization and training
exposure: full Adafactor versus LoRA, six versus four completions per update,
and 250 versus 1,500 updates. It does not isolate one causal change.

## Observed training outcomes

All 1,500 full-run completions were re-scored successfully: **804 verb and
696 noun requests**, covering all seven wordings and zero adjective requests.
Across the complete run, **1,246/1,500** outputs were DAT-scoreable; **971/1,500**
were scoreable and passed all ten dictionary class checks. Mean composite
reward was **0.3057**, and mean DAT among scoreable outputs was **83.55**.

Each row below covers 150 generated outputs. These are training observations,
with differing prompt/class mixtures across windows, rather than a paired
baseline comparison. DAT means exclude unscoreable outputs; composite means
include their negative rewards.

| Updates | DAT-scoreable | Mean composite reward | Mean DAT among scoreable outputs |
| --- | ---: | ---: | ---: |
| 1–25 | 137/150 | 0.4391 | 76.40 |
| 26–50 | 105/150 | -0.0758 | 81.17 |
| 51–75 | 0/150 | -2.0000 | — |
| 76–100 | 141/150 | 0.6338 | 83.17 |
| 101–125 | 146/150 | 0.7076 | 84.62 |
| 126–150 | 145/150 | 0.7052 | 82.47 |
| 151–175 | 147/150 | 0.7112 | 81.68 |
| 176–200 | 148/150 | 0.7852 | 87.28 |
| 201–225 | 131/150 | 0.4350 | 87.83 |
| 226–250 | 146/150 | 0.7152 | 86.63 |

The final 25 updates had **146/150** scoreable outputs, **96.16%** dictionary
word acceptance among those outputs, and **102/150 (68%)** outputs jointly
passing DAT and all ten class checks.

| Final 25 updates | Scoreable outputs | Mean DAT among scoreable outputs | Words accepted among scoreable outputs | Scoreable and all ten accepted |
| --- | ---: | ---: | ---: | ---: |
| Nouns | 40/42 | 88.61 | 95.50% | 24/42 (57.14%) |
| Verbs | 106/108 | 85.88 | 96.42% | 78/108 (72.22%) |

The frozen WordNet verifier accepts any requested noun/verb sense. In the final
window, 61.71% of word tokens in scoreable lists have both senses, including
71.60% in verb lists. These figures measure the verifier's lexical acceptance,
without disambiguating an isolated word's intended sense. Outputs rejected
before word extraction have no class checks; a parsed list can retain class
checks while failing DAT uniqueness/dictionary requirements. Joint compliance
requires both DAT scoreability and all ten accepted words.

## Regressions and limits

Every output in updates 51–75 was unscoreable, predominantly reopening thinking
despite the closed-thinking prompt. The saved state at 75 was resumed in a
fresh worker with the same weights, optimizer, RNG, and settings. Update 76
still failed; later groups recovered. The cause remains unresolved.

Another formatting regression at 204–214 included definitions, truncation,
extra closing brackets, and wrong-length lists. Updates 201–225 contained
19 failures: 16 invalid JSON outputs, two single-word gate failures, and one
wrong-length list. The same worker subsequently produced 60/60 scoreable
outputs in updates 216–225, without a restart or recipe change. The final
25 updates still had four failures: three single-word gate failures and one
invalid JSON output. No thinking tags occurred in that final window.

Improved conditional DAT scores in later training windows do not establish
creativity transfer, preserved capabilities, or improved adjective following.
The subsequent [DAT screen](noun-verb-full250-dat-results-2026-11-03.md) measures
the three word classes and finds adjective-following regressions. General
capability evaluation remains pending. Saved earlier snapshots support the
ongoing comparison.

## Verification and retained models

The final checkpoint and export contain **55 shards, 1,184 tensors, and
27,356,728,560 parameters**. Every saved parameter value was scanned and is
finite. The export shares the exact checkpoint-250 weight files. Comparisons
with the original backbone found nonzero changes in embedding, output-head,
attention, and MLP probes.

The optimizer and scheduler are at exactly 250. All 1,397 floating optimizer
statistics are finite FP32. All 250 loss/gradient agreement pairs passed, and
every canonical update used a single six-row forward in each of its two
passes. Actual gradients cover 851 tensors / 26,895,998,464 text parameters;
the unused vision path receives no text-task gradients. The batch-six capacity
probe performed a 115-prompt-token / 192-completion-token backward with no
weight updates, peaking at **75.694 GiB allocated**. Batch four was unnecessary.

Full model snapshots **50, 75, 100, 150, 200, and 250** are complete. The extra
step-25 model and complete recovery copies at 25/50/75 also remain available.
The latest normal resume checkpoints are 225 and 250. LoRA snapshots
500/1000/1500 and its final adapter were scanned and contain finite values;
the final adapter matches checkpoint 1500. W&B confirms completed targets
1500 and 250, and both final workers exited with their intended terminal states.

Repository-relative artifacts:

- Final model: `runs/qwen38-27b-dat-noun-verb-full-b6-20261102T214703Z/model-final/`
- Full snapshots: the same run's `model-snapshots/step-N/`
- Full weight/reward audit: the same run's `completed-training-verification.json`
- Requirement audit: `runs/noun-verb-lora1500-full250-sequence-20261102T214703Z/completion-audit.json`
- Training summaries: the same sequence directory's `training-outcome-metrics.json`
- Settings: [full batch-six configuration](../configs/qwen38_27b_dat_noun_verb_full_b6.json)
