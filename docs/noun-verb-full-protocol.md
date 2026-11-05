# Conditional noun/verb full GRPO after LoRA step 1,500

Requested 2 November 2026: finish the current LoRA at step 1,500, then run
250 full-weight GRPO updates with the same reward, trying batch six first.
The sequence was queued at 21:47 UTC on 2 November. LoRA completed the selected
1,500-step checkpoint at 01:20 UTC on 3 November. The batch-six full run
completed 250 updates at 09:19 UTC on 3 November, and its saved artifacts passed
the final audit. See the [completed results](noun-verb-full250-results-2026-11-03.md).
The dated entries below preserve the observations and pending status recorded
at each stage of the experiment.

## Experimental settings

The task, prompt templates, fixed dataset seed, backbone revision, verifier,
and reward are copied unchanged from the conditional noun/verb LoRA. There
are seven wordings, optional explicit class reminders, and 1,120 balanced
training rows. Adjectives remain entirely held out. The reward is official
DAT/100 minus 0.2 per wrong or unverified word across all ten requested words;
malformed or unscorable output receives -2. Known WordNet coverage gaps are
left unchanged so the reward is the same experiment.

The full policy starts afresh from `data/inference-bf16`, the original pinned
NF4 backbone dequantized to BF16. No trained LoRA is merged and no weights
from the earlier noun-only full run are loaded. A frozen NF4 reference
remains resident for KL regularization. All policy parameters are trainable.

The optimizer is the previously validated streaming chunked Adafactor,
with BF16 stochastic rounding, activation checkpointing, two checked backward
passes, and global gradient clipping at 0.5. Learning rate is constant 1e-6;
KL coefficient 0.04, GRPO epsilon 0.1, group reward scaling, temperature 1,
top-p 1, and maximum completion length 192 are unchanged. Optimizer choice
differs from LoRA's AdamW, as in the earlier full-weight experiment.

The first candidate uses **six completions of one prompt per GRPO group**,
per-device batch six, and accumulation one. The fallback uses four completions
per group and batch four. This is one group per optimizer update in either
case, with the same 250-update target.

| Candidate | Training configuration | Maximum-length memory probe |
| --- | --- | --- |
| Six first | `configs/qwen38_27b_dat_noun_verb_full_b6.json` | `configs/qwen38_27b_dat_noun_verb_full_b6_preflight.json` |
| Four if capacity fails | `configs/qwen38_27b_dat_noun_verb_full_b4.json` | `configs/qwen38_27b_dat_noun_verb_full_b4_preflight.json` |

Full snapshots retain the prior requested schedule: 50, 75, 100, 150, 200,
and 250, with resume checkpoints every 25 updates and an initial checkpoint
at 10. Archived models survive rotation of the two recent resume checkpoints.

## Durable sequence and validation

`src/dat_grpo/sequence.py` waits for the complete LoRA checkpoint 1,500 and
adapter archive, signals only the verified training process for that run,
waits for its exit, then publishes the selected saved adapter as final.
Existing LoRA snapshots, including 500 and the scheduled 1,000/1,500, remain.
W&B is finalized at the selected stopping step. The already-running trainer
still has max_steps 2,500 in memory; the recorded stop plan and controller
implement the changed stopping point without discarding current updates.

After LoRA exits, each full candidate first runs a separate norm-only memory
probe at the **longest rendered prompt among all 28 requests** and all 192
completion tokens, with its frozen reference resident. The probe checks zero
optimizer updates and unchanged weights. It is not itself training or proof
that the subsequent update pass fits; actual full GRPO must also succeed.
Both probes and real training use the host-memory watchdog, an 18 GiB host
headroom floor, and the prior 72% CUDA allocation cap.

Only an observed memory-capacity failure selects batch four. Non-finite loss,
two-pass disagreement, incorrect metadata, stalled progress, and unrelated
failures stop the queue for investigation. If a six-batch training attempt
fails for capacity after some updates, its diagnostics and outputs remain,
and the four-batch candidate starts from the original backbone for a fresh
250-update run. A live process is reattached rather than restarted on an
observation timeout. Queued configuration hashes prevent unnoticed changes
while waiting.

The previous evaluation-resume controller is explicitly superseded, keeping
the old evaluation campaign paused throughout these training jobs. It cannot
compete for GPU memory at the LoRA/full handoff. Monitoring for probes and
full training uses new W&B runs in the existing project; LoRA retains
[iolv9hr3](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/iolv9hr3).

The persistent plan is
`runs/noun-verb-lora1500-full250-sequence-20261102T214703Z/plan.json`.
`runs/noun-verb-sequence-latest.txt` points to its directory, with live
`status.json` and `sequence.log`. Run-local preflight, memory, watchdog,
loss/gradient agreement, class diagnostics, raw rollouts, and saved checkpoints
provide evidence for the eventual training audit. Completion requires the
full run to reach 250 and have complete checkpoint and final model shards.

Tests cover identical task/reward configurations, six-sample class metadata,
checkpoint completeness, preservation of older adapters when publishing final,
process ownership, memory-only fallback, and final-model completeness. A
simulated sequence exercises LoRA finalization followed by a six-batch capacity
failure and a complete four-batch run, without performing GPU work.

At 21:54 UTC, a separate check instantiated the actual installed Unsloth/TRL
GRPO configuration for both candidates. Batch six resolves to six generations,
generation batch six, accumulation one, and one whole-group policy chunk.
All 28 requests rendered identically to the frozen LoRA training requests.
Prompt lengths range from 73 to 115 tokens, so the queued worst-case capacity
probe must exercise 307 total tokens (115 prompt plus 192 completion). This
configuration/tokenization check loaded no policy weights and did not establish
six-batch memory capacity. Its evidence is `training-stack-validation.json` in
the sequence directory; the actual GPU probe still waits for LoRA to exit.

## Completed-run verification

`python -m dat_grpo.verify_training --run-dir <completed-full-run> --step 250
--reference-run runs/qwen38-27b-dat-noun-verb-lora-20261102T170042Z` is a
read-only check to run from the repository root after full training finishes.
It checks the actual saved optimizer counter and per-parameter counters, every
loss/gradient agreement record, complete archived models, exact final/checkpoint
agreement, original backbone tensor shapes, and finite values across every saved
weight tensor. Separate probes compare original and trained embedding, output
head, attention, and MLP weights. The reward audit recomputes each completed
rollout using the frozen official DAT cache and pinned class verifier, checks
prompt/group metadata, and excludes any uncommitted post-checkpoint rollout.
This is training verification, not a capability benchmark.

The verifier passed five targeted tests for BF16/FP32 finite values, NaN and
infinity rejection, truncated shards, completed-group selection, optimizer
counters, agreement-log coverage, and actual weight differences. Its component
comparison was also checked against the earlier noun-only full model; those
results are explicitly marked as checker validation, not new full-run results.

At 22:18 UTC, the real LoRA step-500 archive passed this check: 992 FP32
tensors, 116,727,808 parameters, all finite. Recomputed rewards agreed for all
2,000 completed samples: 972 noun and 1,028 verb completions across all seven
templates, with zero adjective training prompts. The rendered dataset hash was
unchanged. Evidence is `lora-checkpoint500-verification.json` in the sequence
directory. The full conditional 250-step run remains pending.

At 23:32 UTC, checkpoint 1,000 also passed verification. All 992 AdamW state
entries were at step 1,000 with finite moments, the constant learning rate
remained 1e-6, and the checkpoint recorded the requested 500-update save
interval. The checkpoint and archived adapter hashes matched; the archived
adapter was finite and differed from step 500. Reward recomputation agreed for
all 4,000 completed samples: 2,008 noun and 1,992 verb completions across all
seven templates, with no adjective training samples. Evidence is
`lora-checkpoint1000-verification.json` in the sequence directory. LoRA continues
toward the controller-selected stop at 1,500; the full run has not started.

## Handoff and full batch-six launch, 3 November

LoRA saved checkpoint 1,500 and stopped at 01:20 UTC. The preserved pre-finalization
status confirms exactly 1,500 completed optimizer updates before the planned
interrupt. The final adapter matches the selected checkpoint, and snapshots 500,
1,000, and 1,500 remain available along with the earlier archives. The completed
LoRA audit verified 992 finite FP32 adapter tensors and all 6,000 completed
training rewards: 3,056 noun and 2,944 verb completions, across all seven templates,
with no adjective training samples. The run-local
`completed-training-verification.json` contains the audit.

The six-sample memory probe passed at 01:23 UTC. Its actual single policy forward
was `[6, 307]`, including 115 prompt and 192 completion tokens. Backward covered
851 gradient-bearing tensors; peak CUDA allocation was 75.694 GiB. The frozen
reference remained resident, the probe made zero optimizer updates and changed
no weights, and the independent watchdog recorded at least 38.652 GiB available
host memory with a normal exit. Batch four was not needed.

The real full run then launched from the original backbone:
`runs/qwen38-27b-dat-noun-verb-full-b6-20261102T214703Z`.
W&B: [s1rlnaat](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/s1rlnaat).
Its recorded 27,356,728,560 trainable parameters equal the total policy parameter
count. The policy is unquantized, and its NF4 reference has zero trainable
parameters. Task, reward, DAT provenance, verifier, and rendered dataset
manifests match LoRA exactly. Learning rate remains 1e-6, with six generations
per group, accumulation one, and resume checkpoints every 25 updates.

The first two real updates passed both loss-replay and gradient-norm agreement
checks; the largest relative norm difference was 0.000322. Each norm/update pass
used a single six-row policy forward. Actual gradients and updates covered
851 tensors (26,895,998,464 parameters; the unused vision path receives no
gradient on text requests). Reward recomputation matched all twelve initial
completions, six noun and six verb samples. The observed CUDA peak was 74.881 GiB,
and the watchdog had at least 37.805 GiB host headroom. These integration checks
are saved in `full-training-integration-validation.json` in the sequence directory
and `training-integration-validation.json` in the full run.

At 01:30 UTC, three full updates had completed. The 250-update target and final
model audit remain pending; these initial checks do not establish completion or
any capability/holdout improvement.

## First resumable full checkpoint

Checkpoint 10 finished saving at approximately 01:47 UTC. Verification at 01:48
confirmed 55 complete weight shards, 1,184 tensors, and 54,715,060,192 weight bytes.
The saved optimizer and all active per-parameter counters were at step 10;
1,397 optimizer-statistic tensors were finite FP32 values. Scheduler state was
also at 10. Both loss-replay and gradient-agreement records covered every update.

Independent comparisons with the original backbone found actual numerical
changes in sampled embedding, output-head, attention, and MLP weights: changed
fractions were 33.20%, 12.30%, 16.46%, and 27.59%, respectively. This establishes
real parameter updates in those components, without making a capability claim.
Recomputed rewards matched all 60 samples (24 noun and 36 verb completions),
covering all seven wordings and no adjective training requests. The frozen
dataset hash is unchanged. Evidence is `full-checkpoint10-verification.json`
in the sequence directory and `checkpoint10-verification.json` in the run.
Training remains on batch six with the 250-update target still pending.

## Step-50 model archived

At approximately 03:01 UTC, checkpoint 50 and its independent model archive
finished saving. Verification at 03:02 found the same 55-shard, 1,184-tensor
full-model schema, with every archived weight shard hard-linked to the saved
checkpoint. The archive therefore survives normal resume-checkpoint rotation.
Optimizer and scheduler counters were at 50, optimizer statistics remained finite
FP32, and loss/gradient agreement checks covered all 50 completed updates.
All 300 rewards recomputed correctly: 150 noun and 150 verb completions across
all seven wordings, with no adjective training requests. The rendered dataset
hash remains unchanged. Evidence is `full-checkpoint50-verification.json` in the
sequence directory and `checkpoint50-verification.json` in the full run.

Training continued to step 51 with batch six and over 34.7 GiB minimum observed
host-memory headroom. The 250-update target, later model snapshots, and final
full-model audit remain pending.

## Formatting and thinking-mode regression observed during training

At approximately 03:23 UTC, inspection of the longer updates showed that all
six completions were reaching the 192-token limit. The issue was behavioral,
not a host-memory or numerical failure: available host memory remained over
37 GiB, losses and norms were finite, and the watchdog had no stop condition.

| Completed updates | Formally valid completions | Mean composite reward | Completions starting with `<think>` |
| --- | ---: | ---: | ---: |
| 1–10 | 54/60 | 0.3800 | 0/60 |
| 11–20 | 53/60 | 0.3763 | 0/60 |
| 21–30 | 59/60 | 0.6734 | 0/60 |
| 31–40 | 46/60 | 0.1230 | 0/60 |
| 41–50 | 30/60 | -0.6445 | 0/60 |
| 51–60 | 0/60 | -2.0000 | 58/60 |

Extra JSON closing brackets began appearing before step 50. Immediately after
the step-50 save, outputs switched to visible reasoning, despite the frozen
requests retaining their `<think>\n\n</think>\n\n` non-thinking suffix. The baseline
and saved model retain the same EOS token set. This does not yet establish
whether the shift is weight-mediated or involves hidden runtime state; the
save boundary is a correlation, not proof of causation. Invalid JSON produces
no parsed word list, so the resulting zero POS-correctness metric is not an
independent test of linguistic POS knowledge. When all six rewards equal -2,
their relative DAT advantages are zero, leaving no within-group DAT learning
signal on that update. The KL term remains part of the loss.

Complete recovery copies of checkpoints 25 and 50 have been retained, including
weights, optimizer, scheduler, trainer state, and RNG state. Their model shards
are hard-linked, and an additional permanent step-25 model archive is retained.
These copies survive the regular two-checkpoint rotation. The run and its reward
recipe are unchanged. Evidence is `early-recovery-checkpoints.json` and
`format-regression-through60.json` in the sequence directory.

A controlled resume from checkpoint 75 is planned to test a fresh runtime with
the same saved weights, optimizer, RNG, and configuration. The full target
remains 250. Read-only generation-setting snapshots have been added at session
start and after saves; they take effect in the fresh worker and do not change
sampling or optimization. The plan is `diagnostic-resume-plan.json` in the
sequence directory. The restart and diagnosis are pending at this point.

## Controlled restart from full checkpoint 75

Checkpoint 75 and its permanent model archive finished saving before the old
worker was stopped at 04:08 UTC. A complete recovery copy also retains the
optimizer, scheduler, trainer state, and RNG state. The fresh worker resumed
that exact checkpoint at 04:09, reusing W&B run `s1rlnaat`, the unchanged
configuration hash, and the original 250-update target. It restored optimizer
step 75 and class counters of 234 verb and 216 noun completions. Batch six,
learning rate 1e-6, reward, and all sampling settings remain unchanged.

The actual trainer generation configuration records temperature 1, top-p 1,
top-k 0, a 192-token limit, and EOS IDs `[248046, 248044]`. Actual policy-input
token traces confirm the closed-thinking prompt suffix and six-row groups.
The model's serialized generation defaults are recorded separately and are
not the effective trainer sampling configuration.

| Completed update | Class | Formally valid | Starting with `<think>` | Mean composite reward |
| --- | --- | ---: | ---: | ---: |
| 76 | Verbs | 0/6 | 6/6 | -2.0000 |
| 77 | Verbs | 6/6 | 0/6 | 0.7086 |
| 78 | Nouns | 6/6 | 0/6 | 0.8117 |

The first resumed group remained invalid; four outputs included a word list
after empty thinking tags, and two contained extended reasoning. The next two
groups recovered bare JSON. Both had 59/60 words accepted by the frozen class
verifier. These twelve valid outputs are early evidence of recovery, not proof
that it persists. A fresh runtime and a subsequent KL update both precede the
recovery, so this observation does not isolate the regression's cause.

Independent reward recomputation matched all 468 canonical completions through
step 78, including all seven wordings and no adjective training requests.
Evidence is `diagnostic-resume75-results.json` in the sequence directory, with
generation snapshots under the run's `sessions` directory and actual token
traces in `policy-inputs.jsonl`. Training continues toward 250; later snapshots
and final full-model verification remain pending.

A separate checkpoint-75 audit at 04:27 confirmed all 55 shards and 1,184
weight tensors. The permanent archive and recovery copy share the exact saved
weight files; all optimizer, scheduler, trainer, and RNG files are preserved in
the recovery copy. The outer optimizer and all 851 active parameter optimizers
are at update 75, the scheduler is at 75, and all 1,397 floating optimizer-state
tensors are finite FP32. Loss and gradient agreement records pass for all 75
updates. Evidence is `full-checkpoint75-verification.json` in the sequence
directory and `checkpoint75-verification.json` in the training run.

## Full checkpoint 100 and behavior after saving

Update 100 completed at 04:54 UTC; its full save and permanent model archive
finished at approximately 04:58. Independent verification confirmed 55 shards,
1,184 weight tensors, and 54,715,060,192 weight bytes, with each archived shard
sharing the saved checkpoint file. The outer optimizer and all 851 active
parameter optimizers are at step 100, as is the scheduler. All 1,397 floating
optimizer-state tensors are finite FP32. Loss-replay and gradient-agreement
records pass for all 100 completed updates.

All 600 canonical training rewards independently recomputed correctly: 306 verb
and 294 noun completions across all seven wordings, with no adjective requests.
The rendered dataset hash remains unchanged. Evidence is
`full-checkpoint100-verification.json` in the sequence directory and
`checkpoint100-verification.json` in the full run.

After the first failed resumed update, updates 77–100 produced 141/144 formally
valid lists, with mean composite reward 0.7436 and one thinking-prefixed output.
The last ten updates before the save had 59/60 valid lists, no thinking prefixes,
and mean composite reward 0.7672. These are training observations, not held-out
DAT or capability results.

The generation-state snapshot after saving 100 matches the trainer's pre-save
sampling settings exactly. The model remains in training mode with activation
checkpointing enabled. The first three updates after saving produced:

| Completed update | Class | Formally valid | Starting with `<think>` | Mean composite reward |
| --- | --- | ---: | ---: | ---: |
| 101 | Verbs | 4/6 | 0/6 | -0.1754 |
| 102 | Nouns | 5/6 | 0/6 | 0.3457 |
| 103 | Verbs | 6/6 | 0/6 | 0.7847 |

The three failures were two lists with non-English entries and one malformed
JSON response. The earlier immediate all-group thinking failure did not recur
at this save boundary; this does not identify its original cause or prove that
later behavior will stay stable. Evidence is
`post-checkpoint100-generation-verification.json` in the sequence directory.
W&B independently reports the resumed run as running, with latest checkpoint
100 and target 250. At 05:04 UTC, update 104 was complete and the original
250-update target, later snapshots, and final full-model audit remained pending.

## Halfway recovery checkpoint 125

The halfway checkpoint finished saving at approximately 05:42 UTC. Verification
confirmed 55 complete shards and 1,184 weight tensors. Optimizer and scheduler
counters are at 125, all 1,397 floating optimizer-state tensors are finite FP32,
and both numerical agreement checks pass for all 125 updates. All 750 rewards
recomputed correctly: 384 verb and 366 noun completions, all seven wordings,
and no adjective training requests. Dataset provenance remains unchanged.

The regular checkpoint directories for 25, 50, and 75 have now rotated out.
Their independent recovery copies still contain full weights, optimizer,
scheduler, trainer state, and RNG state. Each recovery model shares the exact
weight files with its permanent archive. Permanent snapshots 50, 75, and 100
also remain structurally complete. Evidence is
`full-checkpoint125-verification.json` in the sequence directory and
`checkpoint125-verification.json` in the training run.

Updates 101–125 produced 146/150 formally valid lists, no thinking prefixes,
and mean composite reward 0.7076. Sampling configuration is unchanged after
the save, with training mode and activation checkpointing still enabled. The
first post-save update, 126, produced six valid lists and no thinking prefixes,
with mean composite reward 0.8412. W&B confirms the run is running with latest
checkpoint 125 and target 250. Training continues on batch six; permanent
snapshots 150/200/250 and the final 250-update verification remain pending.

## Permanent full-model snapshot 150

Verification at 06:25 UTC confirmed checkpoint 150 and its independent
permanent archive: 55 shards, 1,184 weight tensors, and 54,715,060,192 weight
bytes. Every archived weight shard shares the checkpoint file. Optimizer and
scheduler counters are at 150, all 1,397 floating optimizer statistics are
finite FP32, and loss/gradient agreement records pass for all 150 updates.
Every parameter optimizer retains the selected learning rate of 1e-6.

The pinned source configuration hash remains unchanged, and direct comparison
with LoRA confirms identical backbone, task, and reward settings. All 900
training rewards recomputed correctly: 480 verb and 420 noun completions across
all seven wordings, with no adjective training requests. Evidence is
`full-checkpoint150-verification.json` in the sequence directory and
`checkpoint150-verification.json` in the training run.

Updates 126–150 produced 145/150 formally valid lists, zero thinking prefixes,
and mean composite reward 0.7052. Updates 141–150 had 57/60 valid lists and mean
reward 0.6483. These overlapping training windows include format failures and
mix the two requested word classes; they are not held-out DAT measurements.
The post-save generation configuration is unchanged, with training mode and
activation checkpointing enabled. Training continues on batch six toward the
original 250-update target. Permanent snapshots 200/250 and final verification
remain pending.

## Recovery checkpoint 175 and word-class diagnostics

Verification at 07:08 UTC confirmed the full step-175 recovery checkpoint:
55 shards, 1,184 weight tensors, optimizer/scheduler counters at 175, and all
1,397 floating optimizer statistics finite FP32. Both numerical agreement
checks pass for all 175 updates. All 1,050 rewards recomputed correctly:
564 verb and 486 noun completions across all seven wordings, with no adjective
training requests. Permanent archives 50/75/100/150 remain complete. The
learning rate is still 1e-6, and sampling configuration is unchanged after
saving. Evidence is `full-checkpoint175-verification.json` in the sequence
directory and `checkpoint175-verification.json` in the training run.

Updates 151–175 produced 147/150 scoreable lists, no thinking tags anywhere in
the outputs, mean composite reward 0.7112, and mean formal DAT score 81.68 among
scoreable lists. Format validity and dictionary word-class acceptance differ:

| Requested class | Scoreable lists | Mean DAT among scoreable lists | Words accepted in scoreable lists | Scoreable and all ten accepted, all outputs |
| --- | ---: | ---: | ---: | ---: |
| Nouns | 66/66 | 84.37 | 99.85% | 65/66 (98.48%) |
| Verbs | 81/84 | 79.50 | 95.56% | 57/84 (67.86%) |

The frozen WordNet verifier accepts a word with any requested noun/verb sense.
Of the words in scoreable lists, 75.92% have both senses, including 83.95% in
verb lists. These acceptance figures therefore do not disambiguate the intended
sense of an isolated word. Invalid lists contribute zero accepted requested
slots when no word list was extracted. A parsed list can retain class checks
while failing DAT's dictionary/uniqueness gate and receiving reward -2. The
word percentages in the table exclude all DAT-unscoreable lists; its last
column requires both scoreability and ten accepted words. These are training
observations, with adjective and capability results still unmeasured for this run.

W&B confirms checkpoint 175, a running job, and target 250. Update 176 completed
after the save. Training continues on batch six; permanent snapshots 200/250
and final verification remain pending.

## Permanent full-model snapshot 200

Verification at 07:53 UTC confirmed the full step-200 checkpoint and permanent
archive: 55 shards, 1,184 weight tensors, and exact shared weight files. Optimizer
and scheduler counters are at 200, all 1,397 floating optimizer statistics are
finite FP32, and both numerical agreement checks pass for all 200 updates.
Learning rate remains 1e-6. Direct comparison confirms the original LoRA
backbone/task/reward configuration, the source configuration hash is unchanged,
and the actual sampling settings remain unchanged after saving.

All 1,200 rewards recomputed correctly: 636 verb and 564 noun completions, all
seven wordings, and no adjective training requests. Permanent model archives
50/75/100/150/200 remain complete. Evidence is
`full-checkpoint200-verification.json` in the sequence directory and
`checkpoint200-verification.json` in the training run.

Updates 176–200 produced 148/150 DAT-scoreable lists, no thinking tags anywhere
in the outputs, mean composite reward 0.7852, and mean DAT score 87.28 among
scoreable lists. Word acceptance uses the same frozen WordNet any-sense rule:

| Requested class | Scoreable lists | Mean DAT among scoreable lists | Words accepted in scoreable lists | Scoreable and all ten accepted, all outputs |
| --- | ---: | ---: | ---: | ---: |
| Nouns | 77/78 | 89.02 | 97.40% | 62/78 (79.49%) |
| Verbs | 71/72 | 85.40 | 97.61% | 55/72 (76.39%) |

The noun output at update 182 passed all ten class checks but had fewer than
seven DAT-valid unique words; the verb output at update 198 failed the single
English-word gate. Both received -2. The report now labels dictionary acceptance
separately from joint DAT-scoreability/class compliance; this clarifies metrics
without changing the frozen reward. Among scoreable outputs in this window,
47.97% of word tokens have both noun and verb senses. These training windows
are not a paired baseline or adjective comparison.

W&B reports a running job, latest saved checkpoint 200, and target 250.
Training continues on batch six; the final 250-update snapshot and complete
saved-model audit remain pending.

## Definition-style formatting failure at update 204

Four of six responses to the verb `distant_meanings` wording included word
definitions inside list entries and ran into the 192-token limit before
closing valid JSON. The prompt explicitly requires exactly ten strings with
one word per string, alongside its instruction to account for different senses.
The optional class-only reminder was absent in this group. Two responses were
DAT-scoreable; all four malformed responses received -2. None contained
thinking tags. Actual token traces confirm the unchanged closed-thinking
prompt suffix and a six-row, 192-token completion tensor.

This is another instruction-following failure visible during training. The
group still contains differing rewards, so it retains a relative reward
learning signal. Settings and the 250-update target remain unchanged. Evidence
is `generation-style-observation-204.json` in the sequence directory; full
responses remain in the training rollout log.

Updates 207 and 208 each had only one scoreable response out of six. Their
noun requests used different wordings, with the extra reminder enabled in
207 and disabled in 208. The failed responses mostly contained extra closing
brackets or mismatched punctuation after their word lists. No thinking tags
appeared. Numerical loss/gradient agreement still passed through update 209,
with approximately 38.54 GiB host-memory headroom. The experiment continues
unchanged to 250; this format failure is not itself a measurement of retained
word-class knowledge. Evidence is `generation-style-drift-through208.json`.

## Recovery checkpoint 225 and later scoreability

Verification at 08:37 UTC confirmed the full step-225 recovery checkpoint:
55 shards, 1,184 weight tensors, optimizer/scheduler counters at 225, and all
1,397 floating optimizer statistics finite FP32. Both numerical agreement
checks pass for all 225 updates. All 1,350 rewards recomputed correctly:
696 verb and 654 noun completions, all seven wordings, and no adjective
training requests. The source configuration hash, learning rate 1e-6, and
post-save sampling configuration remain unchanged. Permanent model archives
50/75/100/150/200 remain complete. Evidence is
`full-checkpoint225-verification.json` in the sequence directory and
`checkpoint225-verification.json` in the training run.

Updates 201–225 produced 131/150 DAT-scoreable lists, no thinking tags, mean
composite reward 0.4350, and mean DAT score 87.83 among scoreable lists. The
19 failures comprised 16 invalid JSON responses, two failures of the single
English-word gate, and one response with the wrong number of words.

| Requested class | Scoreable lists | Mean DAT among scoreable lists | Words accepted in scoreable lists | Scoreable and all ten accepted, all outputs |
| --- | ---: | ---: | ---: | ---: |
| Nouns | 76/90 | 88.91 | 94.74% | 48/90 (53.33%) |
| Verbs | 55/60 | 86.33 | 96.55% | 41/60 (68.33%) |

The later window, updates 216–225, had 60/60 DAT-scoreable lists, no thinking
tags, mean composite reward 0.7895, and mean DAT score 87.29. Dictionary word
acceptance in that window was 95.83%; 40/60 lists jointly passed DAT and all
ten class checks. No process restart or recipe change occurred between 200
and 225. This recovery in scoreability does not erase the earlier failures or
establish a cause for either regression.

W&B reports a running job at 226, latest checkpoint 225, and target 250.
Training continues on batch six; final snapshot 250 and the full saved-model
verification remain pending. Adjective and capability results remain unmeasured
for this run.

## Completed 250 updates and final audit

The run completed exactly 250 optimizer updates at 09:19:39 UTC. Its worker,
guardian, and sequence coordinator have exited; the final worker's exit code
was zero. W&B confirms finished runs at LoRA 1500 and full 250, with matching
targets and latest checkpoint counters and no failure summary fields.

The final export matches checkpoint 250 and its permanent archive. All
27,356,728,560 saved parameter values were scanned and are finite, across 55
shards / 1,184 tensors. Optimizer and scheduler counters are 250, all 1,397
floating optimizer statistics are finite FP32, and all 250 numerical agreement
pairs passed. Actual forward logs confirm one six-row forward per pass for
each committed update. Independent probes find nonzero embedding, output-head,
attention, and MLP weight changes relative to the original backbone.

All 1,500 full-run rewards recomputed correctly: 804 verb and 696 noun requests,
all seven wordings, and no adjective training requests. The final 25 updates
produced 146/150 scoreable outputs, mean composite reward 0.7152, and mean DAT
score 86.63 among scoreable outputs. Dictionary word acceptance among those
outputs was 96.16%; 102/150 outputs jointly passed DAT and all ten class checks.
The four failures were three single-word gate failures and one invalid JSON
output. No thinking tags occurred in that final window.

Required full snapshots 50/75/100/150/200/250 are complete. LoRA snapshots
500/1000/1500 and its final adapter were scanned and contain finite values;
the adapter matches checkpoint 1500. The six-row capacity probe passed before
full training, and batch four was unnecessary. No training work remains for
this sequence. Adjective transfer and capability preservation remain separate
evaluation questions.

Evidence: the run's `completed-training-verification.json`, the sequence's
`completion-audit.json`, `wandb-completion-verification.json`, and
`training-outcome-metrics.json`. The [final results report](noun-verb-full250-results-2026-11-03.md)
summarizes the complete trajectory and retained model locations.
