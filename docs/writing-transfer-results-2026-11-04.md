# Writing and style transfer results

Recorded on 4 November 2026 for the completed Baseline, conditional noun/verb
LoRA1500, and conditional noun/verb Full250 comparison. All 768 writer responses
and 1,600 judge responses are saved. Nine of nine task/model comparisons are
complete. Generation finished on 3 November at 21:59 UTC, followed by judging
at 23:29 UTC. Mazur aggregation was repaired on 4 November using the saved
judgments.

The main run contains generation diagnostics, final score tables, scalar
summaries, and the Mazur judge reliability warning.
[Main W&B run](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/zcohczve).

The separate scoring run retains the local judge tables and provenance.
[Judging W&B run](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/2gjxnf0j).

## Matched setup and coverage

Baseline is the shared BF16 unpacking of the pinned original NF4 backbone,
served without an adapter. LoRA1500 uses the conditional noun/verb adapter at
optimizer step 1500 on that backbone. Full250 loads the conditional noun/verb
full-weight checkpoint at step 250 without an adapter. The earlier noun-only
checkpoints are excluded from this comparison.

| Panel | Cases per model | Writer responses across three models | Judge responses |
| --- | ---: | ---: | ---: |
| Mazur, 32 briefs with two samples each | 64 | 192 | 256 |
| WritingBench, English style panel | 64 | 192 | 960 |
| SteerBench, two instructions for each of 64 sources | 128 | 384 | 384 |
| Total | 256 | 768 | 1600 |

All writer requests explicitly disable thinking. Each model receives the same
fixed cases and paired seeds. First-attempt responses are retained, including
format failures, excessive lengths, and truncations. The frozen case-file
SHA-256 is `26def4b859de6a58af674c06b64eb284b90b490fc4ff5ae9f4d3cb8136a79a2e`.

| Panel | Temperature | Top-p | Top-k | Completion-token cap |
| --- | ---: | ---: | ---: | ---: |
| Mazur | 1.0 | 1.0 | unrestricted | 2048 |
| WritingBench | 0.7 | 0.8 | 20 | 4096 |
| SteerBench | 0, greedy | 1.0 | unrestricted | 4096 |

The local vLLM deployment uses BF16 weights, a 16,384-token context, 24 GiB of
KV cache, and up to 64 concurrent requests. Exact model paths, source revisions,
selected cases, sampling details, and scorer settings are retained in the
[execution protocol](writing-transfer-protocol-2026-11-03.md) and
[configuration](../configs/benchmark_writing_transfer.json).

## WritingBench

The pinned downloadable WritingBench Qwen2.5-based 7B critic evaluates all five
query-specific criteria, with style-specific criteria also averaged separately.
Scores range from 1 to 10. The six-domain, 64-case panel is a fixed English
style subset. These scores use our local critic configuration.

| Model | Requested cases | Fully scored cases | Overall mean | Style-scored cases | Style mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 64 | 62 | 8.0645 | 64 | 8.4609 |
| LoRA1500 | 64 | 62 | 8.1065 | 64 | 8.6016 |
| Full250 | 64 | 63 | 8.0508 | 64 | 8.3984 |

Five of the 960 criterion judgments contain malformed JSON and retain missing
scores. They affect two baseline cases, two LoRA cases, and one full-model case.
Every style criterion parsed successfully. Overall means above use each model's
fully scored cases. Paired overall differences below use the 61 cases fully
scored for both members of each comparison, so they can differ from subtraction
of the displayed overall means.

## SteerBench

The pinned upstream metric functions evaluate requested changes in reading
grade, MTLD lexical diversity, word count, and formality. Steering error measures
distance from the target. Miscalibration measures error along the requested
direction of change. Side-effect error measures orthogonal drift. Lower is
better for all three metrics.

| Model | Cases | Steering error | Miscalibration | Side-effect error |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 128 | 0.5387 | 0.7769 | 0.5244 |
| LoRA1500 | 128 | 0.5503 | 0.7860 | 0.5270 |
| Full250 | 128 | 0.5633 | 0.8614 | 0.4638 |

The separate Phi4 content-preservation check parsed all 128 judgments per model
and marked every output preserved, giving 100% for all three models. This broad
check provides little discrimination on the panel. Its flags alone cannot
establish semantic fidelity.

The isolated CPU metric environment uses the same packages and normalization
bounds for all variants. Calibration against the 64 released source texts
matches reading-grade, word-count, and formality values exactly. MTLD differs
by a mean absolute 0.1425 and maximum absolute 6.6061 units. Pyphen-only English
syllabification matches the released reading-grade measurements. Dependency
versions and calibration details remain saved with the run.

### Side-effect normalization and stationary rewrites

A follow-up inspection on 4 November reconstructed the upstream metrics from
the saved per-output goal coordinates. The normalized side-effect score divides
orthogonal error by total measured movement, with numerical stabilization.
If a rewrite stays in the source bins on all four attributes, its side-effect
score is almost zero even when the requested change is missed. Every request
in this panel has nonzero requested movement.

| Diagnostic | Baseline | LoRA1500 | Full250 |
| --- | ---: | ---: | ---: |
| No measured movement across the four attribute bins | 32/128 | 34/128 | 46/128 |
| Mean measured movement magnitude | 0.3811 | 0.3838 | 0.3395 |
| Mean unnormalized orthogonal drift | 0.2616 | 0.2727 | 0.2444 |
| Mean normalized side-effect error among moving outputs | 0.6991 | 0.7175 | 0.7238 |

The stationary outputs have mean side-effect scores around 0.0004 and
miscalibration around 1.0 in every model. Their increased prevalence in Full250
pulls its overall side-effect mean downward. Among outputs with measured
movement, Full250 has a higher mean normalized side-effect score, although
these conditional subsets contain different cases and support descriptive
comparison only.

Remaining in the same bins allows changes to wording and changes within each
bin. This diagnostic alone cannot establish literal copying or unchanged
prose. Full250's lower published side-effect error should therefore be read
together with its greater frequency of missed movement and worse calibration.
The corresponding upstream functions are `get_dist_from_source`,
`get_orthogonality`, and `get_miscalibration` in the pinned SteerBench source.

## Paired differences and uncertainty

Deltas are adapted model minus baseline. Intervals are percentile bootstrap
95% intervals from 5,000 resamples with seed 20261103. WritingBench resamples
paired case IDs. SteerBench resamples the 64 source-text clusters, keeping both
instructions for each source together. These intervals have no
multiple-comparison adjustment.

| Panel and metric | Model | Paired items | Clusters | Delta | 95% interval |
| --- | --- | ---: | ---: | ---: | --- |
| WritingBench overall | LoRA1500 | 61 | 61 | +0.0361 | [-0.0525, 0.1279] |
| WritingBench style | LoRA1500 | 64 | 64 | +0.1406 | [0.0000, 0.2891] |
| WritingBench overall | Full250 | 61 | 61 | -0.0557 | [-0.2852, 0.1344] |
| WritingBench style | Full250 | 64 | 64 | -0.0625 | [-0.2969, 0.1484] |
| SteerBench steering error | LoRA1500 | 128 | 64 | +0.0116 | [-0.0125, 0.0360] |
| SteerBench miscalibration | LoRA1500 | 128 | 64 | +0.0092 | [-0.0210, 0.0413] |
| SteerBench side-effect error | LoRA1500 | 128 | 64 | +0.0026 | [-0.0403, 0.0455] |
| SteerBench steering error | Full250 | 128 | 64 | +0.0246 | [-0.0083, 0.0613] |
| SteerBench miscalibration | Full250 | 128 | 64 | +0.0845 | [0.0294, 0.1514] |
| SteerBench side-effect error | Full250 | 128 | 64 | -0.0606 | [-0.1206, -0.0024] |

## Mazur comparative creativity

The independent-family local Phi4 judge receives the pinned upstream comparison
rubric, the required elements, and the stories with writer identities hidden.
Each adapted story is compared with the matched baseline story in both A/B
orders. Story wrappers and cumulative counters are removed for prose judging,
while raw outputs and compliance diagnostics remain available.

A win contributes 1, a tie 0.5, and a loss 0. Each story pair's credit averages
the two orders. The overall estimate averages credits over the 32 brief
clusters, with the two generated samples kept together for each brief.

| Model against baseline | Story pairs with both orders | Brief clusters | Win credit | Clustered 95% interval |
| --- | ---: | ---: | ---: | --- |
| LoRA1500 | 64/64 | 32 | 52.3438% | [46.8750%, 57.8125%] |
| Full250 | 64/64 | 32 | 40.6250% | [33.5938%, 48.4375%] |

### Parser recovery

The initial parser required closing tags and accepted zero Mazur judgments.
Phi4 emitted the requested section headings and explicit winner labels while
omitting closing tags. The corrected parser accepts a winner ending at the
next expected section heading and handles the winner independently of the
signed margin. All 256 winner labels are recoverable.

In 47 judgments, Phi4 declares B the winner but gives a positive signed margin.
Those margins remain invalid. No sign is inferred or corrected. A pair's mean
signed margin is available only when both orders have valid signed margins.
Original judge responses remain unchanged, and pre-correction aggregates are
archived in `judging/parser-recovery/`. Aggregation made no additional writer or
judge inference requests. Regression tests cover unclosed sections, conflicting
margins, ambiguous winners, and incomplete cache rejection.

### Position dependence

| Comparison | A winners | B winners | Ties | Same preferred story after reversal | Valid signed margins |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA1500 versus baseline | 111/128 | 17/128 | 0 | 17/64 (26.56%) | 111/128 |
| Full250 versus baseline | 98/128 | 30/128 | 0 | 28/64 (43.75%) | 98/128 |
| Combined | 209/256 | 47/256 | 0 | 45/128 (35.16%) | 209/256 |

The judge selects position A in 81.64% of judgments. Both-order averaging
balances exposure to A and B, but the low agreement after reversal limits the
reliability of the resulting ranking. Full250's lower observed win credit
therefore remains provisional. The local judge has no expert literary
calibration, and these intervals do not quantify systematic judge bias.

For matched pair `mazur/20`, Phi4 selects baseline when baseline is first,
praising character development, coherent structure, and resolution. After
reversal, it selects Full250 and praises character development, the ending,
and element integration. This shows that fluent explanations can accompany
order-sensitive verdicts. In `mazur/0`, both orders instead favor baseline,
while explicitly crediting Full250 for imaginative atmosphere, experimental
voice, symbolism, and originality. A holistic win can therefore conceal
advantages on particular creative dimensions.

## Generation diagnostics and instruction compliance

Every saved writer response is nonempty. No response is excluded because of
length or formatting. All structured reasoning fields are empty. Literal
thinking tags can still occur in the text, including two baseline Mazur outputs,
and are tracked separately from structured reasoning.

| Panel | Model | Cases | Token-limit truncations | Mean actual words |
| --- | --- | ---: | ---: | ---: |
| Mazur | Baseline | 64 | 23/64 (35.94%) | 1005.0 |
| Mazur | LoRA1500 | 64 | 21/64 (32.81%) | 1010.4 |
| Mazur | Full250 | 64 | 4/64 (6.25%) | 959.2 |
| WritingBench | Baseline | 64 | 0/64 | 1138.8 |
| WritingBench | LoRA1500 | 64 | 2/64 (3.12%) | 1120.4 |
| WritingBench | Full250 | 64 | 4/64 (6.25%) | 1161.2 |
| SteerBench | Baseline | 128 | 0/128 | 666.7 |
| SteerBench | LoRA1500 | 128 | 0/128 | 668.1 |
| SteerBench | Full250 | 128 | 0/128 | 707.4 |

Mazur requires 600–800 actual words, story tags, ASCII English, and cumulative
word counters after every sentence. The diagnostic verifies emitted counter
values against actual word prefixes. That check alone cannot establish a
counter after every sentence.

| Mazur compliance | Baseline | LoRA1500 | Full250 |
| --- | ---: | ---: | ---: |
| Complete story tags | 37/64 (57.81%) | 38/64 (59.38%) | 59/64 (92.19%) |
| Actual 600–800-word range | 7/64 (10.94%) | 5/64 (7.81%) | 9/64 (14.06%) |
| Correct cumulative counter checks | 0/64 | 0/64 | 0/64 |

All three models often exceed the requested length and fail the cumulative
counter check. Full250 improves tag completion and has fewer Mazur truncations,
but complete instruction compliance remains poor. Prose quality and task
validity must therefore be read together.

## Observed changes in writing habits

The same 64 matched Mazur outputs show a substantial change in recurring
openings and names. Story wrappers and cumulative counters are removed before
the opening comparisons.

| Indicator across 64 stories | Baseline | LoRA1500 | Full250 |
| --- | ---: | ---: | ---: |
| Opens with "The air" | 17 | 16 | 2 |
| Contains "Elara" | 31 | 28 | 13 |
| First 20 whitespace-separated tokens match baseline | 64 | 42 | 0 |

These indicators measure recurring writing habits and divergence from baseline.
They establish a style shift while leaving literary preference and quality
open. A reader can reasonably value the new imagery even when a holistic judge
prefers another story's structure or coherence. The current judge's position
dependence further limits comparison with a reader's preferences.

## Findings and limits

LoRA1500 remains close to baseline on the measured writing and steering panels.
Its WritingBench style mean rises by 0.1406, with a paired interval reaching zero.
Its overall WritingBench and all three SteerBench intervals include zero.

Full250 has worse SteerBench miscalibration and lower side-effect error, with
both paired intervals excluding zero before multiple-comparison adjustment.
Its WritingBench overall and style differences remain unresolved on this
panel. Reduced orthogonal drift and worse calibration can occur together, so
the steering results describe a mixed change in control.
Full250's greater frequency of stationary measured attributes also contributes
to its lower side-effect average. That average alone cannot establish better
control of the requested changes.

The observed Mazur win credits favor baseline over Full250, subject to the
position dependence and margin failures above. These results provide limited
evidence of broad creative-writing improvement. A substantial change in
generated prose alone cannot establish better creativity or constraint
following.

The compared training runs also differ in optimizer, group size, update count,
and total sampled completions. This checkpoint comparison cannot isolate the
effect of adapter training versus full-weight training. The local judges and
small fixed subsets also limit generalization to other writing tasks.

The resumed formal capability battery completed all 21 comparisons and 5,823
responses on 4 November. The earlier frozen results and all cached responses
passed preservation checks. Its results and the separate DAT screens are
retained in their existing reports.
[Completed formal capability results](formal-capabilities-final-2026-11-04.md),
[LoRA DAT results](noun-verb-lora1500-dat-results-2026-11-03.md),
[full-weight DAT results](noun-verb-full250-dat-results-2026-11-03.md).

The subsequent independent Sol-medium comparison completed all 64
baseline/Full250 story pairs in both orders. It found higher judged creativity
and imagery for Full250, lower coherence and nearly level overall quality.
The [completed Codex report](writing-codex-results-2026-11-04.md) records this
follow-up alongside its judge and generation limitations. The local-judge
results above remain available as the original measurement.

## Saved evidence

The campaign directory is
`benchmarks/results/writing-transfer-20261103T185135Z/`.

- `config.json`, `protocol.json`, and `assets/` retain the frozen setup and cases.
- Per-task/model `responses.jsonl`, `item-diagnostics.json`, and
  `server-launch.json` retain writer outputs, diagnostics, and serving identity.
- `generation-results.json` contains all nine generation summaries.
- `score-tables.json` contains the published comparison and diagnostic tables.
- `judged-results.json` contains parsed per-case scores and judge diagnostics.
- `judging/*/responses.jsonl` and `jobs.json` retain all 1,600 judge responses,
  exact prompts, writer-case mappings, and prompt hashes.
- `judge-model-sources.json` pins WritingBench critic revision
  `fb5f30a944174bd28042a4f19a1c1de86606a6cd` and Phi4 revision
  `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`.
- `steer-metrics.json` and `steer-metric-preflight.json` retain metric calibration
  and dependency versions. The repository also retains the
  [metric dependency lock](../configs/writing_metrics_requirements.txt).
- `judging/parser-recovery/` retains the original aggregates and recovery log.
- Parser recovery code and regression tests are recorded in commit `1d83e7b`.
