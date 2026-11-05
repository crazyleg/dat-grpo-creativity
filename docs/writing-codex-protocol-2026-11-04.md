# Direct story ranking with Codex Sol

All 128 ordered judgments completed on 4 November at 11:24 UTC. The
[completed results](writing-codex-results-2026-11-04.md) include score intervals,
order checks, a downloadable full-story ranking, and the final audit.

Started on 4 November 2026. The user requested independent 0–10 metrics and
overall A/B and B/A preferences for the full saved baseline/Full250 story panel,
using non-interactive Codex with Sol at medium reasoning effort.

The selected model is `gpt-6.1-sol`, explicitly configured with medium reasoning.
Codex CLI 0.166.0 runs on the Spark using the existing ChatGPT-managed Codex
login for hosted inference. The first real judgment returned valid structured
scores and reported reasoning-token usage.
[Codex non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode).

## Panel and inputs

The panel contains 64 matched cases from 32 briefs with two generation samples
each. Baseline and conditional Full250 contribute one story per case, giving
128 source stories. Every pair is evaluated in both orders, for 128 fresh judge
sessions. The judge input contains the shared required elements and the two
cleaned story texts. Model identity is hidden in each input.

The source is `benchmarks/results/writing-transfer-20261103T185135Z/`. Input
hashes pin the case file and both source response files. The prepared manifest
contains the complete rubric and schema. Each request has its own prompt hash.
The earlier Opus descriptive responses are outside these judge inputs.

Story wrappers and numeric cumulative counters are stripped using the existing
extraction function. Source output lengths and incomplete endings are retained.
Baseline has 23 truncated stories and Full250 has four across the panel. The
judge is instructed to evaluate the text supplied and to avoid rewarding length
or familiarity alone.

## Metrics and preferences

Each story receives the following seven 0–10 scores with at most one decimal
place. Shared anchors are 0 unusable, 1–2 severely weak, 3–4 weak, 5–6 competent,
7–8 strong, and 9–10 exceptional.

| Metric | Definition |
| --- | --- |
| Creativity | Originality of ideas, meaningful novel associations, and inventive construction |
| Imagery | Freshness, specificity, and effectiveness of sensory or figurative images |
| Style | Control of voice, diction, syntax, rhythm, and fit to the chosen mode |
| Coherence | Internal consistency and meaningful causal, emotional, symbolic, or formal connections |
| Character and impact | Character specificity, motivation, agency, and emotional or intellectual effect |
| Integration | Purposeful use of the ten required elements in action, setting, imagery, or structure |
| Overall quality | Holistic literary effectiveness for the brief, considering strengths and weaknesses together |

The prompt permits plain, realist, lyrical, surreal, fragmented, and ambiguous
work. It asks the judge to distinguish control and effective pattern from
apparent accidental disconnection. It allows transformed realizations of a
required element when that element remains clearly identifiable.

The judge also declares an overall winner, A, B, or TIE, and supplies a
comparative rationale for every metric. One exact excerpt from each story and
reading cautions are retained. Overall preference comes from the judge's
explicit holistic choice. The overall score is also holistic.

For the six component metrics, score differences of at least 0.5 favor the
higher score, and smaller absolute differences give a tie. Win credit is 1 for
a Full250 win, 0.5 for a tie, and 0 for a baseline win. Each case averages both
orders before aggregation. The overall-quality win credit uses the explicit
winner field. Overall scores and overall preferences can therefore differ at
the level of rounded scores, which should be inspected in the saved records.

Reports include model means, paired Full250-minus-baseline differences, win
credits, A/B/tie counts, and agreement between reversed orders. Paired 95%
intervals use 5,000 bootstrap resamples of the 32 brief clusters, seed 20261103,
keeping both generated samples and both orders within their brief. There is
no multiple-comparison adjustment or human literary calibration.

## Ranking all source stories

Each of the 128 source stories receives the mean of its two presentation-order
scores. The story ranking sorts by mean overall quality and gives equal scores
the same competition rank. All seven scores remain available for each story.
The ranking spans different briefs and describes this judge's rubric scores.
Differences in brief difficulty and small score gaps limit interpretation of
individual rank positions. The paired model comparison controls the shared
brief within each case.

## Session isolation and validation

Each CLI call uses a fresh ephemeral session in an empty temporary directory,
read-only permissions, explicit model and effort settings, and a structured
output schema. Parent thread metadata is removed from the child environment.
User configuration and project-document injection are disabled for these judge
calls. Shell, delegation, apps, plugins, skill search, memories, hooks, and web
search are disabled. Native event logs are checked for unexpected tool use.

Four sessions run concurrently. The wrapper saves each CLI command, exact
prompt, native JSONL events, stderr, raw answer, token usage, and validated
record. Completed answers are reused by request hash on resume. Invalid scores,
unexpected tools, or incomplete CLI turns trigger a review stop. Ranking and
final model summaries require all 128 ordered judgments.

The non-interactive entry point in this CLI is `codex exec`. The effective
model selection includes `--model gpt-6.1-sol` and
`-c 'model_reasoning_effort="medium"'`. The full per-call command is archived.

## Monitoring and reproducibility

Campaign directory is
`benchmarks/results/writing-codex-sol-medium-20261104T105907Z/`, referenced by
`benchmarks/results/writing-codex-latest.txt`.

The run publishes `results/paired_scores`, `results/story_ranking`, and
`results/ordered_judgments` after complete coverage, alongside progress during
execution. The ranking table includes the required elements and actual story
text. A downloadable `story-ranking.csv` contains all 128 ranked stories and
their seven scores.
[Codex judge W&B run](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/lqteia18).

Configuration is
[benchmark_writing_codex.json](../configs/benchmark_writing_codex.json).

```sh
.venv-inference/bin/python -m dat_grpo.benchmark.writing_codex \
  --config configs/benchmark_writing_codex.json \
  --output benchmarks/results/writing-codex-sol-medium-20261104T105907Z
```

Local regression tests cover full-panel preparation and exact story reversal,
independent direct-story input, explicit Sol medium settings and context
isolation, score validation, score signs, position dependence, separate overall
preferences, incomplete coverage, shared ranks for ties, cached answer reuse,
and rejection of unexpected tools.
