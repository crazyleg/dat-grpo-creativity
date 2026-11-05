# Completed direct-story ranking with Codex Sol

Completed on 4 November 2026 at 11:24 UTC. All 128 ordered comparisons are saved, covering 64 matched baseline/conditional Full250 story pairs from 32 briefs. Each of the 128 source stories was judged in both A and B positions. The judge was `gpt-6.1-sol` with medium reasoning through non-interactive Codex CLI 0.166.0 and the existing ChatGPT-managed login.

Every judgment used the two story texts and their shared required elements in a fresh isolated session. The seven-metric rubric was frozen before the first judgment. Earlier Opus descriptions were outside the input material.

## Model scores

All scores are on a 0–10 scale with higher values preferred. Each story averages its two order-specific scores before model means are calculated. Deltas are Full250 minus baseline. Paired 95% percentile intervals use 5,000 resamples of the 32 brief clusters with seed 20261103. Both generated samples and both presentation orders stay within their brief. Intervals are unadjusted for multiple comparisons.

| Metric | Baseline | Full250 | Delta | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Creativity | 5.2211 | 5.9211 | +0.7000 | [0.4563, 0.9492] |
| Imagery | 5.5922 | 6.2125 | +0.6203 | [0.3718, 0.8633] |
| Style | 4.6375 | 4.8500 | +0.2125 | [-0.1266, 0.5384] |
| Coherence | 5.1922 | 4.2953 | -0.8969 | [-1.2782, -0.5414] |
| Character and impact | 5.0500 | 4.8766 | -0.1734 | [-0.4977, 0.1430] |
| Element integration | 5.8859 | 5.6367 | -0.2492 | [-0.6211, 0.1266] |
| Overall quality | 5.1797 | 5.1273 | -0.0523 | [-0.3618, 0.2430] |

Full250 receives higher creativity and imagery scores, with both paired intervals above zero. Its coherence score falls, with the interval below zero. Style, character/impact, integration, and overall-quality intervals include zero. The overall mean is nearly level. These results support a shift toward judged originality and inventive imagery alongside reduced coherence on this fixed story panel.

## Pairwise preferences and order checks

Component-metric preferences use a half-point score-difference threshold. Smaller differences give a tie. Overall-quality preference uses the judge's explicit A/B/TIE choice. A Full250 win contributes 1, a tie 0.5, and a baseline win 0. Each case averages both orders.

| Metric | Full250 win credit | Clustered 95% interval | Agreement across reversed orders |
| --- | ---: | --- | ---: |
| Creativity | 74.22% | [64.84%, 83.21%] | 84.38% |
| Imagery | 72.66% | [63.28%, 81.64%] | 78.12% |
| Style | 56.64% | [46.09%, 66.42%] | 75.00% |
| Coherence | 32.03% | [23.44%, 41.02%] | 87.50% |
| Character and impact | 49.22% | [39.45%, 58.59%] | 65.62% |
| Element integration | 49.22% | [40.23%, 58.59%] | 75.00% |
| Overall quality | 51.95% | [41.02%, 62.89%] | 82.81% |

The raw counts below cover 128 ordered comparisons. Each matched story pair
contributes A/B and B/A presentations. Component wins use the half-point
score-difference rule, and overall wins use the explicit declared preference.

| Category | Full250 wins | Baseline wins | Ties |
| --- | ---: | ---: | ---: |
| Creativity | 83 | 21 | 24 |
| Imagery | 78 | 20 | 30 |
| Style | 60 | 43 | 25 |
| Coherence | 34 | 80 | 14 |
| Character and impact | 49 | 51 | 28 |
| Element integration | 47 | 49 | 32 |
| Overall quality | 65 | 60 | 3 |

For explicit overall preference, Full250 wins 65 ordered comparisons, baseline wins 60, and three are ties. Overall win credit is 51.95%, with a 95% interval of [41.02%, 62.89%]. This panel leaves the overall model preference unresolved.

Across the 64 story pairs, 28 favor Full250 in both orders, 25 favor baseline in both orders, eight reverse the preferred model, and three combine a decisive choice with a tie. Overall agreement is 53/64, or 82.81%. Position A is selected in 55/128 judgments, position B in 70/128, and ties in 3/128.

The earlier local Phi4 comparison had 43.75% order agreement and selected A in 98/128 judgments for Full250 versus baseline. Both judge and rubric changed for the Codex run. The new agreement figures therefore describe this protocol and cannot isolate the effect of changing model size.

## Ranking all 128 source stories

The full ranking sorts each story by its mean overall-quality score across both orders. Equal scores share a competition rank. The CSV includes all seven scores, required elements, and complete extracted story text. Rank positions span different briefs and should be interpreted as descriptive rubric scores. Brief difficulty and small score differences limit fine-grained cross-brief comparisons.

[Full ranking CSV](../results/raw/sol-story-judgments.tar.gz).

| Rank | Model | Case | Overall quality | Creativity | Coherence |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Full250 | mazur/35 | 7.00 | 7.35 | 6.15 |
| 2 | Full250 | mazur/42 | 6.90 | 7.20 | 6.50 |
| 3 | Full250 | mazur/21 | 6.85 | 7.80 | 5.80 |
| 4 | Full250 | mazur/18 | 6.75 | 7.15 | 6.25 |
| 5 | Baseline | mazur/35 | 6.70 | 6.80 | 7.10 |
| 6 | Baseline | mazur/40 | 6.65 | 6.35 | 7.20 |
| 6 | Full250 | mazur/40 | 6.65 | 7.85 | 5.05 |
| 8 | Full250 | mazur/15 | 6.60 | 6.60 | 6.45 |
| 8 | Full250 | mazur/20 | 6.60 | 6.85 | 5.65 |
| 10 | Full250 | mazur/13 | 6.50 | 6.50 | 6.25 |

## Validation and limitations

- All 128 responses pass the requested schema and numeric score checks. All recorded sessions pass the tool-use audit.
- Two of 256 evidence excerpts fail exact source matching after whitespace and outer-quotation normalization. They are retained in `audit.json` for manual inspection. The affected judgments are `mazur/16:0`, Story A, and `mazur/61:1`, Story A.
- There are zero cases in which a decisive numeric overall-score preference points to the opposite model from a decisive declared winner. Close-score and tie behavior remain separately inspectable.
- Source outputs include 23 baseline truncations and four Full250 truncations. All first-attempt texts remain in the comparison. This asymmetry can affect narrative closure and overall quality.
- Results come from one judge model and a newly specified rubric. Its literary preferences and numerical scale have no human-expert calibration. Bootstrap intervals quantify variation across these briefs.
- The actual trained checkpoints differ in optimizer, group size, update count, and sampled completions. The comparison leaves the causes of the observed changes unresolved.

## Assistant close reading after scoring

The assistant read six selected matched pairs in full, `mazur/0`, `mazur/4`,
`mazur/16`, `mazur/20`, `mazur/35`, and `mazur/42`, with model labels and previous
results visible. This is an unblinded interpretation of illustrative cases.
It supplies textual examples and leaves the prevalence of each pattern
unmeasured.

In `mazur/20`, baseline resolves hopelessness through an upward light and an
extended declaration of victory. Full250 introduces the mosaic sentence
"The fall is only a rehearsal. The rise is the lie." The failed fireworks
designer then argues against that proposition through his craft. His career
and the starless sky remain damaged at the end, while his act defines a new
relationship to them. "But he had defined the vertical" makes the required
gravity motif part of the character's response. This changes the story's
organizing idea and its emotional resolution.

In `mazur/42`, baseline captures a dream through a photographic mechanism and
stores it. Full250's librarian catalogs things that never arrived, and an
accurate photograph threatens the dream. Painting must transform the photograph
to preserve it. "She needed to lie beautifully, to fold light into a secret"
creates a tension among photography, art, truth, and memory. The required
palette and camera acquire a conceptual relationship that changes the action.
Both source stories in this pair finish normally.

The stronger Full250 passages create relationships among concepts and give
those relationships consequences. In other passages the associations continue
without stable referents. The closing part of `mazur/4` shifts through papers,
keys, coins, and an unexplained ball, losing the concrete transaction the
opening established. The assistant could not maintain a consistent reading
of the actions or their stakes there.

Baseline also contains genuine inventions. In `mazur/35`, the wall sustained
by consensus and architecture dissolved through paraphrase are substantive
ideas. Full250's version changes the emotional emphasis, but this pair gives
less reason to infer a dramatic increase in imaginative construction.

The judge's detailed creativity rationales explicitly recognize the changes
in pairs 20 and 42. Their paired creativity gains are +1.45 and +1.20 points,
respectively, while pair 4 falls by 1.40 points. A single global mean pools
different kinds and sizes of change. The score scale also lacks a calibrated
unit for the magnitude of a reader's creative experience.

The creativity-specific win credit is 74.22%, with a 95% interval of
[64.84%, 83.21%]. This is derived from creativity scores using the half-point
threshold, averaging both orders. The 51.95% explicit overall literary win
credit combines a wider set of judgments. The creativity rubric also asks
whether novel associations are meaningful, so realization and coherence can
influence that score alongside novelty.

## Saved evidence and monitoring

Campaign directory is `benchmarks/results/writing-codex-sol-medium-20261104T105907Z/`.

- `manifest.json`, `jobs.json`, and `schema.json` pin inputs, rubric, exact request text, and output structure.
- `judgments/*/launch.json`, `prompt.txt`, `events.jsonl`, `stderr.log`, `answer.json`, and `record.json` retain each CLI invocation and response.
- `results.json` contains model means, intervals, preferences, diagnostics, per-case scores, and the full ranking.
- `story-ranking.csv` contains the 128 scored stories with their required elements.
- `audit.json` retains excerpt flags, winner checks, ordered preference patterns, and native token usage.

The native CLI usage totals are 1,798,235 input tokens, including 433,152 cached input tokens, 142,572 output tokens, and 23,536 reasoning-output tokens. Billing follows the authenticated Codex account.

W&B publishes `results/paired_scores`, `results/story_ranking`, and `results/ordered_judgments`. The ranking table includes the story text, and the ordered-judgment table includes score rationales and quoted evidence. [Completed W&B run](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/lqteia18).

The [execution protocol](writing-codex-protocol-2026-11-04.md) documents the rubric and isolation settings. Earlier [Opus descriptive findings](writing-openrouter-results-2026-11-04.md) and [local-judge results](writing-transfer-results-2026-11-04.md) remain available as separate evaluations.
