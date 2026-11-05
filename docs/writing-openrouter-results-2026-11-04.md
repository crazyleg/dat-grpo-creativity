# OpenRouter style comparison findings

Recorded on 4 November 2026. These are partial descriptive results from the
baseline versus conditional Full250 story comparison using Claude Opus 5.5
through OpenRouter's Anthropic provider. The judge sees the two stories in
both orders with their shared required elements and hidden model identities.
The descriptive pass uses fresh contexts for each ordering.

## Status and coverage

| Item | Current value |
| --- | ---: |
| Requested story pairs | 64 |
| Requested descriptive comparisons | 128 |
| Saved descriptive comparisons | 57 |
| Story pairs with both orders saved | 28 |
| Additional pairs with one order saved | 1 |
| Distinct briefs represented | 15 |
| Briefs with both sampled story pairs complete | 14 |
| Schema-valid descriptive responses | 57/57 |
| Recorded inference cost | $4.024184 |
| Remaining key allowance | $5.975816 |
| Scoring calls completed | 0 |

The descriptive pass started at 10:28 UTC. It stopped at 10:35 UTC when
OpenRouter returned HTTP 402 for exhausted account credits. The account's
funding balance is separate from the key's $10 spending allowance. Every
completed response and its cost receipt remain saved. Resumption will reuse
these 57 responses and submit the missing requests.

The current table has explicit Story A and Story B source labels for readers.
Those labels are metadata added after judging. The request bodies preserve
anonymity. Open `results/style_differences` for profiles, contrasts,
uncertainties, and excerpt flags.
[W&B descriptions](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/i6tr8x0d).

## Summary of the available comparisons

The available descriptions repeatedly identify greater stylistic surprise and
more unusual associations in Full250, together with uneven narrative control.
They support a substantial change in writing behavior. The descriptive stage
reserves numeric quality preferences for subsequent scoring.

| Aspect | Recurring baseline descriptions | Recurring Full250 descriptions |
| --- | --- | --- |
| Imagery | Familiar, legible motifs with relatively stable associations | More unusual, tactile, surreal, or cross-domain images, with some strained or disconnected associations |
| Prose and rhythm | Often uniform declaratives, explicit explanation, and repeated summaries | Larger changes of form across examples, including dense periodic prose, fragments, dialogue, and incantatory rhythms |
| Use of the brief | Frequently names required concepts or traits directly | Often turns concepts into actions, objects, or images, with occasional omissions, substitutions, and contradictions |
| Character | Motivation and traits are commonly stated outright | Some traits emerge through gestures and specific history, while motives remain opaque in other stories |
| Plot and ending | Often a traceable causal sequence and explicit triumph or moral | More reinterpretations, qualified releases, and ambiguous outcomes, alongside repetitive or unresolved codas |
| Failure patterns | Repetition, generic abstraction, and over-explanation | Semantic drift, unstable referents, awkward wording, and occasional language or viewpoint intrusions |

These qualitative themes are grounded in the saved descriptions. Their
prevalence and numeric preferences remain unmeasured. For example, both orders of
`mazur/16` describe Full250 as showing a character trait through behavior, and
both orders of `mazur/20` contrast its qualified ending with baseline's explicit
triumph. Both orders of `mazur/4` instead identify a loss of coherence in
Full250's later passages. The direction and extent of the tradeoffs vary by
story.

## Illustrative matched observations

The following examples describe saved pairs whose two orderings are available.
They illustrate variation across briefs. The incomplete, sequentially generated
sample cannot establish full-panel quality rankings.

- In `mazur/0`, both orderings describe Full250's imagery as lavish, surreal,
  associative, and striking in places. The baseline uses familiar, legible
  technology-versus-nature imagery and a linear arc. The judge also identifies
  disconnected images, grammatical and viewpoint breaks, and foreign-script
  artifacts in Full250. Its account preserves both the increased surprise and
  the uncertainty about narrative control.
- In `mazur/8`, both orderings identify a strong rhythmic difference. Baseline
  uses uniform declaratives on separate lines, while Full250 uses longer,
  ornate sentences and paragraph blocks. Full250 compresses the story into a
  confrontation and integrates elements through plot mechanics. Baseline
  covers more episodes and frequently names the required elements directly.
- In `mazur/16`, Full250 embodies the character's feistiness through gestures
  and uses idiosyncratic images involving rain and roots. Baseline labels the
  trait explicitly and uses more familiar luminous imagery. The judge also
  notes Full250's sprawling closing sequence and repeated endings.
- In `mazur/20`, both orderings describe Full250's ending as partial release
  with continuing loss, while baseline reaches explicit triumph. Full250's
  mosaic inscription drives action, and its syntax mixes fragments with longer
  images. Baseline frequently states the requested tone and motivation.
- In `mazur/4`, both orderings identify a more severe failure in Full250. It
  begins with concrete physical detail and loses referential and semantic
  coherence later. Baseline retains a legible arc despite local slips. The
  increased associative density therefore has uneven outcomes in these examples.

These descriptions capture substantial stylistic differences and distinguish
inventiveness from control more explicitly than the earlier Phi4 win verdicts.
The scoring pass remains pending. Numeric style and creativity scores will
require the predefined scoring rubric, both orders, and brief-clustered
uncertainty estimates.

## Excerpt and response audit

All 57 saved responses finish normally and pass the requested JSON schema. Eight
responses have excerpt-audit flags, comprising ten flagged excerpts. Nine
excerpts exceed the requested 12-word quotation limit. One excerpt cannot be
matched verbatim to its source story. These are separately classified in
`describe/excerpt-audit.json`.

The unmatched excerpt occurs in `mazur/25:0`, Story B, which is Full250 in that
ordering. It requires manual checking before being relied on as textual
evidence. Audit flags are supplied to the subsequent scoring pass together with
the raw prose so the judge can verify its observations. No descriptions were
regenerated because of their content or format.

## Interpretation limits and continuation

The sample is incomplete and comes from the initial portion of the frozen
queue. A few illustrative order-consistent descriptions cannot establish the
judge's full-panel order stability. The descriptive prompt deliberately
reserves numeric scores and winners for a later stage.

The source stories also differ in completion status. Baseline has 23 truncated
stories across the full panel, compared with four for Full250. The comparison
retains those first-attempt outputs. The selected judge's scale and literary
preferences also require attention when interpreting later scores.
Within the 28 fully compared story pairs currently available, baseline has ten
truncated source stories and Full250 has three. This imbalance can affect
judgments of closure and narrative structure in the current subset.

Account funding is required to finish the 71 missing descriptions. At the
observed average cost, those requests would cost approximately $5.01. The key
currently has enough remaining allowance for that estimate, while the account
has no available funds. Completing the later scoring pass will likely also
require a higher key limit than $10. The configured campaign guard remains $30
and is further bounded by current key allowance and available account credits.

## Evidence and protocol

Campaign directory is
`benchmarks/results/writing-openrouter-20261104T100913Z/`.

- `describe/jobs.json` retains all 128 frozen requests and their hashes.
- `describe/responses.jsonl` retains the 57 raw responses, model/provider
  identities, generation IDs, token usage, and cost receipts.
- `describe/parsed.json` retains the validated descriptions.
- `describe/excerpt-audit.json` distinguishes long quotations from unmatched
  quotations.
- `status.json` records `awaiting_credits` and the saved-response count.
- `budget.json` records separate key and account constraints.
- `input-manifest.json` pins the source outputs and configuration.

The [execution protocol](writing-openrouter-protocol-2026-11-04.md) specifies
the descriptive and scoring stages, score anchors, half-point tie threshold,
bootstrap procedure, and resumption commands. The existing
[local-judge results](writing-transfer-results-2026-11-04.md) retain the previous
Phi4 diagnostic findings for comparison.
