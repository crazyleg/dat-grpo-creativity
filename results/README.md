# Results and provenance

The compared models are the shared frozen initialization, the conditional
noun/verb QLoRA checkpoint at step 1500, and conditional noun/verb Full250.
Earlier noun-only runs are not mixed into the tables.

[`summary/`](summary/) contains machine-readable aggregate scores and regenerated
Markdown/CSV tables. [`protocols/`](protocols/) records source revisions,
sampling, original training configurations, software versions, lexical and
embedding provenance, the final judge rubric/schema and completion audits.
Local filesystem prefixes in published metadata have been normalized.

The initial writing/style results also have a [comparison table](summary/initial-writing-table.md)
and [publication figure](../article/figures/initial-writing-evaluations.png).

## Raw bundles

Each compressed archive is downloadable directly from this repository.
[`manifest.json`](manifest.json) records its SHA-256 and every member's hash.

| Bundle | Contents |
| --- | --- |
| [Full250 training](raw/training-full250.tar.gz) | Reward rollouts, training prompts, loss/gradient replay checks and final metrics |
| [LoRA1500 training](raw/training-lora1500.tar.gz) | Reward rollouts, training prompts and final metrics |
| [Full DAT screen](raw/dat-full.tar.gz) | Responses from baseline and saved full snapshots, scores and word-class audit |
| [LoRA DAT screen](raw/dat-lora.tar.gz) | Responses from baseline and LoRA snapshots, scores and word-class audit |
| [Writing transfer](raw/writing-transfer.tar.gz) | All 768 writer responses, local critic/judge outputs, SteerBench metrics and case/source index |
| [Sol story judgments](raw/sol-story-judgments.tar.gz) | All 128 final judgments, exact judge inputs, A/B order jobs, rubric/schema, story ranking and audit |
| [Generic capabilities](raw/generic-capabilities.tar.gz) | All 5,823 responses, native item scores, HumanEval test outcomes, aggregate scores, protocols and case/source indices |
| [Fresh story pilot](raw/fresh-story-pilot.tar.gz) | Separate eight-brief robustness/intervention panel, 48 generations, 80 final judgments, protocol and sensitivity checks |

Example:

```bash
mkdir -p unpacked/sol
tar -xzf results/raw/sol-story-judgments.tar.gz -C unpacked/sol
```

These are raw generated answers and final judge outputs. OpenAI judge internal
event streams are excluded. Open-weight Qwen reasoning fields, when present
in the original responses, are retained. W&B runtime caches, API credentials,
model weights and environments are not in the repository. Upstream benchmark
question banks and prompt collections should be retrieved from their original
projects using the pinned source revisions and case identifiers.

## Interpretation

- The final Sol comparison covers baseline and Full250, not LoRA. Its seven
  dimensions and both presentation orders must remain distinct from the
  earlier local Phi4 holistic comparison.
- Story intervals resample 32 brief clusters, preserving the two samples and
  reversed judge orders. Generic capability intervals resample paired
  questions. Both are exploratory and unadjusted for multiple comparisons.
- The 2,048-token story panel has 23 baseline and four Full250 truncations.
  First attempts were retained. The result is conditional on this protocol.
- Raw adjective DAT increases do not establish transfer: Full250 passes all
  ten adjective checks in only one of 32 lists. WordNet is an any-sense
  lexical audit with coverage limits, not a contextual parser.
- The original training configs scheduled different update counts and group
  sizes. This is a comparison of realized recipes, not a controlled experiment
  on the expressive capacity of LoRA.

## Original working reports

Detailed reports are under [`docs/`](../docs/). They preserve the contemporary
interpretation and run names. Paths quoted as code in those reports describe
the original research workspace; the public files are organized by the index
above. The illustrated article and public summary tables are the entry points
for the final comparison.
