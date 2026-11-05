# OpenRouter style and creativity comparison

Prepared on 4 November 2026 following the user's request for a larger judge,
descriptive A/B and B/A comparisons, and subsequent style and creativity scores.
The 128 descriptive requests are frozen and validated. The descriptive pass
started on 4 November at 10:28 UTC after successful credential authentication.
The first returned comparisons confirm the Anthropic provider, the requested
Opus 5.5 identity, and valid structured output.
[Judge W&B run](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/i6tr8x0d).

At 10:35 UTC, OpenRouter rejected further requests with HTTP 402 because the
account balance was exhausted. All 57 completed descriptions are saved and
published. They cost $4.024184 and include 28 story pairs in both orders plus
one single-order comparison. The key retains $5.975816 of its separate $10
allowance. The account requires credits before requests can continue. The
[partial findings](writing-openrouter-results-2026-11-04.md) record the current
coverage, qualitative examples, and audit limitations.

## Source panel

Reuse the 64 matched baseline and conditional Full250 Mazur stories from
`benchmarks/results/writing-transfer-20261103T185135Z/`. They cover 32 fixed
briefs with two generated samples per brief. Every pair appears in both orders,
giving 128 independent descriptive requests and, later, 128 scoring requests.

The judge receives the required elements and cleaned story prose. Story tags
and cumulative counter artifacts are removed using the existing extraction
function. Model identities, training details, previous judge comments, and
earlier numeric results are excluded from the request material. Cases retain
their saved lengths and endings, including the baseline's 23 and Full250's four
token-limit truncations. This asymmetry limits interpretation of the comparison.

Input hashes pin both writer-response files and the frozen case file. Each
request retains its exact messages, JSON schema, model settings, and SHA-256.

## Judge and settings

The selected judge is `anthropic/claude-opus-5.5`, using the Anthropic provider
through OpenRouter with provider fallbacks disabled. The 4 November catalog
lists input tokens at $4 per million and output tokens at $20 per million.
A copy of the model's catalog entry is saved with the prepared campaign.
[OpenRouter model details](https://openrouter.ai/anthropic/claude-opus-5.5).

Reasoning effort is medium. The model uses adaptive thinking. Temperature,
top-p, and seed are omitted because the catalog does not list them as supported
parameters for this version. Each descriptive request has a 6,144-token output
budget, and each scoring request has a 4,096-token budget. Concurrency is four.
Both stages request strict structured JSON.
[Structured-output documentation](https://openrouter.ai/docs/guides/features/structured-outputs).

The configuration sets a $30 recorded-spend guard for this campaign. The supplied
key has a $10 limit, so the runtime campaign allowance is reduced to $10. The
startup check uses the smallest of the configured limit, the campaign's
recorded cost plus the key's remaining allowance, and the campaign's recorded
cost plus available account credits. The key's allowance and the account's
funding balance are checked separately. This bound is recorded in
`budget.json` and W&B. The runtime reserves
an estimated maximum request cost before dispatch and retains actual reported
costs and token usage. Charges for interrupted requests without receipts can
remain uncertain. Failed requests stop further scheduling and retain completed
in-flight receipts where possible. An interrupted run can resume cached cases
under the same frozen request hashes.

## First pass of descriptive differences

Each A/B ordering receives a fresh context and equal attention to both stories.
The judge describes seven aspects for each story.

- Voice and register.
- Syntax and rhythm.
- Imagery and associations.
- Narrative structure.
- Character and motivation.
- Required-element integration.
- Emotional effect.

The response also contains two to four exact excerpts per story, three to five
contrasts, and reading uncertainties. This pass reserves numeric grades and
rankings for the later stage. It permits realist, plain, lyrical, surreal,
fragmented, and ambiguous work.

Application code checks excerpt presence against the corresponding story and
the requested 12-word limit. Flags remain saved and visible in W&B. Later
scoring receives these flags and the full prose so the judge can verify its
observations.

The initial launch uses the descriptive stage. Its reports are available for
inspection before invoking the scoring stage.

## Subsequent scoring pass

Each ordering uses a new API context with the same ordered stories and its own
first-pass observations. The opposite ordering's analysis is kept separate.
The scoring pass uses the same judge, so it remains dependent on that judge's
preferences and interpretation.

Two distinct 1–10 scores are assigned to each story, using at most one decimal
place and explicit scale anchors.

- **Style** measures control and effectiveness of voice, diction, syntax,
  rhythm, image precision, and fit to the chosen literary mode. Plain and
  experimental prose can both receive high scores.
- **Creativity** measures freshness, inventive associations, non-obvious imagery
  or construction, and original uses of the required concepts. The prompt asks
  whether unusual choices create meaningful new connections. Conventional
  causal plotting and tidy endings are optional. Uncertainties and incoherence
  are recorded explicitly.

The two dimensions receive independent rationales. The application derives
preferences from scores to avoid contradictory winner and margin fields.
A difference of at least 0.5 favors the higher-scoring story. A smaller absolute
difference is a tie. Each adapted-model win contributes 1, a tie 0.5, and a loss
0. Each story pair's credit averages both presentation orders.

Reports include per-model mean scores, paired Full250-minus-baseline score
differences, win credits, A/B/tie counts, and agreement between the two orders.
Paired 95% bootstrap intervals use 5,000 resamples of the 32 brief clusters,
keeping repeat samples and both orders within each brief. Estimates describe
this frozen panel and this judge. They have no expert-literary calibration or
multiple-comparison adjustment.

Malformed, truncated, or out-of-range judge responses retain their raw records
and cause a review stop before the next stage or final aggregation. Cached
receipts are reused without quality-based regeneration. Raw JSON, model and
provider identity, generation IDs, usage, prompt hashes, excerpt checks, and
parsed results remain saved.

## Execution and monitoring

Configuration is
[benchmark_writing_openrouter.json](../configs/benchmark_writing_openrouter.json).
The prepared output directory is
`benchmarks/results/writing-openrouter-20261104T100913Z/`, also referenced by
`benchmarks/results/writing-openrouter-latest.txt`.

Credentials are read from `OPENROUTER_API_KEY` or the plaintext file
`~/.config/openrouter/api_key`. The dedicated W&B run records descriptive
comparisons, progress, recorded cost, excerpt flags, and subsequent scoring
tables when that stage executes.

```sh
.venv-inference/bin/python -m dat_grpo.benchmark.writing_openrouter \
  --config configs/benchmark_writing_openrouter.json \
  --output benchmarks/results/writing-openrouter-20261104T100913Z \
  --stage describe
```

After descriptive inspection, the following invocation reuses saved descriptions
and executes the scoring stage.

```sh
.venv-inference/bin/python -m dat_grpo.benchmark.writing_openrouter \
  --config configs/benchmark_writing_openrouter.json \
  --output benchmarks/results/writing-openrouter-20261104T100913Z \
  --stage all
```

Local regression checks cover exact order reversal, per-order observation
isolation, grounded excerpts, score bounds and signs, tie-boundary precision,
order-sensitive preferences, immutable jobs, receipt reuse, spending checks,
and persistence of completed requests after another request fails.
