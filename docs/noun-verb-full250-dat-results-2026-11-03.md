# Three-class DAT: baseline versus noun/verb full step 250

Completed 3 November 2026: fresh original baseline and full snapshots
50/75/100/150/200/250, 672 responses. The report below details the final model
and early formatting failure; the linked tables contain every snapshot:

- [W&B DAT run](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/8n7l356r)
- [Live raw DAT comparison](../benchmarks/results/noun-verb-full-dat-20261103T102138Z/comparison.md)
- [Live validity comparison](../benchmarks/results/noun-verb-full-dat-20261103T102138Z/word-class-comparison.md)

## First findings

The final full model increases raw DAT substantially on all three prompts.
Adjective class following nevertheless degrades: compatible adjective words
fall from **317/320 (99.06%) to 243/320 (75.94%)**, and lists jointly passing DAT
and all ten adjective checks fall from **29/32 to 1/32**. All 32 full-model
adjective responses are formally DAT-scoreable. Raw distance therefore masks
a large deterioration in the requested-class constraint on this panel.

The full adjective outputs contain 57 known other-class words and 20 unknown
words. Of those known mismatches, 36 occurrences have a noun sense but no
adjective sense. There are known other-class words in 27/32 adjective lists.
Unknowns are reported separately, not automatically called nouns.

## Matched setup

32 fixed cases per class per model: nouns, verbs, and the held-out adjective
request. The 96 cases and sample seeds match the earlier conditional LoRA
screen; these baseline and full-model responses are newly generated.
Temperature 1, top-p 1, unrestricted top-k (-1), min-p 0, presence penalty 0,
repetition penalty 1, thinking disabled, 192 completion tokens, concurrency 64.
The first 192 responses have no reasoning output and no token-limit truncation.

Both variants use BF16 vLLM inference on the same original-backbone export and
the new full-model snapshot. The baseline stores all tensors in BF16; vLLM
casts the full snapshot's 319 FP32 normalization tensors to its requested BF16
inference dtype. This is the matched BF16 serving screen, rather than a replay
of training arithmetic. Actual model paths, launch arguments, case hashes,
seeds, and response coverage are retained in `baseline250-validation.json`.

## Raw official DAT

All means use all 32 requested lists, with unscoreable outputs assigned zero.
Paired 95% intervals use the same 32 case IDs and 5,000 bootstrap resamples,
seed 20261031, without multiple-comparison correction.

| Class | Baseline DAT | Full 250 DAT | Delta | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Nouns | 71.81 | 89.59 | +17.78 | [15.33, 20.52] |
| Verbs | 51.11 | 85.21 | +34.11 | [29.89, 38.81] |
| Adjectives | 60.72 | 86.26 | +25.55 | [23.23, 28.38] |

## Validity and requested word class

WordNet 3.0 accepts any requested sense, including adjective satellites and
polysemous words. It is a lexical proxy with coverage gaps. Typographic quote
normalization is audit-only; formal DAT scores use unmodified final text.
All 192 responses parse into ten-string lists for the lexical audit.

| Class | Model | Compatible words | All ten class-compatible | DAT-scoreable | Scoreable and all ten compatible |
| --- | --- | ---: | ---: | ---: | ---: |
| Nouns | Baseline | 310/320 (96.88%) | 27/32 | 32/32 | 27/32 |
| Nouns | Full 250 | 311/320 (97.19%) | 25/32 | 32/32 | 25/32 |
| Verbs | Baseline | 314/320 (98.12%) | 28/32 | 31/32 | 27/32 |
| Verbs | Full 250 | 311/320 (97.19%) | 23/32 | 32/32 | 23/32 |
| Adjectives | Baseline | 317/320 (99.06%) | 29/32 | 32/32 | 29/32 |
| Adjectives | Full 250 | 243/320 (75.94%) | 1/32 | 32/32 | 1/32 |

Verified DAT retains the official score only for a scoreable list whose ten
words all pass the requested-class check; other lists get zero. Its denominator
remains all 32 requests. This diagnostic does not change raw official DAT.

| Class | Baseline verified DAT | Full 250 verified DAT | Delta | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Nouns | 61.42 | 70.07 | +8.64 | [-9.09, 25.57] |
| Verbs | 43.37 | 61.32 | +17.95 | [2.14, 33.52] |
| Adjectives | 54.42 | 2.56 | -51.86 | [-59.53, -42.88] |

These are small stochastic panels. The adjective result does not support clean
transfer of constrained DAT performance. General capabilities are not evaluated
by this screen. Earlier snapshots locate when class following changes.

## Completed snapshot trajectory

Joint validity requires both formal DAT validity and all ten requested-class
WordNet senses; counts are out of 32. At 50 and 75 every response is unparsed
for the lexical audit, so zero joint validity there reflects output failure.

| Model | Noun DAT | Verb DAT | Adjective DAT | Noun joint valid | Verb joint valid | Adjective joint valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 71.81 | 51.11 | 60.72 | 27/32 | 27/32 | 29/32 |
| Full 50 | 0.00 | 0.00 | 0.00 | 0/32 | 0/32 | 0/32 |
| Full 75 | 0.00 | 0.00 | 0.00 | 0/32 | 0/32 | 0/32 |
| Full 100 | 87.60 | 82.25 | 81.23 | 16/32 | 21/32 | 11/32 |
| Full 150 | 81.31 | 82.08 | 81.77 | 28/32 | 8/32 | 0/32 |
| Full 200 | 78.27 | 87.36 | 79.04 | 13/32 | 20/32 | 0/32 |
| Full 250 | 89.59 | 85.21 | 86.26 | 25/32 | 23/32 | 1/32 |

The separate [LoRA snapshot screen](noun-verb-lora1500-dat-results-2026-11-03.md)
has also completed. LoRA1500 retains 27/32 jointly valid adjective lists and
98.12% compatible adjective words, with raw adjective DAT 64.74. Its smaller raw
distance gain accompanies far better held-out class following in these runs.

## Example adjective response

Full step 250, case `dat_adjectives/3`:

```text
["endpoint", "biblical", "bark", "mule", "contour", "demon", "suture", "crumb", "staple", "gavel"]
```

Eight entries have no adjective sense in the verifier; examples include
endpoint, mule, demon, and gavel.

## Early snapshot 50

All 96 step-50 responses were formally unscoreable: raw DAT and joint validity
are zero for each class. All 32 lists per class were unparsed for the lexical
audit, so this does not establish zero word-class knowledge. Of the 96 outputs,
94 reached the 192-token cap. The two shorter responses stopped with a stray
`</think>` preceding an otherwise recognizable JSON list. This reproduces an
output-format failure when serving saved step-50 weights in a fresh vLLM
process with the same explicit no-thinking request settings. Evidence is
`step50-validation.json` and the saved raw responses.

All artifacts are under
`benchmarks/results/noun-verb-full-dat-20261103T102138Z/`, including
`results.json`, `word-class-audit.json`, `baseline250-validation.json`,
`launch-validation.json`, raw responses, and `protocol.json`.
