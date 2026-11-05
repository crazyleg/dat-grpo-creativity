# Noun/verb LoRA: completed three-class DAT screen

Completed 3 November 2026: fresh baseline plus LoRA 250/500/1000/1500, 480 responses. Adjectives remain held out from training. The final conditional LoRA preserves adjective class following much better than the conditional full-model run, while improving raw noun, verb, and adjective DAT. The stricter all-ten-compatible adjective score is effectively flat, so this is promising raw-distance transfer, not established improvement in the complete constrained task.

## Matched setup

32 fixed cases per class per model, using the same case file and seeds as the conditional full-model DAT screen. Temperature 1, top-p 1, unrestricted top-k (-1), min-p 0, presence penalty 0, repetition penalty 1, thinking disabled, 192 completion tokens, 64 concurrent requests. All 480 responses stop normally, contain no structured reasoning or literal thinking tags, and parse as ten-string lists for the lexical audit. Formal DAT validity remains a separate check.

The baseline and rank-16 adapters use the shared BF16 unpacking of the pinned original NF4 backbone in one vLLM server. The baseline receives no adapter; each adapter alias is pinned to its archived optimizer-step snapshot. vLLM serves the saved FP32 adapters in BF16 without altering the archives. Model paths, registered aliases, payload settings, coverage, and snapshot identities were verified.

These are fresh responses, rather than reused pilot outputs. Small numerical differences from the earlier baseline are retained as observed; every delta uses this run's own fresh baseline. Case-file SHA-256: `db0de89e100577765ed45418b31ec035c1c4683ed8c07ac51d97676bf9535321`.

## Raw official DAT trajectory

All means include all 32 requests, with unscoreable outputs receiving zero. Official DAT does not enforce the requested word class.

| Model | Nouns | Verbs | Adjectives |
| --- | ---: | ---: | ---: |
| Baseline | 71.81 | 51.11 | 60.67 |
| LoRA 250 | 72.73 | 56.22 | 60.83 |
| LoRA 500 | 70.94 | 59.39 | 61.55 |
| LoRA 1000 | 68.60 | 62.34 | 64.88 |
| LoRA 1500 | 79.29 | 70.15 | 64.74 |

Noun scores are not monotonic across snapshots. Verb raw DAT rises across the tested adapters. Adjective raw distance changes little at 250/500, then rises by about four points at 1000/1500.

## Final adapter versus its baseline

Paired 95% intervals use 5,000 bootstrap resamples of the 32 matched case IDs, seed 20261031, without multiple-comparison correction.

| Class | Baseline DAT | LoRA 1500 DAT | Delta | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Nouns | 71.81 | 79.29 | +7.47 | [4.56, 10.46] |
| Verbs | 51.11 | 70.15 | +19.05 | [13.38, 25.12] |
| Adjectives | 60.67 | 64.74 | +4.06 | [1.01, 7.65] |

All three raw-score intervals exclude zero on this small panel. They do not account for the word-class constraint or establish general capability preservation.

## Requested-class and joint validity

WordNet 3.0 accepts any requested-class sense, including adjective satellites and ambiguous words. Unknowns are separate from known other-class entries. This is a lexical proxy with coverage gaps and is not complete contextual grammar annotation. Quote normalization is audit-only; official DAT evaluates unmodified final text.

Joint valid means formally DAT-scoreable and all ten words having a requested-class sense. Counts below are out of 32 requested lists.

| Model | Nouns joint valid | Verbs joint valid | Adjectives joint valid | Adjective-compatible words |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 27/32 | 27/32 | 29/32 | 317/320 (99.06%) |
| LoRA 250 | 27/32 | 24/32 | 29/32 | 317/320 (99.06%) |
| LoRA 500 | 24/32 | 24/32 | 28/32 | 316/320 (98.75%) |
| LoRA 1000 | 26/32 | 31/32 | 25/32 | 309/320 (96.56%) |
| LoRA 1500 | 29/32 | 24/32 | 27/32 | 314/320 (98.12%) |

Final-adapter detail:

| Class | Model | Compatible words | Formal DAT valid | All ten compatible | Joint valid | Known other-class words | Unknown words |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nouns | Baseline | 310/320 (96.88%) | 32/32 | 27/32 | 27/32 | 10 | 0 |
| Nouns | LoRA 1500 | 317/320 (99.06%) | 32/32 | 29/32 | 29/32 | 3 | 0 |
| Verbs | Baseline | 314/320 (98.12%) | 31/32 | 28/32 | 27/32 | 6 | 0 |
| Verbs | LoRA 1500 | 308/320 (96.25%) | 32/32 | 24/32 | 24/32 | 12 | 0 |
| Adjectives | Baseline | 317/320 (99.06%) | 32/32 | 29/32 | 29/32 | 3 | 0 |
| Adjectives | LoRA 1500 | 314/320 (98.12%) | 32/32 | 27/32 | 27/32 | 6 | 0 |

At 1500, adjective-compatible words decrease by three occurrences (317 to 314), and joint-valid lists decrease by two (29 to 27). The paired joint-validity delta is -6.25 percentage points, with a 95% interval of [-21.88, 9.38]; this panel does not resolve a small change reliably. The six flagged adjective words are market, heavily, silk, man, scale, and marry. Market already occurs in the baseline; silk illustrates the ambiguity between attributive noun use and an adjective dictionary sense. Four of the six have a noun sense and no adjective sense.

Verb following deserves attention: final LoRA compatible words fall from 314 to 308 and joint validity from 27 to 24, despite the much higher raw score. Step1000 has the strongest observed verb compliance (319/320 words, 31/32 joint-valid lists), but selecting a snapshot from these same small panels requires independent confirmation.

## DAT requiring full task compliance

Verified DAT retains the raw official score only for joint-valid lists and assigns zero otherwise. Its denominator remains all 32 requests. This separately named diagnostic changes neither official DAT nor the training reward.

| Class | Baseline verified DAT | LoRA 1500 verified DAT | Delta | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Nouns | 61.42 | 71.67 | +10.25 | [-4.01, 23.78] |
| Verbs | 43.37 | 51.46 | +8.09 | [-4.33, 20.06] |
| Adjectives | 54.38 | 53.89 | -0.49 | [-10.33, 8.94] |

All three verified-DAT intervals include zero. For adjectives, the raw +4.06 gain becomes -0.49 in this diagnostic. This does not prove absence of transfer; it means this small screen has not established an improvement while satisfying every requested constraint.

## Comparison with conditional full step250

The separate full-model screen used the same cases and decoding settings. For adjectives, full250 has raw DAT 86.26, 243/320 compatible words (75.94%), 1/32 joint-valid lists, and verified DAT 2.56. LoRA1500 has raw DAT 64.74, 314/320 compatible words (98.12%), 27/32 joint-valid lists, and verified DAT 53.89. The full run therefore obtains much more raw divergence while violating the held-out class much more often.

This is a comparison of the actual trained checkpoints, not an isolated causal test of LoRA versus full training: the runs use different optimizers (AdamW versus Adafactor), group sizes (4 versus 6), update counts, and total sampled completions. No new general-capability benchmarks were run in this DAT screen.

## Artifacts and verification

[LoRA DAT W&B](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/irj22eux). Open `results/score_by_snapshot` for raw DAT, `results/word_class_compliance` for all three classes and verified DAT, and `results/adjective_compliance` for the detailed held-out audit. Training: [conditional LoRA W&B](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/iolv9hr3). Full comparison: [W&B](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/8n7l356r).

Artifacts under `benchmarks/results/noun-verb-lora-dat-20261103T112551Z/`:

- `results.json`: raw means and paired intervals.
- `word-class-audit.json`: per-word senses, class counts, joint validity, and verified-DAT intervals.
- `no_think/*/responses.jsonl`: all raw generations and seeds.
- `completed-validation.json`: 480 responses with 96 unique case IDs per model, 32 per class/model, exact frozen-panel and sampling match, adapter snapshot identity, reasoning/truncation checks.
- `served-models.json` and `protocol.json`: actual server registrations, paths, and inference arguments.
- `wandb-completed-verification.json`: independent W&B summary/table verification.

Config: `configs/benchmark_noun_verb_lora_dat.json`. All snapshots remain archived; the local inference server remains available.
