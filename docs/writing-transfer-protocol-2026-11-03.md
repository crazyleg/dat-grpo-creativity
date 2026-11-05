# Writing and style transfer: execution protocol

The completed [research results](writing-transfer-results-2026-11-04.md) collect
all score tables, paired intervals, compliance diagnostics, and judge limitations.

Started 3 November 2026 at 18:51 UTC, after the user requested stopping and
preserving the formal tests. [Generation W&B](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/zcohczve).
The original baseline, final conditional noun/verb LoRA1500, and final
conditional full250 receive the same fixed prompts and seeds.

## Approved scope

| Task | Per model | All three models | Selection |
| --- | ---: | ---: | --- |
| Mazur | 32 briefs x 2 samples =64 | 192 | Fixed32 briefs from current400;600-800words |
| WritingBench | 64 | 192 | English style cases across six domains |
| SteerBench | 128 | 384 | Two instructions for each of all64 source texts |
| Total | 256 | 768 | One first-attempt response per case/model |

Writer aliases `original`, `step1500`, and `step250` mean Baseline, LoRA1500,
and Full250 respectively. Config pins the newer conditional noun/verb model
paths; the earlier noun-only models are outside this screen. Every model/mode
directory retains the actual serving launch. The existing BF16 server was
validated and adopted after the formal workers stopped. Full250 is loaded
separately with no adapter; baseline and LoRA share the original backbone.

The source revisions and exact selected IDs are frozen in the asset manifest
and `selection.json`. The frozen case file is the authoritative panel, including
prompt ordering. Writer cases SHA-256: `26def4b859de6a58af674c06b64eb284b90b490fc4ff5ae9f4d3cb8136a79a2e`.

WritingBench contains 11 cases each from Academic/Engineering, Advertising/
Marketing, Education, and Finance/Business, and 10 each from Literature/Arts
and Politics/Law. Inputs are at most6500tokens; explicit word requests above
1200words were excluded. Original query text and all five rubric criteria are
retained, together with the style-specific criteria. No material is silently
truncated. SteerBench uses the pinned upstream direct-template implementation
with negative/disambiguation instructions and no-explanation suffix.

## Sampling and budgets

| Task | Temperature | top-p | top-k | Output-token cap |
| --- | ---: | ---: | ---: | ---: |
| Mazur | 1.0 | 1.0 | unrestricted | 2048 |
| WritingBench | 0.7 | 0.8 | 20 | 4096 |
| SteerBench | 0, greedy | 1.0 | unrestricted | 4096 |

All writer requests explicitly disable thinking; no reasoning effort is set.
Min-p and presence penalty are zero; repetition penalty is1. The same generation
seed,20261031 plus case index, is used for every variant. Mazur's two repeat
indices use distinct seeds. Generation batches up to64 requests in vLLM with
16,384-token context and24GiB KV cache. All prompt-plus-output budgets were
checked before launch. The nine task/model comparisons run task by task.

First responses are retained even when tags, length, word counts, requested
style, or generation fail. HTTP transport retries and resuming an interrupted
request are logged infrastructure behavior; there is no quality-based redraw.
Reported progress counts saved writer responses out of768.

## Scoring

Generation diagnostics are immediately separate from judged quality: nonempty
outputs, truncation, word counts, thinking leakage, Mazur story tags, actual
600-800-word compliance, and agreement of emitted cumulative counters with the
actual word prefixes. Counter agreement alone does not establish a marker after
every sentence. Counters/tags are stripped for prose quality comparison;
unmodified outputs remain saved.

The default scoring pipeline is fully local, using frozen weights loaded after
writer generation. An optional question about external judging was presented;
generation proceeds independently. No paid judge API is used by this config.

- **WritingBench:** its official downloadable Qwen2.5-based7B critic and pinned
  evaluation prompt, temperature1/top-p0.95/2048-token cap. Score all five
  query-specific criteria and report overall and style-specific means separately.
- **Mazur:** an independent-family local Phi4 judge uses the current upstream
  comparison prompt/rubric, with writer identities hidden and both A/B orders.
  LoRA-versus-baseline and full-versus-baseline give256 judge calls. Report win
  credit, margins, parsing coverage, and intervals clustered by the32 briefs,
  keeping repeat samples and both orderings within their brief.
- **SteerBench:** the pinned upstream metric functions report steering error,
  miscalibration, and side effects across reading grade, MTLD lexical diversity,
  word count, and Heylighen-Dewaele formality. Phi4 separately applies the
  upstream broad content-preservation prompt. This local judge flag does not
  substitute for the publication's human-adjudicated filtering protocol.

These are matched local diagnostic scores. The local Phi4 judge is not calibrated
against expert literary preferences, and local WritingBench critic scores are
not the current public leaderboard's Claude scores. Interpret quality alongside
validity, representative stories, and judge coverage.

Judging is linked from the generation run as `results/judging_run_url`; its own
W&B run publishes `results/writingbench_critic`, `results/mazur_pairwise`,
`results/steerbench`, and `results/steerbench_content`. Raw judgments, prompt
hashes, model provenance, and parsed scores remain in the campaign directory.
Quality judging adds scoring calls beyond the768 writer generations.

## Metric calibration and reproducibility

CPU metrics have an isolated environment, with exact dependencies saved in
`configs/writing_metrics_requirements.txt`. The upstream pylats fork is pinned.
English syllabification uses Pyphen-only behavior: that reproduces all64
released reading-grade input values exactly. Word-count and formality input
measurements also match exactly across all64 texts. The current pinned MTLD
stack differs slightly on a few source texts (maximum6.61 MTLD units; mean
absolute difference0.14); these differences and package versions are recorded
rather than called an exact leaderboard reproduction. All variants use the
same metric stack and the published normalization bounds.

Public source revisions:

- [Mazur writing](https://github.com/lechmazur/writing), `05408954d0165c3ea68acb834797a64be2baf450`.
- [WritingBench](https://github.com/X-PLUG/WritingBench), `ae2d5176449b7b769815482641d35926f26793eb`.
- [SteerBench code](https://github.com/MLD3/steerability), `97fba84ecba2907ee7c9b59ba2564668b593243c`.
- [SteerBench data](https://huggingface.co/datasets/tchang97/steerbench), `8858ba3fc3cfc90e7dad237ac27d0bf6ae454a98`.
- [WritingBench critic](https://huggingface.co/AQuarterMile/WritingBench-Critic-Model-Qwen-7B), `fb5f30a944174bd28042a4f19a1c1de86606a6cd`.
- [Phi4 local judge](https://huggingface.co/microsoft/phi-4), `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`.

`fetch_writing_sources.py`, `prepare_writing.py`, and
`download_writing_judges.py` provide source/data/scorer preparation. Existing
frozen cases are never overwritten by preparation. Generation resumes saved
case IDs under an unchanged config, and judging caches by prompt hash and judge
identity. Missing/invalid judgments retain explicit coverage counts.

The detached observer restarts an exited failed worker at most three times,
reusing saved writer responses and judge caches. It respects stopped/completed
statuses, never replaces a slow live worker, and waits for any surviving judge
before resuming an unexpectedly exited parent.

Campaign directory: `benchmarks/results/writing-transfer-20261103T185135Z/`.
Configuration: `configs/benchmark_writing_transfer.json`.
The frozen formal-test archive and completed/partial scores are documented in
[the preservation report](formal-capabilities-preserved-2026-11-03.md).

## Completed results on November 4

All 768 writer responses and 1,600 judge responses are saved. Generation finished
at 21:59 UTC on November 3 and judging finished at 23:29 UTC. The following scores
use the local judges and calibrated metric stack described above.

| Metric | Baseline | LoRA 1500 | Full 250 |
| --- | --- | --- | --- |
| WritingBench style, 1–10, 64 cases each | 8.461 | 8.602 | 8.398 |
| WritingBench overall, 1–10 | 8.065 | 8.106 | 8.051 |
| Fully scored WritingBench cases | 62/64 | 62/64 | 63/64 |
| SteerBench steering error, lower is better | 0.539 | 0.550 | 0.563 |
| SteerBench miscalibration, lower is better | 0.777 | 0.786 | 0.861 |
| SteerBench side-effect error, lower is better | 0.524 | 0.527 | 0.464 |
| Mazur win credit against baseline | — | 52.34% | 40.63% |

Paired bootstrap 95% intervals for WritingBench style differences against
baseline are [0.000, 0.289] for LoRA and [-0.297, 0.148] for full fine-tuning.
SteerBench full-model miscalibration increased by 0.0845 [0.0294, 0.1514], while
side-effect error decreased by 0.0606 [-0.1206, -0.0024]. Steering-error intervals
include zero for both variants. These panels give little evidence of a broad
writing-quality improvement.

### Mazur judge recovery and reliability

The first aggregation accepted no Mazur judgments because Phi4 omitted closing
section tags. On November 4, the parser was corrected to accept a scalar winner
ending at the next expected section heading. Explicit winner labels are now
scored separately from signed margins. All 256 winner labels are recoverable,
but 47 signed margins contradict the declared B winner and remain invalid.
Original judgments remain unchanged. Pre-correction aggregates are preserved
under `judging/parser-recovery/`. No writer or judge generations were repeated.

Both A/B orders are present for all 64 story pairs per comparison. Brief-clustered
95% intervals for win credit are [46.88%, 57.81%] for LoRA and [33.59%, 48.44%]
for full fine-tuning. However, Phi4 selects position A in 209/256 judgments
(81.64%). Reversing story order preserves the preferred story in only 17/64
LoRA pairs and 28/64 full-model pairs. This strong position dependence limits
confidence in the comparative creativity scores. The W&B diagnostics table
reports these counts alongside the win credits.

Five WritingBench criterion judgments contain malformed JSON and retain missing
scores. All style criteria parsed successfully. Phi4 marked content preserved
for all 384 SteerBench outputs, so that check offers little discrimination.

Mazur compliance remains poor across all variants. Actual 600–800-word compliance
is 7/64, 5/64, and 9/64 for baseline, LoRA, and full fine-tuning. None achieves
correct cumulative word counts. Token-limit truncations are 23/64, 21/64, and
4/64 respectively. Quality scores must be interpreted alongside these failures.

Completed score tables and judge diagnostics are available in the
[judging run](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/2gjxnf0j).
