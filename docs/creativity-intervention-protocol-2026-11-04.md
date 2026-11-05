# Causal story-creativity pilot, 4 November 2026

The user clarified that the main target is the increase in story creativity
after full fine-tuning. This pilot tests whether a story-conditioned internal
direction contributes to originality and imagery. Coherence, overall quality
and formatting are measured separately.

The earlier Sol story comparison found a Full250 creativity increase of 0.70
points and an imagery increase of 0.62, alongside a coherence decrease of 0.90.
Those estimates come from the existing 32-brief panel. The present generation
panel uses eight different briefs from the same pinned public source.
[Earlier story results](writing-codex-results-2026-11-04.md).

## Direction discovery

Use the first eight existing briefs by frozen case index, Mazur cases
0/2/4/6/8/10/12/14. Each contributes a baseline and a Full250 continuation.
Both checkpoints process both continuations with identical token IDs.
Capture block residual outputs at continuation positions 64, 128 and 256.
The mean Full250-minus-baseline difference over these 16 shared contexts
defines one candidate direction per layer.

The tested boundary is zero-indexed layer 47, a fixed three-quarter-depth
boundary. This exploratory choice follows the initial story activation map
and precedes fresh story generation. The direction has no creativity label
until its behavioral intervention is evaluated. No judge score is used to
select discovery examples, fit the direction or choose its strength.

The layer-47 vector has Euclidean norm 4.98177. A Gaussian random vector with
seed 20261104 is rescaled to the same norm. One direction and one strength,
coefficient magnitude 1, are tested. There is no validation-set strength sweep.

## Fresh validation panel and interventions

Eight prompt files are sampled with seed 20261104 from pinned
`lechmazur/writing` revision `05408954d0165c3ea68acb834797a64be2baf450`,
excluding all 32 briefs used in the previous writing panel. Prompts retain
the original ten required elements and 600–800-word story instructions.
The exact choices and source hashes are frozen before generation.

| Checkpoint | Condition | Intervention |
| --- | --- | --- |
| Baseline | Control | Ordinary generation |
| Baseline | Direction | Add the discovered story-conditioned difference |
| Baseline | Random | Add the norm-matched random vector |
| Full250 | Control | Ordinary generation |
| Full250 | Direction | Subtract the discovered story-conditioned difference |
| Full250 | Random | Subtract the norm-matched random vector |

This retains 48 scored stories. All variants use temperature 1,
top-p 1, unrestricted top-k, zero presence penalty, repetition penalty 1 and
thinking disabled. Each brief has the same seed across conditions. The common
completion cap is 3,072 tokens, increased from the previous 2,048-token writing
screen to reduce truncation. The new baseline/Full250 controls therefore form
the matched comparison for this pilot.

## Intervention implementation

Native Hugging Face hooks collect discovery activations using the bounded
checkpoint loader established in the first internal comparison. Generation
uses vLLM 0.32.0 with the V1 model runner explicitly selected, eager execution,
prefix caching disabled, chunked prefill disabled, tensor parallelism one,
8 GiB KV cache and 8,192-token context.

All 24 stories for one checkpoint run as one cohort. Per-request
`SamplingParams.extra_args` identifies the condition independently of dynamic
batch order. The worker hook reads current request IDs and query boundaries,
applying the appropriate vector to the final query row of each request.
This affects the final prompt token and every generated-token state while
leaving earlier prompt rows without direct injection.

vLLM returns a split residual pair at each block boundary. The hook adds to
one branch and preserves the other branch. Correspondence with the combined
Hugging Face residual is conceptual because BF16 fused arithmetic can have
different rounding. The direction is tested as an additive intervention in
the actual serving implementation.

Preflight compares no hook, the zero intervention and a removed hook, with
fixed greedy generation. Outputs and selected log probabilities must agree.
A nonzero direction must have a measurable effect. The worker retains request
condition mappings, selected-row counts and prefill positions for audit.
CPU regression checks exercise reordering, request completion, prompt-row
masking, sign and hook removal.

Trusted local worker callbacks use vLLM's callable serialization option.
The initial fast-inference preflights exposed its serialization requirement,
a default V2-runner API mismatch and a missing virtual-environment tool path.
These attempts are retained separately and precede scored story generation.

The first complete baseline batch then encountered an audit error because
vLLM adds an eight-character nonce to internal request IDs and returns the
original IDs to its caller. The batch had not been persisted before that check,
so it was lost. Baseline generation is rerun with the same frozen prompts,
seeds, vectors and settings. No literary scores were inspected. The corrected
pipeline persists the raw batch before worker audits, validates the documented
ID mapping and can re-export a retained batch without regenerating it. The
failed run and logs remain archived. This technical rerun is disclosed with
the final results.

## Blinded literary evaluation

Preserve the definitions of the existing seven-metric Sol-medium rubric. Primary endpoints
are creativity and imagery. Coherence, style, character/impact, integration
and overall quality remain separate outputs.

Before literary judging started, the user clarified the hypothesis that some
writing styles may be steered through a simple direction while creativity may
require more complex internal changes. Two additional dimensions were frozen
in `hypothesis-extension.json`. Conceptual invention evaluates premise,
relationships and plot construction as if paraphrased in plain language.
Ornamentation measures intensity of figurative and elaborate expression on a
descriptive scale. Higher ornamentation has no inherent quality preference.
The original seven definitions and the five comparisons remain intact.
This refinement occurred after baseline generation and before Full250
generation finished. It is a pre-judging refinement, with its timing retained.

The five comparisons are Full250 versus baseline, baseline with the direction
versus its control, baseline with random steering versus its control, Full250
with the direction removed versus its control, and Full250 with random steering
versus its control. Each of eight briefs is judged in both A/B and B/A order,
giving 80 fresh isolated Codex sessions. Only required elements and the two
story texts are shown. Source identity and intervention labels are withheld.

Scores and preferences average both orders within each brief. Paired 95%
bootstrap intervals use 5,000 resamples of the eight brief blocks with seed
20261104. Metric preferences require at least a 0.5-point score difference,
and ties receive half credit. Overall preference uses the declared winner.
All intervals are exploratory and unadjusted.
They capture variation over the eight briefs. They exclude uncertainty over
training runs, generation seeds and random-control directions. Only one random
direction is tested. Direction-minus-random changes are indirect contrasts of
separately judged pairs, whose shared control stories may receive different
ratings in the two contexts.

The text audit found one baseline response with a misplaced opening tag at its
end. The existing parser returned an empty string for that response despite
substantial preceding prose. Judging was paused after four unaffected
judgments completed, the parser was corrected with a regression test, and six
changed judgment inputs were refrozen. All completed inputs were unchanged
and retained. No generation was replaced. The prior 64-pair story analysis
was checked and contains no affected extraction. The correction and previous
frozen requests are retained in the experiment directory.

Adding the direction to baseline tests sufficiency of this candidate.
Subtracting it from Full250 tests sensitivity to the opposite fixed translation
within the trained model. This is distinct from removing each activation's
input-dependent projection onto a direction. A random control distinguishes its direction from
a generic perturbation of comparable size. Evidence is strongest when these
effects are consistent across signs and exceed the random controls.

## Evidence limits and artifacts

This is an eight-brief pilot with a single candidate, layer and dose. The
discovered direction can mix stylistic, lexical, formatting and conceptual
changes. A negative result excludes only this particular intervention. A
positive result requires subsequent component and feature localization.
One judge supplies the literary scores, and the original fine-tune is one
training trajectory. No individual creativity neuron is assumed.

## Narrative-person positive control

An additional control under `benchmarks/results/style-direction-control-20261104`
uses 24 matched event prompts to extract a first-person minus third-person
instruction contrast at the same layer. Four separate events calibrate three
initial strengths, 2.5%, 5% and 10% of the median discovery prompt residual
norm. Eight further events form the held-out control panel. All events are
separate from the creative-writing briefs. The same serving hook, random
control, request audit and zero/removal checks apply.

The first 2.5% calibration cohort produced no narrative-person shift. The
fitted instruction contrast itself had norm 22.991, about 21.4% of the median
prompt residual norm of 107.274. Before held-out prompts were prepared for
generation, a timestamped calibration extension added 20% and 40% strengths.
The selection rule and held-out prompts stayed fixed. Singular first-person
pronouns provide a supplementary ambiguity check, and direction-versus-random
differences are computed directly from the objective measurements. These
additions precede held-out generation and have their timing recorded.

The predefined selection chooses the smallest dose whose median increase in
first-person pronoun share reaches 0.25, with a positive increase on at least
three of four calibration events. All direction outputs must have at least
50 words and repeated-trigram fraction at most 0.2. If no dose qualifies,
the maximum observed calibration gain is retained with failure disclosed.
No-pronoun cases are flagged and excluded from ratio comparisons. Word counts,
first-person pronouns per 100 words and repetition remain available.

Style vectors and style-associated neuron sets have primary research support.
These establish feasible interventions without showing that each style maps
to a single native neuron. A dense direction combines native coordinates.
[Style Vectors](https://aclanthology.org/2024.findings-eacl.52/),
[Style-Specific Neurons](https://arxiv.org/abs/2410.00593).

This pilot can detect an effective low-dimensional intervention. Its failure
cannot establish high-dimensional creativity. Alternative layers, directions,
doses, schedules, context-dependent effects and limited statistical power
remain possible explanations. Increasing subspace rank, module patching and
projection ablation would be subsequent tests of concentration and necessity.

The implementation follows contrastive activation-addition ideas and applies
them to the actual paired checkpoint difference on story continuations.
[CAA paper](https://arxiv.org/abs/2312.06681),
[creativity-direction paper](https://arxiv.org/abs/2412.06060).

Artifacts are under `benchmarks/results/creativity-intervention-20261104/`.
The directory contains the frozen discovery panel, validation prompts, protocol,
capture provenance, direction files, preflight checks, all first-attempt
generations, hook audits and blinded judgments.

```sh
.venv-inference/bin/python scripts/prepare_creativity_intervention.py \
  --output benchmarks/results/creativity-intervention-20261104
.venv-inference/bin/python -m dat_grpo.story_steering fit \
  --directory benchmarks/results/creativity-intervention-20261104
.venv-inference/bin/python -m dat_grpo.story_steering generate \
  --directory benchmarks/results/creativity-intervention-20261104 \
  --checkpoint data/inference-bf16 --model original
.venv-inference/bin/python -m dat_grpo.benchmark.creativity_intervention \
  --source benchmarks/results/creativity-intervention-20261104
```

Direction fitting requires completed paired discovery captures. Full250
generation uses the retained full checkpoint and `--model step250`.
