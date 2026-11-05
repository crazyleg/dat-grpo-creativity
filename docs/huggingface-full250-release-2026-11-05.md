# Public Full250 release, 5 November 2026

Full250 is publicly available as a full Hugging Face checkpoint, two quantized
GGUF downloads and a working ZeroGPU prompt interface. Both model repositories
include the inherited Apache-2.0 license and training provenance.

- [Full model](https://huggingface.co/crazyleg11/qwen3.8-27b-dat-grpo-full250)
- [GGUF downloads](https://huggingface.co/crazyleg11/qwen3.8-27b-dat-grpo-full250-GGUF)
- [Public prompt Space](https://huggingface.co/spaces/crazyleg11/full250-creativity)

## Full checkpoint

The source is the archived noun/verb full-weight run at optimizer step 250,
`qwen38-27b-dat-noun-verb-full-b6-20261102T214703Z/model-snapshots/step-250`.
It contains 27,356,728,560 parameters in 1,184 tensors and 55 safetensors shards.
The archive stores large weights in BF16 and normalization tensors in FP32.
The upload preserves every weight file. The config replaces its local
`model_name` path with the Hub model ID. Tokenizer, chat template, generation
config and processor config accompany the model.

The runtime architecture is `Qwen3_5ForConditionalGeneration`. The repository
uses the standard Transformers format. Text loading and generation were
tested with Transformers 5.19.0. The archived checkpoint also powered the
earlier vLLM 0.32.0 evaluations. The card includes usage instructions. SGLang
compatibility follows its Qwen3.5 architecture support and has no separate
local runtime test in this release.

The weight publication is pinned at
`d69f3ff69a4c5c2d84df56876354d6bdef170519`. The updated model card is at
`974c186394d729d6d30de0c4f1b2b47ae8195bb6`. All 67 uploaded files were checked
against their expected remote sizes before the later card update.

The card records the NF4 initialization, GRPO reward and optimizer, DAT
class-following failures, generic capability results and writing measurements.

## Quantized downloads

| File | Bytes | Decimal GB | GiB |
| --- | ---: | ---: | ---: |
| Q4_K_M | 16,547,401,344 | 16.55 | 15.41 |
| Q8_0 | 28,595,764,864 | 28.60 | 26.63 |

Both files contain 26,895,998,464 text parameters in 851 tensors with GGUF
architecture `qwen35`. The full safetensors release additionally contains the
vision path. Runtime memory also includes cache and compute buffers.

Conversion uses llama.cpp commit
`ea40a3a1e1774534b7fe605caa1e85558175e3e8`. A BF16 GGUF intermediate is exported
directly from the archived source. Both quantizations independently use that
intermediate. Each passes structural coverage checks and a local CPU
generation test through `llama-completion`.

The GGUF weight commit is `c431d4e04d43b912e2d03b34354d926f4add6bcc`.
The finalized release, including verified SHA256SUMS line formatting, is
`61734abe5522c241fe4d877b589b4d780cd21da7`. Anonymous HTTP range requests return
206 for both files, with the expected total sizes and valid GGUF version-3
headers. Published SHA256 values agree with the Hub file metadata.

The repo contains instructions for llama.cpp and the documented GGUF import
workflows in Ollama and LM Studio. The latter two applications were not run
locally for this release. The quantized capability and creative-writing
batteries remain unmeasured. Published research scores concern BF16 serving.

Each GGUF retains the original chat template. A separate
`chat-template-no-think.jinja` explicitly sets `enable_thinking` to false.
The tested llama.cpp reasoning-off flag alone still produced a reasoning
trace with the original template. The explicit template produced final
responses directly. Initial checks and corrected tests are retained locally.

## ZeroGPU interface

The public Gradio interface uses full-model revision
`d69f3ff69a4c5c2d84df56876354d6bdef170519` and requests the 96 GB ZeroGPU size
on demand. Defaults are temperature 0.7, top-p 0.8, top-k 20, thinking disabled,
512 maximum new tokens and seed 20261104. Sampling controls and the output cap
are adjustable. Responses stream, with one generation at a time.

The loader materializes one shard at a time and deletes each temporary
download afterward. Parameters are placed on CUDA during module startup,
following ZeroGPU's emulation workflow. Startup loaded all 55 shards and
packed the full model successfully.

| Check | Result | Observed wall time |
| --- | --- | ---: |
| Local Transformers, one sentence | Final response, 31 new tokens | 8.34 s |
| Hosted owner request, 32-token cap | Final lighthouse sentence | 7.49 s |
| Hosted anonymous request, 192-token cap | Completed short lighthouse story | 13.94 s |

These are functional checks with different output lengths and runtime
conditions. They supply no throughput benchmark. The public interface also
returns HTTP 200, exposes its streaming endpoint and enables Generate.

The account reports `isPro=false`. A manual restart API call returned a 403
message requesting PRO. Normal startup after the configuration update and
both real ZeroGPU predictions nevertheless succeeded. The initial assumption
that the denial blocked deployment was corrected after those results.

The model and Space started private. After the user requested an open
release, both model repositories and the Space became public. The model
access token was removed from the Space's secrets after publication.

## Retained artifacts

Release preparation, upload manifests, source revisions, checks, raw
generations, hosted predictions and download audits are under
`runs/hf-full250-release-20261105/`. Maintained publication and Space source
files are under `hf_release/`. The Space repository also contains its runnable
app, loader and pinned dependencies. The full model includes training recipe,
prompt metadata and source provenance.
