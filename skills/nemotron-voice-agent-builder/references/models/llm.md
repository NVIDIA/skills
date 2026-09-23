# LLM

Cascaded Nemotron 3.5 Lightning (default), with Nemotron 3 Super / Ultra when named, plus
Nemotron 3 Omni for the Omni pipeline. Resolve before the proposal table. Self-hosted fit is
mandatory. Cloud skips platform fit.

Self-hosted quantization is verified by the documentation family for the routed service:

| Routed stack | Quantization check | Owned by |
| --- | --- | --- |
| Cascaded LLM NIM, on a workstation serving many users | LLM NIM support matrix, then `list-model-profiles` on the assigned GPU | §Cascaded LLM NIM fit below |
| Omni NIM, on a workstation serving many users | VLM NIM support matrix, then VLM `list-model-profiles` on the assigned GPU | `frameworks/omni.md` §Workstation NIM |
| vLLM plus NeMo-Speech.cpp, on DGX Spark, Jetson Thor, or a low-concurrency workstation | the variant the locked page publishes for the probed compute capability | `platforms/single-gpu.md` §Precision |

Never apply an LLM NIM matrix or profile rule to Omni. Raw vLLM has no NIM profile.

## Discover

| Step | Where |
| --- | --- |
| Family browse | [NVIDIA Nemotron](https://developer.nvidia.com/topics/ai/nemotron#section-nvidia-nemotron-models) |
| Confirm id / run | `build.nvidia.com/nvidia/<model>` · self-hosted: `?nim=self-hosted` |
| Card / API | `…/<slug>/modelcard` · `docs.api.nvidia.com/nim/reference/nvidia-<slug>`, with dots in the slug written as dashes |

### Pages to Open

Every slot has a build page that carries deployment and an API reference that carries the
request contract. Open both for the slot being built.

| Slot | Build page | API reference |
| --- | --- | --- |
| Cascaded LLM | [Nemotron 3.5 Lightning, self-hosted](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b?nim=self-hosted). Drop the parameter for cloud | [Lightning](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-5-lightning-30b-a3b) |
| Omni | [Nemotron 3 Omni](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) | [Omni](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning) |
| Super, only when named | [Super card](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard) | [Super](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-super-120b-a12b) |

The API reference host replaces every dot in the slug with a dash, so
`nemotron-3.5-lightning-30b-a3b` becomes `nvidia-nemotron-3-5-lightning-30b-a3b`. Carrying
the dot across returns 404, so use these links rather than assembling one from §Discover.

These models are the roster. Resolve every proposal from these pages, never from recall of
what the family used to contain, and confirm the slot's model is still current when the page
opens. If a page no longer serves it or marks it superseded, report that and propose the
current model. When a user names a model off the roster, say so and wait
(`models/catalog.md` §Shared rules).

### The Omni Name and Its Ids Differ

The product is **Nemotron 3 Omni**, and its slug, repository, and image ids carry
`nano-omni`. Use the name for people and the ids verbatim in code. Dropping `nano` from a
slug gives a 404, and from an image id a failed pull.

Omni is the audio-native multimodal model. It is not a cascaded text LLM, so exclude it from
that slot.

### Reading These Pages in Realtime

| Page | What a fetch returns |
| --- | --- |
| Omni build page | full self-host content: precision variants, required vLLM version, serve command, a DGX Spark section, and mode parameters |
| Lightning build page `?nim=self-hosted` | the model card only. The deploy panel renders in the browser, so the `nvcr.io` image and run command are usually **not** in fetched content |
| Either API reference | server-rendered and reliable to fetch. Use it for the request contract when a build panel does not come back |

The Lightning card still gives the supported languages, context limit, and quantization
recipe. It does not give the container.

When the panel is not in what you fetched, do not build an `nvcr.io` tag from the slug and a
guessed version. Read the tag from that NIM's NGC catalog page, or ask the user to paste the
panel. State which you used, because an image tag with no source is not verified.

Every card lists the languages the model supports, and that list is usually shorter than the
speech coverage around it. Check the approved response language against it through
`models/language-routing.md` §LLM before locking this slot.

### Resolve Three Names

The catalog slug, the container image, and the served model id are three different
strings. Treating them as one causes `Access Denied` on pull and a wrong model id at
request time.

| Name | Source | Used by |
| --- | --- | --- |
| Container image | the self-hosted build page or NGC command that actually pulls | `compose.yaml` |
| Profile / tags | `list-model-profiles` on the assigned GPU | `NIM_MODEL_PROFILE` |
| Runtime model id | `GET /v1/models` after the service is healthy | agent code |

The slug on `build.nvidia.com/nvidia/<slug>` records the proposal intent only. It is not
proof of the image URI and not proof of the served id. Do not bake it into agent code
until the listing returns a value. A raw vLLM card commonly passes
`--served-model-name`, so the served id can be a short alias that matches neither the slug
nor the repository name.

Query the NIM port for self-hosted, or `https://integrate.api.nvidia.com/v1/models` for
cloud. Drop `embed`, `guard`, and `vlm` entries. If nothing is up yet, use the cloud
listing and re-check after start.

## Reasoning (Voice Default: Off)

Model default is **on** (`enable_thinking: true` when omitted). That burns turn latency
and can leak into TTS (`<think>`, silence, `redacted_thinking`).

| Intake row | When |
| --- | --- |
| off (propose this) | normal voice agent |
| on, budget 8192 | user asked hard multi-step / specialized. Warn if budget > 16384 |

Verify on the locked slot's API reference in §Pages to open, alongside
[NIM reasoning](https://docs.nvidia.com/nim/large-language-models/latest/reasoning-model.html).

The build page is the faster of the two checks, because its prototype panel shows a runnable
request for that exact model with the reasoning fields already in position. Read that
example and the API reference before generating the request, because field names and nesting
move between model generations.

With an OpenAI-compatible SDK these provider fields travel in `extra_body`, which
merges them into the top level of the request body. A raw HTTP call sets them at the
top level directly. Passing `chat_template_kwargs` as a plain SDK keyword argument is
the error, because the client rejects the unknown argument.

```python
# Off (voice default)
extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

# On (table says so)
extra_body = {
    "chat_template_kwargs": {"enable_thinking": True},
    "reasoning_budget": 8192,
}
```

Prefer top-level `reasoning_budget` when the API lists it (some docs also nest it under
`chat_template_kwargs`). When on, size completion tokens from the card's recommendation,
because a thinking trace plus a spoken answer needs far more than a voice turn alone. Treat
≥1024 as a floor rather than a target, and bound thinking with `reasoning_budget` instead of
starving the answer. Pass `extra_body` the way the framework MCP documents. Do not
invent the wrapper. Frameworks rarely accept a raw request dict, so see where the payload
actually goes in `frameworks/pipecat.md` §Local LLM wiring or
`frameworks/livekit.md` §NVIDIA models.

Cloud still needs `enable_thinking` in `extra_body`. Sampling changes with the mode, and a
card commonly gives separate values for reasoning, for reasoning off, and for tool calling,
so take the set that matches the approved row rather than reusing one. Toggle = edit
constants + restart, not mid-session, unless the framework docs say otherwise.

### Reasoning Parser

A model card's serve command enables the reasoning parser independently of each request's
thinking mode. **Keep the parser enabled in both modes.** It separates thinking from
`content` when reasoning is on and applies the model's reasoning-off contract when
`enable_thinking:false`. TTS must read only `content`.

The serve command therefore does not change with the reasoning row. Only the request does,
through `extra_body.chat_template_kwargs.enable_thinking`. With reasoning on, also keep
`reasoning_content` away from TTS.

Parser names and plugin requirements are not shared across this family, so read them from
the locked card for the server you actually run. Both current vLLM paths use
`--reasoning-parser nemotron_v3` alongside
`--enable-auto-tool-choice --tool-call-parser qwen3_coder`, which is a value to confirm
rather than one to copy. When a card names a separate parser-plugin file, download it and
have it present before the server starts. Never adapt a vLLM row for a different server
such as SGLang.

Prove the result on the streaming endpoint in both modes. `scripts/smoke.sh` asserts
non-empty `delta.content` and empty or absent `delta.reasoning_content` when reasoning is
off (`output-contract.md` §Smoke Before Client). One non-streaming answer does not prove it.

## Serve Flags

Every serve-time flag must trace to one of three sources: the locked model card, the memory
controls in §Runtime memory controls, or a fatal log line that names the flag. Nothing else belongs on the
command line.

Performance and scheduling flags are the usual trap. Reaching for eager mode, a scheduling
toggle, or CUDA MPS variables to chase a slow or unstable first boot adds variables while the
real cause is normally an engine build, a kernel compile, or the memory budget. Each
speculative flag then has to be reverted, and a revert is indistinguishable from a fix.

When a flag looks necessary, name the log line that requires it, change that one thing, and
rerun `scripts/smoke.sh` before touching anything else.

## Cascaded LLM NIM Fit

Hardware first from `preflight.md`. Never propose model / precision / TP until verified
for the probed GPU.

This section covers a cascaded LLM on LLM NIM only. Omni NIM uses the VLM NIM sources in
`frameworks/omni.md` §Workstation NIM. The single-GPU stack uses
`platforms/single-gpu.md` §Precision.

### 1. Support Matrix HTML

Fetch HTML only (not UI dropdowns, not `.html.md`):

`https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html`

Match rows using `data-model`, `data-precision`, `data-tp`, and `data-gpus`. Normalize the
probed name (`NVIDIA H100 80GB HBM3` → `NVIDIA-H100-80GB-HBM3`). Keep a row only when
short name, precision, and SKU all match. No row → unsupported. Pick another precision,
TP, model, or cloud. Do not guess a close SKU.

| Model | Section |
| --- | --- |
| Lightning | [#nemotron-3-5-lightning-30b-a3b](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html#nemotron-3-5-lightning-30b-a3b) |
| Super | [#nemotron-3-super-120b-a12b](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html#nemotron-3-super-120b-a12b) |
| Ultra | [#nemotron-3-ultra-550b-a55b](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html#nemotron-3-ultra-550b-a55b) |

“Verified GPUs” under a section is overall. It does not replace per-row `data-gpus`.

### 2. Select a NIM Model Profile

Before proposal, use the support matrix to select a candidate profile and mark shared-GPU
fit provisional. After the user approves self-hosting and container readiness passes, run
`list-model-profiles` on that NIM for the probed GPU. Standard `*/server` deployments use
the result to verify manifest auto-selection. Explicitly pinned self-hosted deployments use
the exact result in generated Compose. Profile verification is mandatory on every NIM LLM
deployment. Buckets:

```bash
docker run --rm --gpus '"device=0"' <nim_llm_image> list-model-profiles
```

Use the exact GPU assignment that the LLM service will see. GPU `0` is the standard
`*/server` assignment. `generic-assistant/server-perf` uses `device=2,3`. Copy
`<nim_llm_image>` from the checked-in Compose service or current self-hosted build page.
Do not infer the image or tag.

| Bucket | Action |
| --- | --- |
| Compatible | use it |
| Low memory | apply the suggested `--max-model-len` / `NIM_MAX_MODEL_LEN` if the user accepts |
| Incompatible | use lower precision on the same image, or a smaller LLM |

Prefer the lightest Compatible precision that fits, in this order: `nvfp4`, `mxfp4`,
`w4a16`, `fp8`, `bf16`. On a local GPU under about 90 GB, prefer Compatible `nvfp4` when
listed. Do not default to a heavier precision when a lighter one is Compatible, unless the
user asked for it.

For an explicitly pinned self-hosted LLM deployment, set `NIM_MODEL_PROFILE` in the launch
environment to one of:

- a description such as `vllm-nvfp4-tp1-pp1` (pattern:
  `<backend>-<precision>-tp<N>-pp1`, backend `vllm` | `sglang` | `trtllm`)
- the 64-char profile id from the listing

For a standard `*/server` deployment, leave `NIM_MODEL_PROFILE` unset so the manifest
can auto-pick.

For a pinned LLM, use `NIM_MODEL_PROFILE` instead of `NIM_TAGS_SELECTOR`. Standard
`*/server` LLMs leave it unset for automatic hardware-compatible selection. The
`generic-assistant/server-perf` benchmark pins
`NIM_MODEL_PROFILE=vllm-nvfp4-tp2-pp1-18.0`. ASR and TTS continue to use their required
`NIM_TAGS_SELECTOR` values. Cloud LLM skips this section. For same-image OOM or latency,
re-run `list-model-profiles`, select a lighter profile or shorter max length, update the
launch command, then health-check. A different model id requires discovery again.

For an explicitly pinned shared-GPU deployment, use the exact Compatible profile and
verify startup logs report the expected backend and precision. For standard `*/server`
auto-selection, verify the selected profile and remaining memory after startup. A profile
being Compatible means the LLM can run on that GPU. It does not mean LLM + ASR + TTS fit
together.

Docs:
[model profiles and selection](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html).

### 3. Precision and Weights for Planning

Matrix / container win over these hints.

| GPU | Precision default |
| --- | --- |
| Blackwell | NVFP4 when supported by the locked model path |
| Ada / Hopper (CC ≥ 8.9) | NVFP4 when Compatible, else the lightest Compatible profile (W4A16 for Lightning, FP8 for Super / Ultra) |
| Ampere and older | W4A16 for Lightning, otherwise BF16 (large, often needs its own GPU) |

Precision is a tag inside `NIM_MODEL_PROFILE`, for example `vllm-bf16-tp1-pp1`, not a separate
env. The Compatible profile listing wins over this planning table. Lightning NVFP4 requires
Blackwell (SM 10.0+), so Ampere and Hopper hosts use W4A16 rather than FP8. Read the row
rather than assuming a precision exists.

| Slot | Weights (est.) |
| --- | --- |
| Lightning NVFP4 | ~19 GB |
| Lightning W4A16 | ~19 GB |
| Lightning BF16 | ~63 GB |

The support matrix lists exact min VRAM per GPU per precision and TP; use it over these
estimates. Super / Ultra: on one workstation GPU, say they will not fit. Offer Lightning
local or that model in the cloud. Do not silently substitute.

## Runtime Memory Controls

This section covers both NIM families and raw vLLM. These controls are required in the
generated `compose.yaml` on any shared GPU, not applied after an OOM. Choose the control
from the actual server path, never from the pipeline name.

| Path | Controls | Flag source | Budget owner |
| --- | --- | --- | --- |
| Cascaded LLM NIM | `NIM_MAX_MODEL_LEN`, plus exactly one documented runtime memory limit, either `--gpu-memory-utilization` through `NIM_PASSTHROUGH_ARGS` or `NIM_KVCACHE_PERCENT` when the image documents it | [LLM NIM environment variables](https://docs.nvidia.com/nim/large-language-models/latest/reference/environment-variables.html) · [LLM NIM memory troubleshooting](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/memory.html) | `preflight.md` §NIM budget |
| Omni NIM | `NIM_MAX_MODEL_LEN`, plus only the runtime memory limit its current VLM NIM instructions document | [VLM NIM environment variables](https://docs.nvidia.com/nim/vision-language-models/2.0.4-variant/environment-variables.html) · `frameworks/omni.md` §Workstation NIM | `preflight.md` §NIM budget |
| Raw vLLM on the single-GPU stack | `--max-model-len`, `--gpu-memory-utilization` | locked model card, then [vLLM `serve` CLI reference](https://docs.vllm.ai/en/stable/cli/serve/) for meaning and current defaults | `platforms/single-gpu.md` §Memory |

Both servers fill their runtime budget with KV cache after weights and overhead, and the
documented default fraction is 0.9 or higher on each. That default starves co-located
speech even when quantized weights are small, so read the current default from the flag
source and set the value from the budget owner instead of accepting it.

Four rules hold on both paths:

1. **Never set both memory controls**, and never copy either value from another image or
   model version.
2. **Set context to the smallest length the approved use case needs.**
3. **A cap that leaves no usable KV cache means the layout does not fit.** Take a lighter
   build, which is a smaller Compatible profile on NIM and a lighter quantization on raw
   vLLM, or move a slot. Do not keep lowering the cap to force co-location.
4. **Verify the measured budget before each service starts**, through
   `platforms/deployment.md` §Shared-GPU memory gate on NIM and
   `platforms/single-gpu.md` §Start and verify on the single-GPU stack. Do not let TTS
   discover an invalid LLM budget through repeated OOM restarts.

## Anti-Patterns

- Voice agent with reasoning left at model default (on).
- Serving without the reasoning parser, or without a plugin file the card requires, in
  either reasoning mode. Dropping the parser because the reasoning row is off.
- Adding a serve flag the card does not list and no fatal log demands.
- `chat_template_kwargs` passed as a plain SDK argument rather than through `extra_body`.
  Reasoning flags in `.env`.
- Cascaded slot → omni model. Model ids in `.env`.
- Propose from memory / Verified GPUs list alone. Click matrix UI or use `.html.md` for fit.
- Override a failed matrix / `list-model-profiles` check. Guess a SKU. Use the deprecated
  `NIM_TAGS_SELECTOR` on a generated LLM deployment.
- Validate Omni with the LLM NIM matrix or copy a cascaded LLM profile onto Omni NIM.
- Treat quantized weight size as total LLM VRAM or leave vLLM at its default memory budget
  while co-locating speech.
