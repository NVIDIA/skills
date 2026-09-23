# Single-GPU vLLM and NeMo-Speech.cpp

vLLM serves the Nemotron LLM, and one NeMo-Speech.cpp service holds ASR and TTS on the same
device. NeMo-Speech.cpp is a native ggml runtime with no Triton and no Python, and its
container publishes `amd64` and `arm64` from one tag.

Read this file when `preflight.md` §4 routes here:

| Host class | Status |
| --- | --- |
| `dgx_spark` | only supported local stack, plus `platforms/dgx-spark.md` |
| `jetson_thor` | only supported local stack, plus `platforms/jetson-thor.md` |
| `workstation`, one user or a few concurrent sessions | recommended local stack |

A workstation serving many users uses NIM instead (`platforms/deployment.md`). DGX Spark and
Jetson Thor have no NIM path, so do not present NIM images, NIM profiles,
`NIM_MODEL_PROFILE`, `NIM_TAGS_SELECTOR`, Magpie NIM, or speech function ids there.

## Before Proposing

This checklist covers all three host classes. The platform file adds only its own identity
signals and coverage limits.

1. Plan against memory the probe reports as available, never advertised capacity. On DGX
   Spark and Jetson Thor the GPU, the operating system, the containers, and page cache all
   draw on one pool.
2. Verify every local slot against §Models, subject to any coverage limit the platform file
   states.
3. Build the runtime budget through §Memory. The speech reserve is measured, not estimated,
   and a draft model counts as a second resident model.
4. Mark provisional co-location until §Start and verify passes.
5. If a slot does not fit, move that slot to NVIDIA cloud and show the hybrid layout in the
   proposal. Never substitute a different model silently.

## Source of Truth

Read these before generating. No value here may come from memory.

| Source | Authority over |
| --- | --- |
| Locked model page in §Models | vLLM container and version, quantization variants, serve flags, parsers, required environment, request format |
| [vLLM `serve` CLI reference](https://docs.vllm.ai/en/stable/cli/serve/) | each flag's spelling, meaning, and current default |
| [`nemo-speech.cpp` on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-speech.cpp) | image tag, architectures, container command line |
| [NeMo-Speech.cpp repository](https://github.com/NVIDIA/NeMo-Speech.cpp) | engine keys ([server](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/server.md), [ASR](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/asr/configuration.md), [TTS](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/tts/configuration.md)), model layout, boosting, text normalization, gRPC limits |
| Hugging Face page per speech model | exact GGUF file names and revisions |

Pass `platforms/readiness.md`, then generate services through `output-contract.md`. This
skill ships no static Compose file.

## Models

| Pipeline | LLM page | Repository family | Speech |
| --- | --- | --- | --- |
| Cascaded | [Nemotron 3.5 Lightning](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b?nim=self-hosted) | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-<variant>` | streaming ASR plus Magpie TTS in one service |
| Omni | [Nemotron 3 Omni](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-<variant>` | TTS only, because Omni performs ASR |

`<variant>` is resolved by §Precision. Read the page for the exact ids rather than
assembling one. The Omni display name and its `nano-omni` ids differ by design
(`models/llm.md` §The Omni name and its ids differ).

Lock locales through `models/language-routing.md`. Resolve speech models through
`models/asr.md` and `models/tts.md`, which route back here.

### Read from the Page

Open the page for the slot being built, immediately before generating. Take:

- exact container tag and CUDA variant. The two models do not pin the same vLLM version
- the platform-specific section when one exists, in preference to the generic example
- extra packages the image lacks (§Omni needs audio extras)
- available precision variants, for §Precision
- reasoning and tool-call parser names
- required environment variables
- sampling values for thinking mode and instruct mode. Voice uses instruct unless reasoning
  was approved
- multimodal limits and media path allowance, which Omni needs before it accepts audio
- per-architecture backend notes, for example a different MoE backend on some cards
- supported response languages, against `models/language-routing.md` §LLM
- published draft model, if any

The page's hardware-compatibility summary is not the authority. Its vLLM section is.

A published launch command is an example, not a budget. These commonly serve a very long
context and size memory for a device not also running speech. Calculate memory and context
yourself in §Memory.

Authenticate with `HF_TOKEN` and verify with `hf auth whoami`. vLLM downloads from the Hub
directly unless the page instructs otherwise.

### Omni Needs Audio Extras

The published vLLM image ships without audio support. Install `vllm[audio]`, pinned to the
image's vLLM version, before `vllm serve`.

Without it the server starts, `/v1/models` answers, and text requests succeed. Only the
first spoken turn fails. Prove the audio path in smoke, not at the microphone. Cascaded does
not need this.

## Precision

Resolve precision per model and GPU. A checkpoint's name and its execution kernel are
different facts. A NIM profile can also name a different precision, because NIM and raw
vLLM use different artifacts. NIM selection belongs to `models/llm.md` §Cascaded LLM NIM
fit or `frameworks/omni.md` §Workstation NIM.

For Cascaded Lightning, use the published NVFP4 checkpoint on the hardware its card
supports:

| Host | Lightning checkpoint and execution | Speculative decoding |
| --- | --- | --- |
| DGX Spark | NVFP4 checkpoint through the published W4A16 Marlin recipe | the DGX Spark draft model |
| Jetson Thor | NVFP4 checkpoint through the published platform recipe | none |
| Blackwell workstation | NVFP4 checkpoint with the backend published for that GPU | the workstation draft model published for that architecture |
| Hopper | NVFP4 checkpoint through W4A16 kernels | only when the card publishes a draft recipe for that workload |
| Ampere | NVFP4 checkpoint through W4A16 kernels | none |
| Ada | use only when the current Lightning page publishes an explicit recipe for the exact GPU |
| Older than Ampere | unsupported. Propose cloud |

Do not replace the Lightning NVFP4 checkpoint with BF16 merely because the GPU executes it
through W4A16 kernels. That increases the weight footprint and can reject a valid local
deployment.

For Omni, do not reuse the Lightning table. Its page publishes separate BF16, FP8, and
NVFP4 repositories. Select the lightest variant with an explicit recipe for the exact GPU,
and propose cloud when the page provides none.

Confirm the repository, execution backend, draft model, flag names, and token count on the
locked page and the vLLM reference. Speculative flag spelling changes between releases.

A draft model is a second download and a second resident model, so budget it in §Memory.

When compute capability was derived from the GPU name rather than read from the driver
(`preflight.md` §Compute capability fallback), confirm it before selecting a variant.

## One-Time Speech Model Setup

Do this before configuring the service, which takes these files as paths. GGUF files load
from disk. Fetch once per machine, keep the tree outside the generated project, mount
read-only.

| Asset | Repository |
| --- | --- |
| Streaming ASR GGUF, English | `nvidia/nemotron-speech-streaming-en-0.6b` |
| Streaming ASR GGUF, multilingual | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| Magpie TTS GGUF and `.nemo` archive | `nvidia/magpie_tts_multilingual_357m` |
| NanoCodec decoder GGUF | `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` |
| Text-normalization grammars | NeMo-Speech.cpp GitHub release asset |

Resolve file names and revisions from each page at fetch time. Quantized names carry a
quantization and version, so a remembered name goes stale.

Four rules decide whether the service starts:

1. **The Magpie tokenizer is in the `.nemo` archive, not the GGUF.** Extract it and pass the
   directory. TTS enables only with the Magpie model, the codec model, and the tokenizer
   directory all present.
2. **The grammars are not on Hugging Face.** They are a release asset pinned to the release
   matching the container tag. Verify the checksum, extract to a staging path, and replace an
   installed set only after the expected files are present.
3. **Assert the grammar layout.** Either a single grammar directory or a parent with
   language-named children, each holding `tokenize_and_classify.far` and `verbalize.far`.
4. **Run the download as the account that owns the path, never with `sudo`**
   (`platforms/readiness.md` §Speech model tree).

Without normalization the service speaks digits, dates, and currency literally. Treat a
missing grammar directory as a defect to fix before handover, and disclose it if the user
accepts it. A container built without normalization support logs a warning at startup and
passes text through unchanged, so read that line rather than assuming the flag took effect.

## Speech Service

One gRPC binary, Riva-compatible. No HTTP listener, no playground, no client tools. Take the
image tag and run command from the NGC page.

One service covers both directions. Capabilities enable from the model paths passed, so
cascaded passes ASR and TTS paths and Omni passes TTS paths only. No enabled capability is
an error.

Precedence is defaults, YAML, environment, then command line. Environment names are the
dotted key uppercased with `.` and `-` becoming `_`, so `asr.model.path` becomes
`NEMO_SPEECH_ASR_MODEL_PATH`. **Unknown keys are errors, not warnings**, so a typo is a
startup failure. Use canonical dotted names, because the short aliases belong to the HTTP
binary.

### Set Explicitly

| Setting | Key | Note |
| --- | --- | --- |
| ASR model | `--asr.model.path` | cascaded only, required for ASR |
| ASR device | `--asr.backend.gpu` | device index, `-1` is CPU |
| Mid-stream end of utterance | `--asr.endpointing.enable` | **off by default.** A live agent requires it |
| End-of-utterance silence | `--asr.endpointing.stop_history_eou_ms` | default 800, usually tuned lower for conversation |
| Magpie token generator | `--tts.magpie-model` | required for TTS |
| NanoCodec decoder | `--tts.codec-model` | required for TTS |
| Tokenizer directory | `--tts.tokenizer-model-dir` | required for TTS, the extracted `.nemo` directory |
| Normalization grammars | `--tts.tn-model-dir` | required for spoken numbers, dates, currency |
| Default locale | `--tts.language-code` | defaults to `en-US` |
| Default voice | `--tts.voice-name` | confirm against the running service |
| TTS CPU threads | `--tts.threads` | Magpie and codec run on CPU |
| Listen address | `--bind` | `0.0.0.0:50051` by default |

Endpointing is the trap. At its default the server emits one final result only when the
client closes the stream, so a live conversation never produces a turn boundary and the
agent appears deaf to the end of a sentence. Verify a mid-stream final in smoke.

### Readiness Is gRPC Only

No health URL exists, so generate no HTTP health check. Prove readiness with
`GetRivaSynthesisConfig`, plus `GetRivaSpeechRecognitionConfig` when cascaded. A TCP connect
shows the port is open, not which model loaded.

### Languages and Voices

Implements `Synthesize`, `SynthesizeOnline`, and `GetRivaSynthesisConfig`. Takes plain text,
returns 16-bit linear PCM at the codec sample rate.

Some Magpie languages are build-time options that are off by default, so the container can
serve fewer languages than the model page lists. `GetRivaSynthesisConfig` is the authority
for this deployment and the page is the upper bound. Disclose it when the requested locale is
one of the optional ones.

Voice names accept a local name, a zero-based speaker index, or a model-qualified name. Lock
only a voice the running service returned.

### gRPC Limits

Confirm on current documentation, and disclose when they touch the use case:

- mono 16-bit linear PCM only, so compressed inbound audio needs client-side transcoding
- one alternative per result, whatever maximum-alternatives the request carries
- confidence is not comparable to NIM. Interim results report zero, cache-aware RNNT finals
  report a fixed value

## LLM with vLLM

Use the image and version the locked page requires, serve the exact repository id, and
translate the page's launch command into the generated `llm` or `omni` service. This is raw
vLLM, so:

- `HF_TOKEN` supplies model access. `NIM_MODEL_PROFILE` and `list-model-profiles` do not
  apply
- the Omni repository name does not enable reasoning by itself
- reasoning off has two parts: `extra_body.chat_template_kwargs.enable_thinking` false in
  agent requests, and the page's reasoning parser in the serve command
  (`models/llm.md` §Reasoning parser)
- add no flag the page does not list unless a fatal log names it
  (`models/llm.md` §Serve flags)
- put the served id from `/v1/models` in the agent, not the repository name
- client traps are the same as any local endpoint (`frameworks/pipecat.md` §Local LLM wiring
  or `frameworks/livekit.md` §NVIDIA models)

### First Boot Compiles Kernels

The NVFP4 path can compile kernels on device at first boot, observed at around 45 minutes
before the server answers. Rising `nvcc` and `cicc` output is progress, not a hang.

Mount a persistent kernel cache beside the Hugging Face cache, with the location resolved
from the vLLM docs for the pinned version. Without it every container replacement recompiles.
No serve flag shortens this.

### Host Memory at Startup

vLLM's startup check has been observed to read Linux **free** memory rather than
**available** memory. Page cache counts against it, so a host with plenty of reclaimable
memory can fail to start after downloads and engine builds fill the cache.

Read both numbers. When free memory is the blocker, reclaim page cache or restart the host
rather than lowering the model configuration. This and a real device OOM look alike and
their fixes are opposite.

## Memory

`--gpu-memory-utilization` and `--max-model-len` are **required Compose entries**, not
post-OOM tuning, because vLLM and speech share one device. The fraction is per-instance
against total device memory and does not account for memory another process holds, so vLLM
will not leave room for speech on its own.

```text
usable     = available_memory - startup_headroom
llm_budget = usable - measured_speech_reserve
```

`available_memory` is the lower of CUDA free memory and host available memory on DGX Spark
and Jetson Thor, which share one pool. On a workstation it is CUDA free memory alone.

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
```

**Measure the speech reserve, do not estimate it.** Start speech alone, let warmup finish,
read both numbers again. The difference is the reserve.

On DGX Spark and Jetson Thor prefer a fixed conservative fraction over one derived at launch,
because free memory moves between a cold boot and a boot after downloads. On a workstation
with dedicated VRAM, derive it from measured free VRAM. Either way round down, leave startup
headroom, and raise it only once the full stack is stable.

Set context to the smallest length the approved use case needs. KV cache grows with context
and a voice turn is short. The repository config carries its own default, so omitting the
flag inherits that. Prefer an absolute KV cache size over a fraction when the current
reference documents one.

Recalculate when context, speech models, thread count, or the draft model changes. If the
budget cannot hold weights, overhead, and a usable KV cache, lower `--max-model-len` first,
then the fraction, then move a slot to cloud.

## Start and Verify

Start speech first, because its footprint is fixed by the models it loads while the LLM
fraction is set against what remains. This is the opposite of the NIM order in
`platforms/deployment.md`.

Assert each result rather than reading a step as done because the previous one produced
output.

1. Start speech alone. `GetRivaSynthesisConfig`, and assert the loaded TTS model, the
   compiled-in languages, and a returned voice.
2. Cascaded: `GetRivaSpeechRecognitionConfig`, and assert the loaded ASR model and language.
3. Measure memory again with speech resident. Write the resulting fraction into Compose
   before starting vLLM.
4. Start vLLM. Wait for `/v1/models` and confirm the locked served id.
5. Prove the input path. Cascaded: one streaming chat completion with reasoning off,
   asserting non-empty `delta.content` and empty or absent `delta.reasoning_content`. Omni:
   one minimal audio request asserting a non-empty text response.
6. Cascaded: synthesize one sentence in the approved locale and feed it back through
   streaming ASR. This is the only check that proves both directions before a microphone,
   and a mid-stream final is what proves endpointing is on.
7. `scripts/smoke.sh`.
8. Start the framework agent.
9. Complete a spoken exchange (`operations/run.md`).

Only after step 9 may the README status change from `provisional co-location` to
`self-hosted, co-located`.

Speech loads GGUFs and warms up rather than building a TensorRT engine, so it starts far
faster than a speech NIM. A long wait there is a symptom to read in the logs. The long first
boot on this stack belongs to vLLM.

## Resource Sharing

Do not prescribe CPU pinning or compute splits. Start from the defaults the current vLLM,
container, and platform releases support. Memory is the exception, set explicitly in §Memory.

Magpie and codec work runs on CPU threads, so leave CPU headroom for the configured thread
count. If measured contention causes audio glitches, lower that count.

## Anti-Patterns

- Following NIM instructions on DGX Spark or Jetson Thor, or using `NIM_MODEL_PROFILE`,
  `NIM_TAGS_SELECTOR`, or a speech function id on this stack.
- Generating an HTTP health check for a gRPC-only service.
- Leaving ASR endpointing at its default, then debugging turn detection in the client.
- Starting vLLM before the speech reserve is measured.
- Copying a page's launch command verbatim, which leaves the memory default in place and
  sizes context for a device not running speech.
- Deriving the memory fraction from launch-time free memory on a unified-memory pool.
- Carrying a draft model across architectures, or adding one to Jetson Thor.
- Serving Omni without the audio extras, installing them unpinned, or calling it working
  from a text request.
- Copying the cascaded vLLM version, tag, or flags onto Omni.
- Assembling a repository id instead of reading it, or dropping `nano` from an id.
- Passing the Magpie GGUF without the tokenizer directory and codec model.
- Treating the model page's language list as what the container serves.
- Shipping without normalization grammars and calling literal digits a model limitation.
- Hardcoding a voice the service never returned.
- Treating advertised unified memory as available to vLLM.
