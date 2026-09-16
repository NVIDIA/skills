# Omni

Read with `frameworks/pipecat.md` when the approved pipeline is Omni. LiveKit is not
supported.

Omni replaces ASR and the text LLM with one audio-in model. TTS remains separate:

```text
transport → user context → Omni service → TTS → transport → assistant context
```

Do not add an ASR service, ASR NIM, or transcription-dependent turn strategy.

## Sources

Pipecat does not provide the required Omni service. Use NVIDIA's
[`nvidia_omni_multimodal_service.py`](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/src/examples/omni_assistant/nvidia_omni_multimodal_service.py)
directly. Copy the current upstream file into the generated project and use the service and
settings classes imported by the current upstream pipeline. They are currently
`NvidiaOmniLLMService` and `NvidiaOmniSettings`. Do not reimplement or simplify them.

Also copy NVIDIA's current
[`audio_only_smart_turn_strategy.py`](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/src/examples/omni_assistant/audio_only_smart_turn_strategy.py).
Pipecat's stock Smart Turn stop strategy waits for an upstream `TranscriptionFrame`, which
does not exist before an Omni audio turn completes.

Generate the surrounding Pipecat pipeline, transport, context, turn handling, and TTS
wiring from the Pipecat docs MCP. Use NVIDIA's current
[`omni_assistant/pipeline.py`](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/src/examples/omni_assistant/pipeline.py)
as the behavioral reference. Read the locked model's API reference or Hugging Face model
card for endpoint settings.

## Model and Deployment

Resolve the exact model through `models/llm.md`. Keep reasoning off for voice.

One page carries the model, both self-hosted paths, and the request format:

[Nemotron 3 Omni](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning)

Read it at build time. It publishes the precision variants, the pinned vLLM container, the
serve command, a DGX Spark section, the thinking and instruct mode parameters, and the
audio request shape. Fetching it returns that content, so there is no reason to work from
memory here. Its API reference in `models/llm.md` §Pages to open carries the request
contract, and both pages answer a fetch.

| Routed stack | Source |
| --- | --- |
| Cloud | model id from `/v1/models`, API shape from the page above |
| Workstation NIM | §Workstation NIM below. Use VLM NIM documentation, not LLM NIM documentation |
| Single-GPU vLLM plus NeMo-Speech.cpp | the vLLM section of the page above, plus `platforms/single-gpu.md` |

Omni self-hosts on both local stacks. It has its own NIM for the workstation NIM path and
its own published checkpoints for the single-GPU stack, so the source follows the routed
stack rather than the pipeline shape. `preflight.md` §4 selects that stack.

The model is named **Nemotron 3 Omni**, while its slug, repository ids, and NIM image still
carry `nano-omni`. Use the product name in the proposal and the README, and the published
ids verbatim everywhere they are strings
(`models/llm.md` §The Omni name and its ids differ).

Budget Omni plus TTS only. `NIM_MODEL_PROFILE` applies only on the NIM path. It never
applies to raw vLLM.

Omni is more memory-hungry than the cascaded LLM on the single-GPU stack, and its coverage
is not identical across platforms. Check the coverage row in `platforms/dgx-spark.md` or
`platforms/jetson-thor.md` before proposing it locally.

### Workstation NIM

Omni is a VLM NIM. Open these current sources before proposing or generating it:

- [Omni VLM NIM support matrix](https://docs.nvidia.com/nim/vision-language-models/2.0.4-variant/support-matrix.html#nemotron-3-nano-omni-30b-a3b-reasoning) for verified GPU, memory, precision, and GPU count
- [VLM NIM quickstart](https://docs.nvidia.com/nim/vision-language-models/2.0.4-variant/get-started/quickstart.html) for the current image and launch shape
- [VLM NIM profiles](https://docs.nvidia.com/nim/vision-language-models/2.0.4-variant/profiles.html) and [utilities](https://docs.nvidia.com/nim/vision-language-models/2.0.4-variant/utilities.html) for profile selection and `list-model-profiles`
- [VLM NIM environment variables](https://docs.nvidia.com/nim/vision-language-models/2.0.4-variant/environment-variables.html) for model, context, cache, and runtime controls

Resolve the release version from the current Omni build page before following versioned
VLM documentation. Select a candidate precision and GPU count from the Omni row in the VLM
matrix. After readiness passes, run that image's `list-model-profiles` on the exact GPU set
and either accept automatic selection or pin a Compatible profile returned there. Verify
the selected profile in startup logs and the served id through `/v1/models`.

Do not use the LLM NIM support matrix, LLM profile naming assumptions, or LLM NIM memory
variables for Omni. Continue through `platforms/deployment.md` for co-location with TTS.

## Audio Is Not in the Image

On the single-GPU stack, the published vLLM image ships without the audio extras, so Omni
will serve text and refuse audio until `vllm[audio]` is installed in the container, pinned
to that image's vLLM version. It also needs explicit per-prompt modality limits and a media
path allowance before it will accept an audio request.

That combination is why an Omni agent can pass a health check, answer a text prompt, and
still fail on the first spoken turn. Prove the audio path in smoke rather than at the
microphone (`platforms/single-gpu.md` §Omni needs audio extras).

A self-hosted Omni endpoint hits the same client traps as the cascaded LLM in
`frameworks/pipecat.md` §Local LLM wiring: an API-key argument may be required even
locally, reasoning-off has to arrive nested inside `extra_body`, and the model string is
the served id from the endpoint rather than the catalog slug.

For domain vocabulary, follow `domain/speech-customization.md`. Only TTS pronunciation
customization applies because Omni has no ASR slot, and the single-GPU stack does not
support even that.

## Turn Handling

Wire transport audio and context into the copied `NvidiaOmniLLMService`, then route its
text output to TTS.

- Start turns from VAD and stop them with `AudioOnlySmartTurnStopStrategy`.
- Use Pipecat's current `MuteUntilFirstBotCompleteUserMuteStrategy` so microphone input
  cannot race the first response.
- Follow the current upstream pipeline for the connect greeting and use one greeting path
  only. The current Omni service supports text turns.
- When user transcripts are enabled, request the structured response shape expected by
  the upstream service. It parses the JSON and sends only `response` text to TTS.
- Keep reasoning off through the request settings in `models/llm.md`.

## Verify

- `scripts/smoke.sh` passes, using the Omni audio request rather than a chat completion.
  A passing text request proves nothing about the audio path.
- No ASR service or ASR sidecar exists.
- The upstream Omni service and audio-only Smart Turn strategy were copied unchanged.
- The model endpoint serves the locked Omni model, and the agent uses its exact served id.
- User audio produces a coherent response and, when supported, a user transcription.
- Exactly one greeting plays and microphone input begins after it completes.
- TTS never speaks reasoning, JSON wrappers, or transcript metadata.
- Barge-in cancels the current Omni response.
- Complete the spoken exchange in `operations/run.md`.

## Anti-Patterns

- LiveKit Omni.
- Stock text-only LLM service or separate ASR used in place of Omni.
- Rewriting `NvidiaOmniLLMService`.
- Stock Smart Turn stop strategy waiting for a transcription.
- Reasoning left at the model default.
- Serving Omni without the audio extras, or declaring the deployment working from a text
  request alone.
- Dropping `nano` from a slug, repository id, or image tag to match the product name.
