# TTS

Magpie and other TTS models on the live Speech matrix. Docs first. Never invent the
roster from memory.

Do not use the LLM support matrix, `list-model-profiles`, or `NIM_MODEL_PROFILE` here.

This file covers cloud TTS and Speech NIM TTS. The single-GPU stack does not run Speech
NIM, so when `preflight.md` §4 routes there, select TTS from
`platforms/single-gpu.md` §One-time speech model setup instead. That path serves a GGUF
Magpie token generator plus a NanoCodec decoder from local files, so there is no
`CONTAINER_ID`, no `NIM_TAGS_SELECTOR`, and no batch-size tag to choose. The voice-locking
rule below still applies, and §Languages and voices in that file owns which languages the
container actually serves.

## Discover

| Step | Where | Why |
| --- | --- | --- |
| Roster + tags + VRAM | [TTS support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html) | every model, `CONTAINER_ID`, `NIM_TAGS_SELECTOR`, GPU memory |
| Confirm | `build.nvidia.com/nvidia/<slug>` | function id, card, languages |
| Deploy copy | `build.nvidia.com/nvidia/<slug>/deploy` | image, docker run, env. Example: [magpie-tts-multilingual/deploy](https://build.nvidia.com/nvidia/magpie-tts-multilingual/deploy) |
| Voice discovery | [HTTP TTS API reference](https://docs.nvidia.com/nim/speech/latest/reference/api-references/tts/http-tts.html) | voice-list path and response shape. Currently `GET /v1/audio/list_voices` |

Index: [NVIDIA Speech NIM](https://docs.nvidia.com/nim/speech/latest/index.html).
Resolve response locale and multilingual routing through `models/language-routing.md`.
For domain pronunciations, also follow `domain/speech-customization.md` and verify the
locked model against the current customization docs.

If the user says “Magpie” without a variant, open the matrix and propose one concrete row.
Do not pick silently.

## Defaults (Hints Only)

| Need | Start from | Then open |
| --- | --- | --- |
| Default voice agent TTS | Magpie TTS Multilingual, `batch_size=8` | [#magpie-tts-multilingual](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html#magpie-tts-multilingual) + [deploy](https://build.nvidia.com/nvidia/magpie-tts-multilingual/deploy) |

Re-check the matrix before the proposal table. After startup, choose only a locale and
voice returned by the running service's documented voice-discovery API.

## Sizing

Planning estimates only. Authoritative GPU memory is the chosen row on the
[TTS support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html).

| Profile | Weights |
| --- | --- |
| Magpie TTS Multilingual, `batch_size=8` (default) | about 13 GB |
| Magpie TTS Multilingual, `batch_size=32` | about 41 GB |
| Magpie TTS Zeroshot, `batch_size=8` (default) | about 13 GB |
| Magpie TTS Zeroshot, `batch_size=32` | about 41 GB |

Self-hosted floor: compute capability 8.0+, ≥16 GB VRAM for the speech NIM. Selection is
`NIM_TAGS_SELECTOR` (not `NIM_MODEL_PROFILE`).

A quantized GGUF Magpie build on the single-GPU stack is far smaller than these figures,
has no batch profile, and shares one service with ASR. Do not plan that stack from this
table. Measure the combined speech reserve through
`platforms/single-gpu.md` §Memory.

Batch size is the TTS memory profile, not only a throughput knob. On a shared single-GPU
layout, select the smallest documented batch profile that satisfies the use case and
write it explicitly into `NIM_TAGS_SELECTOR`. For Magpie TTS Multilingual that is
currently `batch_size=8`, which is also the documented default. Write it anyway, so a
future image default cannot silently change the reservation. Larger batches belong on a
dedicated GPU unless the runtime budget proves they fit.

After the profile fits this GPU, check the combined stack in `preflight.md` §Deployment
fit. Reserve the matrix GPU memory plus startup and engine-build headroom. Verify measured
free VRAM through `platforms/deployment.md` §Shared-GPU memory gate.

## Lock Self-Hosted

1. From the matrix section: `CONTAINER_ID` + `NIM_TAGS_SELECTOR` for language, batch size,
   VRAM. On a shared GPU, choose the smallest supported batch profile and include the
   batch tag explicitly. Tag strings, not LLM profile hashes.
2. Cross-check image and environment on `/deploy` using the precedence in
   `models/catalog.md`.
3. Apply the sizing floor above, then stack fit in `preflight.md`.
4. Languages only from the matrix. Do not invent a locale.
5. Cloud: function id from the build page. Self-hosted: leave function id empty. Never
   look up a speech function id in `/v1/models`.
6. Voice id: after the TTS service is up, query its voice list and lock a voice from that
   response before handover. Speech NIM currently serves `GET /v1/audio/list_voices`, while
   a gRPC service exposes
   [`GetRivaSynthesisConfig`](https://docs.nvidia.com/nim/speech/latest/reference/api-references/tts/protos.html)
   through its client instead. Confirm the path on the API reference above before calling
   it, because a near miss such as `/v1/audio/voices` returns 404 and reads like a mute
   service. Record the path in the README and smoke script, and never hardcode a voice the
   service did not return.

The single-GPU stack is gRPC only, so it has no voice-list URL. It answers
`GetRivaSynthesisConfig`, and that response is also the authority for which languages the
container was built with. See `platforms/single-gpu.md` §Languages and voices.

## Anti-Patterns

- Treating “Magpie” as one model without checking the matrix.
- Skipping the TTS matrix for a remembered build.nvidia.com slug.
- Leaving batch size implicit or selecting it without the combined runtime budget.
- Setting a function id on self-hosted TTS.
- Hardcoding a voice id before querying the running service.
