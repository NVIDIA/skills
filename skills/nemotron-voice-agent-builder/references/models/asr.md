# ASR

Nemotron ASR, Parakeet CTC / RNNT / TDT, and other models on the live Speech matrix.
Docs first. Never invent the roster from memory.

Do not use the LLM support matrix, `list-model-profiles`, or `NIM_MODEL_PROFILE` here.

This file covers cloud ASR and Speech NIM ASR. The single-GPU stack does not run Speech
NIM, so when `preflight.md` §4 routes there, select the ASR model from
`platforms/single-gpu.md` §One-time speech model setup instead. That path serves a GGUF
build of a streaming Nemotron ASR model from local files, so there is no `CONTAINER_ID`,
no `NIM_TAGS_SELECTOR`, and no batch-size tag to choose. §Streaming first and the language
rules below still apply.

## Streaming First

Voice agents need partial transcripts as audio arrives. Always propose a **streaming** ASR
model and a streaming `NIM_TAGS_SELECTOR` (`mode=str` or a streaming-only model such as
Nemotron ASR Streaming).

Prefer streaming even when the user also wants diarization or multilingual. Pick a matrix
row that exposes streaming for that need. Do not default to offline (`mode=ofl`) or to an
offline-only model (Parakeet TDT, Whisper, Canary, and anything the docs mark offline only).

### If the User Chooses Offline

Allow it only when they ask for offline by name, or for a capability that exists only on
an offline profile. Before locking that row, tell them these limits in the proposal (or
when they amend ASR):

- Offline waits for the **full utterance or file**, then returns one complete transcript.
  No partials while the user is still speaking.
- Turn latency is higher. The agent cannot start thinking or speaking from interim text.
- It fits **batch / file** transcription better than a live mic conversation.
- Some offline-only models cannot be switched to streaming later without changing the
  model. If they still want a live voice agent, keep a streaming ASR and use offline only
  as a separate batch path.

Do not silently swap a streaming proposal to offline. Disclose, then wait for confirm.

## Discover

| Step | Where | Why |
| --- | --- | --- |
| Disambiguation | [About ASR](https://docs.nvidia.com/nim/speech/latest/asr/) · [Deploy ASR models](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/index.html) | language, streaming vs offline, diarization, timestamps. Overlaps (Parakeet CTC English vs Nemotron ASR Streaming) resolved here |
| Roster + tags + VRAM | [ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) | every model, `CONTAINER_ID`, `NIM_TAGS_SELECTOR`, GPU memory |
| Confirm | `build.nvidia.com/nvidia/<slug>` | function id, card, languages |
| Deploy copy | `build.nvidia.com/nvidia/<slug>/deploy` | image, docker run, env. Example: [nemotron-asr-streaming/deploy](https://build.nvidia.com/nvidia/nemotron-asr-streaming/deploy) |

Index: [NVIDIA Speech NIM](https://docs.nvidia.com/nim/speech/latest/index.html).
Resolve fixed locale versus automatic detection through `models/language-routing.md`.
For domain word boosting, also follow `domain/speech-customization.md` and verify the
locked model against the current customization docs.

Near-duplicates that must not collapse:

- Parakeet CTC English: more than one size (0.6b and 1.1b on the matrix)
- Parakeet CTC locale containers (Spanish, Mandarin, Vietnamese, Taiwanese, …)
- Parakeet RNNT Multilingual: different family, auto-detect multilingual
- Nemotron ASR Streaming: separate English streaming-focused option

If the user says “Parakeet” without size or decoder, open the docs disambiguation and
propose one **streaming** concrete row. Do not pick silently.

## Defaults (Hints Only)

Re-check the matrix before the proposal table. Every default below is a streaming pick.

| Need | Start from | Then open |
| --- | --- | --- |
| English streaming | Nemotron ASR Streaming | matrix + [deploy](https://build.nvidia.com/nvidia/nemotron-asr-streaming/deploy) |
| English + diarization (still streaming) | Parakeet CTC English streaming profile with diarizer (1.1b unless VRAM needs 0.6b) | matrix + that slug’s `/deploy` |
| One non-English locale | Matching Parakeet CTC locale, **streaming** profile | matrix language table |
| Several languages, auto-detect | Parakeet RNNT Multilingual, **streaming** profile | matrix + [card](https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr) |

Offline profiles and offline-only models are never the default. Use them only after the
disclosure above.

An auto-detecting multilingual row does not become single-language because the request
carries a locale. Resolve that distinction through `models/language-routing.md` before
promising fixed-language recognition.

## Sizing

Planning estimates only. Authoritative GPU memory is the chosen row on the
[ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html).

| Fact | Value |
| --- | --- |
| Nemotron ASR Streaming, `batch_size=32` | about 6 GB |
| Nemotron ASR Streaming, `batch_size=64` | about 15 GB |
| Self-hosted floor | compute capability 8.0+, ≥16 GB VRAM for the speech NIM |
| Selection | `NIM_TAGS_SELECTOR` (not `NIM_MODEL_PROFILE`) |

A quantized GGUF model on the single-GPU stack is far smaller than these figures, and it
shares one service with TTS. Do not plan that stack from this table. Measure the combined
speech reserve through `platforms/single-gpu.md` §Memory.

Batch size moves ASR memory by more than a factor of two, so on a shared GPU take the
smallest documented streaming batch that satisfies the use case and write that tag
explicitly. Read the pair off the matrix row for the locked language type.

After the profile fits this GPU, check the combined stack in `preflight.md` §Deployment
fit. If ASR does not fit beside the LLM, move ASR to another GPU or to the cloud.
When ASR shares a GPU, verify measured fit through `platforms/deployment.md` §Shared-GPU
memory gate.

## Lock Self-Hosted

1. From the matrix section: `CONTAINER_ID` + `NIM_TAGS_SELECTOR` for language, **streaming**
   mode, and VRAM. Prefer `mode=str` (or the model’s streaming-only tags). Tag strings, not
   LLM profile hashes. Use `mode=ofl` or offline-only containers only after the offline
   disclosure and user confirm.
2. Cross-check image and environment on `/deploy` using the precedence in
   `models/catalog.md`.
3. Apply the sizing floor above, then stack fit in `preflight.md`.
4. Languages only from the matrix table. Do not invent a locale.
5. Cloud: function id from the build page. Self-hosted: leave function id empty. Never
   look up a speech function id in `/v1/models`.

## Anti-Patterns

- Defaulting to offline ASR or an offline-only model for a voice agent.
- Locking offline ASR without telling the user the live-conversation limits.
- Treating “Parakeet” as one model.
- Skipping docs.nvidia.com/nim/speech for a remembered build.nvidia.com slug.
- Setting a function id on self-hosted ASR.
- Inventing a locale not on the matrix language table.
