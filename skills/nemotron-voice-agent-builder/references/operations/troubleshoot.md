# Troubleshoot

Use after `operations/run.md` fails to complete a spoken exchange. Diagnose the first
broken stage. Do not change multiple layers at once.

## Method

1. Read the generated agent code and structured events from
   `operations/observability.md`.
2. Confirm every required local service is ready.
3. Trace one turn in order:

   ```text
   cascaded: client → transport → turn detection → ASR → LLM → TTS → transport → client
   omni:      client → transport → turn detection → Omni → TTS → transport → client
   ```

4. Query the selected framework's documentation MCP before changing framework APIs.
5. Make one minimal fix, restart the affected process, and repeat `operations/run.md`.

Do not add a serve flag the model card does not list unless a fatal log names it
(`models/llm.md` §Serve flags). A speculative flag has to be reverted later, and the revert
looks exactly like a fix.

## Quick Index

| Symptom | Check first |
| --- | --- |
| Agent does not start | traceback, missing package, credential |
| Local model service is unhealthy | locked image/profile/tags, memory, platform guide |
| Speech service stuck starting on first boot | download and engine-build logs, cache mount |
| LLM is healthy but TTS / ASR OOM | LLM profile, context, runtime memory cap, TTS batch |
| Smoke fails constructing a service | API-key argument, `extra_body` shape, framework file |
| Agent runs and stays mute with no error | reasoning parser, request mode, and streamed `content` |
| Browser does not connect | framework client path, microphone permission, transport |
| Not enough VRAM to start, or an unexpected OOM | what is already resident, before any model change |
| User speech is not detected | input audio frames, VAD, end-of-turn |
| Transcripts arrive only when the session ends | ASR endpointing on the single-GPU speech service |
| No transcript (cascaded) | streaming ASR endpoint and language |
| Wrong input or reply language | `models/language-routing.md` |
| No reply | LLM/Omni endpoint, model id, request error |
| Reply text exists but no audio | TTS readiness, voice id, output transport |
| Reply is verbose or speaks markup/internal data | `domain/agent-behavior.md` |
| Domain term is wrong | approved glossary, model support, boost or IPA wiring |
| Slow response | stage timings, reasoning, ASR mode, model fit |
| Cannot interrupt | VAD, user-turn strategy, interruption frames |

## Startup and Services

- Docker, Compose, in-container GPU, or cache-path failure: rerun
  `platforms/readiness.md` before changing model configuration.
- Authentication or image-pull failure: re-check `preflight.md` credentials. Never print
  a secret.
- Wrong NIM model profile: Cascaded uses the LLM NIM matrix and profile procedure in
  `models/llm.md`. Omni uses the VLM NIM sources in
  `frameworks/omni.md` §Workstation NIM. Rerun `list-model-profiles` from the selected
  image on the assigned GPU set. Cloud checks model id and API. The single-GPU stack checks
  its vLLM model card, the quantization variant for the probed compute capability, and the
  serve flags.
- Wrong Speech NIM tags: reopen the ASR or TTS matrix and locked `/deploy` page. The
  single-GPU stack instead checks the GGUF paths passed to the speech service and the
  models that actually loaded.
- Speech service still `starting` or `unhealthy` on first boot: read the logs before
  changing anything. Model download and TensorRT engine building routinely take 15 to 30
  minutes or more, and a restart discards that work. See `platforms/deployment.md` §First
  boot takes much longer. Treat it as a failure only on process exit, a fatal log such as
  CUDA OOM, or stalled progress past the documented window.
- Speech ready but deaf or mute: query `GetRivaSpeechRecognitionConfig` or
  `GetRivaSynthesisConfig` and compare the loaded model, language, and voice with the lock.
  On Speech NIM TTS, also call the voice-list path from its current TTS API reference and
  confirm the locked voice is actually returned. A wrong or near-miss path returns 404 or
  an empty list, which looks like a mute service.
- OOM when the budget said it would fit: something is resident that was not there at probe
  time. Read `nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv` and
  `docker ps` first, and identify what holds the memory before changing anything. A stale
  container from an earlier attempt is the common cause, and stopping that named container
  is the fix rather than shrinking the model. Follow
  `preflight.md` §Find out what is holding the memory, and never prune or stop containers
  the user did not approve.
- OOM on the NIM path: stop the failing service. Compare actual per-process GPU use with
  the budget in `preflight.md`. Verify the LLM loaded the pinned quantization profile, then
  reduce its documented runtime memory cap or `NIM_MAX_MODEL_LEN`. Confirm TTS uses the
  smallest approved explicit batch profile. Restart one service at a time through the
  memory gate in `platforms/deployment.md`, or move one slot to cloud. Do not silently
  change the approved model.
- LLM fails with no available KV cache blocks: its cap is too low for that profile. Do not
  reduce it again. Use a smaller Compatible profile, raise the cap only when speech
  reserves still fit, or move a slot to another GPU or cloud.
- OOM on the single-GPU stack: lower `--max-model-len` first, then the memory fraction, and
  restart vLLM. Confirm the fraction was derived from the measured speech reserve rather
  than from free memory at launch. Follow `platforms/single-gpu.md` §Memory.
- vLLM will not start while the host looks like it has memory: compare Linux free memory
  with available memory. The startup check has been observed to read free memory, so page
  cache left by downloads and engine builds can block it. Reclaim cache rather than
  shrinking the model configuration. See `platforms/single-gpu.md` §Host memory at startup.
- vLLM sits silent for a long first boot: look for `nvcc` and `cicc` compile output, which
  is the NVFP4 kernel compile and not a hang. See `platforms/single-gpu.md` §First boot
  compiles kernels.
- Speech service exits at startup or reports a missing model: assert the mounted model tree
  is readable by the container user, and that all three TTS paths and the ASR path resolve
  inside the container. A tree created by Compose before the download is root-owned. See
  `platforms/readiness.md` §Speech model tree.
- Speech service rejects a setting at startup: unknown engine keys are errors on this
  service, so a mistyped or aliased option is a startup failure rather than a default. Use
  the canonical dotted option names.
- Numbers, dates, or currency are spoken literally: the text-normalization grammars are
  missing, or the container was built without normalization support and logged a warning at
  startup. See `platforms/single-gpu.md` §One-time speech model setup.
- Requested TTS locale is refused or silently answered in another language: read the
  languages `GetRivaSynthesisConfig` advertises. Some are build-time options that are off
  by default, so the model's language list is an upper bound rather than the served set.
- Reply text never reaches TTS and no log shows an error: confirm that the serve command
  keeps the reasoning parser required by the locked model card and that the request sets
  `enable_thinking:false`. Reassert non-empty streaming `delta.content` through
  `scripts/smoke.sh`. See `models/llm.md` §Reasoning Parser.

## Input and Transport

First confirm the client has microphone permission and the agent receives audio frames.
Then confirm VAD sees speech and emits a completed user turn.

- Pipecat: check `/status`, the selected transport, and the runner flags in
  `frameworks/pipecat.md`.
- LiveKit: check worker connection, room/participant events, and the current Agents
  console flow from the LiveKit docs MCP.
- Pipecat ICE, TURN, NAT, browser, or one-way-audio issue: follow
  `networking/remote-webrtc.md` before changing network configuration.

## Cascaded Pipeline

If VAD sees speech but no transcript appears:

1. verify the ASR service independently
2. confirm the agent points at its gRPC endpoint
3. confirm the locked profile is streaming
4. confirm its language matches the request

If transcripts appear only after the session closes, the agent is receiving one final
result per stream instead of one per utterance. On the single-GPU stack that is ASR
endpointing left at its default. Enable it, set the end-of-utterance silence threshold, and
reassert a mid-stream final through `scripts/smoke.sh`. See
`platforms/single-gpu.md` §Set explicitly. Do not change VAD or turn strategy
first, because the client-side symptom looks identical.

If a transcript appears but no reply follows, inspect the LLM request and response. Check
the base URL, exact model id, authentication, and reasoning payload from `models/llm.md`.

If only domain terms fail, reopen `domain/speech-customization.md`. Confirm the exact
phrase, approved score, locked model support, and framework wiring before changing models.

## Language

- Wrong input language: inspect the cascaded ASR locale/tags or the Omni model's supported
  audio languages.
- Correct transcript, wrong reply language: inspect the locked response locale and system
  instruction.
- Correct reply text, wrong voice: inspect the TTS locale and running service's
  voice-discovery result.
- Works in one language only: verify every advertised language exists on both input and
  TTS paths. Multilingual ASR or Omni does not imply multilingual TTS.

## Omni Pipeline

Confirm the project copied the current upstream
`nvidia_omni_multimodal_service.py` and instantiates the service class imported by the
current upstream pipeline, which is currently `NvidiaOmniLLMService`.
Confirm it also copied `audio_only_smart_turn_strategy.py`. Do not replace either with a
stock text-only service or transcription-dependent Smart Turn strategy.

If speech produces no request, trace input audio, user-start, and user-stop frames into
the service. If the first greeting races microphone input, confirm
`MuteUntilFirstBotCompleteUserMuteStrategy` is active and compare the greeting against the
current upstream pipeline.

If TTS receives JSON or transcript metadata, verify structured-response and
`emit_transcriptions` settings because the upstream service should emit only the response
field. If it receives reasoning, keep reasoning off through `models/llm.md`. Return to
`frameworks/omni.md` before changing the pipeline.

## TTS and Return Audio

If reply text exists:

1. verify TTS readiness independently
2. query the running platform's voice-discovery API and confirm the configured voice
3. confirm TTS emits audio frames
4. confirm the output transport sends them to the client

Do not debug ASR when reply text already exists. The failure is downstream.
For a wrong domain pronunciation, verify the approved IPA and current TTS customization
format before changing the voice.

## Latency and Interruption

Measure each stage before tuning. Check ASR finalization, LLM/Omni time to first token,
TTS time to first audio, and transport delay separately.

Keep reasoning off unless approved. Never switch streaming ASR to offline to fix
accuracy. For VAD, end-of-turn, or interruption changes, query the framework docs MCP and
change one setting at a time.

## Stop When

- The required documentation MCP is unavailable and the fix needs framework API changes.
- The locked model/profile is unsupported on the probed hardware.
- A proposed fix changes an approved model, deployment, or framework choice.
