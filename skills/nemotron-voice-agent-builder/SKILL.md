---
name: nemotron-voice-agent-builder
description: Create, refine, or fix NVIDIA voice agents (Cascaded or Omni) with Pipecat or LiveKit. Use when building, scaffolding, or iterating on a real-time voice-agent pipeline, including speech (ASR/TTS) customization and cloud or local deployment. Not for offline/batch speech-to-text, text-only chat or RAG, or generic Docker or CUDA work unrelated to a voice agent.
version: "2.2.0"
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA Voice Agent Team <nemotron-voice-agent@nvidia.com>
  tags: "voice-agent, nvidia, nemotron, magpie, pipecat, livekit, nim, vllm, nemo-speech, omni"
---

# Create Voice Agent

Creates a working NVIDIA voice agent or updates an existing project:

- **Cascaded**: ASR transcribes, a text LLM answers, TTS speaks.
- **Omni**: one audio-in LLM replaces ASR and the text LLM. TTS still speaks.

## When to Use This Skill

Use this skill when the user wants to build, scaffold, configure, refine, or fix an
NVIDIA voice agent — a real-time speech pipeline with audio input and spoken output — on
Pipecat or LiveKit. This covers Cascaded (ASR → LLM → TTS) and Omni (audio-in LLM + TTS)
pipelines, speech customization (ASR word boosting, TTS pronunciation), multilingual
routing, and cloud or local deployment (NIM, vLLM, NeMo-Speech.cpp) on workstations, DGX
Spark, or Jetson Thor.

Do not use this skill for text-only chatbots or RAG, standalone batch speech-to-text
transcription, generic Docker or infrastructure help, or unrelated CUDA or model work that
has no voice-agent pipeline.

## Workflow

Classify the starting state before following the phases:

- For an empty project, follow all phases in order.
- For a working existing project, read `references/operations/iterate.md` first, and then route only the references needed for the requested change.
- For a broken existing project, read `references/operations/troubleshoot.md` first. After restoring the baseline, continue with `references/operations/iterate.md`.

For an empty project, follow these phases in order:

| Phase | Read |
| --- | --- |
| 1. Intake | `references/intake.md` resolves pipeline and framework first, then routes preflight, models, language, and domain files |
| 2. Build | `references/platforms/readiness.md` if self-hosted → exact profile or quantization variant in the routed model file → `references/output-contract.md` → routed deployment path → `references/operations/observability.md` → selected framework files |
| 3. Hand over | `references/operations/run.md` gates the client on the generated `scripts/smoke.sh`, then `references/operations/troubleshoot.md` if the spoken exchange fails |
| 4. Iterate | `references/operations/iterate.md` for changes to a working agent |

Wait for approval before writing files.

`references/preflight.md` §4 probes the host and selects the deployment path. The routed
platform file owns that path from there.

Also read `references/networking/remote-webrtc.md` when Pipecat WebRTC crosses hosts or
networks.

## Rules

- Resolve every model id, image tag, profile, and serve flag from current NVIDIA
  documentation at build time. Never from memory.
- Read every routed file before generating code, and keep the language and behavior locks
  approved in intake.
- Handover requires a successful spoken exchange. A running process is not a working voice
  agent.
