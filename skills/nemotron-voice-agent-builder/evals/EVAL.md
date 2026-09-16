# Create Voice Agent Evaluation Guidance

## Questions

- Explicit requests that name `nemotron-voice-agent-builder`.
- Implicit and contextual requests for Cascaded and Omni voice agents.
- DGX Spark, Jetson Thor, low-concurrency workstation, high-concurrency workstation,
  hybrid, and cloud routing.
- Pipecat selection, LiveKit selection, and the LiveKit plus Omni conflict.
- Partial `nvidia-smi` failures and occupied GPU memory.
- Reasoning on and off, multilingual routing, and domain speech customization.
- Approved Cascaded, Omni, and LiveKit builds from generation through non-GPU smoke tests.
- Existing-project iteration and failure recovery through regression verification.
- Negative text-only, batch-transcription, and unrelated NVIDIA development requests.

## Behaviors

- Read `SKILL.md` first, then only the references routed by the selected workflow.
- Resolve framework and pipeline before deployment details.
- Generate framework APIs only from the current Pipecat or LiveKit documentation MCP.
- Separate LLM NIM, VLM NIM, Speech NIM, and raw-vLLM documentation families.
- Route DGX Spark, Jetson Thor, and low-concurrency workstations to vLLM plus
  NeMo-Speech.cpp. Route high-concurrency workstations to NIM.
- Treat probe failures as unknown hardware state rather than proof of no GPU.
- Identify reclaimable GPU memory and obtain approval before stopping a named container.
- Resolve models, profiles, quantization, identifiers, flags, and reasoning settings from
  current authoritative sources.
- Keep reasoning off by default. When enabled, use the locked model's contract and keep
  reasoning away from TTS.
- Wait for approval before writing files and require smoke plus a spoken exchange before
  declaring success.
- For approved builds, generate the complete output contract and run every check supported
  by the evaluation environment.
- Distinguish deterministic mock verification from the final hardware-backed spoken
  exchange. Never report a physical test that did not run.

## Notes

Keep the CI dataset small enough for the one-hour NVSkills CI limit. `evals.json` holds P0
planning and negative cases that verify selection, routing, and safety. Run end-to-end
generation, iteration, and failure-recovery cases separately; they verify project
generation, static checks, unit tests, deterministic mock endpoints, generated smoke paths,
and accurate handover without requiring NVIDIA GPUs, model downloads, external credentials,
or long-running services.

Run physical microphone, speaker, GPU, DGX Spark, Jetson Thor, and workstation deployments
separately. Report those hardware-backed spoken exchanges as extended evidence in
`BENCHMARK.md`.

Negative cases are required. They should ensure the skill does not activate for text-only
chatbots, standalone batch transcription, generic Docker support, or unrelated CUDA work.
