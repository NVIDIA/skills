# SOP Inference Microservice — Generation Prompt

> **Skill**: `deepstream-sop/SKILL.md`
> **Configs**: `deepstream-sop/configs/`
> **Reference code**: `deepstream-sop/references/`

## What to Build

A production-ready FastAPI microservice for SOP compliance monitoring on industrial video feeds:

- GPU-accelerated event boundary detection — **Generic Event Boundary Detection (GEBD)** model, default **DDM** — via DeepStream + Triton CAPI
- Action classification via embedded vLLM (Cosmos Reason 1 or 2; `VLLM_MODEL_PATH` is required — no default)
- SOP sequence and checker compliance (missing / misordered steps)
- OpenAI-compatible SSE chat completions API (`/v1/chat/completions`)
- Video input: files, HTTP/HTTPS URLs, RTSP streams, Basler industrial cameras
- Kafka messaging: publish chunk results via NvProto (VisionLLM protobuf) or JSON schema

All implementation patterns, file mappings, critical rules, and verification steps
are in `deepstream-sop/skill.md`. Follow each section to generate the complete service.