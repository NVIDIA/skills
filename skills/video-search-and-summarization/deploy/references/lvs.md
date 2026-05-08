# VSS LVS Profile — Reference

## What Gets Deployed (adds to base)

| Service | Purpose |
|---|---|
| VSS Agent | Orchestrates tool calls and model inference |
| VSS Agent UI | Web UI at port 3000 |
| VIOS | Video ingestion, recording, playback |
| Nemotron LLM (NIM) | Reasoning and response generation |
| Cosmos Reason 2 (NIM) | Vision-language model |
| VSS Long Video Summarization | Segments and summarizes long-form video |
| ELK (Elasticsearch + Kibana) | Log storage and analysis |
| Phoenix | Observability and telemetry |

## GPU Layout (RTXPRO6000BW)

Same as base profile — LVS runs as a CPU-side microservice:

| Mode | Device 0 | Device 1 |
|---|---|---|
| Shared GPU (default) | LLM + VLM (`local_shared`) | — |
| Dedicated GPU | LLM | VLM |

## Key Capabilities

- Quickly generate a high-level narrative summary of a long video
- Extract timestamped highlights based on user-defined events
- Processes uploaded files from minutes to hours in duration
- Results returned through the AI agent chat interface
- Human-in-the-loop (HITL) prompt editing for report generation
