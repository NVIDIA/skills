# VSS Search Profile — Reference

> **Alpha Feature** — not recommended for production.

## What Gets Deployed

| Service | Purpose |
|---|---|
| VSS Agent | Orchestrates tool calls and model inference |
| VSS Agent UI | Web UI at port 3000 |
| VIOS | Video ingestion, recording, playback |
| Nemotron LLM (NIM) | Reasoning and response generation |
| RTVI-Embed | Generates embeddings for video and text using Cosmos Embed1 |
| ELK (Elasticsearch + Logstash + Kibana) | Indexes and searches embeddings |
| Kafka | Real-time message bus for embedding pipeline |
| Phoenix | Observability and telemetry |

> VLM is **not deployed locally** — `VLM_MODE=remote` is forced by the script for this profile.

## GPU Layout (RTXPRO6000BW)

Both GPUs required:

| Device | Role |
|---|---|
| 0 | RTVI-Embed (Cosmos Embed1) — embedding generation |
| 1 | LLM (`local_shared`) |

## Key Capabilities

- Upload videos; embeddings are generated automatically
- Natural language queries (e.g., "find all instances of forklifts")
- Filter results by similarity score, time range, video name, description, source
- Timestamped results with clip playback

## First Run Note

Downloads Cosmos Embed1 models from NGC — this can take extra time depending on network speed.
