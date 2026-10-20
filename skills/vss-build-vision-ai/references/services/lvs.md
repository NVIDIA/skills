# LVS Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile key |
|---|---|
| Long-video and structured summarization | `lvs-server` |

## Required peers

- Requires one reachable LLM and one reachable VLM/RT-VLM. Not the Agent owner:
  something must *drive* `/v1/summarize` — the agent's summarization tool, or
  `vss summarize run` — but no agent need be deployed.
- The current developer profile uses RT-VLM and ELK/Kafka.
- Graph database backends require a compatible text-embedding endpoint; do not
  point them at RT-Embed unless the adapter explicitly supports that API.

## Configuration knobs

| Environment variable | Use |
|---|---|
| `LVS_IMAGE`, `LVS_TAG`, `BACKEND_HOST_PORT`, `LVS_MCP_HOST_PORT` | Select image and published APIs. |
| `LVS_LLM_MODEL_NAME`, `LLM_BASE_URL`, `OPENAI_API_KEY`, `NVIDIA_API_KEY` | Configure summarization LLM access. |
| `VLM_BASE_URL`, `VLM_NAME`, `RTVI_VLM_BASE_URL` | Configure VLM access and model id. |
| `LVS_DATABASE_BACKEND` | Select `elasticsearch_db`, `graph_db`, or `graph_db_arango`. |
| `LVS_EMB_ENABLE`, `LVS_EMB_MODEL_NAME`, `LVS_EMB_BASE_URL` | Configure graph-backend text embeddings. |
| `KAFKA_ENABLED`, `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_STRUCTURED_SUMMARY_TOPIC` | Configure summary events. |
| `LVS_ENABLE_LLM_MERGING`, `LVS_DISABLE_DB_RESET_ON_REQUEST_DONE` | Configure result persistence and merge behavior. |
| `ENABLE_AUDIO`, `LVS_ENABLE_MCP`, `VSS_LOG_LEVEL` | Toggle audio, MCP, and logging. |

## Sources

- `deploy/docker/services/video-summarization/compose.yml`
- `deploy/docker/services/video-summarization/configs/config.yaml`
- `skills/operations/vss-summarize-video/references/video-summarization-environment-variables.md`
- `skills/operations/vss-summarize-video/references/video-summarization-deployment.md`
