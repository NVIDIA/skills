# Video Summarization Environment Variables

This is the VSS 3.3.0 release `lvs` profile env reference. For
full deployment decisions, use `vss-build-vision-ai`; this file is for quick
video summarization debugging and request construction.

## Profile Env

The checked-in `.env` is the stable-default layer and `overrides.env` is the
runtime/profile default layer. For an actual deployment, initialize the generated
profile env from `overrides.env`, apply overrides there, and resolve Compose with both
`.env` and `generated.env`:

```text
deploy/docker/developer-profiles/dev-profile-lvs/generated.env
```

Password values should be supplied by that profile env or deployment-specific
overrides; the service compose file intentionally does not define password
defaults.

Core deployment:

| Var | Purpose |
|---|---|
| `MODE` | Profile mode, currently `2d`. |
| `BP_PROFILE` | Blueprint profile, `bp_developer_lvs`. |
| `COMPOSE_PROFILES` | Computed profile list. Includes `bp_developer_lvs_2d`. |
| `HARDWARE_PROFILE` | Hardware profile for NIM sizing. |
| `VSS_APPS_DIR` | Absolute path to `deploy/docker`. |
| `VSS_DATA_DIR` | Data root. |
| `HOST_IP` | Host-reachable IP address. |

Model selection:

| Var | Purpose |
|---|---|
| `LLM_MODE` | `local_shared`, `local`, or `remote`. |
| `VLM_MODE` | `local_shared`, `local`, or `remote`; video summarization uses RT-VLM for VLM serving. |
| `LLM_NAME`, `LLM_NAME_SLUG` | LLM model id and service slug. |
| `VLM_NAME` | Model id sent to the video summarization service and RT-VLM. Must match `/v1/models`. |
| `VLM_NAME_SLUG` | VLM service slug, often `none` for integrated RT-VLM. |
| `LLM_BASE_URL`, `VLM_BASE_URL` | Remote endpoints when using remote mode. |

Credentials:

| Var | Purpose |
|---|---|
| `NGC_CLI_API_KEY` | Image/model pulls for local deployment. |
| `NVIDIA_API_KEY` | NVIDIA-hosted remote endpoints and video summarization LLM API key fallback. |
| `OPENAI_API_KEY` | OpenAI-compatible remote endpoints, if used. |
| `HF_TOKEN` | Required for gated Hugging Face checkpoints such as Omni. |

RT-VLM:

| Var | Default / Example | Purpose |
|---|---|---|
| `VSS_RT_VLM_TAG` | `3.3.0` on x86/Tegra; `3.3.0-sbsa` on SBSA | RT-VLM image tag. |
| `RTVI_VLM_BASE_URL` | `http://rtvi-vlm:8000` | Compose service URL injected into LVS as `RTVI_VLM_URL`. |
| `RTVI_VLM_PORT` | `8018` | Host port. |
| `RTVI_VLM_URL` | `http://rtvi-vlm:8000` (derived from `RTVI_VLM_BASE_URL`) | video summarization-facing URL. |
| `RTVI_VLM_MODEL_TO_USE` | `cosmos-reason3` | Default integrated backend selector. |
| `RTVI_VLM_MODEL_PATH` | `ngc:nim/nvidia/cosmos3-nano-reasoner:bf16-final` | Default checkpoint. |
| `RTVI_VLM_DEFAULT_NUM_FRAMES_PER_SECOND_OR_FIXED_FRAMES_CHUNK` | empty; `5` for remote openai-compat LVS deployments | RT-VLM fixed-frame default used when LVS requests omit frame sampling overrides. Leave model-specific local defaults unchanged. |
| `RTVI_VLLM_GPU_MEMORY_UTILIZATION` | empty | Optional vLLM memory fraction. |
| `RTVI_VLM_KAFKA_ENABLED` | `true` | Publish raw caption events. |
| `RTVI_VLM_MESSAGE_BUS` | `kafka` | Generated-output broker type. |
| `RTVI_VLM_MESSAGE_BUS_TOPIC` | `mdx-vlm-captions` | Raw caption Kafka topic or Redis stream. |
| `RTVI_VLM_ERROR_BUS` | `kafka` | Error-output broker type; empty disables errors. |
| `RTVI_VLM_KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | Broker URL from RT-VLM. |

Video summarization service:

| Var | Default / Example | Purpose |
|---|---|---|
| `LVS_BACKEND_URL` | `http://${HOST_IP}:38111` | Agent-facing video summarization URL. |
| `VSS_CONTAINER_REGISTRY` | `nvcr.io/nvidia/vss-core` | Shared image registry root used when the service repository is unset. |
| `VSS_VIDEO_SUMMARIZATION_IMAGE` | `${VSS_CONTAINER_REGISTRY}/vss-video-summarization` | Service image repository override, without a tag. |
| `VSS_VIDEO_SUMMARIZATION_TAG` | `3.3.0` on x86/Tegra; `3.3.0-sbsa` on SBSA | Service image tag override. |
| `CONTAINER_IMAGE` | unset | Complete image reference override; takes precedence over the repository and tag variables. |
| `LVS_ENABLE_MCP` | `false` | Enable optional MCP/SSE port. |
| `LVS_DATABASE_BACKEND` | `elasticsearch_db` | Active CA-RAG database backend. Use `graph_db` for Neo4j or `graph_db_arango` for ArangoDB only with an embedding endpoint configured. |
| `LVS_EMB_ENABLE` | `false` | Required as `true` for Neo4j or ArangoDB graph backends. |
| `LVS_EMB_MODEL_NAME` | unset | Text embedding model id for graph backends, copied from the embedding endpoint's `/models` response. Current LVS graph-RAG examples use `nvidia/llama-3.2-nv-embedqa-1b-v2`. |
| `LVS_EMB_BASE_URL` | unset | OpenAI/NVIDIA-compatible text embedding endpoint for graph backends; include `/v1`, for example `https://integrate.api.nvidia.com/v1` or `http://127.0.0.1:9232/v1`. |
| `KAFKA_ENABLED` | `true` | video summarization Kafka integration. |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | Broker URL from the video summarization service. |
| `KAFKA_STRUCTURED_SUMMARY_TOPIC` | `mdx-structured-events-summary` | Structured summary topic. |
| `LVS_ENABLE_LLM_MERGING` | `true` | Merge duplicate/overlapping events. |

With no image overrides, Compose uses the public release image
`nvcr.io/nvidia/vss-core/vss-video-summarization:3.3.0`. Set
`VSS_VIDEO_SUMMARIZATION_TAG=3.3.0-sbsa` on SBSA. A non-empty
`VSS_VIDEO_SUMMARIZATION_IMAGE` takes precedence over `VSS_CONTAINER_REGISTRY`;
a non-empty `CONTAINER_IMAGE` overrides the entire repository and tag.
`LVS_IMAGE` and `LVS_TAG` are obsolete; use `VSS_VIDEO_SUMMARIZATION_IMAGE`
and `VSS_VIDEO_SUMMARIZATION_TAG` instead. See the
[deployment reference](video-summarization-deployment.md#required-inputs) for
the exact Compose expression.

## Service Compose Env

The video summarization service compose lives at:

```text
deploy/docker/services/video-summarization/compose.yml
```

It maps profile env into container env. Important container env names:

| Container env | Source / value |
|---|---|
| `CA_RAG_CONFIG` | `/app/config.yaml` |
| `BACKEND_PORT` | `${BACKEND_PORT:-38111}` |
| `LVS_MCP_PORT` | `${LVS_MCP_PORT:-38112}` |
| `LVS_LLM_MODEL_NAME` | `${LVS_LLM_MODEL_NAME}` |
| `LVS_LLM_BASE_URL` | `${LLM_BASE_URL:-http://${HOST_IP}:${LLM_PORT}}/v1` |
| `LVS_LLM_API_KEY` | `${OPENAI_API_KEY:-${NVIDIA_API_KEY}}` |
| `VIA_VLM_ENDPOINT` | `${VLM_BASE_URL:-http://${HOST_IP}:${VLM_PORT}}/v1/` |
| `LVS_EMB_ENABLE` | `${LVS_EMB_ENABLE}` |
| `LVS_DATABASE_BACKEND` | `${LVS_DATABASE_BACKEND:-elasticsearch_db}` |
| `ES_HOST`, `ES_PORT` | Elasticsearch connection. |
| `GRAPH_DB_HOST`, `GRAPH_DB_USERNAME`, `GRAPH_DB_PASSWORD`, `GRAPH_DB_HTTP_PORT`, `GRAPH_DB_BOLT_PORT` | Neo4j graph backend connection. |
| `ARANGO_DB_HOST`, `ARANGO_DB_USERNAME`, `ARANGO_DB_PASSWORD`, `ARANGO_DB_PORT` | ArangoDB graph backend connection. |
| `KAFKA_ENABLED` | `${KAFKA_ENABLED:-false}` |
| `KAFKA_BOOTSTRAP_SERVERS` | `${KAFKA_BOOTSTRAP_SERVERS:-kafka:29092}` |
| `KAFKA_STRUCTURED_SUMMARY_TOPIC` | `${KAFKA_STRUCTURED_SUMMARY_TOPIC:-mdx-structured-events-summary}` |
| `RTVI_VLM_URL` | `${RTVI_VLM_URL:-}` |
| `ENABLE_AUDIO` | `${ENABLE_AUDIO:-false}` |
| `ENABLE_DENSE_CAPTION` | `false` |
| `VSS_LOG_LEVEL` | `INFO` |

## Config Map Env

The CA-RAG config at `deploy/docker/services/video-summarization/configs/config.yaml`
uses:

| Env | Purpose |
|---|---|
| `MILVUS_DB_HOST`, `MILVUS_DB_GRPC_PORT` | Milvus backend. |
| `ES_HOST`, `ES_PORT` | Elasticsearch backend. |
| `GRAPH_DB_HOST`, `GRAPH_DB_BOLT_PORT`, `GRAPH_DB_USERNAME`, `GRAPH_DB_PASSWORD` | Neo4j graph backend. |
| `ARANGO_DB_HOST`, `ARANGO_DB_PORT`, `ARANGO_DB_USERNAME`, `ARANGO_DB_PASSWORD` | ArangoDB graph backend. |
| `LVS_LLM_MODEL_NAME`, `LVS_LLM_BASE_URL` | Summarization LLM. |
| `LVS_EMB_ENABLE`, `LVS_EMB_MODEL_NAME`, `LVS_EMB_BASE_URL` | Embedding tool. Auth is read from `NVIDIA_API_KEY`. |
| `KAFKA_ENABLED` | Kafka-backed summarization aggregation. |
| `LVS_ENABLE_LLM_MERGING` | LLM merge behavior. |
| `LVS_DATABASE_BACKEND` | Active DB tool: `elasticsearch_db`, `graph_db`, or `graph_db_arango`. |

## Database Backend Recipes

Default Elasticsearch:

```bash
LVS_DATABASE_BACKEND=elasticsearch_db
```

Neo4j graph backend with the open-source `neo4j:5.26.4` container:

```bash
LVS_DATABASE_BACKEND=graph_db
GRAPH_DB_HOST=graph-db            # Compose service name
GRAPH_DB_USERNAME=neo4j
GRAPH_DB_PASSWORD=<neo4j-password>
GRAPH_DB_HTTP_PORT=7474
GRAPH_DB_BOLT_PORT=7687
LVS_EMB_ENABLE=true
LVS_EMB_MODEL_NAME=nvidia/llama-3.2-nv-embedqa-1b-v2
LVS_EMB_BASE_URL=<embedding-endpoint-with-/v1>
NVIDIA_API_KEY=nvapi-REPLACE_ME
```

ArangoDB graph backend with the open-source `arangodb/arangodb:3.12.4`
container:

```bash
LVS_DATABASE_BACKEND=graph_db_arango
ARANGO_DB_HOST=arango-db          # Compose service name
ARANGO_DB_USERNAME=root
ARANGO_DB_PASSWORD=<arango-password>
ARANGO_DB_PORT=8529
LVS_EMB_ENABLE=true
LVS_EMB_MODEL_NAME=nvidia/llama-3.2-nv-embedqa-1b-v2
LVS_EMB_BASE_URL=<embedding-endpoint-with-/v1>
NVIDIA_API_KEY=nvapi-REPLACE_ME
```

For either graph backend, use a text embedding endpoint compatible with the LVS
embedding adapter. Do not point `LVS_EMB_BASE_URL` at the Search `rtvi-embed`
service unless the adapter has explicitly been changed to support its RTVI
`/v1/generate_text_embeddings` API. Discover the exact model id from the
embedding endpoint and copy it into `LVS_EMB_MODEL_NAME`:

```bash
curl -fsS "${LVS_EMB_BASE_URL%/}/models" | jq -r '.data[].id'
```

The current Docker `lvs-server` uses the default Compose bridge network, and
the checked-in compose uses `graph-db` / `arango-db` as service hostnames.
Override the corresponding DB host only when the database runs outside that
Compose network.

## Runtime Rules

- Do not guess the embedding model id. Verify with the embedding endpoint's
  `/models`; do not use RT-VLM `/v1/models` for `LVS_EMB_MODEL_NAME`.
- Use `LVS_BACKEND_URL` for video summarization API calls and strip trailing `/v1` from VLM
  base URLs before appending `/v1/chat/completions`.
- For VSS 3.3.0 release examples, prefer `/v1/summarize` and omit frame sampling fields
  so RT-VLM applies the model-specific deployment default.
- Do not add development-only API switches to GA instructions.
- Do not switch to `graph_db` or `graph_db_arango` unless `LVS_EMB_ENABLE=true`
  and a reachable embedding endpoint are configured.
