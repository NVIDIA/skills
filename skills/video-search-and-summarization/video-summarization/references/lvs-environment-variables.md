# LVS Environment Variables — Full Reference

Canonical spelling: as declared in the `environment:` block of
`met-blueprints/deployments/lvs/compose.yml`. Values that start with `${VAR}`
come from the `env_file` (`$MDX_SAMPLE_APPS_DIR/lvs/.env`). Variables NOT in
this table are either scaffolded or consumed only by the entrypoint script
(`start_via.sh`) or the docs.

## Required / Effectively Required

Covered in `deploy-lvs-service.md` §7. TL;DR:

- `MDX_SAMPLE_APPS_DIR`
- `HOST_IP` (plus `LLM_PORT`, `VLM_PORT` unless you pass `LLM_BASE_URL` /
  `VLM_BASE_URL` directly)
- `LVS_LLM_MODEL_NAME`
- `NVIDIA_API_KEY` (or `OPENAI_API_KEY`)
- `ES_HOST` + `ES_PORT` (for `elasticsearch_db`) **or**
  `MILVUS_DB_HOST` + `MILVUS_DB_GRPC_PORT` (for `vector_db`)
- `LVS_EMB_ENABLE` — if `true`, also `LVS_EMB_MODEL_NAME` + `LVS_EMB_BASE_URL`

## Core configuration

| Var | Default | Purpose |
|---|---|---|
| `CONTAINER_IMAGE` | `nvcr.io/nvstaging/vss-core/vss-video-summarization:3.2.0-rc1-d3e1a8f` | image pinning |
| `CA_RAG_CONFIG` | `/app/config.yaml` (set in compose) | path to mounted CA-RAG config inside container |
| `BACKEND_PORT` | `38111` | REST API port (host network) |
| `LVS_MCP_PORT` | `38112` | MCP SSE port |
| `FRONTEND_PORT` | `38113` | UI port |
| `LVS_ENABLE_MCP` | `true` (sample `.env`) | toggle MCP server |
| `LVS_DATABASE_BACKEND` | `elasticsearch_db` | or `vector_db` |

## LLM / VLM / embeddings

| Var | Default | Purpose |
|---|---|---|
| `LVS_LLM_MODEL_NAME` | — | summarization LLM model ID |
| `LVS_LLM_BASE_URL` | built from `${LLM_BASE_URL:-http://${HOST_IP}:${LLM_PORT}}/v1` | LLM endpoint |
| `LVS_LLM_API_KEY` | `${OPENAI_API_KEY:-${NVIDIA_API_KEY}}` | LLM auth |
| `VIA_VLM_ENDPOINT` | `${VLM_BASE_URL:-http://${HOST_IP}:${VLM_PORT}}/v1/` | VLM endpoint |
| `VIA_VLM_API_KEY` | `${OPENAI_API_KEY:-${VIA_VLM_API_KEY:-not-used}}` | VLM auth; literal `not-used` if neither is set |
| `VIA_VLM_OPENAI_MODEL_DEPLOYMENT_NAME` | `gpt-4o` (entrypoint default) | VLM model name sent in API requests |
| `VLM_MODEL_TO_USE` | `openai-compat` (sample `.env`) / `vllm-compatible` (docs default) | VLM backend type |
| `LVS_EMB_ENABLE` | (required) | toggle embedding generation |
| `LVS_EMB_MODEL_NAME` | — | embedding model ID |
| `LVS_EMB_BASE_URL` | — | embedding endpoint |

## VLM runtime tuning

| Var | Default | Purpose |
|---|---|---|
| `VLM_INPUT_WIDTH` | `1312` | VLM input frame width |
| `VLM_INPUT_HEIGHT` | `736` | VLM input frame height |
| `VLM_BATCH_SIZE` | auto (entrypoint) | override to pin batch size |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.85` docs / `0.7` auto-set by entrypoint when VRAM ≤50 GB | vLLM VRAM fraction |
| `NUM_VLM_PROCS` | `16` (entrypoint default for openai-compat) | parallel VLM request workers |
| `VLM_DEFAULT_NUM_FRAMES_PER_CHUNK` | entrypoint passes to `--num-frames-per-chunk` | frames per VLM call |
| `TRT_LLM_MODE` | `int4_awq` / forced `fp16` on SM 10.x | quantization mode |
| `TRT_LLM_ATTN_BACKEND` | auto-`FLASHINFER` on SM 12.1 | attention backend |

## RTVI-VLM integration

| Var | Default | Purpose |
|---|---|---|
| `USE_RTVI_VLM` | `false` | route VLM calls to an RTVI sidecar's `/generate_captions` |
| `RTVI_VLM_URL` | empty | RTVI endpoint base URL |
| `RTVI_VLM_URL_PASSTHROUGH` | `false` | pass request through without rewrite |

## Database / storage

| Var | Default | Purpose |
|---|---|---|
| `ES_HOST` | (required when `elasticsearch_db`) | Elasticsearch host |
| `ES_PORT` | `9202` (docs), `9200` (sample `.env`) | Elasticsearch HTTP port |
| `ES_TRANSPORT_PORT` | `9302` (docs), `9300` (sample `.env`) | ES transport port |
| `MILVUS_DB_HOST` | (required when `vector_db`) | Milvus host |
| `MILVUS_DB_GRPC_PORT` | `19530` | Milvus gRPC |
| `MODEL_ROOT_DIR` | `/tmp/model_cache` | bind-mount target + `NGC_MODEL_CACHE` |
| `NGC_MODEL_CACHE` | `${MODEL_ROOT_DIR}` | NGC model download dir |
| `LVS_DISABLE_DB_RESET_ON_REQUEST_DONE` | `true` | keep events in ES after request completes |
| `ASSET_STORAGE_DIR` | `/tmp/assets` (entrypoint default) | uploaded asset dir inside container |
| `MAX_ASSET_STORAGE_SIZE_GB` | unset | cap asset dir size |

## Pipeline toggles

| Var | Default | Purpose |
|---|---|---|
| `DISABLE_GUARDRAILS` | `true` (sample `.env`) | disable NeMo Guardrails filter |
| `DISABLE_CA_RAG` | `false` (entrypoint) | disable the CA-RAG aggregation step |
| `DISABLE_CV_PIPELINE` | `true` (entrypoint) | disable DeepStream + GSAM CV pipeline |
| `ENABLE_AUDIO` | unset (disabled) | enable Riva ASR audio transcription |
| `ENABLE_VIA_HEALTH_EVAL` | `false` | health-eval module toggle |
| `INSTALL_PROPRIETARY_CODECS` | unset | install extra codecs at container start |
| `APPLY_GSTREAMER_RTSP_FIX` | unset | patch gstreamer RTSP manager |

## Audio / Riva ASR (only when `ENABLE_AUDIO=true`)

| Var | Default | Purpose |
|---|---|---|
| `RIVA_ASR_SERVER_URI` | — | Riva ASR host |
| `RIVA_ASR_GRPC_PORT` | — | Riva gRPC port |
| `RIVA_ASR_HTTP_PORT` | — | Riva HTTP port (for readiness probe) |
| `RIVA_ASR_SERVER_IS_NIM` | — | set `true` for NIM-hosted Riva |
| `RIVA_ASR_MODEL_NAME` | — | Riva model name |
| `RIVA_ASR_SERVER_USE_SSL` | — | enable TLS |
| `RIVA_ASR_SERVER_FUNC_ID` | — | NIM function ID |
| `RIVA_ASR_SERVER_API_KEY` | — | bearer token |
| `ENABLE_RIVA_SERVER_READINESS_CHECK` | unset | block entrypoint until Riva ready |

## Observability (OpenTelemetry)

| Var | Default | Purpose |
|---|---|---|
| `VIA_ENABLE_OTEL` | `false` | enable OTEL for VIA engine |
| `VIA_OTEL_ENDPOINT` | `http://localhost:4318` | OTEL collector |
| `VIA_OTEL_EXPORTER` | `console` | exporter format |
| `VIA_CTX_RAG_ENABLE_OTEL` | `false` | enable OTEL for CA-RAG layer |
| `VIA_CTX_RAG_EXPORTER` | `console` | CA-RAG exporter format |
| `VIA_CTX_RAG_OTEL_ENDPOINT` | `http://localhost:4318` | CA-RAG OTEL collector |

## S3 / object storage (docs-mentioned, NOT passed by this compose)

Add to `$MDX_SAMPLE_APPS_DIR/lvs/.env` if you need S3-hosted video URLs — the
`env_file` reference picks them up automatically.

| Var | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | S3 secret |
| `AWS_ENDPOINT_URL_S3` | S3 endpoint (for MinIO etc.) |

## Logging & misc

| Var | Default | Purpose |
|---|---|---|
| `VSS_LOG_LEVEL` | `DEBUG` (sample `.env`) | `DEBUG`/`INFO`/`WARN`/`ERROR` |
| `VIA_LOG_DIR` | `/tmp/via-logs/` | log output directory inside container |
| `MODE` | `release` | `release` runs packaged code; `dev` runs from `src/` |
| `VSS_EXTRA_ARGS` | unset | appended to `via_server.py` command line |
| `TRT_ENGINE_PATH` | unset | pre-built TRT engine dir |
| `MODEL_PATH` | `/opt/models/...` (entrypoint default) | VLM weight path for non-openai-compat backends |
| `ENABLE_NSYS_PROFILER` | `false` | wrap server in `nsys profile` |
