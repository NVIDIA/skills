# Integration Reference: Long Video Summarization (LVS)

## Overview

Long Video Summarization (LVS) summarizes video files and recorded clips through
the VSS video summarization REST API. Include LVS when a generated profile must
summarize videos uploaded to VIOS or stored by the VIOS recorder and return the
summary through `POST /v1/summarize`. This reference covers stored/uploaded
video summarization only; live streaming video summarization is out of scope for
the current build-vision-ai add-on.

## Required Peer Services

- **VIOS** - Required. Supplies uploaded files, recorded clips, and playback
  URLs that LVS can summarize.
- **RT-VLM** - Required for the default local VLM path. LVS calls RT-VLM as its
  video-language backend while processing the stored file or clip.
- **Elasticsearch** - Required for the default `elasticsearch_db` backend used
  by the VSS developer profile.
- **Kafka / Logstash** - Required when `KAFKA_ENABLED=true`, the VSS default for
  structured summary and caption/event integration.
- **LLM endpoint or LLM NIM** - Required for LLM-backed summarization and event
  merging. This can be a local NIM, a shared-GPU local NIM, or a remote
  OpenAI-compatible endpoint.

The `component_services:` block for build-vision-ai is intentionally not in
this neutral contract. It lives in
`skills/vss-build-vision-ai/references/patch-lvs.md`, where the orchestrator
declares `lvs` and the optional local LLM placement variants without
making this service skill depend back on the orchestrator.

## Integration Interfaces

### Inputs

- **Method:** REST API
  **Endpoint:** `POST /v1/summarize`
  **Source:** `url` for an HTTP(S), S3, or VIOS-retrievable video URL, or `id`
  for a file/clip id already known to LVS.
  **Schema:** `SummarizationQuery` from
  `long-video-summarization/api_spec/openapi.json`; required keys are `model`,
  `scenario`, and `events`, plus a source field such as `url` or `id`.
  **Authentication:** OpenAPI declares bearer auth. Local developer deployments
  usually expose the endpoint without an auth header.

- **Method:** REST API
  **Endpoint:** `GET /models`
  **Purpose:** Discover the model id to send in `POST /v1/summarize`.
  **Authentication:** Same deployment-level auth behavior as the summarize API.

- **Method:** REST API
  **Endpoint:** `GET /v1/ready`, `GET /v1/live`, `GET /v1/startup`,
  `GET /v1/healthz`
  **Purpose:** Health and readiness probes.
  **Authentication:** Same deployment-level auth behavior as the summarize API.

- **Method:** Peer REST call
  **Endpoint:** `RTVI_VLM_URL`, populated from `RTVI_VLM_BASE_URL` (the
  developer profile sets `http://rtvi-vlm:8000`)
  **Purpose:** LVS calls RT-VLM while summarizing a stored video file or clip.
  **Authentication:** Local default is unauthenticated; remote endpoints use the
  configured API key.

### Outputs

- **Method:** REST API response
  **Endpoint:** `POST /v1/summarize`
  **Schema:** `CompletionResponse`. The summary payload is in
  `choices[0].message.content`; VSS examples often encode structured summary
  JSON there.
  **Trigger:** One response per summarize request.

- **Method:** Kafka publish, when enabled
  **Topic:** `mdx-structured-events-summary` by default
  **Schema:** Structured summary events produced by LVS.
  **Trigger:** Per summarize request when the Kafka integration is enabled.

- **Method:** Elasticsearch documents, when using the default backend
  **Indexing path:** LVS/Logstash writes structured summary or caption-derived
  records to Elasticsearch using the shared VSS infra.
  **Trigger:** Per summarize request and per backend write.

## API Schema

OpenAPI source: `long-video-summarization/api_spec/openapi.json`.
Operational summary is in
`skills/operations/vss-summarize-video/references/video-summarization-api.md`.

Required `POST /v1/summarize` request fields:

| Field | Type | Notes |
|---|---|---|
| `model` | string | Must match an id returned by `GET /models`. |
| `scenario` | string | Required use-case context. |
| `events` | array[string] | Required event or topic names to summarize. |
| `url` | string or null | HTTP(S), S3, or VIOS-provided video URL. |
| `id` | UUID, array[UUID], or null | File or clip id known to LVS. |

Common optional fields include `prompt`, `system_prompt`, `chunk_duration`,
`chunk_overlap_duration`, `summary_duration`,
`num_frames_per_second_or_fixed_frames_chunk`, `use_fps_for_chunking`,
`enable_audio`, `enable_reasoning`, `temperature`, `top_p`, and structured
output controls. Do not invent fields outside the OpenAPI schema.

Stored-video request shape:

```json
{
  "model": "nim_nvidia_cosmos3-nano-reasoner_bf16-final",
  "url": "http://<host>:30888/<vios-video-or-clip-url>",
  "scenario": "warehouse safety review",
  "events": ["person activity", "forklift interaction"],
  "chunk_duration": 10,
  "num_frames_per_second_or_fixed_frames_chunk": 20,
  "use_fps_for_chunking": false
}
```

Response assertion:

```bash
jq -e '.choices[0].message.content | length > 0'
```

## Environment Variables

| Variable | Purpose | Default | Required? |
|---|---|---|---|
| `LVS_BACKEND_URL` | Host-facing LVS API URL used by operators and agents | `http://${HOST_IP}:38111` | Yes |
| `VSS_VIDEO_SUMMARIZATION_IMAGE` | LVS image repository | `nvcr.io/nvidia/vss-core/vss-video-summarization` | Yes |
| `VSS_VIDEO_SUMMARIZATION_TAG` | LVS image tag | `3.3.0`; use `3.3.0-sbsa` on SBSA | Yes |
| `LVS_ENABLE_MCP` | Optional MCP/SSE endpoint | `false` | No |
| `LVS_DATABASE_BACKEND` | Active database backend | `elasticsearch_db` | Yes |
| `KAFKA_ENABLED` | Enable Kafka integration | `true` in the LVS developer profile | Yes for shared VSS infra |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker address from the LVS container | `kafka:29092` | Yes when Kafka is enabled |
| `KAFKA_STRUCTURED_SUMMARY_TOPIC` | Structured summary topic | `mdx-structured-events-summary` | Yes when Kafka is enabled |
| `LVS_ENABLE_LLM_MERGING` | Merge duplicate or overlapping events with the LLM | `true` | No |
| `RTVI_VLM_URL` | Container-internal RT-VLM URL populated by Compose from `RTVI_VLM_BASE_URL`; do not set directly | `http://rtvi-vlm:8000` | Yes for local RT-VLM |
| `RTVI_VLM_BASE_URL` | Host-side Compose input for the RT-VLM peer | `http://rtvi-vlm:8000` | Yes for local RT-VLM |
| `RTVI_VLM_MODEL_TO_USE` | RT-VLM backend selector | `cosmos-reason3` | Yes for integrated RT-VLM |
| `RTVI_VLM_MODEL_PATH` | Integrated RT-VLM checkpoint | `ngc:nim/nvidia/cosmos3-nano-reasoner:bf16-final` | Yes for integrated RT-VLM |
| `RTVI_VLM_MESSAGE_BUS` | Generated-output broker type | `kafka` | Yes when RT-VLM Kafka is enabled |
| `RTVI_VLM_MESSAGE_BUS_TOPIC` | Raw RT-VLM caption topic | `mdx-vlm-captions` | Yes when RT-VLM Kafka is enabled |
| `RTVI_VLM_ERROR_BUS` | Error-output broker type | `kafka` | Yes when RT-VLM Kafka is enabled |
| `VLM_NAME` | Model id sent to LVS | `nim_nvidia_cosmos3-nano-reasoner_bf16-final` | Yes |
| `LLM_NAME` | LLM model id | `nvidia/nemotron-3.5-lightning-30b-a3b` | Required for local LLM |
| `LLM_BASE_URL` | Remote or local LLM OpenAI-compatible base URL | `http://${HOST_IP}:${LLM_PORT}` when local | Required |
| `NVIDIA_API_KEY` / `OPENAI_API_KEY` | Remote endpoint auth and LVS LLM API key fallback | empty | Required when endpoint enforces auth |
| `VSS_APPS_DIR` | Optional build/deploy root when wrapper composes are used; source-of-truth LVS compose uses repo-relative config mounts instead | none | Conditional |
| `VSS_DATA_DIR` | Data root for models, videos, logs, and caches | none | Yes |
| `HOST_IP` | Host-reachable IP address | none | Yes |

## Network Requirements

- **Ports exposed:** LVS binds `38111` for the REST API and optionally `38112`
  for MCP/SSE when enabled.
- **Inbound:** Operators, generated deploy skills, or agents call
  `http://${HOST_IP}:38111/v1/summarize`.
- **Outbound:** LVS calls RT-VLM at `RTVI_VLM_URL`, Elasticsearch at
  `ES_HOST:ES_PORT`, Kafka at `KAFKA_BOOTSTRAP_SERVERS`, and the configured LLM
  endpoint.
- **DNS / hostname assumptions:** The Docker service uses its Compose bridge
  network and explicit host port mappings. Use Compose service names for
  container-to-container peers and `${HOST_IP}` for host-facing URLs.
- **`network_mode`:** Compose bridge (default).

## Known Integration Constraints

- **Stored-video scope for build-vision-ai.** This add-on is for videos
  uploaded to VIOS or stored by VIOS recorder. Do not generate live stream
  summarization wiring unless a future prompt explicitly requests it.
- **Use `/v1/summarize` for output.** `/v1/stream_summarize` and
  `/v1/generate_captions` are not the output path for this add-on.
- **Model id must be discovered.** `VLM_NAME` must match `GET /models`; a
  friendly model name that RT-VLM does not advertise causes summarize requests
  to fail.
- **`lvs` has optional peer `depends_on` entries.** Standalone generated builds
  must keep selected Nemotron/Cosmos NIM and `rtvi-vlm` peers, and strip only
  undefined optional peers when a wrapper or selected deployment shape does not
  declare them.
- **LVS config bind mount must resolve to a file.** The compose mount for
  `config.yaml` must point to a real file, not a Docker-created directory.
- **Image variable isolation.** Generated profiles must keep the LVS image
  isolated from unrelated service image variables, using the source compose's
  image contract or the `VSS_VIDEO_SUMMARIZATION_IMAGE` /
  `VSS_VIDEO_SUMMARIZATION_TAG` pair.
- **Single-file processing concurrency.** A busy LVS instance can return 503 for
  concurrent summarize requests. Generated smoke tests should run one request at
  a time.

## Example Compose Snippet

Excerpted shape from `deploy/docker/services/video-summarization/compose.yml`:

```yaml
services:
  lvs-server:
    image: ${CONTAINER_IMAGE:-${VSS_VIDEO_SUMMARIZATION_IMAGE:-${VSS_CONTAINER_REGISTRY:-nvcr.io/nvidia/vss-core}/vss-video-summarization}:${VSS_VIDEO_SUMMARIZATION_TAG:-3.3.0}}
    container_name: vss-lvs
    profiles: ["lvs-server"]
    ports:
      - ${BACKEND_HOST_PORT:-38111}:${BACKEND_PORT:-38111}
    volumes:
      - ${VSS_APPS_DIR:?VSS_APPS_DIR must be set}/services/video-summarization/configs/config.yaml:/app/config.yaml:ro
    environment:
      CA_RAG_CONFIG: /app/config.yaml
      LVS_DATABASE_BACKEND: ${LVS_DATABASE_BACKEND:-elasticsearch_db}
      KAFKA_ENABLED: ${KAFKA_ENABLED:-false}
      KAFKA_BOOTSTRAP_SERVERS: ${KAFKA_BOOTSTRAP_SERVERS:-kafka:29092}
      KAFKA_STRUCTURED_SUMMARY_TOPIC: ${KAFKA_STRUCTURED_SUMMARY_TOPIC:-mdx-structured-events-summary}
      RTVI_VLM_URL: ${RTVI_VLM_BASE_URL:-}
      RTVI_VLM_URL_PASSTHROUGH: "true"
    depends_on:
      nvidia-nemotron-nano-9b-v2-fp8:
        condition: service_healthy
        required: false
      nemotron-3.5-lightning-30b-a3b:
        condition: service_healthy
        required: false
      nvidia-nemotron-nano-9b-v2-fp8-shared-gpu:
        condition: service_healthy
        required: false
      nemotron-3.5-lightning-30b-a3b-shared-gpu:
        condition: service_healthy
        required: false
      cosmos3-reasoner:
        condition: service_healthy
        required: false
      cosmos3-reasoner-shared-gpu:
        condition: service_healthy
        required: false
      rtvi-vlm:
        condition: service_healthy
        required: false
```
