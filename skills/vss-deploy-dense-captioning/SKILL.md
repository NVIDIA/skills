---
name: vss-deploy-dense-captioning
description: Use this skill when deploying standalone RT-VLM dense captioning or calling its REST API (uploads, captions, streams, chat-completions, Kafka). Not for VSS profile deploy or video-search ingestion.
license: Apache-2.0
metadata:
  version: "3.2.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational deployment"
---
## Purpose

Stand up the RT-VLM dense-captioning microservice on its own and exercise every endpoint it exposes (file upload, generate_captions, stream add/delete, chat-completions, Kafka topics).

## Prerequisites

For standalone RT-VLM deployment:
- Docker, Docker Compose, NVIDIA Container Toolkit, and a visible GPU.
- NGC registry credentials in `$NGC_CLI_API_KEY` for `docker login nvcr.io`,
  image pulls, and local NGC model/artifact downloads.
- `curl`, `jq`, and any writable working directory for the standalone compose copy.

For API calls against an existing service:
- Running RT-VLM service reachable at `$BASE_URL`.
- Bearer token in `$RTVI_VLM_API_KEY` or `$NGC_CLI_API_KEY`, depending on how the
  service was configured.

For full VSS profile deployment:
- Use `../vss-deploy-profile/SKILL.md`; this skill does not deploy full VSS profiles.

## Instructions

Follow the routing tables and step-by-step workflows below. Each section that ends in *workflow*, *quick start*, or *flow* is intended to be executed top-to-bottom. Detailed reference material lives in `references/` and helper scripts live in `scripts/` — call them via `run_script` when the skill points to a script by name.

## Examples

Worked end-to-end examples are kept under `evals/` (each `*.json` manifest contains a runnable scenario) and inline in the per-workflow `curl` blocks below. Run a Tier-3 evaluation with `nv-base validate <this-skill-dir> --agent-eval` to replay them.

## Limitations

- Requires either a standalone RT-VLM service deployed via this skill or an
  existing RT-VLM service reachable from the caller.
- NGC-hosted models and NIMs may be subject to rate-limits, GPU memory requirements, and license restrictions.
- Concurrency, GPU memory, and storage limits depend on the host hardware and the profile's compose file.

## Troubleshooting

- **Error**: REST call returns connection refused. **Cause**: target microservice not running. **Solution**: probe `/docs` or `/health`; redeploy via `vss-deploy-profile` or the matching `vss-deploy-*` skill.
- **Error**: HTTP 401/403 from NGC pulls. **Cause**: missing/expired `NGC_CLI_API_KEY`. **Solution**: `docker login nvcr.io` and re-export the key before retrying.
- **Error**: container OOM or model fails to load. **Cause**: insufficient GPU memory for the selected profile. **Solution**: switch to a smaller variant or free GPUs via `docker compose down`.

# Deploy and Use RT-VLM Dense Captioning (VSS 3.2)

RT-VLM is NVIDIA's real-time vision-language microservice: decode video (file or
RTSP), segment it into chunks, run a VLM (`cosmos-reason1`, `cosmos-reason2`, or any
OpenAI-compatible model), stream dense captions back over SSE/HTTP, and publish
captions, incident alerts, and errors to Kafka. Use this skill to deploy the
standalone RT-VLM service when a full VSS profile is not already running, then call
its `/v1/...` API for caption generation, file upload, live-stream management, health
checks, NIM-compatible chat completions, or Prometheus metrics. API reference:
<https://docs.nvidia.com/vss/latest/real-time-vlm-api.html>.

## Deployment Routing

If the user asks to deploy a full VSS profile, use
[`../vss-deploy-profile/SKILL.md`](../vss-deploy-profile/SKILL.md). That skill
owns profile routing, `generated.env`, `resolved.yml`, multi-service sizing, and
full-stack deploy/teardown.

If the user asks for standalone RT-VLM dense captioning, or no VSS profile is
already running, use the standalone RT-VLM flow in
[`references/deploy-rt-vlm-service.md`](references/deploy-rt-vlm-service.md)
before calling the API. This follows the same compose-centric pattern as
`vss-deploy-profile`: gather context, run preflights, work from a local copy,
dry-run with `docker compose config`, review, deploy, then wait for health.

## Standalone Deployment Flow

Always follow this sequence. Never skip the dry-run.

```bash
# 1. Copy deploy/docker/services/rtvi/rtvi-vlm/rtvi-vlm-docker-compose.yml
#    into any writable standalone working directory.
# 2. Derive RTVI_VLM_IMAGE_TAG from that compose copy.
# 3. Strip the standalone-only dangling depends_on block from the copy.
# 4. Create a gitignored .env with the required RT-VLM values.
# 5. Prepare host bind paths such as $VSS_DATA_DIR/data_log/vst/clip_storage.
# 6. docker compose --env-file .env -f rtvi-vlm-docker-compose.yml config --quiet
# 7. docker pull the exact RT-VLM image tag.
# 8. docker compose ... up -d rtvi-vlm, wait for ready, then smoke test.
```

Run preflights before any pull or `up`; stop and fix failures here before
debugging RT-VLM itself:

```bash
nvidia-smi --query-gpu=index,name --format=csv,noheader
nvidia-container-cli info
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

For standalone single-file deployments, do not run the raw
`deploy/docker/services/rtvi/rtvi-vlm/rtvi-vlm-docker-compose.yml` directly: it
contains `depends_on` references to sibling VLM/NIM services that are only
defined in the full VSS/met-blueprints compose project. The standalone reference
shows how to copy the compose file, derive the current image tag from it, strip
the `depends_on` block, and validate the result before `up`.

If `docker pull` fails with a containerd snapshotter/unpack error on Docker 28+,
apply the `/etc/docker/daemon.json` `containerd-snapshotter=false` fix in the
standalone reference before retrying.

Minimum standalone `.env` values:

| Host env var | Required when | Purpose |
|---|---|---|
| `NGC_CLI_API_KEY` | Standalone deploy path | NGC registry image pull and NGC model/artifact download |
| `RTVI_VLM_API_KEY` or `NGC_CLI_API_KEY` | Authenticated API calls | RT-VLM bearer auth after the service is running |
| `RTVI_VLM_PORT` | Always | Host API port mapped to container `8000` |
| `HOST_IP` | Always | Kafka bootstrap host (`${HOST_IP}:9092`) |
| `VSS_DATA_DIR` | Always | Required clip-storage bind mount |
| `RTVI_VLM_MODEL_TO_USE` | Always for standalone | Backend selector; use `cosmos-reason2` for the default local model or `openai-compat` for a remote/sibling endpoint |
| `RTVI_VLM_MODEL_PATH` | Local self-hosted model | Source-backed Cosmos Reason 2 path: `ngc:nim/nvidia/cosmos-reason2-8b:0303-fp8-dynamic-kv8` |
| `RTVI_VLM_ENDPOINT` | `RTVI_VLM_MODEL_TO_USE=openai-compat` | Remote/sibling OpenAI-compatible VLM endpoint |
| `VLM_NAME` | `RTVI_VLM_MODEL_TO_USE=openai-compat` | Model/deployment name exposed by that endpoint |

## Setup

```bash
export BASE_URL="http://localhost:${RTVI_VLM_PORT:-8018}"  # host-side RT-VLM port
export API_KEY="${NGC_CLI_API_KEY:-${RTVI_VLM_API_KEY:-}}" # bearer token used by host-side curl commands
: "${API_KEY:?Set NGC_CLI_API_KEY or RTVI_VLM_API_KEY before calling authenticated endpoints}"
```

Every request below uses `Authorization: Bearer $API_KEY`. Health endpoints
(`/v1/health/*`, `/v1/ready`, `/v1/live`, `/v1/startup`) typically work without auth.

**Smoke test before use:**
```bash
curl -fsS "$BASE_URL/v1/health/ready"
MODEL_ID="$(curl -fsS "$BASE_URL/v1/models" -H "Authorization: Bearer $API_KEY" | jq -r '.data[0].id // .id')"
curl -fsS "$BASE_URL/openapi.json" | jq -r '.paths | keys[]' | sort
```

## Quick Start — dense captions from a local video

```bash
# 1. Upload the video, capture its file id
FILE_ID=$(curl -fsS -X POST "$BASE_URL/v1/files" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@/path/to/warehouse.mp4" \
  -F "purpose=vision" \
  -F "media_type=video" | jq -r '.id')

# 2. Generate captions + alerts (SSE stream of chunked responses)
curl -N -X POST "$BASE_URL/v1/generate_captions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"id\": \"$FILE_ID\",
    \"prompt\": \"Write a concise dense caption for each 10-second segment of this warehouse video.\",
    \"model\": \"$MODEL_ID\",
    \"chunk_duration\": 10,
    \"stream\": true
  }"
```

## Endpoints

### Captions
> Generate VLM captions and alerts for videos and live streams.

#### `POST /v1/generate_captions` — Generate VLM captions (and alerts) for video/stream

**Required:**
| Field | Type | Description |
|-------|------|-------------|
| `id` | string \| array | UUID of a previously-uploaded file, or id of an active live stream. Accepts a list of ids for batch |
| `prompt` | string | User prompt to the VLM (e.g. dense-caption instruction) |
| `model` | string | Exact model id returned by `GET /v1/models`, for example `nim_nvidia_cosmos-reason2-8b_0303-fp8-dynamic-kv8`; backend selector aliases such as `cosmos-reason2` are not request model ids |

**Key optional fields:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | string | — | System prompt; use `<think></think><answer></answer>` tags to enable reasoning on Cosmos Reason |
| `enable_reasoning` | boolean | false | Turn on reasoning for Cosmos Reason models |
| `enable_audio` | boolean | false | Transcribe audio (via Riva) and fold into captions |
| `chunk_duration` | integer | — | Segment video into N-second chunks (`0` = no chunking) |
| `chunk_overlap_duration` | integer | 0 | Overlap between consecutive chunks |
| `num_frames_per_second_or_fixed_frames_chunk` | number | — | FPS (if `use_fps_for_chunking=true`) or fixed frames per chunk |
| `use_fps_for_chunking` | boolean | false | Interpret above as FPS vs. fixed-frame count |
| `vlm_input_width` / `vlm_input_height` | int | — | Resize frames before inference (0 = native) |
| `media_info` | object | — | `{"type":"offset","start_offset":0,"end_offset":10}` to process a slice of a file (not live streams) |
| `stream` | boolean | false | SSE: emit per-chunk caption deltas as `data:` events (recommended for long videos) |
| `max_tokens` / `temperature` / `top_p` / `top_k` / `seed` / `ignore_eos` | | | Standard sampling controls |
| `response_format` | object | — | Query response format object |
| `mm_processor_kwargs` | object | — | Extra kwargs for the multimodal processor (e.g. size, shortest/longest edge) |

```bash
curl -N -X POST "$BASE_URL/v1/generate_captions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "prompt": "Dense-caption this warehouse video, one sentence per 10s chunk.",
    "model": "nim_nvidia_cosmos-reason2-8b_0303-fp8-dynamic-kv8",
    "chunk_duration": 10,
    "stream": true
  }'
```

**Response shape:** live 26.05 responses use `chunk_responses` with
`start_time`/`end_time`; SSE streams terminate with `data: [DONE]`. See
[`references/api-surface-26.05.md`](references/api-surface-26.05.md).

#### `DELETE /v1/generate_captions/{stream_id}` — Stop caption generation for a live stream, if exposed

Some deployments expose this companion stop endpoint. Check the live OpenAPI
(`curl -fsS "$BASE_URL/openapi.json" | jq '.paths | keys[]'`) before using it.
Always pair live-stream cleanup with `DELETE /v1/streams/delete/{stream_id}` to
un-register the RTSP source.

```bash
curl -X DELETE "$BASE_URL/v1/generate_captions/$STREAM_ID" -H "Authorization: Bearer $API_KEY"
```

### Files
> Upload and manage media files consumed by `/v1/generate_captions`.

#### `POST /v1/files` — Upload a media file (multipart)
```bash
curl -X POST "$BASE_URL/v1/files" -H "Authorization: Bearer $API_KEY" \
  -F "file=@./video.mp4" -F "purpose=vision" -F "media_type=video"
```
**Response:** `{ "id", "object": "file", "bytes", "created_at", "filename", "purpose" }`.
Optional metadata such as `sensor_name` may be accepted by newer builds; check
the live OpenAPI before sending it.

#### `GET /v1/files?purpose=vision` — List uploaded files
#### `GET /v1/files/{file_id}` — File metadata
#### `GET /v1/files/{file_id}/content` — Download original file content
#### `DELETE /v1/files/{file_id}` — Delete file (releases asset storage)

### Live Stream
> RTSP stream lifecycle.

#### `POST /v1/streams/add` — Register one or more RTSP streams
**Required per stream:** `liveStreamUrl` (must start with `rtsp://`), `description`.
Optional: `username`, `password`, `sensor_name`, and placement metadata
(`place_name`, `place_type`, `place_lat`, `place_lon`, `place_alt`,
`place_coordinate_x`, `place_coordinate_y`).

Precheck public or external RTSP sources before registering them. A probe exit
code alone is not enough; `gst-discoverer-1.0` can exit `0` while reporting an
unknown media type. Treat the stream as usable only when a probe output
identifies at least one video stream/caps entry. If one probe is inconclusive,
cross-check with another tool such as `ffprobe` before failing or registering:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_type -of csv=p=0 "$RTSP_URL" | grep -qx video
```

```bash
STREAM_ID=$(curl -fsS -X POST "$BASE_URL/v1/streams/add" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"streams":[{"liveStreamUrl":"rtsp://cam:8554/live","description":"warehouse cam 1"}]}' \
  | jq -r '.results[0].id')
```

#### `GET /v1/streams/get-stream-info` — List active streams
#### `DELETE /v1/streams/delete/{stream_id}` — Remove a single stream
#### `DELETE /v1/streams/delete-batch` — Remove many (`{"stream_ids":[...]}`)

#### CV-style singular stream endpoints

26.05 deployments also expose CV-style stream control paths:
`POST /v1/stream/add`, `GET /v1/stream/get-stream-info`, and
`POST /v1/stream/remove`. Use these when a workflow or release note explicitly uses
the key/value envelope; otherwise prefer the plural RT-VLM stream endpoints
above. See [`references/api-surface-26.05.md`](references/api-surface-26.05.md)
for examples and the `stream_count:0` compatibility caveat.

### NIM Compatible
> OpenAI-compatible endpoints for interop with OpenAI/NVIDIA-API clients.

#### `POST /v1/chat/completions` — OpenAI-compatible chat (text + multimodal)
**Required:** `messages`, `model`. Text-only requests work and omit `id`,
`video_url`, and `image_url`. For uploaded-video, direct `video_url`,
direct `image_url`, streaming, and RTSP-backed chat examples, see
[`references/api-surface-26.05.md`](references/api-surface-26.05.md).

```bash
curl -X POST "$BASE_URL/v1/chat/completions" -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Summarize this scene.\"}]}"
```

#### `POST /v1/completions` — OpenAI-compatible legacy completions
This endpoint exists for compatibility, but on current 26.05 builds text-only legacy
completion requests return HTTP 400 by design. Use `/v1/chat/completions` for
text-only and multimodal requests.

#### `GET /v1/version` — `{ "version": "3.2.0-..." }`
#### `GET /v1/manifest` — NIM manifest
#### `GET /v1/health/live` · `GET /v1/health/ready` — NIM-style probes

Do not assume `/v1/license` exists. The current 26.05 live OpenAPI does not expose it
and the endpoint returns 404; only call it after checking `GET /openapi.json`.

### Models · Metadata · Metrics · Health Check
#### `GET /v1/models` — List loaded VLMs: `{ "data": [{ "id", "object": "model", "owned_by" }] }`
#### `GET /v1/metadata` — Service metadata (build, release, image tag)
#### `GET /v1/assets/stats` — Asset storage counts, TTL, and oldest-asset age
#### `GET /v1/metrics` — Prometheus metrics (plain text)
#### `GET /v1/ready` · `GET /v1/live` · `GET /v1/startup` — Kubernetes-style probes

---

## Common Workflows

The four standard dense-captioning scenarios.

### 1. Dense captions from a stored video file

```bash
# Upload → capture file id → generate captions (SSE stream)
FILE_ID=$(curl -fsS -X POST "$BASE_URL/v1/files" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@warehouse.mp4" -F "purpose=vision" -F "media_type=video" | jq -r '.id')

curl -N -X POST "$BASE_URL/v1/generate_captions" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d "{
    \"id\": \"$FILE_ID\",
    \"prompt\": \"Describe warehouse events in 1 sentence per 10s chunk.\",
    \"model\": \"$MODEL_ID\",
    \"chunk_duration\": 10,
    \"stream\": true
  }"

# When done, free storage:
curl -X DELETE "$BASE_URL/v1/files/$FILE_ID" -H "Authorization: Bearer $API_KEY"
```

### 2. Dense captions from an RTSP live stream

```bash
# Register the stream
STREAM_ID=$(curl -fsS -X POST "$BASE_URL/v1/streams/add" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"streams":[{"liveStreamUrl":"rtsp://10.0.0.5:8554/warehouse","description":"warehouse cam"}]}' \
  | jq -r '.results[0].id')

# Start continuous caption generation
curl -N -X POST "$BASE_URL/v1/generate_captions" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d "{
    \"id\": \"$STREAM_ID\",
    \"prompt\": \"Describe each event; start each sentence with a timestamp.\",
    \"model\": \"$MODEL_ID\",
    \"chunk_duration\": 10,
    \"num_frames_per_second_or_fixed_frames_chunk\": 2,
    \"use_fps_for_chunking\": true,
    \"stream\": true
  }" &

# Tear down when finished. If the live OpenAPI exposes
# DELETE /v1/generate_captions/{stream_id}, call it before unregistering.
curl -X DELETE "$BASE_URL/v1/streams/delete/$STREAM_ID"  -H "Authorization: Bearer $API_KEY"
```

### 3. Dense captions with alerts from an RTSP stream

```bash
# Pre-req: Kafka is enabled and topics match the deployment source.
# The checked-in rtvi-vlm/.env and VSS alerts profiles use:
#   RTVI_VLM_KAFKA_ENABLED=true
#   RTVI_VLM_KAFKA_TOPIC=mdx-vlm
#   RTVI_VLM_KAFKA_INCIDENT_TOPIC=mdx-vlm-incidents
#   RTVI_VLM_ERROR_MESSAGE_TOPIC=vision-llm-errors
#   HOST_IP=<kafka-host>
# A copied compose without those env overrides falls back to vision-llm-* topics.
# Confirm the live container before consuming:
#   docker exec vss-rtvi-vlm printenv KAFKA_TOPIC KAFKA_INCIDENT_TOPIC ERROR_MESSAGE_TOPIC

STREAM_ID=$(curl -fsS -X POST "$BASE_URL/v1/streams/add" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"streams":[{"liveStreamUrl":"rtsp://10.0.0.5:8554/warehouse","description":"warehouse cam"}]}' \
  | jq -r '.results[0].id')

curl -N -X POST "$BASE_URL/v1/generate_captions" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d "{
    \"id\": \"$STREAM_ID\",
    \"prompt\": \"You are a warehouse monitoring system. Describe the scene in one sentence, then on a new line output exactly:\\nAnomaly Detected: Yes/No\\nReason: <one sentence>\\nFlag an anomaly if any worker is missing a hard hat or high-vis vest.\",
    \"system_prompt\": \"Answer the user's question correctly in yes or no.\",
    \"model\": \"$MODEL_ID\",
    \"chunk_duration\": 60,
    \"chunk_overlap_duration\": 10,
    \"stream\": true
  }"
```

**Consume alerts from Kafka**. Kafka values are NvSchema protobuf payloads, so
use `print.value=false` for a clean validation pass that shows timestamp, key,
and headers without dumping binary payload bytes. The VSS alerts/profile source
uses `mdx-vlm-incidents`; a bare copied compose may fall back to
`vision-llm-events-incidents` if no `RTVI_VLM_KAFKA_INCIDENT_TOPIC` override is
loaded. Prefer the live container environment over hard-coded topic names.
```bash
INCIDENT_TOPIC="${INCIDENT_TOPIC:-$(docker exec vss-rtvi-vlm printenv KAFKA_INCIDENT_TOPIC 2>/dev/null || true)}"
INCIDENT_TOPIC="${INCIDENT_TOPIC:-mdx-vlm-incidents}"

docker exec mdx-kafka kafka-console-consumer \
  --bootstrap-server 127.0.0.1:9092 \
  --topic "$INCIDENT_TOPIC" \
  --from-beginning \
  --timeout-ms 5000 \
  --max-messages 10 \
  --property print.timestamp=true \
  --property print.key=true \
  --property print.headers=true \
  --property print.value=false
```

If Kafka is not running in the VSS `mdx-kafka` container, use the Kafka CLI from
the host or container running the broker:
```bash
INCIDENT_TOPIC="${INCIDENT_TOPIC:-mdx-vlm-incidents}"

kafka-console-consumer \
  --bootstrap-server "$HOST_IP:9092" \
  --topic "$INCIDENT_TOPIC" \
  --from-beginning \
  --timeout-ms 5000 \
  --max-messages 10 \
  --property print.timestamp=true \
  --property print.key=true \
  --property print.headers=true \
  --property print.value=false
```

For standalone validation, remember that the RT-VLM compose maps Kafka through
`KAFKA_BOOTSTRAP_SERVERS=${HOST_IP}:9092`; setting `KAFKA_BOOTSTRAP_SERVERS`
directly in `.env` is ignored unless the compose is changed. The broker must
advertise a listener reachable from the `vss-rtvi-vlm` container. `localhost`
inside the broker and service containers is not the host, and a broker alias
such as `kafka:9092` only works when both containers share that Docker network.
For RT-VLM-only validation, prefer the self-contained broker in
`references/kafka-workflows.md` over the full repo infra compose; the latter
expects full-profile SDRC env/config. If Kafka is already running, ask the user
whether to reuse it or launch a dedicated broker before stopping or replacing
anything. Run CLI checks inside the actual broker container, but still configure
the advertised listener so RT-VLM can connect from its container network.

Incident protobuf (`ext.proto :: Incident`) key fields: `sensorId`, `timestamp`, `end`,
`objectIds`, `frameIds`, `place`, `analyticsModule`, `category`, `isAnomaly` (`true` for
alerts), `llm` (nested VisionLLM), `info` map including `triggerPhrase`, `verdict`,
`requestId`, `chunkIdx`, `streamId`, `alertCategory` (if the deployment supports the
`alert_category` query field — post-3.1).

### 4. Kafka workflows (alerts + message bus)

Dense captioning with alerts on an RTSP stream and the HTTP-vs-Kafka response model are documented in [`references/kafka-workflows.md`](references/kafka-workflows.md).

## Error Reference

| Code | Meaning | Common Cause |
|------|---------|--------------|
| 400 | Bad Request | Missing required field (`id`, `prompt`, `model`); unsupported `media_type`; unknown `model` name |
| 401 | Unauthorized | Missing/invalid `Authorization: Bearer $API_KEY` — or wrong key format (expect `nvapi-...`) |
| 404 | Not Found | `file_id` deleted / stream_id not registered / wrong endpoint path (note: `{stream_id}` is required on `DELETE /v1/streams/delete/{stream_id}`) |
| 413 | Payload Too Large | Uploaded file exceeds server `MAX_FILE_SIZE`; increase or pre-chunk the video |
| 422 | Unprocessable Entity | Pydantic schema violation — e.g. `use_fps_for_chunking=true` without `num_frames_per_second_or_fixed_frames_chunk`; stream ids supplied to a file-only field like `media_info` |
| 429 | Rate Limited | Too many concurrent streams — raise `VLM_BATCH_SIZE` or spread across instances |
| 500 | Server Error | VLM inference exception (OOM, model unavailable) — check `docker logs vss-rtvi-vlm` |
| 503 | Service Busy | Startup not complete (model still downloading) or upstream NIM dependency unhealthy |

---

## Gotchas

- **Use the live OpenAPI as the source of truth.** For VSS 3.2, the caption-generation endpoint is `/v1/generate_captions`. Some older references and images used `/v1/generate_captions_alerts`; do not assume that path exists unless `GET /openapi.json` shows it.
- **URL-based input support depends on the deployed service version.** If the live schema does not expose `url`/`media_type`/`creation_time`, upload via `POST /v1/files` first and pass the returned `id`.
- **Alert trigger = the tokens `"yes"` or `"true"` in the VLM response (case-insensitive)**. There is no per-request alert flag. Design prompts with an explicit `Anomaly Detected: Yes/No` line and set `system_prompt` to constrain the model to Yes/No answers (per the VSS docs). Every chunk is published to `KAFKA_TOPIC`; matched chunks additionally go to `KAFKA_INCIDENT_TOPIC` with `isAnomaly=true`, `info["triggerPhrase"]` set to the matched tokens, and `info["verdict"]="confirmed"`.
- **`alert_category` support depends on the deployed service version.** If the live OpenAPI schema does not expose it, Kafka incidents default `incident.category = "vlm-alert"`.
- **Kafka topics are server-side config, not per-request.** The `KAFKA_*` env vars (via compose `RTVI_VLM_KAFKA_*` rewrites) are fixed at container start — clients can't override topics on a per-request basis. Kafka publish is *additive* to the HTTP response, never a replacement.
- **Topic names differ by deployment source.** The checked-in RT-VLM `.env` and VSS alerts/profile sources use `mdx-vlm` and `mdx-vlm-incidents`; a bare copied compose with no `RTVI_VLM_KAFKA_*` overrides falls back to `vision-llm-messages` and `vision-llm-events-incidents`. Always trust the live `vss-rtvi-vlm` environment before consuming.
- **Standalone Kafka must advertise `${HOST_IP}:9092`.** The RT-VLM compose uses `KAFKA_BOOTSTRAP_SERVERS=${HOST_IP}:9092`; a broker that advertises `localhost:9094` or `kafka:9092` may pass producer/consumer tests inside the broker container while RT-VLM publish fails.
- **Start Kafka before RT-VLM when Kafka is enabled.** For deterministic standalone validation, make the broker reachable at `${HOST_IP}:9092` first. If you start Kafka later or change its advertised listener, restart/recreate `rtvi-vlm` before expecting Kafka offsets to move.
- **`stream=true` returns Server-Sent Events, not chunked JSON.** Use `curl -N` (no buffering). Each event is `data: {...}\n\n` with per-chunk fields such as `content`, `start_time`, and `end_time`, terminated by `data: [DONE]`. Without `stream=true` the server buffers until the full video is processed — fine for short clips (<1 min), avoid for live streams.
- **Trust live OpenAPI for optional NIM-compatible endpoints.** `/v1/license` is not exposed by current 26.05 builds and returns 404, even though older generic NIM docs may mention it.
- **Prefer `/v1/chat/completions` over `/v1/completions`.** Text-only legacy completions return HTTP 400 by design on current 26.05 builds; text-only chat completions work.
- **`chunk_duration=0` disables chunking** — the entire video is sent to the VLM as one shot. Only meaningful for short clips; long videos will OOM or exceed `max_model_len`.
- **Default frame budget caps at `VLLM_MM_PROCESSOR_VIDEO_NUM_FRAMES` (256).** Requesting FPS that implies >256 frames per chunk is silently capped; drop FPS or shorten `chunk_duration` to stay within budget.
- **`enable_reasoning` requires a Cosmos Reason model.** Passing it with Qwen3-VL or other non-reasoning models is a no-op.
- **`/v1/metrics` is unauthenticated on current 26.05 standalone builds.** A Bearer token is harmless if a deployment has stricter auth, but do not fail validation when `/v1/metrics` returns HTTP 200 without auth.
- **File upload is multipart, not JSON.** Use `-F file=@path -F purpose=vision -F media_type=video`; a `-d` body returns 422.
- **Live-stream lifecycle cleanup must unregister the stream:** `DELETE /v1/streams/delete/{stream_id}` removes the RTSP source. If the live schema also exposes `DELETE /v1/generate_captions/{stream_id}`, call it first to stop inference explicitly.
