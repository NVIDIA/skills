---
name: rt-vlm
description: >
  Use this skill when working with the RTVI VLM or RT-VLM microservice API on VSS 3.1.
  Generate dense captions and alerts for stored video files and live RTSP streams via
  `/v1/generate_captions_alerts`; upload media via `/v1/files`; add and remove live
  streams with `/v1/streams/add` and `/v1/streams/delete/{stream_id}`; call
  OpenAI-compatible `/v1/chat/completions`; consume Kafka caption, incident, and error
  topics; or debug rtvi-vlm responses. For deployment, read
  `references/deploy-rt-vlm-service.md` first.
license: Apache-2.0
metadata:
  version: "3.1.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint operational deployment"
---

# RTVI VLM Usage API (VSS 3.1)

RTVI VLM is NVIDIA's real-time vision-language microservice: decode video (file or
RTSP) → segment into chunks → run a VLM (`cosmos-reason1`, `cosmos-reason2`, or any
OpenAI-compatible model) → stream dense captions back over SSE/HTTP and publish
captions + incident alerts + errors to Kafka. Use this skill whenever you need to hit
any `/v1/...` endpoint on the VSS 3.1 rtvi-vlm microservice: caption generation, file
upload, live-stream management, health checks, NIM-compatible chat completions,
Prometheus metrics. API reference: <https://docs.nvidia.com/vss/latest/real-time-vlm-api.html>.

## Setup

```bash
export BASE_URL="http://localhost:8000"     # RTVI VLM host:port — matches $RTVI_VLM_PORT in compose
export API_KEY="$NGC_API_KEY"               # Bearer token (NGC key works if the service was deployed with NGC auth)
```

Every request below uses `Authorization: Bearer $API_KEY`. Health endpoints
(`/v1/health/*`, `/v1/ready`, `/v1/live`, `/v1/startup`) typically work without auth.

**Smoke test before use:**
```bash
curl -fsS "$BASE_URL/v1/health/ready" && curl -fsS "$BASE_URL/v1/models" | jq
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
curl -N -X POST "$BASE_URL/v1/generate_captions_alerts" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"id\": \"$FILE_ID\",
    \"prompt\": \"Write a concise dense caption for each 10-second segment of this warehouse video.\",
    \"model\": \"cosmos-reason1\",
    \"chunk_duration\": 10,
    \"stream\": true
  }"
```

## Endpoints

### Captions
> Generate VLM captions and alerts for videos and live streams.

#### `POST /v1/generate_captions_alerts` — Generate VLM captions (and alerts) for video/stream

**Required:**
| Field | Type | Description |
|-------|------|-------------|
| `id` | string \| array | UUID of a previously-uploaded file, or id of an active live stream. Accepts a list of ids for batch |
| `prompt` | string | User prompt to the VLM (e.g. dense-caption instruction) |
| `model` | string | Model name — see `GET /v1/models` |

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
| `media_info` | object | — | `{"start_offset_ms": ..., "end_offset_ms": ...}` to process a slice of a file (not live streams) |
| `stream` | boolean | false | SSE: emit per-chunk caption deltas as `data:` events (recommended for long videos) |
| `max_tokens` / `temperature` / `top_p` / `top_k` / `seed` / `ignore_eos` | | | Standard sampling controls |
| `response_format` | object | — | Query response format object |
| `mm_processor_kwargs` | object | — | Extra kwargs for the multimodal processor (e.g. size, shortest/longest edge) |

```bash
curl -N -X POST "$BASE_URL/v1/generate_captions_alerts" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "prompt": "Dense-caption this warehouse video, one sentence per 10s chunk.",
    "model": "cosmos-reason1",
    "chunk_duration": 10,
    "stream": true
  }'
```

**Response (200, SSE when `stream=true`):** each event payload has `start_ts`, `end_ts`,
`content`, and a terminal `{"status": "completed"}` event.
**Response (200, non-stream):** `{ "id", "object": "caption", "choices": [{...}], "usage": {...} }`.

#### `DELETE /v1/generate_captions_alerts/{stream_id}` — Stop caption generation for a live stream

Stops inference while leaving the stream registered. Pair with
`DELETE /v1/streams/delete/{stream_id}` to also un-register the RTSP source.

```bash
curl -X DELETE "$BASE_URL/v1/generate_captions_alerts/$STREAM_ID" -H "Authorization: Bearer $API_KEY"
```

### Files
> Upload and manage media files consumed by `/v1/generate_captions_alerts`.

#### `POST /v1/files` — Upload a media file (multipart)
```bash
curl -X POST "$BASE_URL/v1/files" -H "Authorization: Bearer $API_KEY" \
  -F "file=@./video.mp4" -F "purpose=vision" -F "media_type=video"
```
**Response:** `{ "id", "object": "file", "bytes", "created_at", "filename", "purpose" }`.

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
```bash
STREAM_ID=$(curl -fsS -X POST "$BASE_URL/v1/streams/add" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"streams":[{"liveStreamUrl":"rtsp://cam:8554/live","description":"warehouse cam 1"}]}' \
  | jq -r '.results[0].id')
```

#### `GET /v1/streams/get-stream-info` — List active streams
#### `DELETE /v1/streams/delete/{stream_id}` — Remove a single stream
#### `DELETE /v1/streams/delete-batch` — Remove many (`{"stream_ids":[...]}`)

### NIM Compatible
> OpenAI-compatible endpoints for interop with OpenAI/NVIDIA-API clients.

#### `POST /v1/chat/completions` — OpenAI-compatible chat (text + multimodal)
**Required:** `messages`, `model`. Text-only requests omit `id` / `video_url` / `image_url`.
```bash
curl -X POST "$BASE_URL/v1/chat/completions" -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"cosmos-reason1","messages":[{"role":"user","content":"Summarize this scene."}]}'
```

#### `POST /v1/completions` — OpenAI-compatible legacy completions
#### `GET /v1/version` — `{ "version": "3.1.0-..." }`
#### `GET /v1/license` — license text
#### `GET /v1/manifest` — NIM manifest
#### `GET /v1/health/live` · `GET /v1/health/ready` — NIM-style probes

### Models · Metadata · Metrics · Health Check
#### `GET /v1/models` — List loaded VLMs: `{ "data": [{ "id", "object": "model", "owned_by" }] }`
#### `GET /v1/metadata` — Service metadata (build, release, image tag)
#### `GET /v1/metrics` — Prometheus metrics (plain text)
#### `GET /v1/ready` · `GET /v1/live` · `GET /v1/startup` — Kubernetes-style probes

---

## Common Workflows

The four scenarios from the VSS 3.1 RT-VLM Usage Skill requirements.

### 1. Dense captions from a stored video file

```bash
# Upload → capture file id → generate captions (SSE stream)
FILE_ID=$(curl -fsS -X POST "$BASE_URL/v1/files" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@warehouse.mp4" -F "purpose=vision" -F "media_type=video" | jq -r '.id')

curl -N -X POST "$BASE_URL/v1/generate_captions_alerts" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d "{
    \"id\": \"$FILE_ID\",
    \"prompt\": \"Describe warehouse events in 1 sentence per 10s chunk.\",
    \"model\": \"cosmos-reason1\",
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

# Start continuous caption generation (runs until stream stops or DELETE)
curl -N -X POST "$BASE_URL/v1/generate_captions_alerts" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d "{
    \"id\": \"$STREAM_ID\",
    \"prompt\": \"Describe each event; start each sentence with a timestamp.\",
    \"model\": \"cosmos-reason1\",
    \"chunk_duration\": 10,
    \"num_frames_per_second_or_fixed_frames_chunk\": 2,
    \"use_fps_for_chunking\": true,
    \"stream\": true
  }" &

# Tear down when finished:
curl -X DELETE "$BASE_URL/v1/generate_captions_alerts/$STREAM_ID" -H "Authorization: Bearer $API_KEY"
curl -X DELETE "$BASE_URL/v1/streams/delete/$STREAM_ID"  -H "Authorization: Bearer $API_KEY"
```

# Pre-req: the container was started with:
#   RTVI_VLM_KAFKA_ENABLED=true
#   RTVI_VLM_KAFKA_TOPIC=vision-llm-messages
#   RTVI_VLM_KAFKA_INCIDENT_TOPIC=vision-llm-events-incidents
#   RTVI_VLM_ERROR_MESSAGE_TOPIC=vision-llm-errors
#   HOST_IP=<kafka-host>

STREAM_ID=$(curl -fsS -X POST "$BASE_URL/v1/streams/add" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"streams":[{"liveStreamUrl":"rtsp://10.0.0.5:8554/warehouse","description":"warehouse cam"}]}' \
  | jq -r '.results[0].id')

curl -N -X POST "$BASE_URL/v1/generate_captions_alerts" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d "{
    \"id\": \"$STREAM_ID\",
    \"prompt\": \"You are a warehouse monitoring system. Describe the scene in one sentence, then on a new line output exactly:\\nAnomaly Detected: Yes/No\\nReason: <one sentence>\\nFlag an anomaly if any worker is missing a hard hat or high-vis vest.\",
    \"system_prompt\": \"Answer the user's question correctly in yes or no.\",
    \"model\": \"cosmos-reason2\",
    \"chunk_duration\": 60,
    \"chunk_overlap_duration\": 10,
    \"stream\": true
  }"
```

**Consume alerts from Kafka** (when using the VSS foundational Kafka container).
Kafka values are NvSchema protobuf payloads, so use `print.value=false` for a
clean validation pass that shows timestamp, key, and headers without dumping
binary payload bytes:
```bash
docker exec mdx-kafka kafka-console-consumer \
  --bootstrap-server 127.0.0.1:9092 \
  --topic vision-llm-events-incidents \
  --from-beginning \
  --timeout-ms 5000 \
  --max-messages 10 \
  --property print.timestamp=true \
  --property print.key=true \
  --property print.headers=true \
  --property print.value=false
```

If Kafka is not running in the VSS `mdx-kafka` container, use the Kafka CLI from
the host running the broker:
```bash
kafka-console-consumer \
  --bootstrap-server "$HOST_IP:9092" \
  --topic vision-llm-events-incidents \
  --from-beginning \
  --timeout-ms 5000 \
  --max-messages 10 \
  --property print.timestamp=true \
  --property print.key=true \
  --property print.headers=true \
  --property print.value=false
```
Incident protobuf (`ext.proto :: Incident`) key fields: `sensorId`, `timestamp`, `end`,
`objectIds`, `frameIds`, `place`, `analyticsModule`, `category`, `isAnomaly` (`true` for
alerts), `llm` (nested VisionLLM), `info` map including `triggerPhrase`, `verdict`,
`requestId`, `chunkIdx`, `streamId`, `alertCategory` (if the deployment supports the
`alert_category` query field — post-3.1).

### 3. Kafka workflows (alerts + message bus)

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
| 500 | Server Error | VLM inference exception (OOM, model unavailable) — check `docker logs rtvi-vlm-*` |
| 503 | Service Busy | Startup not complete (model still downloading) or upstream NIM dependency unhealthy |

---

## Gotchas

- **3.1 GA endpoint is `/v1/generate_captions_alerts`, not `/v1/generate_captions`.** The rename lands in a post-3.1 build. For VSS 3.1 releases (`rtvi_vlm/26.01.x`–`26.02.3`), always use the `_alerts` suffix. `https://docs.nvidia.com/vss/latest/real-time-vlm-api.html` is the canonical reference.
- **No URL-based input in 3.1 GA** — the `url`/`media_type`/`creation_time` fields were added post-3.1. You **must** upload via `POST /v1/files` first and then pass the returned `id`.
- **Alert trigger = the tokens `"yes"` or `"true"` in the VLM response (case-insensitive)**. There is no per-request alert flag. Design prompts with an explicit `Anomaly Detected: Yes/No` line and set `system_prompt` to constrain the model to Yes/No answers (per the VSS docs). Every chunk is published to `KAFKA_TOPIC`; matched chunks additionally go to `KAFKA_INCIDENT_TOPIC` with `isAnomaly=true`, `info["triggerPhrase"]` set to the matched tokens, and `info["verdict"]="confirmed"`.
- **No `alert_category` query field in the 3.1 OpenAPI spec.** The Kafka incident topic defaults `incident.category = "vlm-alert"` on 3.1. Post-3.1 builds expose an optional `alert_category` request field to override `incident.category`.
- **Kafka topics are server-side config, not per-request.** The `KAFKA_*` env vars (via compose `RTVI_VLM_KAFKA_*` rewrites) are fixed at container start — clients can't override topics on a per-request basis. Kafka publish is *additive* to the HTTP response, never a replacement.
- **`stream=true` returns Server-Sent Events, not chunked JSON.** Use `curl -N` (no buffering). Each event is `data: {"content": "...", "start_ts": ..., "end_ts": ...}\n\n`, terminated by `data: {"status":"completed"}\n\n`. Without `stream=true` the server buffers until the full video is processed — fine for short clips (<1 min), avoid for live streams.
- **`chunk_duration=0` disables chunking** — the entire video is sent to the VLM as one shot. Only meaningful for short clips; long videos will OOM or exceed `max_model_len`.
- **Default frame budget caps at `VLLM_MM_PROCESSOR_VIDEO_NUM_FRAMES` (256).** Requesting FPS that implies >256 frames per chunk is silently capped; drop FPS or shorten `chunk_duration` to stay within budget.
- **`enable_reasoning` requires a Cosmos Reason model.** Passing it with Qwen3-VL or other non-reasoning models is a no-op.
- **`/v1/metrics` requires auth**, unlike `/v1/health/*`. Prometheus scrapers need the Bearer token.
- **File upload is multipart, not JSON.** Use `-F file=@path -F purpose=vision -F media_type=video`; a `-d` body returns 422.
- **Live-stream lifecycle requires two deletes to fully tear down:** `DELETE /v1/generate_captions_alerts/{stream_id}` stops inference; `DELETE /v1/streams/delete/{stream_id}` un-registers the stream. Skipping the second leaks RTSP connection resources.
