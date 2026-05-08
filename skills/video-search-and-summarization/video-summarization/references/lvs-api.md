# LVS API Reference

This reference documents the LVS 3.1.0 GA OpenAPI surface. Do not infer API fields or behaviors
from newer branches, deployment runbooks, or implementation code unless the user explicitly asks
to go beyond this OpenAPI spec.

LVS provides video summarization and insight extraction endpoints. It accepts a summarization
query, returns OpenAI-style completion objects, lists configured models, exposes health probes,
returns Prometheus metrics, and recommends chunking parameters.

## Setup

The OpenAPI spec declares a relative server URL (`/`), so `BASE_URL` is deployment-specific.
Confirm the deployed LVS host and port from the compose/Helm output, service runbook, or the
operator before calling the API. Common deployments expose LVS on a host port such as `38111`,
but some dev containers use `8000`.

If an Agent's sandbox cannot reach `localhost:<port>`, do not assume LVS is down. The sandbox
may have a different network view than the host. Confirm the port from the host/deployment
context and retry from a host-visible shell or with the externally reachable host:port.

```bash
export BASE_URL="http://localhost:38111"
export API_KEY="your-bearer-token"
```

The spec declares bearer auth globally. Use this header on calls:

```bash
-H "Authorization: Bearer $API_KEY"
```

## Quick Start

List models, then summarize a video URL:

```bash
MODEL=$(curl -s "$BASE_URL/models" \
  -H "Authorization: Bearer $API_KEY" | jq -r '.data[0].id')

curl -s -X POST "$BASE_URL/v1/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"scenario\": \"warehouse\",
    \"events\": [\"safety violation\", \"unauthorized access\"],
    \"url\": \"https://www.example.com/video.mp4\",
    \"prompt\": \"Write a concise summary with timestamps.\"
  }" | jq '.choices[0].message.content'
```

## Endpoints

### Health Check

#### `GET /v1/live` - Liveness

Get LVS liveness status.

```bash
curl -s "$BASE_URL/v1/live" \
  -H "Authorization: Bearer $API_KEY"
```

#### `GET /v1/ready` - Readiness

Get LVS readiness status.

```bash
curl -s "$BASE_URL/v1/ready" \
  -H "Authorization: Bearer $API_KEY"
```

#### `GET /v1/startup` - Startup

Get LVS startup status.

```bash
curl -s "$BASE_URL/v1/startup" \
  -H "Authorization: Bearer $API_KEY"
```

#### `GET /v1/metadata` - Metadata

Get LVS service metadata information.

```bash
curl -s "$BASE_URL/v1/metadata" \
  -H "Authorization: Bearer $API_KEY"
```

### Models

#### `GET /models` - List models

Lists currently available models and basic model information.

```bash
curl -s "$BASE_URL/models" \
  -H "Authorization: Bearer $API_KEY" \
  | jq '.data[] | {id, created, object, owned_by, api_type}'
```

**Response (200) top-level fields:** `object`, `data`.

Each model has: `id`, `created`, `object`, `owned_by`, `api_type`.

### Summarization

The spec exposes both `POST /v1/summarize` and `POST /summarize`; both use the same
`SummarizationQuery` request schema and `CompletionResponse` response schema.

#### `POST /v1/summarize` - Summarize a video

Required request fields:

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model to use for this query, for example `cosmos-reason1`. |
| `scenario` | string | Scenario or use-case context, for example `warehouse`, `retail`, or `security`. |
| `events` | array[string] | Events to detect or extract from the video. Use `[]` if there are no target events. |

Source-related optional fields:

| Field | Type | Description |
|-------|------|-------------|
| `url` | string or null | Video URL. Examples include `https://www.example.com/video.mp4` and `s3://bucket/video.mp4`. |
| `id` | UUID string, array[UUID], or null | Unique ID or list of IDs of files or live streams to summarize. |
| `media_info` | object | Segment selection. Use `{"type":"offset","start_offset":0,"end_offset":60}` for files or `{"type":"timestamp","start_timestamp":"2024-05-30T01:41:25.000Z","end_timestamp":"2024-05-30T02:14:51.000Z"}` for live streams. |

Common optional fields from the OpenAPI schema:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | string | `""` | System prompt for the VLM. The spec notes Cosmos Reason1 reasoning can be enabled by adding `<think></think>` and `<answer></answer>` tags. |
| `prompt` | string | `""` | Prompt for summary generation. |
| `max_tokens` | integer | - | Maximum tokens to generate in a call. |
| `temperature` | number | - | Sampling temperature, range 0 to 1. |
| `top_p` | number | - | Top-p sampling mass, range 0 to 1. |
| `top_k` | number | - | Number of highest probability vocabulary tokens to keep, range 1 to 1000. |
| `seed` | integer | - | Seed value. |
| `chunk_duration` | integer | `0` | Chunk videos into this many seconds. `0` means no chunking. |
| `chunk_overlap_duration` | integer | `0` | Overlap between chunks in seconds. `0` means no overlap. |
| `summary_duration` | integer | `0` | Summarize every N seconds of video. Spec says applicable to live streams only. |
| `num_frames_per_chunk` | integer | `0` | Number of frames per chunk to use for the VLM. |
| `vlm_input_width` | integer | `0` | VLM input width. |
| `vlm_input_height` | integer | `0` | VLM input height. |
| `custom_metadata` | object | - | Key-value metadata. Spec says supported only with user-managed Milvus DB collections. |
| `delete_external_collection` | boolean | `false` | Delete the external collection at the end of the summarization request. |
| `schema` | string | - | JSON schema string for structured output extraction. |
| `batch_response_method` | string | - | Batch response method. Examples: `json_schema`, `text`. |
| `auto_generate_prompt` | boolean | - | Generate a prompt from schema and events. |
| `override_vlm_prompt` | boolean | `false` | Override the VLM prompt with the supplied prompt. |
| `enable_vlm_structured_output` | boolean | `true` | Enable VLM structured output. |
| `objects_of_interest` | array[string] | `[]` | Objects to detect or extract from the video. |
| `min_tokens` | integer or null | - | Minimum tokens to generate. Used with `ignore_eos` for benchmarking. |
| `ignore_eos` | boolean or null | - | Ignore EOS and continue until `max_tokens`; useful for fixed-output benchmarking. |

Basic URL request:

```bash
curl -s -X POST "$BASE_URL/v1/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosmos-reason1",
    "scenario": "security",
    "events": ["intrusion", "loitering"],
    "url": "https://www.example.com/video.mp4",
    "chunk_duration": 60,
    "chunk_overlap_duration": 10,
    "prompt": "Summarize the video and call out any target events."
  }'
```

Structured extraction request:

```bash
SCHEMA='{"type":"object","properties":{"events":{"type":"array","items":{"type":"object","properties":{"timestamp":{"type":"string"},"event_type":{"type":"string"},"description":{"type":"string"}}}}}}'

curl -s -X POST "$BASE_URL/v1/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg model "cosmos-reason1" \
    --arg scenario "warehouse" \
    --argjson events '["safety violation","forklift near person"]' \
    --arg url "https://www.example.com/warehouse.mp4" \
    --arg schema "$SCHEMA" \
    '{
      model: $model,
      scenario: $scenario,
      events: $events,
      url: $url,
      schema: $schema,
      auto_generate_prompt: true,
      enable_vlm_structured_output: true
    }')"
```

**Response (200) top-level fields:** `id`, `video_id`, `choices`, `created`, `model`,
`media_info`, `object`, `usage`.

`choices[].message` has `content`, `tool_calls`, and `role`. Tool calls use type `alert`
and include alert fields such as `name`, `detectedEvents`, `details`, plus `offset` for
files or `ntpTimestamp` for live streams.

#### `POST /summarize` - Summarize a video

Legacy or unversioned route with the same schema as `/v1/summarize`.

```bash
curl -s -X POST "$BASE_URL/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosmos-reason1",
    "scenario": "retail",
    "events": ["theft"],
    "url": "https://www.example.com/store.mp4"
  }'
```

### Recommended Config

#### `POST /recommended_config` - Recommend chunking parameters

The `RecommendedConfig` schema has no required fields, but it defines these optional fields:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `video_length` | integer | 1 to 864000000 | Video length in seconds. |
| `target_response_time` | integer | 1 to 86400 | Target LVS response time in seconds. |
| `usecase_event_duration` | integer | 1 to 86400 | Duration of the target event, for example how long a box-falling event takes. |

```bash
curl -s -X POST "$BASE_URL/recommended_config" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "video_length": 300,
    "target_response_time": 60,
    "usecase_event_duration": 5
  }'
```

**Response (200) top-level fields:** `chunk_size`, `text`.

### Metrics

#### `GET /metrics` - Prometheus metrics

Get LVS metrics in Prometheus format.

```bash
curl -s "$BASE_URL/metrics" \
  -H "Authorization: Bearer $API_KEY"
```

## Common Workflows

### Summarize A Video With The First Available Model

```bash
until curl -sf "$BASE_URL/v1/ready" -H "Authorization: Bearer $API_KEY" >/dev/null; do
  sleep 5
done

MODEL=$(curl -s "$BASE_URL/models" \
  -H "Authorization: Bearer $API_KEY" | jq -r '.data[0].id')

curl -s -X POST "$BASE_URL/v1/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"scenario\": \"security\",
    \"events\": [\"intrusion\", \"fighting\"],
    \"url\": \"https://www.example.com/security.mp4\",
    \"chunk_duration\": 60
  }" | jq '{video_id, model, content: .choices[0].message.content}'
```

### Ask For A Recommended Chunk Size Then Summarize

```bash
CHUNK=$(curl -s -X POST "$BASE_URL/recommended_config" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"video_length": 600, "target_response_time": 120, "usecase_event_duration": 5}' \
  | jq -r '.chunk_size')

curl -s -X POST "$BASE_URL/v1/summarize" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"cosmos-reason1\",
    \"scenario\": \"warehouse\",
    \"events\": [\"safety violation\"],
    \"url\": \"https://www.example.com/warehouse.mp4\",
    \"chunk_duration\": $CHUNK
  }"
```

## Error Reference

| Code | Spec description | Common API cause |
|------|------------------|------------------|
| 400 | Bad Request | Invalid syntax or bad request body. |
| 401 | Unauthorized request | Missing or invalid bearer token. |
| 422 | Failed to process request | Schema validation failure, including extra fields where schemas set `additionalProperties: false`. |
| 429 | Rate limiting exceeded | Too many requests for the service limits. |
| 500 | Internal Server Error | Server-side LVS processing failure. |
| 503 | Server is busy processing another file / live-stream | Summarize endpoint is busy; try again later. |

## Gotchas

- **Only OpenAPI fields are valid here** - do not add fields absent from `SummarizationQuery`.
- **`model`, `scenario`, and `events` are required** - the schema requires all three for both
  `/v1/summarize` and `/summarize`.
- **Most request/response schemas set `additionalProperties: false`** - extra fields are schema
  violations and can produce 422 responses.
- **`schema` is a string** - pass a JSON schema serialized as a string, not a nested JSON object.
- **`enable_vlm_structured_output` defaults to `true`** - set it explicitly only when you want to
  override the default.
- **`chunk_duration: 0` means no chunking** and `chunk_overlap_duration: 0` means no overlap.
- **`summary_duration` is for live streams** according to the field description.
- **`media_info` has two shapes** - use `type: "offset"` with second offsets for files and
  `type: "timestamp"` with ISO timestamp strings for live streams.
