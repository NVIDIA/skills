# Video Summarization API Reference

This reference explains the video summarization API workflows used by
`vss-summarize-video`. The tables and examples below are illustrative guidance
only for live calls. The running service's `/openapi.json` is authoritative
because its image version may expose a newer or different schema; fetch it under
the recorded `/lvs` mount (see Runtime OpenAPI Discovery).

Use `/v1/summarize` for new file-summarization examples. `/summarize` is still
present with the same request and response schema as a compatibility route.

## Setup

The OpenAPI spec declares a relative server URL (`/`), so `BASE_URL` is the
`/lvs` mount `vss configure` recorded, on Compose and Kubernetes alike. Do not
rebuild it from `VSS_PUBLIC_URL`, `HOST_IP`, or a port:

```bash
DEPLOYMENT=$(vss configure show) || exit $?
BASE_URL=$(printf '%s' "$DEPLOYMENT" | jq -er '.services.lvs.url')   # no /v1 suffix
```

## Runtime OpenAPI Discovery

Before constructing or issuing any live API operation, fetch the
schema from the same service instance that will receive the request. The
bootstrap `/openapi.json` fetch and health probes are the only exceptions.

```bash
LVS_OPENAPI=/tmp/vss-lvs-openapi.json
curl -fsS --connect-timeout 3 --max-time 15 \
  "$BASE_URL/openapi.json" > "$LVS_OPENAPI"
jq -e '.openapi and (.paths | type == "object")' "$LVS_OPENAPI" >/dev/null
```

LVS is published as a Prefix mount at `/lvs`, so `${BASE_URL}/openapi.json`
and `${BASE_URL}/models` are reachable alongside `${BASE_URL}/v1/ready` and
`${BASE_URL}/v1/summarize`. The origin's own `/openapi.json` — no prefix — is
the **Agent** document on Kubernetes; do not treat it as the LVS schema.
Resolve `model` from `${BASE_URL}/models` (LVS), from `/v1/models` under
`services.rt_vlm.url` (RT-VLM), or from `VLM_NAME`.

Use the runtime document to confirm the operation exists and inspect
its request body before building a payload. For example:

```bash
OPERATION_PATH=/v1/summarize
OPERATION_METHOD=post

jq -e --arg path "$OPERATION_PATH" --arg method "$OPERATION_METHOD" \
  '.paths[$path][$method]' "$LVS_OPENAPI"

REQUEST_REF=$(jq -r --arg path "$OPERATION_PATH" --arg method "$OPERATION_METHOD" '
  .paths[$path][$method].requestBody.content["application/json"].schema["$ref"] // empty
' "$LVS_OPENAPI")

if [[ "$REQUEST_REF" == '#/components/schemas/'* ]]; then
  REQUEST_SCHEMA_NAME=${REQUEST_REF##*/}
  jq -e --arg name "$REQUEST_SCHEMA_NAME" \
    '.components.schemas[$name]' "$LVS_OPENAPI"
else
  jq -e --arg path "$OPERATION_PATH" --arg method "$OPERATION_METHOD" '
    .paths[$path][$method].requestBody.content["application/json"].schema
  ' "$LVS_OPENAPI"
fi
```

Follow the runtime schema's required fields, types, enums, and
`additionalProperties` policy. Do not infer live request fields solely from
the tables or examples below. If `/openapi.json` is unavailable or the desired
operation is absent, stop before a mutating or inference request and report
that the deployed API contract could not be verified. The checked-in snapshot
may still be used to explain expected behavior, but not to guess a live
payload.

The OpenAPI declares bearer auth globally, but local VSS developer deployments
usually expose these endpoints without an auth header. If the deployment
requires auth, add:

```bash
-H "Authorization: Bearer $API_KEY"
```

to each `curl` call.

## Endpoint Examples

Confirm these paths against the runtime OpenAPI document before calling them.

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/ready` | GET | Readiness probe. HTTP 200 means ready; HTTP 503 means warming or dependency unavailable. |
| `/v1/live` | GET | Liveness probe. |
| `/v1/startup` | GET | Startup probe. |
| `/v1/healthz` | GET | VIA service health status. |
| `/v1/metadata` | GET | Service metadata. |
| `/models` | GET | List models available to the video summarization service. |
| `/recommended_config` | POST | Recommend chunking parameters. |
| `/metrics` | GET | Prometheus metrics. |
| `/v1/summarize` | POST | Summarize a video file. Canonical 3.2 route. |
| `/summarize` | POST | Compatibility route with the same schema as `/v1/summarize`. |
| `/v1/generate_captions` | POST | Start RTVI stream captioning for a stream id. |
| `/v1/stream_summarize` | POST | Summarize an already-captioned stream from database captions. |

## Health And Metadata

Readiness checks should use the HTTP status only. Do not parse the body; it can
be empty on success.

```bash
curl -sf --max-time 15 "$BASE_URL/v1/ready" >/dev/null
curl -sf --max-time 15 "$BASE_URL/v1/live" >/dev/null
curl -sf --max-time 15 "$BASE_URL/v1/startup" >/dev/null
curl -sf --max-time 15 "$BASE_URL/v1/healthz" >/dev/null
curl -sf --max-time 15 "$BASE_URL/v1/metadata" | jq .
```

## Models

Always discover the model id from the endpoint that will receive the request.
Use `${VLM_NAME}` only when it matches an advertised id; otherwise use the sole
advertised id. Do not guess when multiple unmatched ids are returned.

```bash
curl -sf "$BASE_URL/models" | jq '.data[] | {id, object, owned_by, api_type}'
```

## File Summarization

`POST /v1/summarize` and `POST /summarize` both use `SummarizationQuery`.
The OpenAPI schema requires `model`, `scenario`, and `events` on every request;
omitting `scenario` (or any other required key) returns HTTP 422.

Required fields:

| Field | Type | Notes |
|---|---|---|
| `model` | string | Required. Must match an available model id. |
| `scenario` | string | Required. User-provided use-case context. |
| `events` | array[string] | Required. User-provided event names to detect or summarize. |

Source fields:

| Field | Type | Notes |
|---|---|---|
| `url` | string or null | HTTP(S) or S3 video URL. |
| `id` | UUID, array[UUID], or null | File or live stream ids known to the video summarization service. |
| `media_info` | object | Offset or timestamp segment selector. |

Common optional fields:

| Field | Notes |
|---|---|
| `prompt`, `system_prompt` | Prompt overrides. |
| `chunk_duration`, `chunk_overlap_duration`, `summary_duration` | Chunking and live-stream summary cadence. |
| `num_frames_per_second_or_fixed_frames_chunk`, `use_fps_for_chunking` | Optional 3.2 frame sampling overrides. Omit in the standard `/v1/summarize` workflow so RT-VLM uses its model-specific deployment default. |
| `num_frames_per_chunk` | Deprecated compatibility field; avoid in new examples. |
| `enable_audio`, `enable_reasoning` | Optional audio and reasoning controls. |
| `vlm_input_width`, `vlm_input_height` | VLM input dimensions. |
| `schema`, `batch_response_method`, `auto_generate_prompt`, `override_vlm_prompt`, `enable_vlm_structured_output` | Structured output controls. |
| `objects_of_interest`, `alert_category`, `creation_time`, `mm_processor_kwargs` | Extraction and model-processing context. |
| `temperature`, `top_p`, `top_k`, `max_tokens`, `min_tokens`, `ignore_eos`, `seed` | Generation controls. |

Most request schemas set `additionalProperties: false`; do not invent fields
that are absent from the OpenAPI schema.

Basic request:

```bash
LVS_REQUEST=/tmp/vss-summarize-video-request.json
LVS_RESPONSE=/tmp/vss-summarize-video-response.json
LVS_MODEL=$(curl -fsS "$BASE_URL/models" | jq -er --arg preferred "${VLM_NAME:-}" '
  [.data[]?.id | select(type == "string" and length > 0)] | unique as $ids
  | if $preferred != "" and ($ids | index($preferred)) != null then $preferred
    elif ($ids | length) == 1 then $ids[0]
    else empty end
') || { echo "Set VLM_NAME to an advertised model id"; return 1 2>/dev/null || exit 1; }

jq -n \
  --arg model "$LVS_MODEL" \
  --arg url "https://www.example.com/video.mp4" \
  --arg scenario "warehouse monitoring" \
  --argjson events '["boxes falling","forklift stuck"]' \
  '{
      model: $model,
      url: $url,
      scenario: $scenario,
      events: $events,
      chunk_duration: 10,
      seed: 1
  }' > "$LVS_REQUEST"

HTTP_CODE=$(curl -sS -o "$LVS_RESPONSE" -w '%{http_code}' \
  -X POST "$BASE_URL/v1/summarize" \
  -H "Content-Type: application/json" \
  --data-binary "@$LVS_REQUEST")
```

Response shape: `CompletionResponse` with top-level fields such as `id`,
`video_id`, `choices`, `created`, `model`, `media_info`, `object`, and `usage`.
For the VSS summarization workflow, the actual summary payload is a JSON string
inside `choices[0].message.content`.

```bash
jq '{
  usage: (.usage // {}),
  result: (.choices[0].message.content | fromjson | {video_summary, events})
}' "$LVS_RESPONSE"
```

Preserve `usage.total_chunks_processed` when presenting an empty summary and
event list. A positive value proves media was processed; zero or missing usage
does not.

The POST above is the only summarize request. Use the saved HTTP status and
`$LVS_RESPONSE` for all error handling and diagnostics; never repeat the POST
just to inspect the unfiltered response.

## Stream Captioning And Stream Summarization

For streams, the OpenAPI directs callers to start captioning first, then
summarize the stored captions.

Start captioning:

```bash
STREAM_MODEL=$(curl -fsS "$BASE_URL/models" | jq -er --arg preferred "${VLM_NAME:-}" '
  [.data[]?.id | select(type == "string" and length > 0)] | unique as $ids
  | if $preferred != "" and ($ids | index($preferred)) != null then $preferred
    elif ($ids | length) == 1 then $ids[0]
    else empty end
') || { echo "Set VLM_NAME to an advertised model id"; return 1 2>/dev/null || exit 1; }

curl -s -X POST "$BASE_URL/v1/generate_captions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg id "<stream_uuid>" \
    --arg model "$STREAM_MODEL" \
    --arg scenario "traffic monitoring" \
    --argjson events '["accident","pedestrian crossing"]' \
    '{
      id: $id,
      model: $model,
      scenario: $scenario,
      events: $events,
      chunk_duration: 10,
      num_frames_per_second_or_fixed_frames_chunk: 20,
      use_fps_for_chunking: false
    }')"
```

The response has `id`, `status`, and `model`.

Summarize existing stream captions:

```bash
curl -s -X POST "$BASE_URL/v1/stream_summarize" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg id "<stream_uuid>" \
    --arg model "$STREAM_MODEL" \
    '{
      id: $id,
      model: $model,
      start_time: 0,
      end_time: 0,
      enable_vlm_structured_output: true
    }')"
```

`/v1/stream_summarize` uses `StreamSummarizeRequest`; `id` and `model` are
required.

## Recommended Config

```bash
curl -s -X POST "$BASE_URL/recommended_config" \
  -H "Content-Type: application/json" \
  -d '{
    "video_length": 300,
    "target_response_time": 60,
    "usecase_event_duration": 5
  }' | jq .
```

The response includes `text` and may include `chunk_size`.

## Metrics

```bash
curl -sf "$BASE_URL/metrics" | head
```

## Errors And Gotchas

- `400` means invalid syntax or malformed request.
- `401` means auth was required but missing or invalid.
- `422` usually means a schema validation failure. Check for missing required
  keys (`model`, `scenario`, `events` on `/v1/summarize`) or extra fields.
- `429` means rate limiting.
- `503` from readiness means warming or dependencies unavailable.
- `503` from summarize means the service is busy processing another file.
- Treat the runtime OpenAPI as authoritative for GA fields. Some internal sanity
  scripts exercise non-spec streaming flags on `/v1/summarize`; do not teach
  those as public GA fields unless the OpenAPI is updated.
