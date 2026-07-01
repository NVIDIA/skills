# § 10 — Test Suite Coverage

> **Reference file to copy**: `test_api_endpoints_reference.py` → `tests/test_api_endpoints.py`
> **Related sections**: § 1 (endpoints), § 7 (SSE streaming), § 13 (curl examples)

---

## Running Tests

```bash
# Start service first (requires GPU + DDM model + vLLM)
TEST_VIDEO_PATH=/path/to/video.mp4 python tests/test_api_endpoints.py
# OR with pytest
TEST_VIDEO_PATH=/path/to/video.mp4 pytest tests/test_api_endpoints.py -v
```

---

## Test Classes and Coverage

| Test Class | Endpoints Tested |
|-----------|-----------------|
| `TestHealthEndpoints` | `/v1/live`, `/v1/startup`, `/v1/ready` |
| `TestModelEndpoints` | `/v1/models` |
| `TestMetadataEndpoint` | `/v1/metadata` |
| `TestMetricsEndpoint` | `/v1/metrics` (format, content, increments) |
| `TestFileEndpoints` | Full workflow: upload → list → download → chat → delete |
| `TestChatCompletionEndpoint` | Non-streaming, streaming, invalid content-type |
| `TestUniformChunkingEndpoint` | `algorithm:"uniform"` non-streaming + streaming; `chunk_length_sec<=0` → 422; extra DDM fields → 422; ddm-net regression |
| `TestEdgeCases` | 404 on invalid paths, 405 on wrong HTTP method |

---

## Critical Assertions

```python
# --- Health endpoints ---
# /v1/live
assert data["object"] == "health.response"
assert data["message"] == "Service is live."

# /v1/startup
assert "started successfully" in data["message"].lower()

# /v1/ready
assert "ready" in data["message"].lower() or "dummy" in data["message"].lower()

# --- Model endpoint ---
# /v1/models
assert data["object"] == "list"
assert data["data"][0]["object"] == "model"

# --- Metadata endpoint ---
# /v1/metadata
assert "version" in data and "modelInfo" in data and "licenseInfo" in data
assert all(k in data["licenseInfo"] for k in ["name", "path", "size", "content"])

# --- File management ---
# /v1/files (POST)
assert upload_data["object"] == "file"
assert upload_data["id"].startswith("file-")

# --- Chat completions (non-streaming) ---
# /v1/chat/completions
assert data["object"] == "chat.completion"
assert data["id"].startswith("chatcmpl-")
assert data["choices"][0]["message"]["role"] == "assistant"

# --- Chat completions (streaming) ---
# /v1/chat/completions (stream=true)
assert "text/event-stream" in response.headers["content-type"]
assert first_chunk["object"] == "chat.completion.chunk"

# --- Uniform chunking (algorithm="uniform") ---
# {"algorithm":"uniform","chunk_length_sec":2.5} → accepted, returns chunks
assert data["object"] == "chat.completion"
# chunk_length_sec <= 0 → 422 ; extra DDM fields (e.g. threshold) with uniform → 422 (extra="forbid")
assert response.status_code == 422

# --- Error cases ---
# Content-type validation
# POST with Content-Type: text/plain → 415
# POST /v1/live → 405
# GET /v1/invalid_endpoint → 404

# --- Metrics ---
assert "api_requests_total" in metrics_text
assert "api_request_latency_seconds" in metrics_text
```
