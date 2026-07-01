# § 13 — Verification & curl Examples

> Use these commands to verify the service after build and deploy (§ 9).

---

## Build & Test

```bash
# 1. Build Docker image
docker compose -f deploy/compose.yaml build

# 2. Start service (requires GPU + DDM model + embedded vLLM)
docker compose -f deploy/compose.yaml up -d

# 3. Run full API test suite (24 tests)
TEST_VIDEO_PATH=test_video_whole_sop_h264.mp4 pytest tests/test_api_endpoints.py -v
```

---

## curl Examples

```bash
BASE_URL="http://localhost:8300"

# --- Health checks ---
# NOTE: model loading takes several minutes on first run.
# Poll every 10s until HTTP 200 — see § 9 (skill_09) for the full readiness wait script.
curl $BASE_URL/v1/live
curl $BASE_URL/v1/ready
curl $BASE_URL/v1/startup

# --- List models ---
curl $BASE_URL/v1/models

# --- Upload a video file ---
curl -X POST $BASE_URL/v1/files \
  -F "file=@test_video.mp4" \
  -F "purpose=inference"

# --- List files ---
curl $BASE_URL/v1/files

# --- Chat completion (non-streaming, base64 video) ---
# Encodes video as base64 data URI → sent to Stage 1 pipeline.
# NOTE: Do NOT include a {"type":"text"} item in the message content.
#   The VLM prompt is controlled by VLM_PROMPT_PATH (e.g. configs/assy17_vlm_prompts.txt).
#   If a text item is present, it overrides the config-file prompt and breaks SOP classification.
VIDEO_B64=$(base64 -w0 test_video.mp4)
curl -X POST $BASE_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ds_sop_model",
    "messages": [{"role": "user", "content": [
      {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,'$VIDEO_B64'"}}
    ]}],
    "stream": false,
    "chunking_options": {"algorithm": "ddm-net", "threshold": 0.8}
  }'

# --- Streaming response ---
# Returns SSE chat.completion.chunk events as each chunk is processed.
# NOTE: Omit {"type":"text"} — VLM prompt comes from VLM_PROMPT_PATH config file.
curl -N -X POST $BASE_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ds_sop_model","messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"data:video/mp4;base64,'$VIDEO_B64'"}}]}],"stream":true}'

# --- Uniform (fixed-length) chunking ---
# algorithm:"uniform" cuts fixed chunk_length_sec chunks and BYPASSES the DDM model (§ 3, § 6).
# extra="forbid": do NOT mix DDM fields (threshold / min_length_sec / max_length_sec) here → 422.
curl -N -X POST $BASE_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ds_sop_model","messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"data:video/mp4;base64,'$VIDEO_B64'"}}]}],"stream":true,"chunking_options":{"algorithm":"uniform","chunk_length_sec":2.0}}'

# --- Physical Basler camera (serial 40748152, PFS optional) ---
# NOTE: Use max_length_sec <= 5.0 for responsive live streaming (default 60.0 is too slow)
curl -N -X POST $BASE_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ds_sop_model","messages":[{"role":"user","content":[
    {"type":"input_camera","input_camera":{
      "camera_id":"40748152","camera_vendor":"Basler",
      "camera_format":"RGB","camera_width":1280,"camera_height":720,
      "camera_fps_num":30,"camera_fps_den":1
    }}
  ]}],"stream":true,"chunking_options":{"algorithm":"ddm-net","threshold":0.8,"max_length_sec":2.0}}'

# --- Camera emulation (serial 0815-0000, PYLON_CAMEMU=1 in service env required) ---
curl -N -X POST $BASE_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ds_sop_model","messages":[{"role":"user","content":[
    {"type":"input_camera","input_camera":{
      "camera_id":"0815-0000","camera_vendor":"Basler",
      "config":"/opt/nvidia/nvds_sop/configs/Emulation_0815-0000.pfs",
      "camera_format":"RGB","camera_width":1280,"camera_height":720,
      "camera_fps_num":30,"camera_fps_den":1
    }}
  ]}],"stream":true,"chunking_options":{"algorithm":"ddm-net","threshold":0.8,"max_length_sec":2.0}}'

# --- Prometheus metrics ---
curl $BASE_URL/v1/metrics
```
