# SOP Inference Microservice Evaluation Workflow

Use this reference when evaluating a generated DeepStream SOP microservice from
scratch. It defines the phase order, required environment checks, build/test
commands, optional live camera/Kafka checks, and final report checklist.

Load the supporting references as needed:

- `skill_09_docker_build_deploy.md` for Docker build, compose, and `.env`
- `skill_10_test_suite.md` for unittest coverage and expectations
- `skill_13_verification_curl.md` for file/video API checks
- `skill_15_latency_measurement.md` for TTFC and C2C file-input latency
- `skill_17_camera_latency_measurement.md` for Basler/live-stream latency
- `skill_16_message_schema.md` for Kafka message schema checks (JSON vs `NvProtoSchema`; the
  Kibana SOP dashboard requires the flat `JSON` schema — see Phase 6.3)
- `skill_18_rtsp_streaming_output.md` for the **optional** RTSP streaming-output checks (Phase 5B; only if the feature was generated)

## What to Evaluate

Evaluate the generated microservice project `@ds_sop_microservice` from scratch.
Throughout this workflow, **`@ds_sop_microservice`** refers to the target project directory
provided by the user (e.g. `ds_sop_auto_ms_1`, `ds_sop_v2`, etc.).

- Build the SOP docker image from `@ds_sop_microservice/deploy/compose.yaml`.
  The pylon-25.10.2 binary is located at `@ds_sop_microservice/binaries/pylon-25.10.2_linux-x86_64_setup.tar.gz`.
- Launch the microservice with `@ds_sop_microservice/deploy/compose.yaml`
- Run unittest with `@ds_sop_microservice/tests/test_api_endpoints.py`
- API test with a local video file provided by the user

Follow `@ds_sop_microservice/deploy/.env` settings. All prerequisites, critical rules,
and verification steps are in `deepstream-sop/SKILL.md`.

**IMPORTANT — Model paths must be configured before GPU tests.**
The user MUST provide local paths for the two GPU models via environment variables
(in `deploy/.env` or exported in shell). There are no usable defaults — the default
paths point to locations that may not exist on the host:

| Variable | Default (may not exist) | What it is |
|----------|------------------------|------------|
| `DDM_MODEL_PATH` | `/models/gbed_models/ddm/checkpoint.pth.tar` | GEBD model checkpoint (default architecture: DDM) |
| `VLLM_MODEL_PATH` | _(none — required)_ | Cosmos Reason 1 or 2 (HuggingFace repo id or local path under `MODEL_ROOT_DIR`). No default; fails fast if empty |
| `MODEL_ROOT_DIR` | `/models` | Host directory mounted into container for DDM model access |
| `TEST_VIDEO_PATH` | *(none)* | Path to a test video file (H.264 MP4) for unit tests and API integration tests |

Before Phase 1, ensure the user has provided model paths and a test video path,
then create or update `deploy/.env`.

---

## Phase 0: Pre-Build Validation

Before building, verify the generated project structure is complete and correct.

### 0.1 File Completeness Check

Verify every file listed in `@ds_sop_microservice/docker/package_file_list.txt` exists
under `@ds_sop_microservice/`. Report any missing files.

### 0.2 Critical Rules Static Check

Inspect the generated source files against `deepstream-sop/SKILL.md` Critical Rules:

1. **Rule 1 — SOPProcessManager init in `main()`**: Open `nvds_action_detector/api_server.py`
   and confirm `SOPProcessManager()` and `wait_for_model_ready()` are called inside `main()`
   **before** `uvicorn.run()`, and NOT inside the async `lifespan()` context manager.

2. **Rule 2 — `create_video_processor()` named kwargs**: In the chat completion handler inside
   `api_server.py`, confirm the call uses named kwargs:
   `create_video_processor(file_path=..., chunk_params=..., camera_serial_number=..., camera_config=..., ...)`
   and NOT positional args or a packed dict.

3. **Rule 3 — stream required for live inputs**: Confirm RTSP and camera inputs return HTTP 400
   when `stream: false`.

4. **Rule 6 — VLM warmup with 3 frames**: Confirm warmup calls
   `inference("Say Hi", [zeros, zeros, zeros])` with exactly 3 zero frames.

5. **Rule 7 — 4 thread pools**: Confirm `cv_process_pool(32)`, `clip_process_pool(32)`,
   `vlm_inference_pool(64)`, `vlm_request_pool(64)` exist with correct sizes.

6. **Rule 8 — MediaInfo (`MEDIA_INFO_PYMEDIAINFO`)**: Confirm `pymediainfo.MediaInfo.parse()`
   is used for file sources (not `pyservicemaker.utils.MediaInfo`, ffprobe, or other
   alternatives). For live sources (`_is_live == True`), media info is skipped and
   `chunk_params.fps`/`duration_sec` are set directly. This matches the authoritative
   `MEDIA_INFO_PYMEDIAINFO` rule in `SKILL.md` / `skill_06_sop_process_manager.md`.

### 0.3 Reference File Integrity

Verify files that must be copied verbatim from `deepstream-sop/references/` were not
regenerated. Diff these against their reference counterparts:

- `nvds_action_detector/vllm_inference.py` ↔ `references/vllm_inference_reference.py`
- `nvds_action_detector/utils.py` ↔ `references/utils_reference.py`
- `nvds_action_detector/custom_postprocess/nvds_custom_postprocess_tensor.cpp` ↔ `references/nvds_custom_postprocess_tensor_reference.cpp`
- `nvds_action_detector/custom_postprocess/Makefile` ↔ `references/Makefile_custom_postprocess_reference`
- `nvds_action_detector/protos/__init__.py` ↔ `references/protos_init_reference.py`
- `docker/copy_sources.sh` ↔ `references/copy_sources_reference.sh`
- `docker/package_file_list.txt` ↔ `references/package_file_list_reference.txt`
- `docker/ddm_pytorch2.patch` ↔ `references/ddm_pytorch2.patch`

Report any diffs. These files contain machine-generated blobs, exact compiler flags,
or complex logic that cannot be reliably regenerated.

**Do NOT diff the generated protobuf bindings (`nvds_action_detector/protos/nv_pb2.py`,
`ext_pb2.py`) against the references.** The project ships only the `.proto` source of truth
(`nv.proto`, `ext.proto`); the `*_pb2.py` files do not exist in the source tree and are
generated at image-build time by `protoc` (see `docker/Docker.build`:
`protoc -I. --python_out=. nv.proto ext.proto`). A direct text diff against
`references/*_pb2_reference.py` produces spurious mismatches caused purely by protoc-version
formatting (single- vs double-quoted serialized blob, `AddSerializedFile` line-wrapping,
`_builder` import ordering) even though the serialized FileDescriptor is functionally
identical. To verify the bindings, instead confirm they import and serialize inside the
built container:
`docker exec <container> python3 -c "from nvds_action_detector.protos import nv_pb2, ext_pb2; nv_pb2.VisionLLM()"`.

---

## Phase 1: Docker Image Build

### 1.1 Environment Setup

Check if `@ds_sop_microservice/deploy/.env` exists. If not, create it with the user's
model paths. If the user has not supplied `DDM_MODEL_PATH` and `VLLM_MODEL_PATH`, ask
for them before proceeding; do not assume the defaults will work.

```bash
cat > <ds_sop_microservice>/deploy/.env << 'EOF'
NV_DS_SOP_IMAGE=nvds-sop:latest
API_SERVER_PORT=8300
API_DUMMY_TEST=false
NVIDIA_VISIBLE_DEVICES=0
INSTALL_PROPRIETARY_CODECS=true

# === GPU Model Paths (REQUIRED for GPU tests) ===
# DDM checkpoint: host path mounted via MODEL_ROOT_DIR
DDM_MODEL_PATH=/models/gbed_models/ddm/checkpoint.pth.tar
# VLM model (REQUIRED, no default): HuggingFace repo id or local path under MODEL_ROOT_DIR
VLLM_MODEL_PATH=/models/<your-finetuned-vlm-checkpoint>
# Host directory containing DDM model (mounted into container)
MODEL_ROOT_DIR=/models
# HuggingFace/vLLM cache on host
HOST_CACHE=$HOME/.cache/ds_sop
# Media storage for uploaded files
MEDIA_STORAGE_DIR=/tmp/nvds_sop_storage
EOF
```

If `.env` already exists, verify it contains valid `DDM_MODEL_PATH` and `VLLM_MODEL_PATH`
entries and that the paths are accessible on the host.

### 1.2 Pylon Binary Check

Verify the Pylon SDK binary exists at the expected location. The Dockerfile expects it
at `./binaries/pylon-25.10.2_linux-x86_64_setup.tar.gz` relative to the build context.

```bash
ls -lh <ds_sop_microservice>/binaries/pylon-25.10.2_linux-x86_64_setup.tar.gz
```

If missing, it must be placed there before build. The Dockerfile will attempt to download
it from Basler's CDN as fallback, but this requires network access during build.

### 1.3 Build the Docker Image

```bash
cd <ds_sop_microservice>
docker compose -f deploy/compose.yaml build nvds-action-sop
```

**Expected outcome**: Image builds successfully with tag `nvds-sop:latest`.

**Common build failures to watch for**:
- `copy_sources.sh` fails: a file in `package_file_list.txt` doesn't exist in the project
- `make` fails in `custom_postprocess/`: incorrect Makefile flags or missing DeepStream headers
- `git apply ddm_pytorch2.patch` fails: patch file is corrupted or different from reference
- `gst-plugin-pylon` build fails: DeepStream lib hide/restore trick missing (Rule 11)

---

## Phase 2: Launch the Microservice

### 2.1 Verify Model Paths Before Launch

Before starting, confirm the model files are accessible:

```bash
# Verify DDM checkpoint exists on host (path from .env MODEL_ROOT_DIR + DDM_MODEL_PATH)
ls -lh /models/gbed_models/ddm/checkpoint.pth.tar

# For VLLM_MODEL_PATH: if it's a HuggingFace ID (e.g. "nvidia/cosmos-reason1-7b"),
# vLLM will auto-download on first run. If it's a local path, verify it exists.
```

### 2.2 Start Services

```bash
cd <ds_sop_microservice>
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d
```

This starts both `nvds-action-sop` and `kafka` containers.

### 2.3 Verify Container Health

```bash
# Check containers are running
docker compose -f deploy/compose.yaml ps

# Watch logs for startup sequence
docker compose -f deploy/compose.yaml logs -f nvds-action-sop
```

**Expected startup sequence**:
1. VLM model loads first (Cosmos Reason 1 or 2 via vLLM — defaults to Reason 1 7B)
2. DDM dummy pipeline warmup runs (`SLIDING_WINDOWS_SIZE` = 18 frames)
3. `wait_for_model_ready()` completes
4. Uvicorn starts on port 8300
5. Log message: service is ready

**Startup failure indicators**:
- Hangs at VLM warmup for >300s → Rule 1 violated (blocking in async context)
- `TypeError: unhashable type: 'ChunkParams'` → Rule 2 violated
- GPU OOM → Check `VLLM_GPU_MEMORY_UTILIZATION` (default 0.3)
- `FileNotFoundError: checkpoint.pth.tar` → `DDM_MODEL_PATH` or `MODEL_ROOT_DIR` misconfigured
- vLLM download hangs → `VLLM_MODEL_PATH` HuggingFace ID incorrect or no network access

### 2.4 Quick Health Check

```bash
curl -s http://localhost:8300/v1/live | python3 -m json.tool
curl -s http://localhost:8300/v1/ready | python3 -m json.tool
curl -s http://localhost:8300/v1/startup | python3 -m json.tool
```

**Expected**: All return `{"object": "health.response", "message": "..."}` with HTTP 200.

### 2.5 Server-Log + GPU Watchdog (agent instructions)

Run for every test in Phases 3–6. Arm before each in-flight request, disarm after.

**Stall = (no new server log line in window) AND (GPU util < 5% in window).**
Either signal alone is unreliable: GPU busy with no logs = healthy long inference; logs scrolling but GPU idle = polling loop or warmup. Both quiet ⇒ stuck.

```bash
log_progressed() {           # 0 if last log line < $2 s old, else 1
  local c="$1" w="$2" t
  t=$(docker logs --tail 1 --timestamps "$c" 2>/dev/null | awk '{print $1}')
  [ -z "$t" ] && return 1
  t=$(date -d "$t" +%s 2>/dev/null) || return 1
  [ $(( $(date +%s) - t )) -lt "$w" ]
}
gpu_busy() {                 # 0 if any GPU sample > $1 % in last $2 s
  local thr="$1" w="$2" i u
  for i in $(seq 1 "$w"); do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${u:-0}" -gt "$thr" ] && return 0
    sleep 1
  done
  return 1
}
stuck() {                    # 0 if BOTH log-silent AND gpu-idle for $2 s
  local c="$1" w="$2"
  ! log_progressed "$c" "$w" && ! gpu_busy 5 3
}
```

**Window per test** (request in flight; pick the larger): `W = max(30, max_length_sec + 10)`.

| Test | `max_length_sec` | Window `W` |
|---|---:|---:|
| `test_api_endpoints.py` (no clip) | — | 30s |
| file-video non-stream | 60 | 70s |
| file-video SSE between chunks | 60 | 70s |
| camera SSE between chunks | 2.0 | 30s |
| Kafka consume while expecting | — | 30s |

**Inline arm pattern** (alongside curl):
```bash
curl -N -X POST ... &
CURL_PID=$!
W=30   # set per table above
while kill -0 "$CURL_PID" 2>/dev/null; do
  if stuck deploy-nvds-action-sop-1 "$W"; then
    echo "[$(date +%H:%M:%S)] STUCK ${W}s: log silent + GPU<5%" >&2
    kill "$CURL_PID" 2>/dev/null
    break
  fi
  sleep 5
done
```

**Hung signatures (in captured log, after a stall fires)**:
- `Add Element ... srcbin` + `LINKING:` then silence (no `Main Loop Running` → `End of stream` → `Main Loop Exited`).
- Multiple pipeline-build blocks back-to-back, no EOS between → Rule 16 (`ABORT_INFLIGHT_VLM`) violated.
- `/v1/metrics` 200 + no `chunk_idx` for > `max_length_sec + 10s`.
- `sop_step_checker: SOP steps:` then no next chunk for > 2 × `max_length_sec`.

**On stall: diagnose → fix → continue (max 2 retries per test)**:

1. Kill client: `kill $CURL_PID 2>/dev/null`.
2. Snapshot: `docker logs --tail 300 deploy-nvds-action-sop-1 > /tmp/stuck_$(date +%s).log; nvidia-smi > /tmp/stuck_$(date +%s).gpu`.
3. Match signature → root-cause class:
   - **Rule 16 violation** (multi-pipeline no-EOS) → patch `nvds_action_detector/ds_sop_process.py` `ABORT_INFLIGHT_VLM` path; rebuild image.
   - **VLM engine hang** (vLLM stacktrace, CUDA timeout, `vllm_inference` silence + GPU<5%) → lower `VLLM_GPU_MEMORY_UTILIZATION` or `VLM_MAX_PIXELS`; restart.
   - **VLM inference loop** (GPU=99% + log silent > 2× max_length_sec) → cap `vllm_inference` `max_tokens` in `ds_sop_process.py`; rebuild.
   - **Pipeline never starts** (`Add Element` block + no `Main Loop Running`) → fix link order in `ds_3d_action_pipeline.py`; rebuild.
   - **DDM/Triton hang** (`TrtISBackend`/`sequence_image_process` silence + GPU<5%) → check `triton_model_repo/ddm/1/model.py`; restart.
   - **DDM TRT engine build in progress** (`DDM_TRT_OPTIMIZATION=true`; log shows `Building DDM TRT engine on the fly`, GPU busy, no requests served) → NOT a hang; the first on-the-fly build takes ~5–15 min. Wait for `TRT engine ready` (then `Created TRT thread-local state` on first inference). To avoid per-restart rebuilds, persist `DDM_TRT_ENGINE_OUTPUT_PATH` under `MODEL_ROOT_DIR` (§ 5). A build failure logs a PyTorch-fallback message and the service still starts.
   - **Idle, no request in flight** → false positive; ignore.
   - **Unknown** → add debug logs (next step), restart, retry once.
4. **Add debug logs** when signature is "Unknown" or you want confirmation:
   - Use `get_logger(__name__)` (Rule LOGGER_EXPORT_GET_LOGGER) — DO NOT use `print`.
   - For pipeline issues: log entry/exit of each callback in `ds_sop_process.py` (`_on_chunk_end`, `_run_vlm_inference`, `_publish_kafka_message`).
   - For VLM issues: log start/end + token count around the `vllm_inference()` call.
   - For Kafka issues: log start/end around `produce()` / `flush()`.
   - Live-patch without full rebuild: `docker cp <local.py> deploy-nvds-action-sop-1:/opt/nvidia/nvds_sop/<path>.py && docker compose --env-file deploy/.env -f deploy/compose.yaml restart nvds-action-sop`.
   - After fix verified, fold the useful debug lines into source (don't leave noisy `DEBUG` prints).
5. Restart: `docker compose --env-file deploy/.env -f deploy/compose.yaml restart nvds-action-sop && until curl -fs http://localhost:8300/v1/ready >/dev/null; do sleep 2; done`.
6. Re-run the failing test once.
7. Same signature twice → hard FAIL that test; record `stuck_*.log` paths + root-cause class + fix + outcome in eval report; do NOT loop further.

---

## Phase 3: Run Unit Tests

### 3.1 API Dummy Mode Test (Optional Fast Path)

For a quick API-only test without GPU models, restart with `API_DUMMY_TEST=true`:

```bash
API_DUMMY_TEST=true docker compose -f deploy/compose.yaml up -d nvds-action-sop
```

Then run the test suite (health, models, metadata, metrics, file management endpoints):

```bash
cd <ds_sop_microservice>
python3 tests/test_api_endpoints.py
```

### 3.2 Full Test Suite with GPU Models

With the service running in normal mode (GPU models loaded):

```bash
cd <ds_sop_microservice>
TEST_VIDEO_PATH=<TEST_VIDEO_PATH> python3 tests/test_api_endpoints.py
```

**Test classes and expected results**:

| Test Class | Tests | What it validates |
|-----------|-------|-------------------|
| `TestHealthEndpoints` | 3 | `/v1/live`, `/v1/startup`, `/v1/ready` |
| `TestModelEndpoints` | 1 | `/v1/models` returns model list |
| `TestMetadataEndpoint` | 1 | `/v1/metadata` with version, modelInfo, licenseInfo |
| `TestFileEndpoints` | 2 | Upload → List → Chat → Download → Delete → 404 workflow |
| `TestChatCompletionEndpoint` | 4 | Non-streaming, streaming SSE, 415 content-type, validation |
| `TestEdgeCases` | 2 | 404 invalid endpoint, 405 method not allowed |
| `TestMetricsEndpoint` | 6 | Prometheus metrics format, counters, GPU metrics |
| `TestUniformChunkingEndpoint` | 5 | `algorithm:"uniform"` accepted; mixing DDM fields → 422 |

**Expected**: All 24 tests pass → `Results: 24/24 passed`

---

## Phase 4: API Integration Test with Local Video

### 4.1 Non-Streaming Chat Completion

```bash
# Base64-encode the test video and send as non-streaming request.
# NOTE: send no {"type":"text"} item — the prompt comes from VLM_PROMPT_PATH
#       (§ 6 USER_PROMPT_PRIORITY; request text would override it and break SOP classification).
VIDEO_B64=$(base64 -w0 <TEST_VIDEO_PATH>)

curl -s -X POST http://localhost:8300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ds_sop_model",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,'"$VIDEO_B64"'"}}
      ]
    }],
    "stream": false,
    "chunking_options": {
      "algorithm": "ddm-net",
      "threshold": 0.8,
      "min_length_sec": 1.0,
      "max_length_sec": 60.0
    }
  }' | python3 -m json.tool
```

**Expected response structure**:
```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": "<unix_timestamp>",
  "model": "ds_sop_model",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "<SOP analysis results>"},
    "finish_reason": "stop",
    "chunk_metadata_list": ["..."]
  }]
}
```

### 4.2 Streaming Chat Completion (SSE)

```bash
# NOTE: no {"type":"text"} item (see § 4.1) — prompt comes from VLM_PROMPT_PATH.
curl -s -N -X POST http://localhost:8300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ds_sop_model",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,'"$VIDEO_B64"'"}}
      ]
    }],
    "stream": true
  }'
```

**Expected**: Server-Sent Events stream with `data: {"object": "chat.completion.chunk", ...}`
chunks, ending with `data: [DONE]`.

### 4.3 File Upload + Chat Workflow

```bash
# Upload video file
FILE_RESP=$(curl -s -X POST http://localhost:8300/v1/files \
  -F "file=@<TEST_VIDEO_PATH>" \
  -F "purpose=test")
echo "$FILE_RESP" | python3 -m json.tool

FILE_ID=$(echo "$FILE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# Chat with uploaded file_id
curl -s -X POST http://localhost:8300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ds_sop_model",
    "messages": [{
      "role": "user",
      "content": [{"type": "input_video", "file_id": "'"$FILE_ID"'"}]
    }],
    "stream": false
  }' | python3 -m json.tool

# Cleanup
curl -s -X DELETE http://localhost:8300/v1/files/$FILE_ID | python3 -m json.tool
```

### 4.4 Uniform Chunking Smoke Test

Verify the fixed-length chunking path (selectable per request; bypasses the DDM model — § 3, § 6):

```bash
VIDEO_B64=$(base64 -w0 <TEST_VIDEO_PATH>)

# Accepted → returns fixed-length chunks
curl -s -X POST http://localhost:8300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ds_sop_model","messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"data:video/mp4;base64,'$VIDEO_B64'"}}]}],"stream":false,"chunking_options":{"algorithm":"uniform","chunk_length_sec":2.0}}' \
  | python3 -m json.tool

# Mixing DDM fields with algorithm=uniform is rejected (extra="forbid") → expect HTTP 422
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ds_sop_model","messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"data:video/mp4;base64,'$VIDEO_B64'"}}]}],"stream":false,"chunking_options":{"algorithm":"uniform","chunk_length_sec":2.0,"threshold":0.8}}'
```

Also covered by `TestUniformChunkingEndpoint` in the test suite (§ 10).

---

## Phase 5: Live Camera / RTSP Streaming Test (Optional)

Test live streaming via a physical Basler camera, an RTSP URL, **or** the Basler **camera
emulation** path. Emulation needs no hardware — you generate the frames yourself from a
video (§ 5.0), so prefer it whenever the user hasn't provided a physical camera. Do NOT ask
the user to supply a directory of PNGs.

**IMPORTANT — pick `max_length_sec` for the goal (latency vs. classification):**
- **Low latency** (responsiveness / `chunk_e2e`): `max_length_sec` ≤ 5.0 (e.g. 2.0). The default 60.0
  delays the first SSE chunk for over a minute; 2.0–5.0 gives the first chunk in ~3–6 s.
- **Correct SOP classification**: use **action-length chunks** — DDM boundaries (`threshold` ~0.8) with a
  larger `max_length_sec` (e.g. 15.0), matching the file-input run. A 2 s cap fragments each action so the
  VLM labels every chunk the out-of-scope action (a chunk-duration effect, NOT a camera/color bug).
  Do NOT evaluate camera classification with a 2 s cap — use the same chunking as the file run.

### 5.0 Camera Emulation (no physical camera required)

The emulated Basler device (serial `0815-0000`, `PYLON_CAMEMU=1`) replays PNG frames from
`CAMERA_EMULATION_DIR`. Generate those frames from the test video with GStreamer inside the
`nvds-sop` image — see **§ 8 → "Generating emulation frames from a video"** for the full
command. In short:

```bash
VIDEO=<TEST_VIDEO_PATH>
SIM_DIR=<host dir>/streams/simulation       # becomes CAMERA_EMULATION_DIR
mkdir -p "$SIM_DIR"
docker run --rm --gpus all \
  -v "$(dirname "$VIDEO")":/in:ro -v "$SIM_DIR":/out \
  --entrypoint gst-launch-1.0 nvds-sop:latest \
  -e filesrc location="/in/$(basename "$VIDEO")" ! decodebin ! nvvideoconvert ! \
     pngenc ! multifilesink sync=false location=/out/sop_sample_frame_%06d.png
N=$(ls -1 "$SIM_DIR"/*.png | wc -l)
```

Then set in `deploy/.env` (and restart the service so the mount + env apply):

```
PYLON_CAMEMU=1
CAMERA_EMULATION_DIR=<host dir>/streams/simulation
CAMERA_NUM_BUFFERS=<N>          # the frame count printed above → one bounded pass then EOS
```

Send the emulation request (set `camera_width`/`camera_height` to the generated PNG size):

```bash
timeout 30 curl -s -N -X POST http://localhost:8300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ds_sop_model","messages":[{"role":"user","content":[
    {"type":"input_camera","input_camera":{
      "camera_id":"0815-0000","camera_vendor":"Basler",
      "config":"/opt/nvidia/nvds_sop/configs/Emulation_0815-0000.pfs",
      "camera_format":"RGB","camera_width":1280,"camera_height":720,
      "camera_fps_num":30,"camera_fps_den":1
    }}
  ]}],"stream":true,"chunking_options":{"algorithm":"ddm-net","threshold":0.8,"max_length_sec":2.0}}'
```

**Expected**: live SSE `chat.completion.chunk` events every ~2–3 s with `chunk_idx`,
`cv_boundary_score`, `response`, `checker_result`; stream ends after `CAMERA_NUM_BUFFERS`
frames. Verify the emulated camera is immediately reusable after disconnect (§ 5.2).

**Verify the VLM output is meaningful** — the per-chunk `response` should classify real SOP actions
(matching what the *same video* produces via file input in § 4.1), NOT label every chunk the
out-of-scope action. If you see all-out-of-scope while DDM still segments (boundary scores vary), check
these IN ORDER:
1. **Chunking (most common):** are you using a small `max_length_sec` (e.g. 2.0)? That caps chunks below
   action length and the VLM labels the fragments out-of-scope. Re-run with the **same chunking as the
   file run** (`threshold` ~0.8, `max_length_sec` 15.0) before suspecting anything else.
2. **Frame color:** only if (1) is ruled out — emulation frames must be 3-channel **RGB** (match the
   `.pfs` `PixelFormat=RGB8Packed`); RGBA loads and DDM still fires on motion but the VLM sees wrong
   colors (see § 8 "Generating emulation frames").

### 5.1 Physical Basler Camera

```bash
# Replace <CAMERA_SERIAL> with actual serial (e.g. 40748152)
# Use timeout to auto-disconnect after collecting some chunks
timeout 20 curl -s -N -X POST http://localhost:8300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ds_sop_model",
    "messages": [{"role": "user", "content": [
      {"type": "input_camera", "input_camera": {
        "camera_id": "<CAMERA_SERIAL>",
        "camera_vendor": "Basler",
        "camera_format": "RGB",
        "camera_width": 1280,
        "camera_height": 720,
        "camera_fps_num": 30,
        "camera_fps_den": 1
      }}
    ]}],
    "stream": true,
    "chunking_options": {"algorithm": "ddm-net", "threshold": 0.8, "max_length_sec": 2.0}
  }'
```

**Expected**: SSE `chat.completion.chunk` events appear every ~2–3 seconds with VLM
classification results. Each chunk has `chunk_idx`, `start_time`, `end_time`, `cv_boundary_score`, `response`, and `checker_result`.

### 5.2 Camera Cleanup Verification

After disconnecting from the camera stream, verify the camera can be immediately reused:

```bash
# Send another camera request — should succeed without "device is controlled" error
timeout 10 curl -s -N -X POST http://localhost:8300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{...same camera request...}' | head -2
```

**Expected**: Pipeline starts and SSE chunks appear. If you see
`"The device is controlled by another application"` in container logs, the pipeline
cleanup (Rule 14) is not working correctly.

### 5.3 Common Camera Issues

- **"Device is controlled by another application"**: Another process (e.g. `pylonviewer`,
  another container) holds the camera. Kill it and retry.
- **No SSE output for >30 seconds**: Check `max_length_sec` — the default 60.0 is too large
  for interactive testing. Use ≤ 5.0.
- **Pipeline starts but VLM hangs**: Previous VLM requests may not have been aborted (Rule 16).
  Restart the container to clear the engine state.

---

## Phase 5B: RTSP Streaming Output Test (Optional — § 18)

> **Run this phase ONLY if the microservice was generated with the optional RTSP
> streaming-output feature (ds-sop-agent § 18).** It is opt-in and not part of the default
> build. Skip the whole phase if the source has no `RTSPStreamingServer` /
> `ENABLE_RTSP_OUTPUT` wiring (see the gate check in 5B.0 below).

RTSP streaming output re-streams the **decoded input video** over RTSP, in parallel with the
inference pipeline, at `rtsp://127.0.0.1:{RTSP_PORT}/ds-out/{sensor_id}`. It is auto-enabled
for **live inputs** (`rtsp://` / `camera://`) when `ENABLE_RTSP_OUTPUT=true`, and can be
forced for any input via the `--rtsp-port` CLI flag on the standalone entry points.

> **Network note:** for low-latency RTSP the container should run with `network_mode: host`
> (so `127.0.0.1:{RTSP_PORT}` on the host reaches the in-container RTSP server). If the
> container is on a bridge network, publish/forward `RTSP_PORT` (default `8554`) instead.

### 5B.0 Feature Gate Check (skip phase if absent)

```bash
# If this prints nothing, RTSP output was NOT generated → skip Phase 5B entirely.
grep -rl "RTSPStreamingServer\|ENABLE_RTSP_OUTPUT" \
  <ds_sop_microservice>/nvds_action_detector <ds_sop_microservice>/deploy 2>/dev/null
```

### 5B.1 In-Container RTSP Component Checks

Confirm the built image actually shipped the RTSP plugins/typelib (mirrors the § 18 post-build
checks). Run inside the running container:

```bash
C=deploy-nvds-action-sop-1   # adjust to your container/compose project name

# GstRtspServer typelib importable
docker exec "$C" python3 -c \
  "import gi; gi.require_version('GstRtspServer','1.0'); from gi.repository import GstRtspServer; print('GstRtspServer OK')"

# Software H.264 encoder registered (SW_ENCODER=true path)
docker exec "$C" python3 -c \
  "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print('x264enc', bool(Gst.ElementFactory.find('x264enc')))"
```

**Expected**: `GstRtspServer OK` and `x264enc True`. If either fails, the RTSP GStreamer
packages / registry-cache clear are missing from the Dockerfile (§ 9, § 18) — fix and rebuild.

### 5B.2 Enable RTSP Output

Ensure RTSP output is enabled in `deploy/.env` and the container can serve the port:

```
ENABLE_RTSP_OUTPUT=true
RTSP_PORT=8554
# false = GPU nvv4l2h264enc (default); true = CPU x264enc
# NOTE: Some GPUs (e.g. H100) lack HW NVENC — set SW_ENCODER=true to fall back to x264enc.
SW_ENCODER=false
```

Restart so the settings take effect:

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d nvds-action-sop
until curl -fs http://localhost:8300/v1/ready >/dev/null; do sleep 2; done
```

### 5B.3 Re-stream a Video File as a Live RTSP Input, then Probe the RTSP Output

RTSP output is auto-injected for **live** inputs (`rtsp://…` / `camera://…`) — a plain file
input does NOT trigger it (use 5B.4 for that). To exercise the full **API auto-injection path**
(`api_server.py` → `camera_kwargs["rtsp_port"]/["rtsp_path"]`) **without a physical camera or
external live source**, re-stream the test video file as a local RTSP source and feed that
`rtsp://` URL to `/v1/chat/completions` as the live input.

The output path stem is the `sensor_id`, which for an `rtsp://` input is the input URL's last
path segment sans extension (`api_server.py`: `url.split('/')[-1].split('.')[0]`). So an input
of `rtsp://127.0.0.1:8552/sensor_0` re-streams out at `rtsp://127.0.0.1:8554/ds-out/sensor_0`.

> The re-streamer below runs **inside the SOP container** (which ships `GstRtspServer` +
> the H.264 GStreamer plugins from 5B.1), so it needs no host RTSP dependencies. With
> `network_mode: host`, both the input source (`:8552`) and the service RTSP output (`:8554`)
> are reachable on `127.0.0.1`. The input source port (`8552`) must differ from `RTSP_PORT`
> (`8554`).

```bash
C=deploy-nvds-action-sop-1        # adjust to your container/compose project name
SENSOR_ID=sensor_0
SRC_PORT=8552                     # local re-streamer (the LIVE input source)
RTSP_IN="rtsp://127.0.0.1:${SRC_PORT}/${SENSOR_ID}"
RTSP_OUT="rtsp://127.0.0.1:8554/ds-out/${SENSOR_ID}"   # service RTSP output (sensor_id = input URL last segment)

# 1) Copy the test video into the container and write a tiny looping RTSP re-streamer
#    (H.264 passthrough: filesrc ! qtdemux ! h264parse ! rtph264pay, seeking to 0 on EOS).
docker cp "<TEST_VIDEO_PATH>" "$C":/tmp/rtsp_src.mp4
docker exec -i "$C" bash -c 'cat > /tmp/rtsp_src.py' <<'PY'
import sys, gi
gi.require_version("Gst", "1.0"); gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib
Gst.init(None)
PATH, PORT, MOUNT = sys.argv[1], sys.argv[2], sys.argv[3]
class Factory(GstRtspServer.RTSPMediaFactory):
    def do_create_element(self, url):
        b = Gst.parse_launch(
            f'filesrc location="{PATH}" ! qtdemux ! h264parse ! '
            'identity name=idn single-segment=true ! rtph264pay name=pay0 pt=96 config-interval=1')
        b.get_by_name("idn").get_static_pad("sink").add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM, self._loop, b)
        return b
    def _loop(self, pad, info, b):
        e = info.get_event()
        if e.type == Gst.EventType.EOS:
            GLib.idle_add(lambda: (b.seek_simple(
                Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0), False)[1])
            return Gst.PadProbeReturn.DROP
        if e.type in (Gst.EventType.FLUSH_START, Gst.EventType.FLUSH_STOP):
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK
srv = GstRtspServer.RTSPServer.new(); srv.set_service(PORT)
f = Factory(); f.set_shared(True)
srv.get_mount_points().add_factory(MOUNT, f); srv.attach(None)
print(f"RTSP source ready at rtsp://127.0.0.1:{PORT}{MOUNT}", flush=True)
GLib.MainLoop().run()
PY
docker exec -d "$C" python3 /tmp/rtsp_src.py /tmp/rtsp_src.mp4 "$SRC_PORT" "/$SENSOR_ID"

# 2) Confirm the LIVE input source is actually serving before driving inference through it.
sleep 4
ffprobe -v error -rtsp_transport tcp -show_streams "$RTSP_IN" 2>&1 | grep -E "codec_name|codec_type" | head || \
  echo "WARN: input RTSP source not up yet (check: docker exec $C ps aux | grep rtsp_src)"

# 3) Kick off a live inference request against the rtsp:// input in the background. This
#    drives the API auto-injection path, keeping the pipeline + RTSP output branch up.
timeout 30 curl -s -N -X POST http://localhost:8300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ds_sop_model",
    "messages": [{"role": "user", "content": [
      {"type": "video_url", "video_url": {"url": "'"$RTSP_IN"'"}}
    ]}],
    "stream": true,
    "chunking_options": {"algorithm": "ddm-net", "threshold": 0.8, "max_length_sec": 2.0}
  }' >/tmp/rtsp_live_req.log 2>&1 &
REQ_PID=$!

# 4) Give the pipeline a few seconds to build the RTSP branch, then confirm the server log line.
sleep 6
docker logs --tail 200 "$C" 2>&1 | grep -i "RTSP Server started at\|linked RTSP stream" || \
  echo "WARN: no RTSP server start log yet"

# 5) Probe the OUTPUT stream — expects an H.264 video stream.
ffprobe -v error -rtsp_transport tcp -show_streams "$RTSP_OUT" 2>&1 | grep -E "codec_name|codec_type" | head

# Probe the OUTPUT stream resolution — MUST match the source resolution (e.g. 1920x1080), NOT 224x224.
# If it is 224x224, the pipeline is misconfigured and squashing live inputs.
ffprobe -v error -rtsp_transport tcp -show_entries stream=width,height -of csv=p=0 "$RTSP_OUT"

wait $REQ_PID 2>/dev/null

# 6) Stop the input re-streamer.
docker exec "$C" pkill -f rtsp_src.py || true
```

**Expected**:
- Step 2 `ffprobe` on the input source `$RTSP_IN` reports `codec_type=video` / `codec_name=h264`
  (the looping re-stream of the test video is live).
- Container logs show `RTSP Server started at rtsp://127.0.0.1:8554/ds-out/sensor_0` and
  `linked RTSP stream on port 8554`.
- Step 5 `ffprobe` on the output `$RTSP_OUT` reports `codec_type=video` and `codec_name=h264`
  (or `mjpeg` if the encoder fell back to MJPEG), and the resolution matches the source resolution (e.g., `1920,1080`), **NOT** `224,224`. A live frame can also be grabbed with:

```bash
ffmpeg -y -rtsp_transport tcp -i "$RTSP_OUT" -frames:v 1 /tmp/rtsp_out_frame.jpg && \
  ls -lh /tmp/rtsp_out_frame.jpg
```

> **Substituting a real live source:** a physical Basler camera (`{"type":"input_camera", …}`,
> `sensor_id` = camera serial) or an external `rtsp://` URL works identically — they hit the
> same auto-injection path. Use the video re-stream above when no camera/live source is
> available, which is the common eval-host case.

### 5B.4 (Optional) File Input via CLI `--rtsp-port`

File inputs don't auto-enable RTSP output; force it via the standalone entry point inside the
container, then probe `rtsp://127.0.0.1:8554/ds-out/{video_name}`:

```bash
docker exec -d "$C" python -m nvds_action_detector.ds_sop_process \
  --video-path /path/in/container/video.mp4 --rtsp-port 8554
sleep 6
ffprobe -v error -rtsp_transport tcp -show_streams \
  rtsp://127.0.0.1:8554/ds-out/video 2>&1 | grep -E "codec_name|codec_type" | head
```

### 5B.5 Common RTSP Output Issues

- **`ffprobe` connection refused / 404**: the live request isn't running (the RTSP branch only
  exists while a pipeline with `rtsp_port` is active), `network_mode: host` is not set, or
  `ENABLE_RTSP_OUTPUT` is false. Confirm the start log line first.
- **`GstRtspServer not found` warning in logs**: the typelib/packages are missing from the
  image (5B.1 failed) — add the § 9/§ 18 apt packages and rebuild.
- **Inference stalls when an RTSP client connects/disconnects**: the RTSP queue is not
  `leaky=2` + tiny (`RTSP_LEAKY_QUEUE_TINY`, § 18); a slow client is back-pressuring `tee1`
  and starving inference. Fix the queue caps in `ds_3d_action_pipeline.py` and rebuild.
- **Recorded clips won't seek/cut**: the encoder is missing `key-int-max=30`
  (`RTSP_KEYINT_MAX_30`, § 18).

---

## Phase 6: Kafka Messaging Test

Test that chunk results are published to Kafka when `ENABLE_MESSAGING=true`.

### 6.1 Enable Messaging

Add to `deploy/.env`:
```
ENABLE_MESSAGING=true
KAFKA_BROKER=localhost:9092
# Serialization schema (§ 16):
#   JSON          → flat JSON doc, indexable as-is by the Kibana SOP dashboard (use for 6.3)
#   NvProtoSchema → VisionLLM protobuf, nested under llm/sensor/info (for NVIDIA analytics)
SOP_MESSAGING_SCHEMA=JSON
```

Restart the service:
```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d nvds-action-sop
```

> **Schema choice drives the dashboard.** The Kibana SOP dashboard
> (`mdx-vlm-captions-*` index pattern) reads **flat** top-level fields
> (`response`, `sensor_id`, `chunk_idx`, `cv_execute_time`, `vlm_execute_time`,
> `start_time`, `end_time`). The `JSON` schema publishes the `chunk_info` dict as-is, so
> the Kafka message **is** the document and those fields land flat. `NvProtoSchema` nests
> them under `llm`/`sensor`/`info`, which the flat dashboard field references cannot resolve.
> Use `SOP_MESSAGING_SCHEMA=JSON` for the dashboard-structure test in 6.3.

### 6.2 Send Video Request and Verify Messages

Run the **§ 4.3 file upload → chat workflow** (with `ENABLE_MESSAGING=true` each chunk is
published to Kafka), then count and inspect the resulting messages:

```bash
# Count messages in Kafka topic
docker exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic mdx-vlm-captions \
  --from-beginning --timeout-ms 5000 2>/dev/null | wc -l

# Deserialize protobuf messages with Python consumer
docker exec <container-name> timeout 10 python3 -c "
from nvds_action_detector.messager import create_consumer
import json
consumer = create_consumer('localhost:9092', 'eval-group')
for i, msg in enumerate(consumer.consume(topic='mdx-vlm-captions')):
    if msg is None: continue
    print(json.dumps(msg, indent=2, default=str))
    if i >= 2: break
"
```

**Expected**:
- Kafka topic `mdx-vlm-captions` is auto-created
- Number of Kafka messages matches the number of chunks from chat completion
- With `SOP_MESSAGING_SCHEMA=JSON`: each message is a flat JSON object (validated in 6.3).
  With `SOP_MESSAGING_SCHEMA=NvProtoSchema`: each message is a `VisionLLM` protobuf with
  `version`, `timestamp`, `sensor`, `llm.queries[].response` (deserialize via the Python
  consumer shown above).

### 6.3 Validate JSON Message Structure Against the Kibana Dashboard

This test confirms the **published Kafka JSON message structure matches the fields the Kibana
SOP dashboard expects**. With `SOP_MESSAGING_SCHEMA=JSON` the Kafka message is ingested
verbatim into the `mdx-vlm-captions-*` Elasticsearch index, so each message must carry these
**flat top-level** fields (derived from `sop-kibana-objects.ndjson`):

| Field | Type | Dashboard usage |
|-------|------|-----------------|
| `sensor_id` | string | "Sensor Selector" control + per-series split (`sensor_id.keyword`) |
| `chunk_idx` | int | X-axis ("Chunk Index") on the CV / VLM / chunk-length charts |
| `response` | string | `action_id_display` / `action_id_sort` runtime fields parse the leading `(N)` from `response.keyword` |
| `cv_execute_time` | number | "Average CV execute time over chunk idx" line |
| `vlm_execute_time` | number | "Average VLM execute time over chunk idx" line |
| `start_time` | number | `chunk_length_seconds` + `vlm_execute_time_normalized` runtime fields (need `end_time > start_time`) |
| `end_time` | number | `chunk_length_seconds` + `vlm_execute_time_normalized` runtime fields |

> `@timestamp` (the dashboard time field) is **not** produced by the service — the downstream
> Logstash/ELK ingest stamps it from a real wall-clock pipeline timestamp. This test only
> validates the message structure the service controls; `@timestamp` correctness is covered by
> the blueprint's ELK pipeline test.

Confirm `SOP_MESSAGING_SCHEMA=JSON` is set (6.1), run a request that produces chunks
(reuse 6.2 or 4.1), then consume the raw topic and validate each message:

```bash
# Dump the raw JSON messages from the topic (plain JSON — no protobuf decode needed).
docker exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic mdx-vlm-captions \
  --from-beginning --timeout-ms 8000 2>/dev/null > /tmp/kafka_msgs.jsonl

python3 - /tmp/kafka_msgs.jsonl <<'PY'
import json, re, sys

# Flat fields the Kibana `mdx-vlm-captions-*` dashboard reads from each document.
REQUIRED = {
    "sensor_id":        (str,),
    "chunk_idx":        (int,),          # bool is rejected explicitly below
    "response":         (str,),
    "cv_execute_time":  (int, float),
    "vlm_execute_time": (int, float),
    "start_time":       (int, float),
    "end_time":         (int, float),
}

lines = [ln for ln in open(sys.argv[1]) if ln.strip()]
assert lines, "No Kafka messages — is ENABLE_MESSAGING=true and a request producing chunks?"

docs, parse_errs = [], []
for i, ln in enumerate(lines):
    try:
        docs.append(json.loads(ln))
    except Exception as e:
        parse_errs.append(f"msg {i}: not valid JSON ({e}) — NvProtoSchema (protobuf) is NOT "
                          "dashboard-indexable; set SOP_MESSAGING_SCHEMA=JSON")

# Per-chunk docs are the ones the charts plot (they carry chunk_idx).
chunks = [d for d in docs if isinstance(d, dict) and "chunk_idx" in d]
errs = list(parse_errs)
if not chunks:
    errs.append("no per-chunk messages with 'chunk_idx' — CV/VLM/chunk-length charts (plotted "
                "over chunk_idx) would be empty; check the message schema and that chunks were produced")

for i, d in enumerate(chunks):
    if ("response" not in d) and ("llm" in d or "sensor" in d):
        errs.append(f"chunk {i}: nested protobuf shape (llm/sensor) — dashboard needs flat "
                    "'response'/'sensor_id'; use SOP_MESSAGING_SCHEMA=JSON")
        continue
    for k, t in REQUIRED.items():
        if k not in d:
            errs.append(f"chunk {i}: missing '{k}'")
        elif isinstance(d[k], bool) or not isinstance(d[k], t):
            errs.append(f"chunk {i}: '{k}'={d[k]!r} is {type(d[k]).__name__}, expected {'/'.join(x.__name__ for x in t)}")
    s, e = d.get("start_time"), d.get("end_time")
    if isinstance(s, (int, float)) and isinstance(e, (int, float)) and not isinstance(s, bool) and e <= s:
        errs.append(f"chunk {i}: end_time({e}) <= start_time({s}) — chunk_length_seconds / "
                    "vlm_execute_time_normalized resolve to 0 on the dashboard")
    r = d.get("response")
    if isinstance(r, str) and not re.match(r"^\s*\(\d+\)", r):
        errs.append(f"chunk {i}: response {r[:40]!r} does not start with '(N)' — "
                    "action_id_display / action_id_sort cannot parse the action id")

print(f"Validated {len(chunks)} per-chunk JSON message(s) against the Kibana dashboard contract")
if errs:
    print("FAIL — dashboard-incompatible messages:")
    for e in errs[:25]:
        print("  -", e)
    sys.exit(1)
print("PASS — all messages match the mdx-vlm-captions dashboard schema")
PY
```

**Expected**: `PASS — all messages match the mdx-vlm-captions dashboard schema`, i.e. every
per-chunk message is a flat JSON object carrying all seven fields with the right types,
`end_time > start_time`, and a `response` that starts with `(N)` so the action-id runtime
fields parse. Any `FAIL` line names the exact field/message — fix the producer
(`ds_sop_process._publish_message` / `chunk_info` assembly in § 6) or the schema setting,
then re-run.

---

## Phase 7: Cleanup

```bash
cd <ds_sop_microservice>
docker compose -f deploy/compose.yaml down
```

---

## Evaluation Summary Checklist

| # | Check | Pass/Fail |
|---|-------|-----------|
| 0.1 | All files in `package_file_list.txt` exist | |
| 0.2 | Critical Rules 1-9, 14-18 static check passes | |
| 0.3 | Reference files match (no unintended regeneration) | |
| 1.3 | Docker image builds successfully | |
| 2.1 | Model paths verified (DDM checkpoint + VLM model accessible) | |
| 2.3 | Container starts, models load, service ready | |
| 2.4 | Health endpoints return 200 | |
| 3.2 | All 24 unit tests pass (incl. 5 `TestUniformChunkingEndpoint`) | |
| 4.1 | Non-streaming chat completion returns valid response | |
| 4.2 | Streaming SSE returns chunks + `[DONE]` | |
| 4.3 | File upload → chat → delete workflow works | |
| 4.4 | Uniform chunking accepted; mixing DDM fields → HTTP 422 | |
| 5.0 | Camera emulation: frames generated from video, live SSE chunks produced | |
| 5.0b | Camera emulation: VLM classifies real actions (not all out-of-scope → RGB frame check) | |
| 5.1 | Live camera SSE produces real-time chunks (if physical camera available) | |
| 5.2 | Camera released after disconnect (reuse works immediately) | |
| 5B | *(optional, § 18)* RTSP output: `GstRtspServer`/`x264enc` present + a re-streamed video file as live `rtsp://` input re-streams H.264 at `rtsp://127.0.0.1:8554/ds-out/{sensor_id}` with original source resolution, **NOT** 224x224 (skip if feature not generated) | |
| 6.2 | Kafka messages published for all chunks (`ENABLE_MESSAGING=true`) | |
| 6.3 | JSON Kafka message structure matches the Kibana dashboard fields (flat `response`, `sensor_id`, `chunk_idx`, `cv_execute_time`, `vlm_execute_time`, `start_time`, `end_time`) | |
