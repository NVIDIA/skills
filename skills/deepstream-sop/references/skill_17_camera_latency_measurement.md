# § 17 — Camera / Live-Stream Chunk-E2E Latency Measurement

Defines how to measure **chunk_e2e** latency for **camera and live-stream** (RTSP) inputs.

> **Contrast with § 15** (file input): File latency uses wall-clock around the HTTP request
> because request-start is well-defined. For live sources there is no HTTP "start" — the
> camera streams continuously. Instead, use internal pipeline timestamps that are already
> written into every `chunk_metadata` dict.

---

## Metric Definition

| Metric | Formula | Meaning |
|--------|---------|---------|
| **chunk_e2e** | `pipeline_vlm_ready_timestamp − pipeline_chunk_end_timestamp` | Wall-clock latency from "last frame of the chunk first seen by the frame branch" → "VLM result ready for this chunk" |
| **pipeline_chunk_end_timestamp** | `wall_clock_entry` at `consume()` entry for the last frame ≤ chunk end_time | Earliest possible "chunk frame boundary" wall-clock (before GPU ops) |
| **pipeline_vlm_ready_timestamp** | `self._tm_e2e.now()` in `vlm_inference_response_process()` after future resolves | Wall-clock when VLM result is available |

Both fields are written into `chunk_info` and appear in SSE `chunk_metadata` (under
`choices[0]["chunk_metadata"]`). **No client-side timing is needed** — these are server-side
wall-clock values carried in the payload.

---

## How the Fields Are Captured

### `pipeline_chunk_end_timestamp` (§ 6 — `submit_vllm_inference`)

```python
# CRITICAL [WALL_CLOCK_BEFORE_GPU]: Capture wall_clock BEFORE GPU extract+dlpack.
# This is the earliest "frame arrived at frame branch" timestamp.
def consume(self, buffer):
    wall_clock_entry = time.time()        # ← capture here, before GPU
    pts_ns = buffer.timestamp
    tensor = buffer.extract(0).clone()
    torch_tensor = torch.utils.dlpack.from_dlpack(tensor)
    timestamp = pts_ns / 1e9
    self.decoded_frame_queue.put((timestamp, wall_clock_entry, torch_tensor), ...)
```

```python
# submit_vllm_inference: set pipeline_chunk_end_timestamp on last frame of chunk
timestamp, wall_clock, tensor = frame
if timestamp + 1e-3 >= end_time:
    chunk_info["pipeline_chunk_end_timestamp"] = wall_clock
    break
```

### `pipeline_vlm_ready_timestamp` (§ 6 — `vlm_inference_response_process`)

```python
chunk_info["pipeline_vlm_ready_timestamp"] = self._tm_e2e.now()
```

Both are written before the chunk enters `final_queue` / SSE.

---

## chunk_e2e Decomposition

For a **2-second chunk** at 30 fps, chunk_e2e is dominated by three stages:

```
pipeline_chunk_end_timestamp
       │
       ├─► DDM wait        ≈ (FRAMES_PER_SIDE + extra) / fps + DDM_inference
       │   (trailing context frames DDM needs before it can score the chunk boundary)
       │   Best case:  5/30 + 0.034s ≈ 0.200 s
       │   Average:    ~0.318 s  (depends on where boundary falls in SEQUENCE_BATCH=8 cycle)
       │   Worst case: 12/30 + 0.034s ≈ 0.434 s
       │
       ├─► VLM preprocess  ≈ 0.042 s
       │   (frame sampling from chunk + resize to VLM input resolution)
       │   Runs in vlm_request_pool (64 workers) — does NOT block vlm_inference_request_process
       │
       └─► VLM inference   ≈ 0.180–0.190 s  (measured with Cosmos Reason 1 7B, 20 frames, 256 tokens; Reason 2 latency may differ)
               │
       pipeline_vlm_ready_timestamp

chunk_e2e ≈ DDM_wait + VLM_preprocess + VLM_inference
          ≈ 0.200–0.434 s  +  0.042 s  +  0.186 s
          ≈ 0.427 – 0.662 s  (best → worst DDM alignment)
```

**Key insight — DDM wait is NOT DDM inference time:**
DDM inference per batch takes ~33 ms. The wait comes from trailing context:
the DDM model scores a frame only after seeing `FRAMES_PER_SIDE` (5) additional frames
past it. The chunk boundary frame must wait for those 5 trailing context frames to
accumulate before DDM can emit a boundary score. The exact wait depends on how many
extra frames are needed to complete the next `SEQUENCE_BATCH` (8-frame) cycle:

```
DDM_wait = (FRAMES_PER_SIDE + extra_frames) / fps + δ_ddm
extra_frames = (SEQUENCE_BATCH - ((N + FRAMES_PER_SIDE - SLIDING_WINDOWS_SIZE + 1) % SEQUENCE_BATCH)) % SEQUENCE_BATCH
```

where N = number of frames in the chunk, SLIDING_WINDOWS_SIZE = 18.

---

## Forcing Fixed Chunk Size for Reproducible Measurement

> **Latency-only — do NOT judge classification with this.** The fixed 2 s chunks below are for
> reproducible `chunk_e2e`/TTFC timing. A 2 s cap is shorter than most SOP actions, so the VLM will
> label these chunks the out-of-scope action — that is expected here and is NOT a bug. For meaningful
> SOP classification on live/camera, use action-length chunks (`threshold` ~0.8, larger `max_length_sec`
> e.g. 15.0), as in the file-input run (§ 8, § 12).

Use the same fixed-chunk `ddm-net` `CHUNKING` as file input (`threshold=0.99` so DDM
never fires a boundary; `min_length_sec = max_length_sec = 2.0` so the cap controls
chunk duration deterministically). See **§ 15 → "Forcing fixed chunk sizes"** for the
`CHUNKING` block and why `algorithm:"uniform"` is **not** used here — uniform would
bypass DDM (§ 3, § 6) and exclude it from the `chunk_e2e` latency, whereas this measures
the full live pipeline including DDM.

---

## Rules for Generating a Camera chunk_e2e Test Script

Use these rules to generate a measurement script for camera or RTSP inputs.
**Do not copy `api_client_perf.py` verbatim** — generate a purpose-built script
using these rules:

### Rule 1 — Use SSE streaming
Camera inputs **require** `"stream": true` (HTTP 400 otherwise, see `LIVE_REQUIRES_STREAM_TRUE`).
All chunk_metadata arrives only in SSE events.

### Rule 2 — Read `chunk_e2e` from chunk_metadata, not wall-clock
```python
meta = obj["choices"][0].get("chunk_metadata", {})
if meta and meta.get("chunk_idx") is not None and meta["chunk_idx"] >= 0:
    t_end = meta.get("pipeline_chunk_end_timestamp")
    t_vlm = meta.get("pipeline_vlm_ready_timestamp")
    if t_end and t_vlm:
        chunk_e2e = t_vlm - t_end
```

Do NOT compute latency from `time.time()` on the client side — SSE delivery jitter
makes it inaccurate for live sources.

### Rule 3 — Skip summary chunk
The last SSE event has `chunk_idx=-1` (summary) — skip it, exactly as in **§ 15**
(`if meta.get("chunk_idx", -1) < 0: continue`).

### Rule 4 — Wait for server ready before starting
Poll `/v1/ready` until HTTP 200, then wait 2 s before sending the camera request.
The camera pipeline takes ~0.5–1 s to start streaming.

### Rule 5 — Use the config-file VLM prompt by default
Same prompt-source rule as file input — see **§ 15 → "Prompt source rule"**: send only
the media content item and omit `{"type":"text", ...}` so the service uses
`VLM_PROMPT_PATH`. Request text overrides it; label such runs as prompt-override
experiments.

### Rule 6 — Use current camera / RTSP content schemas

At the API boundary, camera requests use the `input_camera` content schema. The
server then converts `input_camera.camera_id` to the internal source URI
`camera://<serial>` before creating the DeepStream pipeline. Do not send
`{"type":"camera_input","serial":"..."}` from client scripts.

```python
payload = {
    "model": "ds_sop_model",
    "messages": [{
        "role": "user",
        "content": [{
            "type": "input_camera",
            "input_camera": {
                "camera_id": "<serial>",
                "camera_vendor": "Basler",
                "camera_format": "RGB",
                "camera_width": 1280,
                "camera_height": 720,
                "camera_fps_num": 30,
                "camera_fps_den": 1,
            },
        }]
    }],
    "stream": True,
    "chunking_options": CHUNKING,
}
```
For RTSP: use `{"type": "video_url", "video_url": {"url": "rtsp://..."}}` instead.

### Rule 7 — Collect N chunks then kill the connection
Camera streams indefinitely. Kill after collecting enough chunks:
```python
if len(chunks) >= MAX_CHUNKS:
    proc.kill()
    break
```

### Rule 8 — Report per-chunk and aggregate stats
```python
import statistics
print(f"chunk_e2e: mean={statistics.mean(e2e_list):.3f}s  "
      f"min={min(e2e_list):.3f}s  max={max(e2e_list):.3f}s  "
      f"stdev={statistics.stdev(e2e_list):.3f}s  n={len(e2e_list)}")
```

### Rule 9 — Also log DDM wait separately (optional)
`pipeline_chunk_end_timestamp` is the last frame wall-clock. The chunk `end_time`
(chunk boundary PTS) is in `chunk_metadata["end_time"]`. Their difference approximates
DDM wait (if clocks are synchronized). Otherwise use VLM time from `vlm_execute_time`
field to separate DDM_wait from VLM_time within chunk_e2e.

---

## Example Skeleton (for generation reference)

```python
"""Camera chunk_e2e latency measurement — generate from the rules above, not from api_client_perf.py."""
import json, subprocess, time, statistics

URL    = "http://localhost:8300"
SERIAL = "40748152"        # or "0815-0000" for emulation
CHUNKING = {"algorithm": "ddm-net", "threshold": 0.99, "min_length_sec": 2.0, "max_length_sec": 2.0}
MAX_CHUNKS = 10

def stream_camera(serial, max_chunks=MAX_CHUNKS):
    payload = json.dumps({
        "model": "ds_sop_model",
        "messages": [{"role": "user", "content": [{
            "type": "input_camera",
            "input_camera": {
                "camera_id": serial,
                "camera_vendor": "Basler",
                "camera_format": "RGB",
                "camera_width": 1280,
                "camera_height": 720,
                "camera_fps_num": 30,
                "camera_fps_den": 1,
            },
        }]}],
        "stream": True,
        "chunking_options": CHUNKING,
    })
    proc = subprocess.Popen(
        ["curl", "-N", "-s", "--max-time", "120", "-X", "POST",
         f"{URL}/v1/chat/completions", "-H", "Content-Type: application/json", "-d", payload],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    e2e_list, chunks = [], []
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("data:"): continue
        data = line[5:].strip()
        if data == "[DONE]": break
        try:
            obj = json.loads(data)
            meta = obj["choices"][0].get("chunk_metadata", {})
            if not meta or meta.get("chunk_idx", -1) < 0: continue   # skip summary
            t_end = meta.get("pipeline_chunk_end_timestamp")
            t_vlm = meta.get("pipeline_vlm_ready_timestamp")
            if t_end and t_vlm:
                e2e_list.append(t_vlm - t_end)
            chunks.append(meta)
            if len(chunks) >= max_chunks:
                proc.kill(); break
        except Exception:
            pass
    proc.wait()
    return e2e_list, chunks

# Wait for server ready
for _ in range(60):
    try:
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                            f"{URL}/v1/ready"], capture_output=True, text=True, timeout=5)
        if r.stdout.strip() == "200": break
    except Exception:
        pass
    time.sleep(10)
time.sleep(2)  # let camera settle

e2e_list, chunks = stream_camera(SERIAL)
if e2e_list:
    print(f"chunk_e2e  mean={statistics.mean(e2e_list):.3f}s  "
          f"min={min(e2e_list):.3f}s  max={max(e2e_list):.3f}s  n={len(e2e_list)}")
```

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `pipeline_chunk_end_timestamp` missing from metadata | `consume()` uses old 2-tuple `(timestamp, tensor)` | Update to 3-tuple with `wall_clock_entry`; update `submit_vllm_inference` to unpack 3 values |
| `pipeline_vlm_ready_timestamp` missing | `vlm_inference_response_process` doesn't set it | Add `chunk_info["pipeline_vlm_ready_timestamp"] = self._tm_e2e.now()` before putting to queue |
| chunk_e2e > 1 s | VLM not warmed up, or DDM worst-case alignment (12 trailing frames) | Expected range 0.43–0.66 s; values > 1 s suggest model cold-start or GPU contention |
| HTTP 400 on camera request | `stream=false` for live input | Camera requires `"stream": true` (LIVE_REQUIRES_STREAM_TRUE) |
| Accuracy or latency differs from production eval | Request includes `{"type":"text"}` and overrides `VLM_PROMPT_PATH` | Omit request text, or label the run as prompt override |
