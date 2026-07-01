# § 15 — Latency Measurement (File Input)

Defines how to measure **TTFC** (Time-To-First-Chunk) and **C2C** (Chunk-To-Chunk)
latency for file input via SSE streaming.

> **Note**: Live-stream latency (camera / RTSP) cannot use the same wall-clock method
> because there is no well-defined HTTP request start. For live sources, use internal
> pipeline timestamps: `pipeline_vlm_ready_timestamp - pipeline_starting_timestamp`
> from chunk metadata.

---

## Definitions

| Metric | Definition |
|--------|-----------|
| **TTFC** | Wall-clock time from HTTP request sent → first SSE `chunk_metadata` event received. Covers DDM pipeline start-up, first boundary detection, VLM inference, and SOP checker for chunk 0. |
| **C2C** | Wall-clock time between consecutive SSE `chunk_metadata` arrivals (chunk N-1 → chunk N). |
| **TTFC_int** | Internal: `pipeline_vlm_ready_timestamp − pipeline_starting_timestamp` in chunk 0 metadata. Excludes HTTP/network overhead. Available for both file and live inputs. |

---

## Measurement Method

Always use **SSE streaming** (`"stream": true`). Non-streaming collects all chunks
before returning and cannot measure per-chunk latency.

### Prompt source rule

For production-equivalent TTFC/C2C measurement, send only the media content item
(`input_video`) and omit `{"type":"text", ...}`. The service will then use the
prompt loaded from `VLM_PROMPT_PATH`.

Before measuring, verify that `VLM_PROMPT_PATH` is set in `.env`/compose and that
the prompt file is mounted in the container. If a request text prompt is included,
label the run as a prompt-override experiment and do not compare it with
config-prompt latency numbers.

### Forcing fixed chunk sizes

Set `threshold=0.99` so DDM never triggers a boundary; `max_length_sec` controls
chunk duration deterministically:

```python
CHUNKING = {
    "algorithm": "ddm-net",
    "threshold": 0.99,       # boundary never fires → max_length always triggers
    "min_length_sec": 2.0,
    "max_length_sec": 2.0,
}
```

> **Why not `algorithm:"uniform"` here?** Uniform chunking also yields fixed-length chunks, but it
> **bypasses the DDM model entirely** (§ 3, § 6), so it would exclude DDM inference from the latency
> numbers. These measurements profile the **full pipeline including DDM**, so we keep `ddm-net` and
> use `threshold=0.99` to force deterministic chunk sizes while DDM still runs.

### SSE parsing — `chunk_metadata` location

`chunk_metadata` is at `choices[0]["chunk_metadata"]`, **not** inside `delta`:

```python
obj  = json.loads(sse_line[5:].strip())          # strip "data: " prefix
meta = obj["choices"][0].get("chunk_metadata", {})
# skip summary chunk (chunk_idx == -1)
if meta and meta.get("chunk_idx") is not None and meta["chunk_idx"] >= 0:
    ...
```

### Measurement script

```python
import subprocess, json, time, statistics

URL      = "http://localhost:8300"
VIDEO    = "/path/to/test.mp4"
CHUNKING = {"algorithm":"ddm-net","threshold":0.99,"min_length_sec":2.0,"max_length_sec":2.0}
RUNS     = 3
MAX_CHUNKS = 10

def upload(video):
    r = subprocess.run(["curl","-s","-X","POST",f"{URL}/v1/files",
                        "-F",f"file=@{video}","-F","purpose=eval"],
                       capture_output=True, text=True, timeout=20)
    return json.loads(r.stdout)["id"]

def stream_chunks(fid, max_chunks=MAX_CHUNKS):
    payload = json.dumps({
        "model": "ds_sop_model",
        "messages": [{"role":"user","content":[{"type":"input_video","file_id":fid}]}],
        "stream": True, "chunking_options": CHUNKING,
    })
    proc = subprocess.Popen(
        ["curl","-N","-s","--max-time","60","-X","POST",f"{URL}/v1/chat/completions",
         "-H","Content-Type: application/json","-d",payload],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    t_req = time.time()
    chunks, t_arr = [], []
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("data:"): continue
        data = line[5:].strip()
        if data == "[DONE]": break
        try:
            meta = json.loads(data)["choices"][0].get("chunk_metadata", {})
            if meta and meta.get("chunk_idx") is not None and meta["chunk_idx"] >= 0:
                t_arr.append(time.time())
                chunks.append(meta)
                if len(chunks) >= max_chunks:
                    proc.kill(); break
        except Exception:
            pass
    proc.wait()
    return t_req, t_arr, chunks

all_ttfc, all_c2c = [], []
for _ in range(RUNS):
    fid = upload(VIDEO)
    try:
        t_req, t_arr, chunks = stream_chunks(fid)
    finally:
        subprocess.run(["curl","-s","-X","DELETE",f"{URL}/v1/files/{fid}"],
                       capture_output=True)
    if t_arr:
        all_ttfc.append(t_arr[0] - t_req)
        all_c2c.extend(t_arr[i] - t_arr[i-1] for i in range(1, len(t_arr)))

print(f"TTFC: mean={statistics.mean(all_ttfc):.3f}s  "
      f"min={min(all_ttfc):.3f}s  max={max(all_ttfc):.3f}s")
if all_c2c:
    print(f"C2C:  mean={statistics.mean(all_c2c):.3f}s  "
          f"min={min(all_c2c):.3f}s  max={max(all_c2c):.3f}s  "
          f"stdev={statistics.stdev(all_c2c):.3f}s")
```

---

## DDM TensorRT Latency Comparison

The optional TensorRT path (§ 5, `DDM_TRT_OPTIMIZATION`) accelerates the Stage-1 DDM
inference. To quantify the speedup, run the **same fixed-chunk `ddm-net` measurement above**
twice — once on PyTorch, once on TRT — keeping the video, `CHUNKING`, and all VLM settings
identical so the delta isolates DDM. (The fixed-chunk `ddm-net` path is required here precisely
because `uniform` would bypass DDM — see the note above.)

### Procedure

1. **Baseline (PyTorch):** set `DDM_TRT_OPTIMIZATION=false` in `deploy/.env`, restart the
   container (the runtime is chosen once at Triton init), wait for `/v1/ready`, run the script.
2. **TRT:** set `DDM_TRT_OPTIMIZATION=true` (optionally `DDM_TRT_PRECISION=fp16`), restart, and
   **wait out the one-time engine build** — the first start logs `Building DDM TRT engine on the
   fly` → `TRT engine ready` (~5–15 min). Persist it via `DDM_TRT_ENGINE_OUTPUT_PATH` under
   `MODEL_ROOT_DIR` so it is not rebuilt on the next restart. Then run the same script.
3. **Compare** the two runs.

### What to compare

- **`cv_execute_time`** — per-chunk DDM/CV stage time (set in `_make_chunk_info`). If your chunk
  schema surfaces it, this is the **cleanest DDM-only metric** — this is where TRT shows up directly.
- **C2C** — secondary; steady-state throughput, still includes DDM.
- **TTFC** — least sensitive: dominated by VLM inference + SOP checker, so DDM speedup is partly
  masked. Report it, but expect a smaller relative change than the CV-stage time.

```python
# Add to the measurement script to capture the DDM/CV stage time per chunk:
cv_times = [m["cv_execute_time"] for m in chunks if m.get("cv_execute_time") is not None]
if cv_times:
    print(f"DDM cv_execute_time: mean={statistics.mean(cv_times):.4f}s  min={min(cv_times):.4f}s")
# Run once with DDM_TRT_OPTIMIZATION=false and once =true; compare the means.
```

### Caveats

- The runtime is selected at init — you **must restart the container** between PyTorch and TRT runs.
- **Discard the first TRT run** if it triggered the engine build (the build time would pollute it).
- Precision is NOT re-checked against a cached engine — to compare fp16 vs fp32, delete the engine
  file (or change `DDM_TRT_ENGINE_OUTPUT_PATH`) before switching `DDM_TRT_PRECISION` (§ 5).
- A TRT build failure silently falls back to PyTorch — confirm the TRT run actually used TRT
  (look for `Created TRT thread-local state` in the logs) before trusting the numbers.

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `chunk_metadata` not found in SSE | Accessing `delta` instead of `choices[0]` | Use `choices[0]["chunk_metadata"]` |
| TTFC > 5s | VLM model not warmed up | Poll `/v1/ready` every 10s; start measuring only after 200 is returned (see § 9) |
| Accuracy or latency differs from production eval | Request includes `{"type":"text"}` and overrides `VLM_PROMPT_PATH` | Omit request text, or label the run as prompt override |
