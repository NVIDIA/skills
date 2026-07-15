# § 1 — FastAPI Endpoints (`api_server.py`)

> **Generates**: `nvds_action_detector/api_server.py`
> **Critical Rules**: MANAGER_INIT_IN_MAIN, NAMED_KWARGS, CLEANUP_ON_DISCONNECT, METADATA_LICENSE_FROM_FILE
> **Pipeline stage**: Entry point — receives requests, dispatches to SOPVideoProcessor (§ 6),
> streams results via SSE (§ 7)

---

## All Required Endpoints

```
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI Server (port 8300)                                        │
│                                                                    │
│  POST /v1/chat/completions ──► SOPProcessManager ──► 4-stage pipe  │
│  POST /v1/files            ──► upload video to MEDIA_STORAGE_DIR   │
│  GET  /v1/files             ──► list uploaded files                │
│  GET  /v1/files/{id}/content──► download file                     │
│  DELETE /v1/files/{id}      ──► delete file                       │
│  GET  /v1/models            ──► model info                        │
│  GET  /v1/metadata          ──► version + license info            │
│  GET  /v1/live              ──► liveness probe                    │
│  GET  /v1/startup           ──► startup probe                     │
│  GET  /v1/ready             ──► readiness (models loaded?)        │
│  GET  /v1/metrics           ──► Prometheus text format             │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# api_server.py — complete endpoint set

@app.post("/v1/chat/completions", tags=["Chat Completions"],
          response_model=Union[ChatCompletionResponse, ChatCompletionStreamResponse],
          dependencies=[Depends(validate_json_request)])
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Main inference endpoint. Supports stream=true (SSE) and stream=false.

    Pipeline flow:
      Request → SOPProcessManager.create_video_processor()
             → SOPVideoProcessor runs 4-stage pipeline (§ 6)
             → results streamed via SSE (§ 7) or collected for non-streaming response

    CRITICAL [CLEANUP_ON_DISCONNECT]: Must call trigger_stop_processors() in try/finally
    to release hardware (camera lock, GPU pipeline) on client disconnect.
    """
    ...

@app.post("/v1/files", response_model=FileObject, tags=["File and Stream Management"])
async def upload_file(file: UploadFile = File(...), purpose: str = Form(...)):
    """Upload a video file. Returns FileObject with id=file-<uuid>.
    Saved to MEDIA_STORAGE_DIR; metadata persisted to metadata.json."""
    ...

@app.get("/v1/files", response_model=FileList, tags=["File and Stream Management"])
async def list_files():
    """List all uploaded files."""
    ...

@app.get("/v1/files/{file_id}/content", tags=["File and Stream Management"])
async def get_file_content(file_id: str):
    """Download file content. Returns application/octet-stream."""
    ...

@app.delete("/v1/files/{file_id}", response_model=DeletionStatus, tags=["File and Stream Management"])
async def delete_file(file_id: str):
    """Delete a file. Returns {"id": file_id, "object": "file.deleted", "deleted": true}."""
    ...

@app.get("/v1/models", tags=["Models"])
async def list_models():
    """Returns {"object": "list", "data": [{"id": "ds_sop_model", "object": "model", ...}]}."""
    ...

@app.get("/v1/metadata", response_model=DSSOPMetadataResponse, tags=["Metadata"])
async def show_metadata():
    """Return {"version", "modelInfo", "licenseInfo"} as DSSOPMetadataResponse.

    CRITICAL [METADATA_LICENSE_FROM_FILE]: read licenseInfo from the SHIPPED license
    file; never hard-code license text and never read a path the build does not ship.
    Ship `license.txt` via `docker/package_file_list.txt` (copy_sources copies it to
    WORK_DIR `/opt/nvidia/nvds_sop/license.txt`). Default DS_SOP_LICENSE_PATH to that
    shipped path so /v1/metadata works on a clean build. Populate every
    LicenseInfoResponse key (name/path/size/url/type/content):

        path = os.environ.get("DS_SOP_LICENSE_PATH", "/opt/nvidia/nvds_sop/license.txt")
        with open(path) as f:
            content = f.read()
        licenseInfo = LicenseInfoResponse(
            name=os.path.basename(path), path=path, size=os.path.getsize(path),
            url="", type="file", content=content,
        )
    """
    ...

@app.get("/v1/live", response_model=HealthSuccessResponse, tags=["Health Check"])
async def health_live():
    """Returns {"object": "health.response", "message": "Service is live."}."""
    ...

@app.get("/v1/startup", response_model=HealthSuccessResponse, tags=["Health Check"])
async def health_startup():
    """Returns {"object": "health.response", "message": "Service started successfully."}."""
    ...

@app.get("/v1/ready", response_model=HealthSuccessResponse, tags=["Health Check"])
async def health_ready():
    """Returns ready or 503 if models not loaded."""
    ...

@app.get("/v1/metrics", tags=["Metrics"])
async def metrics():
    """Prometheus text format. Updates GPU metrics via nvidia-smi."""
    _update_gpu_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

---

## Content-Type Validation Middleware

```python
async def validate_json_request(raw_request: Request):
    """Reject non-JSON POST bodies with HTTP 415.
    Used as a FastAPI dependency on /v1/chat/completions."""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise HTTPException(status_code=415, detail="Only 'application/json' is allowed")
```

---

## Server Initialization (CRITICAL — MANAGER_INIT_IN_MAIN)

`SOPProcessManager` and model warmup MUST happen in `main()` BEFORE `uvicorn.run()`.
Do NOT put blocking `wait_for_model_ready()` inside the async `lifespan()`.

**Why**: `wait_for_model_ready()` calls `threading.Event.wait()` which blocks the asyncio
event loop. vLLM's `AsyncLLMEngine` schedules background tasks that need the event loop
to process requests. When the loop is blocked, the warmup inference request is added
but never processed, causing a 300s timeout.

```python
# api_server.py — initialization pattern
#
# Startup sequence:
#   1. main() creates SOPProcessManager (synchronous, no event loop yet)
#   2. SOPProcessManager → ModelInitializer → VLM warmup + DDM dummy pipeline
#   3. wait_for_model_ready() blocks until both models are warm
#   4. uvicorn.run() starts the asyncio event loop and HTTP server
#   5. lifespan() loads metadata (non-blocking) — NO model init here

global_sop_manager = None

def _ensure_writable_dir(path, purpose, env_var):
    """Create `path` and verify the container user can write it. The most common
    first-run failure is a uid/gid mismatch: a host bind mount auto-created as
    root:root while the container runs as the default non-root uid 1001. Raise an
    actionable error instead of an opaque permission failure deeper in the flow."""
    try:
        os.makedirs(path, exist_ok=True)
    except PermissionError:
        pass  # fall through to the unified os.access message below
    if not os.access(path, os.W_OK):
        uid, gid = os.getuid(), os.getgid()
        raise RuntimeError(
            f"No write permission for {purpose} directory: {path}\n"
            f"  The container runs as uid={uid} gid={gid}, but this path is not "
            f"writable by that user (often a host bind mount created as root:root).\n"
            f"  Fix it by either: (1) running as your host user — set USER_ID=$(id -u) "
            f"and GROUP_ID=$(id -g) in deploy/.env, or (2) granting the container user "
            f"write access to the host path backing {env_var} (e.g. chown/chmod)."
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_metadata
    # Fail fast with an actionable message if the storage dir isn't writable by
    # the container user (uid 1001 by default) — see _ensure_writable_dir.
    _ensure_writable_dir(MEDIA_STORAGE_DIR, "media storage", "MEDIA_STORAGE_DIR")
    # Emit the camera-emulation notice once at startup (the pipeline builds per
    # request, so a per-stream log would repeat). PYLON_CAMEMU defaults to 1, so a
    # customer with a real Basler camera silently runs emulation unless they opt out.
    if os.environ.get("PYLON_CAMEMU") == "1":
        logger.warning(
            "PYLON_CAMEMU=1: Basler camera EMULATION mode is enabled — live Basler "
            "cameras will NOT be used. Set PYLON_CAMEMU=0 in deploy/.env to use real "
            "camera hardware."
        )
    global_metadata = _load_metadata()
    yield                                  # NO blocking calls here
    if global_sop_manager is not None:
        global_sop_manager.close()

app = FastAPI(title="...", lifespan=lifespan)

def main():
    global global_sop_manager
    if not API_DUMMY_TEST:
        global_sop_manager = SOPProcessManager()
        global_sop_manager.wait_for_model_ready()   # safe: no event loop yet
    uvicorn.run(app, host="0.0.0.0", port=API_SERVER_PORT)
```

---

## Live Input: stream=true Required (LIVE_REQUIRES_STREAM_TRUE)

RTSP and Basler camera inputs MUST use `"stream": true`. The endpoint raises HTTP 400
for live sources with `stream=false` — live pipelines never reach EOS so a non-streaming
response would block forever:

```python
is_live = is_rtsp or is_camera
if is_live and not request.stream:
    raise HTTPException(400, "Live stream requests must enable streaming response (stream=true)")
```

---

## File Upload: Metadata Async Lock

Concurrent uploads can corrupt `metadata.json` if not serialized. Use an async lock:

```python
METADATA_PATH = os.path.join(MEDIA_STORAGE_DIR, "metadata.json")

async with global_metadata_lock:          # asyncio.Lock(), module-level
    global_metadata[file_id] = file_object.model_dump()
    with open(METADATA_PATH, "w") as f:
        json.dump(global_metadata, f)
```

On startup, load existing metadata from `METADATA_PATH` so uploaded files survive
a service restart within the same `MEDIA_STORAGE_DIR`.

---

## Chat Completion: create_video_processor Calling Convention (CRITICAL — NAMED_KWARGS)

`SOPProcessManager.create_video_processor()` takes **named kwargs** — see § 6
(`NAMED_KWARGS`) for the canonical signature and the rule. The chat-completion handler
calls it as follows (camera args are separate named kwargs, never packed into a dict):

```python
# In the chat completion endpoint handler:
#
# This dispatches a new SOPVideoProcessor for the request.
# The processor runs the full 4-stage pipeline:
#   Stage 1: DeepStream GEBD inference, e.g. DDM (§ 3)
#   Stage 2: Clip post-processing (boundary detection → chunk segmentation)
#   Stage 3: VLM inference (Cosmos Reason 1 or 2) per chunk
#   Stage 4: SOP sequence validation

# Build ChunkParams from the discriminated chunking_options union (§ 2).
# "uniform" selects fixed-length chunking and skips the DDM model (§ 3, § 6).
chunking_options = request.chunking_options
if chunking_options and chunking_options.algorithm == "ddm-net":
    chunk_params = ChunkParams(
        algorithm="ddm-net",
        min_length_sec=chunking_options.min_length_sec,
        max_length_sec=chunking_options.max_length_sec,
        threshold=chunking_options.threshold,
    )
elif chunking_options and chunking_options.algorithm == "uniform":
    chunk_params = ChunkParams(
        algorithm="uniform",
        chunk_length_sec=chunking_options.chunk_length_sec,
    )
else:
    chunk_params = ChunkParams(min_length_sec=1, max_length_sec=10, threshold=0.8)

processor = global_sop_manager.create_video_processor(
    file_path=video_file_path,       # str: file path, rtsp://, or camera://serial
    chunk_params=chunk_params,       # ChunkParams instance (MUST be named kwarg)
    camera_serial_number=cam_serial, # str or None — separate kwarg, NOT in a dict
    camera_config=cam_config,        # str or None — separate kwarg
    prompt=text_content,             # str — user's text prompt (see NOTE below)
    temperature=request.temperature,
    max_completion_tokens=request.max_completion_tokens,
    seed=request.seed,
    top_p=request.top_p,
    **camera_extra_args,             # camera_format, camera_width, camera_height,
                                     # plus rtsp_port/rtsp_path for RTSP output (see below)
)
# NOTE: The VLM prompt priority in SOPVideoProcessor.__init__ is:
#   user request text (prompt kwarg)  >  VLM_PROMPT_PATH file content
# If the request includes a {"type":"text"} item, that text overrides the config-file prompt.
# VLM_PROMPT_PATH is only used as fallback when the request contains no text.
```

---

## RTSP Streaming Output: Auto-Injection for Live Inputs (§ 18)

> **OPTIONAL (opt-in) — generate this block only when the user explicitly requests RTSP
> streaming output (§ 18).** For the default build, omit the `ENABLE_RTSP_OUTPUT`/`RTSP_PORT`
> constants and the `camera_kwargs["rtsp_port"]`/`["rtsp_path"]` auto-injection below.

When `ENABLE_RTSP_OUTPUT` is true (default), the chat-completion handler auto-injects
`rtsp_port`/`rtsp_path` into the camera kwargs for **RTSP and Basler camera** inputs, so the
service re-streams the decoded live feed at `rtsp://127.0.0.1:{RTSP_PORT}/ds-out/{sensor_id}`.
File inputs do not get RTSP output unless explicitly requested via the `--rtsp-port` CLI.

```python
ENABLE_RTSP_OUTPUT = os.getenv("ENABLE_RTSP_OUTPUT", "true").lower() in ("1", "true", "yes")
RTSP_PORT = int(os.environ.get("RTSP_PORT", 8554))

# RTSP input — sensor id is the last URL path segment (sans extension):
if url.startswith("rtsp://"):
    file_path = url
    is_rtsp = True
    if ENABLE_RTSP_OUTPUT:
        camera_kwargs["rtsp_port"] = RTSP_PORT
        camera_kwargs["rtsp_path"] = f"/ds-out/{url.split('/')[-1].split('.')[0]}"

# Basler camera — sensor id is the camera serial:
cam_serial = cam.camera_id
file_path = f"camera://{cam.camera_id}"
cam_config = cam.config
if ENABLE_RTSP_OUTPUT:
    camera_kwargs["rtsp_port"] = RTSP_PORT
    camera_kwargs["rtsp_path"] = f"/ds-out/{cam_serial}"
```

`camera_kwargs` is spread into `create_video_processor(**camera_kwargs)` (NAMED_KWARGS),
forwarding `rtsp_port`/`rtsp_path` through to `SOPVideoProcessor` (§ 6) and on to
`create_inference_pipeline()` (§ 3).

---

## Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# --- Prometheus metric definitions ---
# These track API usage and GPU health for monitoring dashboards

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["path", "method"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["path", "method"])
CHAT_COMPLETIONS_COUNT = Counter("chat_completions_total", "Number of chat completion requests")
CHAT_COMPLETIONS_LATENCY = Histogram(
    "chat_completions_latency_seconds", "Latency in seconds", buckets=[0.5, 1, 5, 10]
)
GPU_UTILIZATION = Gauge("gpu_utilization_percent", "GPU utilization percent", ["gpu"])
GPU_MEMORY = Gauge("gpu_memory_used_megabytes", "GPU memory used in MB", ["gpu"])

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Middleware: automatically records request count and latency for every endpoint."""
    REQUEST_COUNT.labels(path=request.url.path, method=request.method).inc()
    with REQUEST_LATENCY.labels(path=request.url.path, method=request.method).time():
        response = await call_next(request)
    return response

def _update_gpu_metrics():
    """Scrape nvidia-smi for GPU utilization and memory usage.
    Called on each /v1/metrics request to provide fresh data."""
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ]).decode()
        for idx, line in enumerate(out.strip().split("\n")):
            util, mem = line.split(", ")
            GPU_UTILIZATION.labels(gpu=str(idx)).set(float(util))
            GPU_MEMORY.labels(gpu=str(idx)).set(float(mem))
    except Exception:
        pass
```
