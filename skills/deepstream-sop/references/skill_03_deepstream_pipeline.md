# § 3 — DeepStream Pipeline (`ds_3d_action_pipeline.py`)

> **Generates**: `nvds_action_detector/ds_3d_action_pipeline.py`
> **Critical Rules**: VLM_REAL_GPU_FRAMES, FRAME_RETRIEVER_PRIORITY, MUX_ORIGINAL_RESOLUTION, FILE_URI_NO_DOUBLE_PREFIX, UNIFORM_CHUNKING_BYPASSES_DDM, RTSP_OUTPUT_TAPS_TEE1
> **Pipeline stage**: Stage 1 — GPU-accelerated **Generic Event Boundary Detection (GEBD)** via DeepStream + Triton CAPI. Default GEBD model is **DDM** (see § 5 for the swap contract).
> **RTSP streaming output (OPTIONAL, opt-in)**: the `tee1`-tapped RTSP output branch + `RTSPStreamingServer` are documented in [`skill_18_rtsp_streaming_output.md`](skill_18_rtsp_streaming_output.md). Generate it as part of this file **only when the user explicitly requests RTSP output**; otherwise omit the `GstRtspServer` import, `SW_ENCODER` flag, and the RTSP branch entirely (the `RTSP_OUTPUT_TAPS_TEE1` rule applies only then).

---

## Pipeline Architecture (Stage 1)

```
Stage 1: DeepStream Pipeline (GPU) — runs inside SOPVideoProcessor._run_cv_pipeline()
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  Input Source ──► nvstreammux ──► tee1                                          │
│  (file/RTSP/      (batching)       │                                            │
│   Basler cam)                      ├─► queue1 ──► nvdspreprocess               │
│                                    │              (3D temporal preprocessing)   │
│                                    │   ──► nvinferserver (Triton CAPI + DDM)    │
│                                    │   ──► queue2 ──► fakesink                  │
│                                    │         ↑ InferenceOutputTensorParser      │
│                                    │           → score_queue (boundary scores)  │
│                                    │                                            │
│                                    └─► queue3 ──► nvvideoconvert                │
│                                        ──► capsfilter (RGB/NVMM)               │
│                                        ──► appsink                             │
│                                               ↑ DecodedFrameRetriever          │
│                                                 (GPU zero-copy via dlpack)     │
│                                               → frame_queue for VLM (Stage 3)  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
       │ score_queue                              │ frame_queue
       ▼                                          ▼
  Stage 2: Clip Post-Process                 Stage 3: VLM Inference
```

---

## Pipeline branch map — follow this to read/verify `create_inference_pipeline`

> **CRITICAL [UNIFORM_CHUNKING_BYPASSES_DDM]**: build the DDM preprocess/inference branch ONLY when
> `uniform_chunk == False`. When `True`, skip only that DDM branch; keep the shared `tee1`
> fanout so frame capture and optional RTSP output can still run from the same decoded stream.
> Stage 2 cuts fixed-length chunks from frame timestamps (§ 6).
> **CRITICAL [MUX_ORIGINAL_RESOLUTION]**: set nvstreammux `width`/`height` to the source W×H, never `DS_ACTION_IN_RESOLUTION`.
> **CRITICAL [FILE_URI_NO_DOUBLE_PREFIX]**: pass a URI that already starts with `file://` through unchanged.
> **CRITICAL [FRAME_RETRIEVER_PRIORITY] / [VLM_REAL_GPU_FRAMES]**: prefer `frame_retriever` over `frame_queue`; the retriever supplies GPU zero-copy frames for the VLM.

```
create_inference_pipeline(file_path, score_queue, gpu_id, frame_queue, uniform_chunk, **kwargs)
│
├─ SOURCE  (select one by file_path prefix)
│   ├─ "camera://<serial>" → pylonsrc → caps(format,W,H,fps)
│   │                         → [videoconvert]            # only if CAMERA_FORMAT == YUY2
│   │                         → nvvideoconvert → caps(NVMM,NV12)            [is_live=True]
│   ├─ "rtsp://…"           → nvurisrcbin(uri, low-latency + reconnect)     [is_live=True]
│   └─ else (file)          → nvurisrcbin(uri = file:// + path)             [is_live=False]
│                                       └─ FILE_URI_NO_DOUBLE_PREFIX: keep an existing file:// as-is
│
├─ nvstreammux(width=mux_width, height=mux_height)        # MUX_ORIGINAL_RESOLUTION (source W×H)
│
├─ SHARED FANOUT
│   mux ─► tee1
│
├─ DDM INFERENCE branch ── built ONLY when uniform_chunk == False  (UNIFORM_CHUNKING_BYPASSES_DDM)
│   tee1 ─► queue ─► nvdspreprocess (5D tensor [1,3,SLIDING_WINDOWS_SIZE,R,R])
│         ─► nvinferserver (Triton CAPI → DDM runtime: PyTorch | TensorRT, see § 5)
│         ─► queue ─► fakesink
│              ▲ InferenceOutputTensorParser probe ─► score_queue (boundary scores)
│   └─ uniform_chunk == True → skip this DDM branch; score_queue unused
│
└─ FRAMES branch  ── built when frame_retriever or frame_queue is set (uniform REQUIRES a retriever)
    tee1 ─► queue ─► nvvideoconvert ─► capsfilter(NVMM, RGB) ─► appsink
                              ▲ DecodedFrameRetriever (GPU zero-copy via dlpack)   # VLM_REAL_GPU_FRAMES
                              └─► frame_queue ─► Stage 3 VLM     # FRAME_RETRIEVER_PRIORITY over frame_queue
```

---

## Core Pipeline Creation

```python
import socket

from pyservicemaker import (
    BatchMetadataOperator, Buffer, BufferProvider, BufferRetriever,
    ColorFormat, EOSMessage, Feeder, Pipeline, Probe, Receiver,
    StateTransitionMessage, as_tensor,
)

# --- Guarded GstRtspServer import for RTSP streaming output (§ 18) ---
# OPTIONAL (opt-in): include this import block ONLY when the user requested RTSP
# streaming output (§ 18). Omit it entirely for the default build.
# The inference graph uses pyservicemaker; the RTSP output branch needs the
# GObject GstRtspServer binding. Guard it so the module still loads where the
# typelib is absent (the RTSP branch is simply disabled in that case).
import gi

try:
    gi.require_version("Gst", "1.0")
    gi.require_version("GstRtspServer", "1.0")
    from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402,F401
except (ValueError, ImportError):
    GstRtspServer = None
    Gst = None
    print("WARNING: GstRtspServer not found, RTSP streaming will not be available")

# --- RTSP streaming-output encoder selection (§ 18) ---
# true → software H.264 (x264enc/avenc_h264/openh264enc); false → GPU nvv4l2h264enc.
SW_ENCODER = os.getenv("SW_ENCODER", "true").lower() in ("1", "true", "yes")

# --- DDM sliding window constants (configurable temporal window) ---
# The DDM model uses a temporal sliding window of SLIDING_WINDOWS_SIZE frames:
#   FRAMES_PER_SIDE frames of context on each side (checkpoint-bound; lower it for a
#   smaller-context checkpoint) + SEQUENCE_BATCH center frames (runtime grouping/stride).
# Defaults give 2*5 + 8 = 18; e.g. FRAMES_PER_SIDE=1 → 2*1 + 8 = 10.
FRAMES_PER_SIDE = int(os.getenv("FRAMES_PER_SIDE", "5"))
SEQUENCE_BATCH = int(os.getenv("SEQUENCE_BATCH", "8"))
SLIDING_WINDOWS_SIZE = 2 * FRAMES_PER_SIDE + SEQUENCE_BATCH  # = 18 with defaults

# --- Input resolution for DDM model ---
DS_ACTION_IN_RESOLUTION = int(os.getenv("DS_ACTION_IN_RESOLUTION", "224"))
DS_ACTION_IN_RESIZE_METHOD = os.getenv("DS_ACTION_IN_RESIZE_METHOD", "nearest")
DS_ACTION_IN_RESIZE_METHOD_MAP = {
    "nearest": 0, "bilinear": 1, "cubic": 2, "super": 3, "lanzos": 4
}

def create_inference_pipeline(file_path, score_queue, gpu_id=0, frame_queue=None,
                              uniform_chunk=False, **kwargs):
    """Create the Stage 1 DeepStream pipeline for DDM temporal action detection.

    Args:
        file_path: Video source — file path, "rtsp://..." URL, or "camera://<serial>"
        score_queue: Queue to receive (frame_id, pts, boundary_score) tuples
        gpu_id: GPU device index
        frame_queue: Optional queue for decoded frame capture (legacy path)
        uniform_chunk: When True, skip the DDM preprocess/inference branch entirely
                       (fixed-length chunking from frame timestamps — § 6). Frames are
                       taken straight off the mux; score_queue receives nothing.
        **kwargs:
            frame_retriever: DecodedFrameRetriever for GPU zero-copy frames (VLM_REAL_GPU_FRAMES)
                             Takes priority over frame_queue when provided (FRAME_RETRIEVER_PRIORITY)
            camera_*: Basler camera parameters (see § 8)
            rtsp_port: when set, build the RTSP streaming-output branch and serve the
                       decoded input at rtsp://127.0.0.1:<rtsp_port><rtsp_path> (see § 18)
            rtsp_path: RTSP mount path for the output stream (default "/ds-test")

    Returns:
        pyservicemaker.Pipeline ready to start
    """
    pipeline = Pipeline("ds_action_detector")
    is_camera = file_path.startswith("camera://")
    # CRITICAL [FRAME_RETRIEVER_PRIORITY]: frame_retriever takes priority over frame_queue
    frame_retriever = kwargs.get("frame_retriever", None)

    if is_camera:
        # --- Basler camera source (§ 8) ---
        # Pipeline: pylonsrc → capsfilter(raw) → [videoconvert if YUY2] → nvvideoconvert → capsfilter(NVMM/NV12)
        serial = file_path.split("camera://")[1]
        fmt = kwargs.get("camera_format", os.getenv("CAMERA_FORMAT", "RGB"))
        w = kwargs.get("camera_width", int(os.getenv("CAMERA_WIDTH", "1280")))
        h = kwargs.get("camera_height", int(os.getenv("CAMERA_HEIGHT", "720")))
        # fps + num-buffers fall back to env (CAMERA_FPS_NUM/DEN, CAMERA_NUM_BUFFERS — § 11)
        fps_num = kwargs.get("camera_fps_num") or os.getenv("CAMERA_FPS_NUM")
        fps_den = kwargs.get("camera_fps_den") or os.getenv("CAMERA_FPS_DEN")
        num_buffers = os.getenv("CAMERA_NUM_BUFFERS")

        # num-buffers makes pylonsrc emit EOS after N frames — emulation only (PYLON_CAMEMU=1)
        pylonsrc_props = {"device-serial-number": serial, "capture-error": 1}
        if num_buffers and os.getenv("PYLON_CAMEMU") == "1":
            pylonsrc_props["num-buffers"] = int(num_buffers)
        pipeline.add("pylonsrc", "pylonsrc", pylonsrc_props)
        cam_caps = f"video/x-raw, format={fmt}, width={w}, height={h}"
        if fps_num and fps_den:
            cam_caps += f", framerate={fps_num}/{fps_den}"
        pipeline.add("capsfilter", "cam_caps1", {"caps": cam_caps})

        # YUY2 format requires CPU conversion before GPU upload
        need_cpu = (fmt == "YUY2")
        if need_cpu:
            pipeline.add("videoconvert", "cpu_convert1")
        pipeline.add("nvvideoconvert", "cam_convert1", {
            "nvbuf-memory-type": 2, "gpu-id": gpu_id, "compute-hw": 1
        })
        pipeline.add("capsfilter", "cam_caps2", {
            "caps": "video/x-raw(memory:NVMM), format=NV12"
        })
        pipeline.add("queue", "last_src")

        # Set Pylon Feature Stream (.pfs) config if provided
        camera_config = kwargs.get("camera_config", "")
        if camera_config and camera_config.endswith(".pfs"):
            pipeline["pylonsrc"].set({"pfs-location": camera_config})

        if need_cpu:
            pipeline.link("pylonsrc", "cam_caps1", "cpu_convert1", "cam_convert1", "cam_caps2", "last_src")
        else:
            pipeline.link("pylonsrc", "cam_caps1", "cam_convert1", "cam_caps2", "last_src")
        is_live = True

    elif file_path.startswith("rtsp://"):
        # --- RTSP stream source ---
        pipeline.add("nvurisrcbin", "srcbin", {"uri": file_path})
        pipeline["srcbin"].set({
            "latency": 100, "leaky": 2, "max-size-buffers": 2,
            "num-extra-surfaces": 10, "init-rtsp-reconnect-interval": 10
        })
        is_live = True
    else:
        # --- File source ---
        # CRITICAL: file_path may already have "file://" prefix (passed from API as video URL).
        # Do NOT prepend "file://" again — that gives "file://file:///path" which fails to play.
        uri = file_path if file_path.startswith("file://") else "file://" + file_path
        pipeline.add("nvurisrcbin", "srcbin", {"uri": uri})
        is_live = False

    # --- Common pipeline: mux → tee → [queue1 → preprocess → infer → queue2 → fakesink] ---
    # CRITICAL [MUX_ORIGINAL_RESOLUTION]: mux must use the original video resolution (e.g. 1280×720),
    # NOT DS_ACTION_IN_RESOLUTION (224). nvdspreprocess handles DDM-specific scaling internally.
    # Using 224×224 for the mux squashes the aspect ratio → wrong smart_resize output for VLM
    # (336×336 instead of 196×364 for a 1280×720 source) → incorrect VLM classification.
    # Pass mux_width/mux_height from SOPVideoProcessor via kwargs; default to DS_ACTION_IN_RESOLUTION
    # only when original resolution is unavailable (e.g. get_media_info failed).
    mux_width = kwargs.get("mux_width", DS_ACTION_IN_RESOLUTION)
    mux_height = kwargs.get("mux_height", DS_ACTION_IN_RESOLUTION)
    pipeline.add("nvstreammux", "mux", {
        "batch-size": 1, "width": mux_width,
        "height": mux_height, "batched-push-timeout": -1,
        "live-source": is_live, "buffer-pool-size": 16, "gpu-id": gpu_id
    })
    # Link source → mux (always), then mux → tee1 shared fanout.
    src = "last_src" if is_camera else "srcbin"
    pipeline.link((src, "mux"), ("", "sink_%u"))
    pipeline.add("tee", "tee1")
    pipeline.link("mux", "tee1")
    frame_branch_src = "tee1"

    # --- DDM inference branch (skipped entirely for uniform chunking) ---
    # CRITICAL [UNIFORM_CHUNKING_BYPASSES_DDM]: with uniform_chunk=True, chunk boundaries
    # come from frame timestamps (§ 6), so the DDM preprocess/inference branch and its
    # score probe are NOT built. tee1 remains as the shared fanout for frames/RTSP.
    if not uniform_chunk:
        # nvdspreprocess: builds the 5D tensor [1, 3, SLIDING_WINDOWS_SIZE, 224, 224] for DDM
        pipeline.add("nvdspreprocess", "preprocess_3d", {
            "config-file": "configs/nvds_preprocess_rendered.txt", "gpu-id": gpu_id
        })
        # nvinferserver: runs DDM via Triton CAPI Python backend (§ 5)
        pipeline.add("nvinferserver", "inferencer", {
            "config-file-path": "configs/nvds_inference_rendered.txt"
        })
        pipeline.add("queue", "queue1")
        pipeline.add("queue", "queue2")
        pipeline.add("fakesink", "fakesink", {"sync": False, "qos": False})
        # Attach boundary score parser probe to extract DDM output scores
        meta_probe = Probe("probe", InferenceOutputTensorParser(queue=score_queue))
        pipeline.attach("queue2", meta_probe)
        pipeline.link("tee1", "queue1", "preprocess_3d", "inferencer", "queue2", "fakesink")

    # --- Optional RTSP streaming-output branch (taps tee1) — OPT-IN (§ 18) ---
    # Generate this block ONLY when the user requested RTSP output; omit it for the default build.
    # CRITICAL [RTSP_OUTPUT_TAPS_TEE1]: link from the existing tee1; never add a second tee.
    # Built only when a rtsp_port kwarg is present (live inputs / --rtsp-port). The branch is
    # leaky+tiny so it can never backpressure tee1 and starve inference. Full encoder-fallback
    # (x264enc → avenc_h264 → openh264enc → MJPEG, or HW nvv4l2h264enc), key-int-max=30 keyframe
    # interval, RTSPStreamingServer, and get_free_port() live in § 18 — generate them here too.
    rtsp_port = kwargs.get("rtsp_port", None)
    rtsp_path = kwargs.get("rtsp_path", "/ds-test")
    if rtsp_port is not None and GstRtspServer is not None:
        udp_port = get_free_port()
        pipeline.add("queue", "queue_rtsp",
                     {"leaky": 2, "max-size-buffers": 2, "max-size-bytes": 0, "max-size-time": 0})
        pipeline.add("nvvideoconvert", "convert_rtsp", {"gpu-id": gpu_id})
        # ... encoder selection per SW_ENCODER → sets rtsp_encoding "H264"/"JPEG" (see § 18) ...
        pipeline.add("rtph264pay", "pay_rtsp", {"pt": 96})   # rtpjpegpay on MJPEG fallback
        # Create the server AFTER the encoder is known so udpsrc caps match the payloader.
        pipeline.rtsp_server = RTSPStreamingServer(rtsp_port, udp_port, stream_path=rtsp_path,
                                                   encoding=rtsp_encoding)
        pipeline.add("udpsink", "sink_rtsp",
                     {"host": "127.0.0.1", "port": udp_port, "async": False, "sync": False})
        pipeline.link("tee1", "queue_rtsp", "convert_rtsp", "caps_rtsp", "enc_rtsp", "pay_rtsp", "sink_rtsp")

    # --- Optional frame capture branch (for VLM inference in Stage 3) ---
    # CRITICAL [VLM_REAL_GPU_FRAMES]: When frame_retriever (DecodedFrameRetriever) is provided,
    # it enables GPU zero-copy frame sharing via dlpack for VLM inference.
    # CRITICAL [FRAME_RETRIEVER_PRIORITY]: frame_retriever takes priority over frame_queue.
    # Uniform mode REQUIRES a retriever (it supplies the timestamps Stage 2 cuts on — § 6).
    if frame_queue is not None or frame_retriever is not None:
        pipeline.add("queue", "queue3")
        # Convert to RGB on GPU for VLM input
        pipeline.add("nvvideoconvert", "frame_converter", {
            "nvbuf-memory-type": 2, "gpu-id": gpu_id, "compute-hw": 1
        })
        pipeline.add("capsfilter", "frame_capsfilter", {
            "caps": "video/x-raw(memory:NVMM), format=RGB"
        })
        pipeline.add("appsink", "frame_sink", {
            "emit-signals": True, "sync": False, "qos": False, "async": True
        })
        pipeline.link(frame_branch_src, "queue3", "frame_converter", "frame_capsfilter", "frame_sink")
        # Use DecodedFrameRetriever (GPU zero-copy) if available, else legacy queue
        retriever = frame_retriever if frame_retriever else FrameBufferRetriever(frame_queue)
        pipeline.attach("frame_sink", Receiver("receiver", retriever), tips="new-sample")
    elif uniform_chunk:
        # Uniform mode with no retriever: terminate the mux so the pipeline can still run.
        pipeline.add("fakesink", "fakesink", {"sync": False, "qos": False})
        pipeline.link("tee1", "fakesink")

    return pipeline
```

---

## Inference Output Tensor Parser

```python
class InferenceOutputTensorParser(BatchMetadataOperator):
    """Extracts action boundary scores from DDM inference output.

    Attached as a probe to queue2 in the pipeline. For each frame:
      1. Records (frame_number, pts) in a circular buffer
      2. Extracts boundary scores from NvDsObjectMeta (created by C++ postprocess, § 5b)
      3. Puts (frame_id, pts_at_frame, confidence) into score_queue

    score_queue feeds into Stage 2: Clip Post-Process (§ 6).
    """

    def __init__(self, queue):
        super().__init__()
        self._queue = queue
        self._temporal_pts = []       # circular buffer of (frame_id, pts) tuples
        self._first_metadata = True

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            frame_num = frame_meta.frame_number
            pts = float(frame_meta.buffer_pts) / 1e9   # nanoseconds → seconds
            self._temporal_pts.append((frame_num, pts))

            # Signal first frame arrival (sentinel value -1 = not a boundary)
            if self._first_metadata:
                self._first_metadata = False
                self._queue.put((frame_num, pts, -1))

            # Keep buffer to SLIDING_WINDOWS_SIZE (default 18 frames)
            if len(self._temporal_pts) > SLIDING_WINDOWS_SIZE:
                self._temporal_pts.pop(0)

            # Extract boundary scores from object metadata
            # Each object_id maps to a frame in the sliding window;
            # confidence is the DDM boundary probability score
            boundary_items = sorted(
                [(obj.object_id, obj.confidence) for obj in frame_meta.object_items],
                key=lambda x: x[0]
            )
            for i, (frame_id, confidence) in enumerate(boundary_items):
                pts_at_frame = self._temporal_pts[FRAMES_PER_SIDE + i][1]
                self._queue.put((frame_id, pts_at_frame, confidence))
```

---

## Dummy Pipeline for Model Warmup

```python
def create_dummy_pipeline(gpu_id=0):
    """Warm up the DDM model by running SLIDING_WINDOWS_SIZE (default 18) dummy frames.

    Called by ModelInitializer (§ 6) during startup. Feeds synthetic zero-tensors
    through the full DDM pipeline to trigger Triton model loading and TensorRT
    engine build. The pipeline sends EOS after all frames, signaling warmup complete.

    Pipeline: appsrc → nvvideoconvert → capsfilter(NVMM/NV12) → nvstreammux → nvdspreprocess → nvinferserver → fakesink
    """

    class TensorBufferProvider(BufferProvider):
        """Generates SLIDING_WINDOWS_SIZE dummy RGB frames for DDM warmup."""
        def __init__(self):
            super().__init__()
            self.format = "RGB"
            self.width = DS_ACTION_IN_RESOLUTION
            self.height = DS_ACTION_IN_RESOLUTION
            self.framerate = 30
            self.device = "gpu"
            self.max_count = SLIDING_WINDOWS_SIZE
            self.frame_idx = 0

        def generate(self, size):
            if self.frame_idx >= self.max_count:
                return Buffer()  # Empty buffer signals EOS
            # Use CPU tensor (pyservicemaker workaround for torch 2.x)
            torch_tensor = torch.zeros(self.height, self.width, 3, dtype=torch.int8)
            ds_tensor = as_tensor(torch_tensor, "HWC").to_gpu(0)
            self.frame_idx += 1
            return ds_tensor.wrap(ColorFormat.RGB)

    provider = TensorBufferProvider()
    caps1 = f"video/x-raw, format=RGB, width={provider.width}, height={provider.height}, framerate={provider.framerate}/1"
    caps2 = f"video/x-raw(memory:NVMM), format=NV12, width={provider.width}, height={provider.height}, framerate={provider.framerate}/1"

    pipeline = Pipeline("ds_dummy_pipeline")
    pipeline.add("appsrc", "appsrc", {"caps": caps1, "do-timestamp": True})
    pipeline.add("nvvideoconvert", "convert_to_nv12", {"nvbuf-memory-type": 2, "gpu-id": 0, "compute-hw": 1})
    pipeline.add("capsfilter", "caps2", {"caps": caps2})
    pipeline.add("nvstreammux", "mux", {
        "batch-size": 1, "width": DS_ACTION_IN_RESOLUTION, "height": DS_ACTION_IN_RESOLUTION,
        "batched-push-timeout": -1, "live-source": False, "buffer-pool-size": 4
    })
    pipeline.add("nvdspreprocess", "preprocess_3d", {"config-file": PREPROCESS_CONFIG})
    pipeline.add("nvinferserver", "inferencer", {"config-file-path": INFERENCE_CONFIG})
    pipeline.add("queue", "queue1")
    pipeline.add("fakesink", "fakesink", {"sync": False, "qos": False})
    pipeline.attach("appsrc", Feeder("feeder", provider), tips="need-data/enough-data")
    pipeline.link("appsrc", "convert_to_nv12", "caps2")
    pipeline.link(("caps2", "mux"), ("", "sink_%u"))
    pipeline.link("mux", "preprocess_3d", "inferencer", "queue1", "fakesink")
    return pipeline
```
