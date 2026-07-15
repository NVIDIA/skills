# § 18 — RTSP Streaming Output

> **⚠️ OPTIONAL — OPT-IN FEATURE. Do NOT generate by default.**
> This section is generated **only when the user explicitly requests RTSP streaming output**
> (e.g. *"…generate a SOP microservice **with rtsp streaming output feature**…"*). When the
> user runs the plain generation prompt (`example_sop_prompt.md` without asking for RTSP),
> **skip this section entirely** — do not add the RTSP branch, the `GstRtspServer` import, the
> auto-injection, the Docker RTSP packages, or the `ENABLE_RTSP_OUTPUT`/`RTSP_PORT`/`SW_ENCODER`
> env vars. The base microservice (§§ 1, 3, 6, 9) is fully functional without any of it.

> **Generates / extends (only when requested)**: `nvds_action_detector/ds_3d_action_pipeline.py` (§ 3),
> `nvds_action_detector/ds_sop_process.py` (§ 6), `nvds_action_detector/api_server.py` (§ 1),
> `docker/Docker.build` (§ 9), `deploy/compose.yaml` (§ 9)
> **Critical Rules**: RTSP_OUTPUT_TAPS_TEE1, RTSP_LEAKY_QUEUE_TINY, RTSP_OUTPUT_ORIGINAL_RESOLUTION, RTSP_KEYINT_MAX_30, RTSP_ENCODER_FALLBACK, RTSP_OUTPUT_TEARDOWN
> **Env vars**: `ENABLE_RTSP_OUTPUT`, `RTSP_PORT`, `SW_ENCODER` (see § 11)

## When to generate this section

RTSP streaming output is an **optional, opt-in feature**. Generate it **only** when the user
asks for it, for example:

```
Please follow instructions in @example_sop_prompt.md to generate a SOP microservice
with rtsp streaming output feature in folder @ds_sop_microservice
```

Trigger phrases include *"with rtsp streaming output"*, *"re-stream the input over RTSP"*,
*"RTSP output"*, *"restream live feed"*, etc. If none of these are present, **do not generate
any of the RTSP wiring described below** — every other section already guards its RTSP block
behind this same condition.

---

RTSP streaming output re-streams the **decoded input video** over RTSP, in parallel with
the inference pipeline, so a downstream Video Management System (VMS / recorder) or a
plain player can pull the live feed the service is analysing. It is an **optional branch**
that taps the existing `tee1` element — the inference path (DDM → VLM → SOP checker) is
unchanged and unaffected.

When generated, it is a **first-class feature** (no post-generation patching). Once the
RTSP code is present, output is enabled:

- **Automatically for live inputs** (`rtsp://…` and `camera://…`) when `ENABLE_RTSP_OUTPUT`
  is true (default), at per-stream path `/ds-out/{sensor_id}`.
- **Explicitly for any input** via the `--rtsp-port` CLI flag on the standalone entry points.

When no RTSP port is resolved (e.g. file input with `ENABLE_RTSP_OUTPUT=false`), the branch
is simply not built and the pipeline behaves exactly as in § 3.

---

## Pipeline Topology (with RTSP branch)

```
                                 ┌─ queue1 ─► preprocess_3d ─► inferencer ─► queue2 ─► fakesink
Source ─► mux ─► tee1 ──────────┤
                                 ├─ queue_rtsp ─► nvvideoconvert ─► caps_rtsp ─► enc_rtsp ─► pay_rtsp ─► udpsink ──► [RTSP Server]
                                 └─ queue3 ─► frame_converter ─► frame_capsfilter ─► queue4 ─► frame_sink (DecodedFrameRetriever → VLM)
```

The RTSP branch encodes H.264 (or MJPEG fallback), RTP-payloads it, sends it to a local
`udpsink`, and an in-process `GstRtspServer.RTSPServer` republishes that UDP feed at
`rtsp://127.0.0.1:{RTSP_PORT}{rtsp_path}`.

---

## Critical Rules

- **`RTSP_OUTPUT_TAPS_TEE1`**: The RTSP branch links from the existing `tee1` element (the
  same tee that feeds the inference and frame-capture branches). Add it **after** the main
  inference link line so `tee1` already exists. Never insert a second tee or re-link the
  inference branch.

- **`RTSP_LEAKY_QUEUE_TINY`**: The RTSP queue MUST be `leaky=2` (drop oldest, downstream)
  with a **tiny** cap (`max-size-buffers=2`, others 0). Reasons:
  1. The RTSP output is best-effort. Without `leaky`, a slow encoder or a disconnected RTSP
     client backpressures `tee1` and **starves the inference branch** — DDM never scores →
     no chunks → no VLM → no messages.
  2. This queue holds NVMM frames from the shared `nvstreammux` buffer pool
     (`buffer-pool-size=16`). A large cap can hold more NVMM buffers than the whole pool,
     exhausting it and stalling inference. Keep it tiny **and** leaky.

- **`RTSP_OUTPUT_ORIGINAL_RESOLUTION`**: The RTSP branch capsfilter (`caps_rtsp`) MUST
  include explicit `width` and `height` matching the original source resolution (from
  `mux_width`/`mux_height` kwargs). Without explicit dimensions, the capsfilter inherits
  whatever the mux outputs — if the mux is misconfigured to 224×224 (e.g. if the `MUX_ORIGINAL_RESOLUTION`
  rule is violated for live sources in § 6), the RTSP stream is also 224×224 and any downstream VMS / player shows a tiny
  video. Setting explicit dimensions in `caps_rtsp` makes `nvvideoconvert` (`convert_rtsp`)
  scale to the correct output resolution as defense-in-depth.

- **`RTSP_KEYINT_MAX_30`**: The H.264 encoder MUST set `key-int-max=30` (IDR keyframe at
  least every 30 frames ≈ 3 s at 10 fps) **and disable B-frames** (`bframes=0` /
  `num-B-Frames=0`). This lets a downstream recorder/VMS seek into any recorded segment for
  clip extraction. Without a bounded keyframe interval an RTSP client that connects mid-stream
  waits indefinitely for the first IDR — it **stalls on media-info / SDP negotiation** (no
  decodable frame to report resolution/codec), and short clips cut from the recording are
  undecodable / fail to multiplex. The `x264enc` element therefore needs
  `{"key-int-max": 30, "bframes": 0, "tune": "zerolatency"}`, not just bitrate.

> **Adapting from upstream / canonical source:** the upstream `ds_3d_action_pipeline.py` RTSP
> branch may ship a plain `queue` (no `leaky`) and an encoder without `key-int-max`/`bframes`.
> When generating from canonical source rather than this section, **re-apply both
> `RTSP_LEAKY_QUEUE_TINY` and `RTSP_KEYINT_MAX_30`** — dropping them reintroduces the
> inference-starvation stall and the RTSP client media-info stall respectively.

- **`RTSP_ENCODER_FALLBACK`**: Select the encoder from `SW_ENCODER`:
  - `SW_ENCODER=true` (default, CPU): prefer `x264enc`, else `avenc_h264`, else
    `openh264enc`, else fall back to MJPEG (`jpegenc` + `rtpjpegpay`). Probe availability
    with `Gst.ElementFactory.find(...)` (call `Gst.init(None)` first).
  - `SW_ENCODER=false` (GPU): use `nvv4l2h264enc` on NVMM frames.
  Keep this preference order consistent with the deploy-time RTSP verification scripts.

---

## Conditional GStreamer-RTSP Import (`ds_3d_action_pipeline.py`)

The pipeline module uses `pyservicemaker` for the inference graph but needs the GObject
`GstRtspServer` binding for the RTSP server. Import it **guarded** so the module still loads
on hosts without the typelib (the branch is just disabled):

```python
import os
import socket
import string

import torch

import gi

try:
    gi.require_version("Gst", "1.0")
    gi.require_version("GstRtspServer", "1.0")
    from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402,F401
except (ValueError, ImportError):
    GstRtspServer = None
    Gst = None
    print("WARNING: GstRtspServer not found, RTSP streaming will not be available")
```

Module-level encoder-selection flag:

```python
# --- RTSP streaming-output encoder selection ---
# When true, use a software H.264 encoder (x264enc/avenc_h264/openh264enc) for the
# RTSP output branch; otherwise use the hardware nvv4l2h264enc.
SW_ENCODER = os.getenv("SW_ENCODER", "true").lower() in ("1", "true", "yes")
```

---

## RTSP Server + Free-Port Helper (`ds_3d_action_pipeline.py`)

Place before `create_inference_pipeline()`:

```python
def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class RTSPStreamingServer:
    def __init__(self, port, udp_port, stream_path="/ds-test", encoding="H264"):
        # Always define attributes so stop() is safe even if GstRtspServer is absent.
        self.server = None
        self.stream_path = stream_path
        self._source_id = None
        if GstRtspServer is None:
            return
        self.server = GstRtspServer.RTSPServer.new()
        self.server.set_service(str(port))
        mounts = self.server.get_mount_points()
        factory = GstRtspServer.RTSPMediaFactory.new()
        # The udpsrc caps MUST match the payloader actually feeding udp_port
        # (RTSP_ENCODER_FALLBACK): rtph264pay → encoding-name=H264, the MJPEG
        # fallback rtpjpegpay → encoding-name=JPEG. Advertising the wrong
        # encoding makes clients fail to negotiate / decode the stream.
        if encoding == "JPEG":
            caps = "application/x-rtp, media=video, clock-rate=90000, encoding-name=JPEG, payload=96"
        else:
            caps = "application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96"
        factory.set_launch(f'( udpsrc name=pay0 port={udp_port} caps="{caps}" )')
        factory.set_shared(True)
        mounts.add_factory(stream_path, factory)
        # attach() returns the GSource id of the listen socket on the default main
        # context; keep it so stop() can detach it and free the RTSP port.
        self._source_id = self.server.attach(None)
        logger.info(f"RTSP Server started at rtsp://127.0.0.1:{port}{stream_path} (encoding={encoding})")

    def stop(self):
        """Tear down the RTSP server so RTSP_PORT is freed for the next request
        (RTSP_OUTPUT_TEARDOWN). Without this, each request leaks a GstRtspServer
        bound to RTSP_PORT whose udpsrc points at a now-dead udp_port; later clients
        then hit the stale server and get no data / 503."""
        if self.server is None:
            return
        try:
            mounts = self.server.get_mount_points()
            if mounts is not None and self.stream_path:
                mounts.remove_factory(self.stream_path)
        except Exception as e:
            logger.warning(f"RTSP server mount cleanup failed: {e}")
        try:
            if self._source_id is not None:
                GLib.Source.remove(self._source_id)
        except Exception as e:
            logger.warning(f"RTSP server source removal failed: {e}")
        finally:
            self._source_id = None
            self.server = None
            logger.info(f"RTSP Server stopped for {self.stream_path}")
```

> **`RTSP_OUTPUT_TEARDOWN`**: the RTSP output server is created **per request** (per
> `create_inference_pipeline`) bound to the fixed `RTSP_PORT`. When the request's pipeline
> stops, `SOPVideoProcessor.stop()` MUST call `pipeline.rtsp_server.stop()` (after
> `self._inference_pipeline.stop()`, before dropping the reference) so the listen socket /
> `GSource` on `RTSP_PORT` is removed. Otherwise the finished request leaks a server bound to
> `RTSP_PORT` whose `udpsrc` feeds from a dead udp_port, and the **next** live request's RTSP
> output answers DESCRIBE with no data / `503 Service Unavailable`. The teardown call:
>
> ```python
> rtsp_server = getattr(self._inference_pipeline, "rtsp_server", None)
> if rtsp_server is not None:
>     await loop.run_in_executor(None, rtsp_server.stop)
> ```

> The `encoding` arg keeps the server's `udpsrc` caps in lock-step with the payloader chosen by
> `RTSP_ENCODER_FALLBACK`: pass `encoding="JPEG"` on the MJPEG fallback (`jpegenc` + `rtpjpegpay`)
> and `encoding="H264"` (default) for every hardware/software H.264 encoder path. The server is
> therefore created **after** the encoder is selected (see the branch below).

---

## RTSP Output Branch (`create_inference_pipeline`, `ds_3d_action_pipeline.py`)

Add the branch **immediately after** the main inference link line
(`pipeline.link("mux", "tee1", "queue1", "preprocess_3d", "inferencer", "queue2", "fakesink")`)
and **before** the optional frame-capture branch. The branch is built only when a
`rtsp_port` kwarg is present:

```python
    # --- Optional RTSP streaming-output branch (taps tee1) ---
    rtsp_port = kwargs.get("rtsp_port", None)
    rtsp_path = kwargs.get("rtsp_path", "/ds-test")
    if rtsp_port is not None:
        if GstRtspServer is not None:
            udp_port = get_free_port()
            use_mjpeg = False
            # RTSP_LEAKY_QUEUE_TINY: leaky + tiny cap so a slow/absent RTSP client
            # never backpressures tee1 and starves the inference branch, and so this
            # queue can't exhaust the shared nvstreammux NVMM pool.
            pipeline.add(
                "queue",
                "queue_rtsp",
                {"leaky": 2, "max-size-buffers": 2, "max-size-bytes": 0, "max-size-time": 0},
            )
            pipeline.add("nvvideoconvert", "convert_rtsp", {"gpu-id": gpu_id})

            # RTSP_OUTPUT_ORIGINAL_RESOLUTION: include explicit width/height in caps
            # so nvvideoconvert scales to the original source resolution. Without this,
            # the capsfilter inherits the mux dimensions — if the mux is 224×224 the
            # RTSP stream is also 224×224 (tiny video on downstream VMS/player).
            rtsp_w = kwargs.get("mux_width", mux_width)
            rtsp_h = kwargs.get("mux_height", mux_height)

            if SW_ENCODER:
                rtsp_caps_str = f"video/x-raw, format=I420, width={rtsp_w}, height={rtsp_h}"
                pipeline.add("capsfilter", "caps_rtsp", {"caps": rtsp_caps_str})

                if not Gst.is_initialized():
                    Gst.init(None)

                # RTSP_ENCODER_FALLBACK: x264enc → avenc_h264 → openh264enc → MJPEG
                if Gst.ElementFactory.find("x264enc"):
                    logger.info("Using x264enc for software encoding")
                    pipeline.add("x264enc", "enc_rtsp", {"bitrate": 4000, "tune": "zerolatency", "speed-preset": "superfast", "bframes": 0, "key-int-max": 30})
                elif Gst.ElementFactory.find("avenc_h264"):
                    logger.info("Using avenc_h264 for software encoding")
                    pipeline.add("avenc_h264", "enc_rtsp", {"bitrate": 4000000})
                elif Gst.ElementFactory.find("openh264enc"):
                    logger.info("Using openh264enc for software encoding")
                    pipeline.add("openh264enc", "enc_rtsp", {"bitrate": 4000000})
                else:
                    factories = Gst.Registry.get().get_feature_list(Gst.ElementFactory)
                    available_encs = [f.get_name() for f in factories if "enc" in f.get_name()]
                    logger.warning(f"No suitable H264 software encoder found. Falling back to mjpeg encoding using jpegenc. Available encoders: {available_encs}")
                    pipeline.add("jpegenc", "enc_rtsp", {})
                    use_mjpeg = True
            else:
                rtsp_caps_str = f"video/x-raw(memory:NVMM), format=NV12, width={rtsp_w}, height={rtsp_h}"
                pipeline.add("capsfilter", "caps_rtsp", {"caps": rtsp_caps_str})
                pipeline.add(
                    "nvv4l2h264enc",
                    "enc_rtsp",
                    {"bitrate": 4000000, "preset-level": 1, "insert-sps-pps": 1, "bufapi-version": 1, "num-B-Frames": 0},
                )

            # Payloader + server caps MUST agree on the encoding (RTSP_ENCODER_FALLBACK):
            # only the software path that ran out of H.264 encoders falls back to MJPEG.
            if use_mjpeg:
                pipeline.add("rtpjpegpay", "pay_rtsp", {"pt": 96})
                rtsp_encoding = "JPEG"
            else:
                pipeline.add("rtph264pay", "pay_rtsp", {"pt": 96})
                rtsp_encoding = "H264"

            # Create the RTSP server AFTER the encoder is known so its udpsrc caps
            # match the payloader feeding udp_port (H264 vs JPEG).
            pipeline.rtsp_server = RTSPStreamingServer(
                rtsp_port, udp_port, stream_path=rtsp_path, encoding=rtsp_encoding
            )

            pipeline.add("udpsink", "sink_rtsp", {"host": "127.0.0.1", "port": udp_port, "async": False, "sync": False})
            pipeline.link("tee1", "queue_rtsp", "convert_rtsp", "caps_rtsp", "enc_rtsp", "pay_rtsp", "sink_rtsp")
            logger.info(f"######## linked RTSP stream {rtsp_w}x{rtsp_h} on port {rtsp_port} (encoding={rtsp_encoding})")
        else:
            logger.warning("RTSP streaming requested but GstRtspServer is not available")
```

> `num-B-Frames=0` / `bframes=0`: B-frames break low-latency live seeking and add decode
> reordering latency. Always disable them on the RTSP output encoder.

---

## Forwarding RTSP Params (`ds_sop_process.py`, § 6)

`SOPVideoProcessor` does not pass `**kwargs` transparently to `create_inference_pipeline()`;
it assembles an explicit `pipeline_kwargs` dict. Read the RTSP params in `__init__` and
forward them in `_run_cv_pipeline`:

```python
# In SOPVideoProcessor.__init__ (alongside the other camera kwargs):
self._rtsp_port = kwargs.get("rtsp_port")
self._rtsp_path = kwargs.get("rtsp_path")
```

```python
# In _run_cv_pipeline, before create_inference_pipeline(...):
if self._rtsp_port is not None:
    pipeline_kwargs["rtsp_port"] = self._rtsp_port
if self._rtsp_path is not None:
    pipeline_kwargs["rtsp_path"] = self._rtsp_path
```

### sensor_id (per-stream tagging for messaging)

Derive a `sensor_id` once in `__init__` so every published Kafka message can be filtered
per camera/stream by a downstream dashboard. Prefer the RTSP output path stem, then the
camera serial, then the source file stem:

```python
# In SOPVideoProcessor.__init__ (after _is_live is computed):
if self._rtsp_path:
    self._sensor_id = self._rtsp_path.rstrip("/").split("/")[-1]
elif self._camera_serial_number:
    self._sensor_id = str(self._camera_serial_number)
else:
    base = os.path.basename(file_path) if file_path else ""
    self._sensor_id = base.split(".")[0] if base else "unknown"
```

Stamp it onto every published message (chunk + end-of-stream summary) in
`_publish_message` (KAFKA_USE_CREATE_PRODUCER, § 6):

```python
def _publish_message(self, chunk_result):
    if self._messager is None:
        return
    # Ensure every published message carries the sensor_id flat field.
    chunk_result.setdefault("sensor_id", self._sensor_id)
    try:
        self._messager.produce(chunk_result, request_id=self._request_uuid)
    ...
```

---

## Auto-Injection for Live Inputs (`api_server.py`, § 1)

When `ENABLE_RTSP_OUTPUT` is true (default), the chat-completion handler auto-injects
`rtsp_port`/`rtsp_path` into the camera kwargs for **RTSP and Basler camera** inputs. The
sensor id is the last URL path segment (sans extension) for RTSP, or the camera serial for
a Basler source:

```python
ENABLE_RTSP_OUTPUT = os.getenv("ENABLE_RTSP_OUTPUT", "true").lower() in ("1", "true", "yes")
RTSP_PORT = int(os.environ.get("RTSP_PORT", 8554))
```

```python
# RTSP input branch:
if url.startswith("rtsp://"):
    file_path = url
    is_rtsp = True
    if ENABLE_RTSP_OUTPUT:
        camera_kwargs["rtsp_port"] = RTSP_PORT
        camera_kwargs["rtsp_path"] = f"/ds-out/{url.split('/')[-1].split('.')[0]}"

# Basler camera branch:
cam_serial = cam.camera_id
file_path = f"camera://{cam.camera_id}"
cam_config = cam.config
if ENABLE_RTSP_OUTPUT:
    camera_kwargs["rtsp_port"] = RTSP_PORT
    camera_kwargs["rtsp_path"] = f"/ds-out/{cam_serial}"
```

`camera_kwargs` is spread into `create_video_processor(**camera_kwargs)` (NAMED_KWARGS, § 1),
which forwards `rtsp_port`/`rtsp_path` through to `SOPVideoProcessor`.

---

## Docker Packages (`docker/Docker.build`, § 9)

RTSP output needs the GStreamer RTSP-server dev lib, its GObject-introspection typelib, and
the ugly/bad/libav plugin sets (H.264 encode + RTP payload + muxing). Add them to the apt
block right after `libgstreamer-plugins-base1.0-dev`:

```dockerfile
    libgstrtspserver-1.0-dev \
    gir1.2-gst-rtsp-server-1.0 \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
```

Clear any stale GStreamer plugin registry cache so the freshly installed rtsp-server / encode
plugins are (re)discovered at runtime (anchor on the existing `pip install … qwen-vl-utils`
line):

```dockerfile
RUN rm -rf ~/.cache/gstreamer-1.0/registry.x86_64.bin
RUN pip install --no-cache-dir qwen-vl-utils==0.0.14 ipdb==0.13.13
```

---

## Compose env (`deploy/compose.yaml`, § 9)

These three MUST be added to the service `environment:` block (`COMPOSE_ENV_PASSTHROUGH`,
§ 9) — `--env-file` only substitutes `${VAR}`; a var absent here never reaches the container
and the service falls back to its in-code default (`ENABLE_RTSP_OUTPUT`→`true`,
`RTSP_PORT`→`8554`), so `.env` overrides are silently ignored.

```yaml
      # === RTSP streaming output ===
      # Re-stream live inputs over RTSP at /ds-out/{sensor_id}
      ENABLE_RTSP_OUTPUT: "${ENABLE_RTSP_OUTPUT:-true}"
      RTSP_PORT: "${RTSP_PORT:-8554}"
      # true = CPU x264enc; false = GPU nvv4l2h264enc
      SW_ENCODER: "${SW_ENCODER:-true}"
```

> For low-latency RTSP, run the container with `network_mode: host` (see the compose
> reference comments on `NETWORK_MODE`).

---

## CLI Usage

Both standalone entry points accept `--rtsp-port`:

- Standalone pipeline:
  `python -m nvds_action_detector.ds_3d_action_pipeline --video-path /path/to/video.mp4 --rtsp-port 8554`
  → `rtsp://127.0.0.1:8554/ds-test`
- SOP process:
  `python -m nvds_action_detector.ds_sop_process --video-path /path/to/video.mp4 --rtsp-port 8554`
  → `rtsp://127.0.0.1:8554/ds-out/{video_name}`

The API server reads `RTSP_PORT`/`ENABLE_RTSP_OUTPUT` from env and auto-configures RTSP
output for RTSP/Basler inputs at `/ds-out/{sensor_id}`.

### Viewing the stream

```bash
ffplay rtsp://127.0.0.1:8554/ds-out/my_camera
vlc    rtsp://127.0.0.1:8554/ds-out/my_camera
gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/ds-out/my_camera latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
```

---

## Post-Build Verification

Validate RTSP support inside the built image. The first six are the standard RTSP-component
checks (shared with deploy-time verification); fix each before moving on.

| # | Check | Validates | Fix |
|---|-------|-----------|-----|
| 1 | `GstRtspServer` importable | RTSP server typelib (`libgstrtspserver-1.0-dev`, `gir1.2-gst-rtsp-server-1.0`) | add the packages to the Dockerfile |
| 2 | `x264enc` available | Software H.264 encoder plugin registered | install `gstreamer1.0-plugins-{ugly,bad}` + `libav`, rebuild registry |
| 3 | GStreamer registry plugins | `x264enc`, `rtph264pay`, `udpsink`, `nvvideoconvert`, `jpegenc`, `rtpjpegpay` all findable | clear cache + re-inspect; add the matching plugin package |
| 4 | Shared library deps satisfied | No `not found` in `ldd` of `gstreamer-1.0/libgst*.so` (codec plugins fail silently otherwise) | reinstall `libvpx9 libzvbi0t64 libmp3lame0 libx265-199 libunibreak5 libmpg123-0t64` |
| 5 | `RTSPStreamingServer` instantiable | End-to-end RTSP server attaches to a GLib MainContext | re-check checks 1 and 3 (usually a missing typelib) |
| 6 | Live re-stream | `ffprobe rtsp://127.0.0.1:8554/ds-out/<id>` returns an H.264 video stream while a live request runs | verify `SW_ENCODER`/encoder availability and `network_mode: host` |
| 7 | Stream resolution | `ffprobe` output shows the source resolution (e.g. 1280×720), **not** 224×224 | verify `RTSP_OUTPUT_ORIGINAL_RESOLUTION`: caps_rtsp must include explicit `width`/`height`; check that `MUX_ORIGINAL_RESOLUTION` (§ 6) populates `mux_width`/`mux_height` for live sources |

Quick smoke test (inside the container):

```bash
python3 -c "import gi; gi.require_version('GstRtspServer','1.0'); from gi.repository import GstRtspServer; print('OK', GstRtspServer)"
python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print('x264enc', bool(Gst.ElementFactory.find('x264enc')))"
```

---

## Integration Note — VMS clip downloads

If a downstream consumer reports `MediaInfo.parse: Unsupported file type` when downloading
GStreamer-created clips, force base64-encoded video transfer instead of file links in that
consumer's config. This is a downstream MIME-parsing quirk, not an RTSP-output defect.
