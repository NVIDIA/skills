# Video Streaming Sources

## Overview

DeepStream pipelines ingest video from HTTP progressive download, HLS, MPEG-DASH, RTSP, and
local files through a single element: `nvurisrcbin`. This reference covers how to build source
pipelines for each protocol, how to scale to multiple simultaneous streams, and the codec and
format constraints that apply.

All pipeline examples use the `pyservicemaker` Python API. GStreamer
command-line equivalents are included where useful.

---

## Quick Reference

| Source type | URI format | Notes |
|---|---|---|
| Local file | `file:///path/to/video.mp4` | No server required |
| HTTP progressive (MP4) | `http://host/video.mp4` | Server must support Range requests |
| HLS (VoD or live) | `http://host/stream.m3u8` | Requires `gstreamer1.0-plugins-bad` |
| MPEG-DASH | `http://host/stream.mpd` | Requires DASH demux support; AdaptationSet selection is automatic |
| RTSP | `rtsp://camera/stream` | Live source; set `live-source=1` on `nvstreammux` |

---

## nvurisrcbin

**Purpose**: Universal source bin for URI-based video ingestion. It creates the appropriate
source, demux, parse, and decode path for URI protocols such as file, HTTP, HLS, DASH, and RTSP.
For HTTP/HLS/DASH inputs this commonly involves `souphttpsrc` plus the relevant demuxer
(`qtdemux`, `hlsdemux`, or `dashdemux`) before hardware decode.

**Key Properties**:

| Property | Type | Default | Description |
|---|---|---|---|
| `uri` | string | - | Source URI (`file://`, `http://`, `rtsp://`) |
| `gpu-id` | int | 0 | GPU device for NVDEC decoding |
| `num-buffers` | int | -1 | Limit decoded buffer count (-1 = unlimited) |
| `drop-on-latency` | bool | false | Drop frames when downstream is too slow |

**Output**: `video/x-raw(memory:NVMM)` frames in GPU memory, directly compatible
with `nvstreammux`.

**Usage (pyservicemaker)**:
```python
p.add("nvurisrcbin", "src", {"uri": "http://host/video.mp4", "gpu-id": 0})
p.link(("src", "mux"), ("", "sink_%u"))
```

**Usage (GStreamer CLI)**:
```bash
nvurisrcbin uri=http://host/video.mp4 gpu-id=0
```

For source selection, NVMM memory expectations, and `sink_%u` dynamic request-pad syntax,
follow the critical rules in [../SKILL.md](../SKILL.md#critical-rules). Audio pads from muxed
sources are not linked to `nvstreammux`; it accepts video NVMM pads only.

---

## Pipeline Topology

The downstream inference and output chain is the same for these source types. Live sources still
need live-specific muxer and sink settings; see the protocol notes below.

```
nvurisrcbin  <- URI (http://, rtsp://, file://)
     |
     | video/x-raw(memory:NVMM)
     v
nvstreammux      batch N streams; batch-size must equal source count
     v
nvinfer          TensorRT inference (object detection or classification)
     v
[nvtracker]      optional; use only when tracking or object IDs are requested
     v
nvosdbin         GPU bounding-box and label rendering
     v
nvvideoconvert
     v
capsfilter       video/x-raw(memory:NVMM), format=NV12; required before encoder
     v
nvv4l2h264enc    NVENC hardware H.264 encoder
     v
h264parse -> qtmux -> filesink      output.mp4
```

---

## Local Files

**When to use**: Video files already available inside the container or on a mounted host path.
No HTTP server is required.

Convert plain filesystem paths to `file://` URIs before passing them to `nvurisrcbin`:

```python
from pathlib import Path

uri = Path(video_path).resolve().as_uri()
p.add("nvurisrcbin", "src", {"uri": uri, "gpu-id": 0})
```

**Notes**:
- Containerized apps must mount the host directory containing the video file.
- For MP4/MOV/MKV files, `nvurisrcbin` handles demuxing internally.
- If the user explicitly needs parser-level control for raw elementary streams such as `.h264`
  or `.h265`, use the manual parser patterns in [use_cases_pipelines.md](use_cases_pipelines.md).

---

## HTTP Progressive Download

**When to use**: Local or CDN-hosted MP4 files served over plain HTTP. The simplest setup
for offline or batch processing.

**Server requirement**: `souphttpsrc` issues `Range: bytes=X-Y` requests to seek to the
MP4 `moov` atom before decoding. The server must respond with `206 Partial Content` and
include `Accept-Ranges: bytes`. Python's built-in `SimpleHTTPRequestHandler` does not
implement byte-range serving - see [Local Test Servers](#local-test-servers).

**Pipeline**:
```python
p.add("nvurisrcbin", "src", {"uri": "http://host:8080/video.mp4", "gpu-id": 0})
```

**GStreamer CLI**:
```bash
gst-launch-1.0 \
  nvurisrcbin uri=http://host:8080/video.mp4 gpu-id=0 ! mux.sink_0 \
  nvstreammux name=mux batch-size=1 width=1280 height=720 ! \
  nvinfer config-file-path=pgie.yml ! \
  nvosdbin ! nvvideoconvert ! \
  "video/x-raw(memory:NVMM),format=NV12" ! \
  nvv4l2h264enc ! h264parse ! qtmux ! filesink location=output.mp4
```

**Notes**:
- Muxed MP4 (video + audio) works without modification - audio pads are silently discarded.
- MP4 files with multiple video tracks: `qtdemux` selects the first video track.
- Verify Range support: `curl -I -H "Range: bytes=0-0" http://host:8080/video.mp4` should
  return `206 Partial Content` with `Accept-Ranges: bytes`.

---

## HLS

**When to use**: Live or VoD streams exposed as `.m3u8` playlists - CDN delivery, broadcast
workflows, or local live-stream simulation. No Range request support is required; HLS segments
are complete files.

**The pipeline code is identical to HTTP progressive** - only the URI changes:

```python
# HTTP MP4
p.add("nvurisrcbin", "src", {"uri": "http://host:8080/video.mp4", "gpu-id": 0})

# HLS VoD or live - only the URL changes
p.add("nvurisrcbin", "src", {"uri": "http://host:8080/stream.m3u8", "gpu-id": 0})
```

GStreamer's `hlsdemux` (from `gstreamer1.0-plugins-bad`) is selected automatically from the
URI. Output remains `video/x-raw(memory:NVMM)`; everything downstream is unchanged.

**Container package requirement**: `gstreamer1.0-plugins-bad` must be installed. See
[docker_containers.md](docker_containers.md).

**Known limitations**:

| Limitation | Detail |
|---|---|
| Master playlist variant selection | `hlsdemux` picks a quality level automatically; bitrate/resolution targeting is not configurable through `nvurisrcbin` properties |
| CMAF / fMP4 segments | Modern HLS increasingly uses `.mp4`/`.m4s` segments instead of MPEG-TS; behavior is not covered by this reference |
| Encrypted segments (`EXT-X-KEY`) | Encryption handling is not covered by this reference |
| Separate audio renditions (`EXT-X-MEDIA`) | Emitted as dynamic pads and silently discarded |

**Notes**:
- Direct CDN `.m3u8` URLs can be passed straight to `nvurisrcbin` - no local server needed.
- For local testing, generate HLS segments with ffmpeg as shown in
  [Local Test Servers](#local-test-servers).
- For live HLS, set `live-source=1` on `nvstreammux`; use `sync=0` on display sinks.

---

## MPEG-DASH

**When to use**: Sources that expose a `.mpd` manifest URL directly.

```python
p.add("nvurisrcbin", "src", {"uri": "http://host/stream.mpd", "gpu-id": 0})
```

GStreamer's `dashdemux` is selected automatically when the DASH plugin is available. Output is
`video/x-raw(memory:NVMM)`.

**Known limitations**:

| Limitation | Detail |
|---|---|
| AdaptationSet selection | `dashdemux` selects a video AdaptationSet automatically; codec and bitrate targeting is not configurable |
| Codec variants | If the manifest lists AV1 and H.264 AdaptationSets, `dashdemux` may select AV1, which NVDEC cannot decode on Turing/Ampere GPUs (see [Codec Support](#codec-support)) |
| Audio AdaptationSets | Emitted as dynamic pads and silently discarded |
| Subtitle / text AdaptationSets | Behavior is untested |

**Notes**:
- Standard DASH streams with a proper `.mpd` manifest URL pass directly to `nvurisrcbin` -
  no intermediate download step is required.
- DASH is not suitable for sources that only expose a CDN segment URL without a manifest.

---

## RTSP

**When to use**: Live camera streams or RTSP servers.

```python
p.add("nvurisrcbin", "src", {"uri": "rtsp://camera/stream", "gpu-id": 0})
p.add("nvstreammux", "mux", {
    "batch-size": 1,
    "width": 1280,
    "height": 720,
    "live-source": 1,
    "batched-push-timeout": 33000,
    "gpu-id": 0,
})
```

**Notes**:
- Set `live-source=1` on `nvstreammux` for RTSP inputs.
- Use `sync=0` on display sinks for live pipelines to avoid clock-related stalls.
- Put credentials in the URI only when appropriate for the deployment; otherwise use the
  application's credential handling path.
- Validate camera URLs with a simple player before debugging the DeepStream pipeline.
- For multi-RTSP inference examples, see [use_cases_pipelines.md](use_cases_pipelines.md).

---

## Multi-Stream (Batched Inference)

Set `batch-size` on `nvstreammux` to N and add N `nvurisrcbin` sources. Sources can mix
protocols - HTTP, HLS, RTSP, and local files can coexist in the same batch:

```python
urls = [
    "http://host/cam1.mp4",
    "http://host/stream.m3u8",
    "rtsp://camera/stream",
]
n = len(urls)

p.add("nvstreammux", "mux", {
    "batch-size": n,
    "width": 1280,
    "height": 720,
    "batched-push-timeout": 33000,   # microseconds; about 33 ms at 30 fps
    "gpu-id": 0,
})

for i, url in enumerate(urls):
    name = f"src{i}"
    p.add("nvurisrcbin", name, {"uri": url, "gpu-id": 0})
    p.link((name, "mux"), ("", "sink_%u"))

p.link("mux", "infer", ...)
```

**Notes**:
- `batch-size` on `nvinfer` must match `nvstreammux` `batch-size`.
- `batched-push-timeout` controls how long `nvstreammux` waits for lagging streams before
  pushing a partial batch.
- For live sources (RTSP, live HLS), set `live-source=1` on `nvstreammux` and `sync=0`
  on display sinks.
- Each source's detections carry a `source_id` field in `NvDsFrameMeta` matching its
  `sink_%u` slot index.

---

## Inference Configuration

`nvinfer` is model-agnostic: source handling does not change when the model changes. Pass the
model-specific YAML or INI file with `config-file-path`:

```python
p.add("nvinfer", "infer", {"config-file-path": pgie_cfg})
```

For nvinfer config keys, detector/classifier `network-type` rules, engine caching, dynamic
shape `infer-dims`, and custom parser requirements, use
[nvinfer_config.md](nvinfer_config.md). The bundled TrafficCamNet sample config is under the
unversioned DeepStream root:

```text
/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml
```

---

## Codec Support

Codec support depends on the target GPU or Jetson generation and on the decoder selected by
GStreamer. For x86 dGPU source selection, use this quick reference when choosing online stream
variants:

| Codec | Turing / Ampere dGPU | Ada Lovelace dGPU (RTX 40+) |
|---|---|---|
| H.264 | yes | yes |
| H.265 | yes | yes |
| VP9 | yes | yes |
| AV1 | no | yes |

When a platform or CDN offers multiple encodings, prefer H.264 or H.265 for the widest DeepStream
compatibility. Avoid adding decoder plugin-rank overrides in source examples; keep decoder
selection and troubleshooting guidance in [gstreamer_plugins.md](gstreamer_plugins.md) and
[docker_containers.md](docker_containers.md).

---

## Local Test Servers

### HTTP Server (Range-aware)

For local MP4 testing, `souphttpsrc` requires byte-range serving. The server must return
`206 Partial Content` and `Accept-Ranges: bytes` when the client sends a `Range` header.
If no Range-aware server is already available, create this local helper as `range_server.py`.
It implements the single byte-range requests needed for local MP4 testing; it is not a production
HTTP server.

```python
#!/usr/bin/env python3
import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()

        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(f.fileno()).st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")

        if range_header:
            try:
                unit, spec = range_header.split("=", 1)
                if unit.strip() != "bytes":
                    raise ValueError
                start_s, end_s = spec.split("-", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                if start < 0 or end >= size or start > end:
                    raise ValueError
            except ValueError:
                f.close()
                self.send_error(416, "Invalid Range")
                return None
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)

        self.range = (start, end)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        f.seek(start)
        return f

    def copyfile(self, source, outputfile):
        start, end = getattr(self, "range", (0, None))
        if end is None:
            return super().copyfile(source, outputfile)
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    handler = partial(RangeRequestHandler, directory=args.dir)
    ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()
```

Run and verify before starting the pipeline:

```bash
python3 range_server.py --dir /path/to/videos --port 8080
curl -I -H "Range: bytes=0-0" http://localhost:8080/sample.mp4
```

The curl response must include `206 Partial Content` and `Accept-Ranges: bytes`.

### HLS Server

For local HLS testing, generate segments with ffmpeg and serve the generated directory. No
custom Python helper is required.

```bash
mkdir -p hls
ffmpeg -y -i sample.mp4 -c:v libx264 -profile:v high -pix_fmt yuv420p -an \
  -f hls -hls_time 2 -hls_list_size 0 hls/stream.m3u8
python3 -m http.server 8080 --directory hls
```

Pipeline URL: `http://localhost:8080/stream.m3u8`

For live-style testing, run ffmpeg and the HTTP server in separate terminals. Start the server
first; it can serve the directory while ffmpeg updates the playlist and segments.

Terminal 1:
```bash
mkdir -p hls
python3 -m http.server 8080 --directory hls
```

Terminal 2:
```bash
mkdir -p hls
ffmpeg -re -stream_loop -1 -i sample.mp4 -c:v libx264 -profile:v high \
  -pix_fmt yuv420p -an -f hls -hls_time 2 -hls_list_size 3 \
  -hls_flags delete_segments hls/stream.m3u8
```

### Generate a Synthetic Test Video

```bash
ffmpeg -f lavfi -i "testsrc2=duration=30:size=1280x720:rate=30" \
       -c:v libx264 -profile:v high -pix_fmt yuv420p -preset fast \
       -an sample.mp4
```

---

## Docker and Container Notes

For Docker image selection, pyservicemaker installation, GPU runtime flags, codec package
installation, environment variables, and common container failures, use
[docker_containers.md](docker_containers.md).

Source-specific container notes:

- HLS requires `gstreamer1.0-plugins-bad` for `hlsdemux`.
- DASH requires the GStreamer DASH demuxer; install `gstreamer1.0-plugins-bad` if it is missing.
- HTTP MP4 inputs may require the codec packages covered in `docker_containers.md` if the
  source includes audio or codecs stripped from the base image.
- Install pyservicemaker with the wildcard wheel path documented in `docker_containers.md`:
  `/opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl`.

---

## Related Pipeline Patterns

For a compact URI-source detection-to-MP4 pattern, see
[URI Source Inference Pattern](use_cases_pipelines.md#uri-source-inference-pattern).
Use this file for source URI selection, protocol constraints, and local stream setup details.
