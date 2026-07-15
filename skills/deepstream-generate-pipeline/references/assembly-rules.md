# Pipeline Assembly Rules

When the script is not available or fails, Claude can assemble the pipeline directly using these rules. These rules also serve as validation for the script output.

## Source Elements

| Source Type | Elements |
| --- | --- |
| Local video (.mp4) | `filesrc location=<path> ! qtdemux ! h264parse ! nvv4l2decoder` |
| Local video (.h264) | `filesrc location=<path> ! h264parse ! nvv4l2decoder` |
| Local video (.h265) | `filesrc location=<path> ! h265parse ! nvv4l2decoder` |
| Local image (.jpg) | `filesrc location=<path> ! jpegparse ! nvjpegdec ! nvvideoconvert` or `filesrc location=<path> ! jpegparse ! nvv4l2decoder` |
| RTSP stream | `uridecodebin uri=rtsp://<url>` |
| USB camera | `v4l2src device=/dev/video0 ! video/x-raw,width=1280,height=720 ! nvvideoconvert` |
| Test pattern | `videotestsrc num-buffers=1000 ! video/x-raw,format=NV12,width=1920,height=1080` |

> **Note:** `nvjpegdec` outputs system memory (`video/x-raw`), so it **always** needs `nvvideoconvert` after it before connecting to `nvstreammux` or any NVMM-requiring element. `nvv4l2decoder` outputs NVMM directly and does not need this conversion.

## Multi-Stream Pattern

For N > 1 sources, use `nvstreammux` with named pads:

```text
<source_0_elements> ! m.sink_0
nvstreammux name=m batch-size=<N> width=1920 height=1080 ! <rest_of_pipeline>
<source_1_elements> ! m.sink_1
...
<source_N-1_elements> ! m.sink_<N-1>
```

Always add `nvmultistreamtiler width=1920 height=1080` after inference for multi-stream.

## Inference Chain

| Mode | Elements |
| --- | --- |
| Primary only | `nvinfer config-file-path=<config> batch-size=<N> unique-id=1` |
| Primary + Secondary | `nvinfer <primary_config> ! ... ! nvinfer <secondary_config> infer-on-gie-id=1` |
| With preprocessor | `nvdspreprocess config-file=<config> ! nvinfer <config> input-tensor-meta=1` |
| Primary only (Triton) | `nvinferserver config-file-path=<config> batch-size=<N> unique-id=1` |
| Primary + Secondary (Triton) | `nvinferserver <primary_config> ! ... ! nvinferserver <secondary_config> infer-on-gie-id=1` |

## Tracker Elements

| Tracker | Config |
| --- | --- |
| NvDCF | `nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ll-config-file=config_tracker_NvDCF_perf.yml` |
| IOU | `nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ll-config-file=config_tracker_IOU.yml` |
| NvSORT | `nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ll-config-file=config_tracker_NvSORT.yml` |
| DeepSORT | `nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ll-config-file=config_tracker_DeepSORT.yml` |

Tracker goes **after primary inference** and **before secondary inference** (if any).

## Sink Elements

| Sink | Elements |
| --- | --- |
| Display (x86 dGPU) | `nvvideoconvert ! nvdsosd ! nveglglessink` |
| Display (aarch64 Jetson / SBSA) | `nvvideoconvert ! nvdsosd ! nv3dsink` |
| Save JPG | `nvvideoconvert ! nvdsosd ! nvvideoconvert ! jpegenc ! filesink location=out.jpg` |
| Save PNG | `nvvideoconvert ! nvdsosd ! nvvideoconvert ! pngenc ! filesink location=out.png` |
| Save MP4 | `nvvideoconvert ! nvdsosd ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=out.mp4` |
| Save H264 | `nvvideoconvert ! nvv4l2h264enc ! filesink location=out.h264` |
| RTSP out | `nvvideoconvert ! nvv4l2h264enc ! h264parse ! rtph264pay ! udpsink host=<ip> port=<port>` |
| Fakesink | `fakesink sync=0` |

## Extra Operations

**Placement:** Resize, rotate, flip, and crop operate on the **incoming video** and must be inserted **immediately after `nvstreammux`** (before `nvinfer`) so inference receives correctly oriented/scaled frames. The pipeline order is:

```text
nvstreammux ! nvvideoconvert <extra_ops> ! nvinfer ! ...
```

| Operation | How to add |
| --- | --- |
| Resize | `nvvideoconvert ! "video/x-raw(memory:NVMM),format=NV12,width=W,height=H"` — insert after muxer, before infer |
| Rotate/Flip | `nvvideoconvert flip-method=<0-7>` — insert after muxer, before infer. Values: 0=none, 1=ccw90, 2=180, 3=cw90, 4=hflip, 5=ur-ll, 6=vflip, 7=ul-lr |
| Crop | `nvvideoconvert src-crop=X:Y:W:H` or `dest-crop=X:Y:W:H` — insert after muxer, before infer |
| Color convert | `nvvideoconvert ! "video/x-raw,format=RGB"` |

**Example** (rotate 90° clockwise before inference):
```text
nvstreammux name=m ... ! nvvideoconvert flip-method=3 ! nvinfer ... ! nvtracker ... ! nvdsosd ! nveglglessink
```

> **Note:** `nvdsosd` is not listed here — it is automatically included in the sink chain when inference is present (see Sink Elements table above).
