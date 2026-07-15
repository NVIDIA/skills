# Step 1 — Requirement Extraction Reference

Detailed lookup tables, the full `AskUserQuestion` question bank, and the
extraction/question-reduction examples for **Step 1 — Collect Pipeline
Requirements** of the DeepStream Pipeline Builder skill. SKILL.md instructs the
agent to read this file before performing Step 1; apply everything here exactly.

---

**First, extract what you already know from the user's query.** Parse the original request for any parameters that are obvious — do NOT re-ask what's already clear. Be aggressive about inferring from context: if the user says "infer on 3 videos and display" you already know input type (video), num sources (3), inference (primary), and sink (display) — that's 4 out of 7 parameters resolved without asking.

| If the query mentions... | You already know |
| --- | --- |
| "image", "jpg", "png", "picture", "photo" | Input = Local image file, Num sources = 1 (unless stated otherwise) |
| "video", "mp4", "h264", "h265", "clip" | Input = Local video file |
| "rtsp", "stream", "camera stream", "ip camera" | Input = RTSP stream |
| "usb camera", "webcam", "/dev/video" | Input = USB camera |
| "infer", "detect", "inference", "detection", "model" | Inference = Primary detector |
| "classify", "secondary", "vehicle type", "car color" | Inference = Primary + Secondary |
| "triton", "nvinferserver", "inference server" | Inference = Primary detector (Triton) |
| "track", "tracker", "tracking", "re-id" | Tracker = present (ask which one) |
| "display", "show", "render", "screen", "view" | Sink = Display |
| "save", "output file", "write to file", "record", "store" | Sink = Save (ask which format) |
| "benchmark", "throughput", "fps", "performance" | Sink = Fakesink |
| "jetson", "orin", "xavier", "nano" | Platform = Jetson |
| "dgpu", "server", "T4", "A100", "RTX", "L40" | Platform = dGPU |
| "N streams", "N sources", "N videos", "N cameras" | Num sources = N |
| A number followed by "video/stream/camera/file" | Num sources = that number AND Input = corresponding type |
| "rotate 90", "rotate clockwise", "cw90" | Extras = Rotate, flip-method = 3 |
| "rotate 180", "flip 180" | Extras = Rotate, flip-method = 2 |
| "rotate 270", "rotate counter-clockwise", "ccw90" | Extras = Rotate, flip-method = 1 |
| "mirror", "horizontal flip", "hflip" | Extras = Rotate, flip-method = 4 |
| "vertical flip", "vflip", "flip upside down" | Extras = Rotate, flip-method = 6 |
| "rotate", "flip" (without specific direction) | Extras = Rotate (ask which direction) |
| "resize", "scale", "resolution" | Extras = Resize (ask target W x H) |
| "crop" | Extras = Crop (ask X:Y:W:H) |

> **Compound extraction examples:**
> - *"infer on 3 videos"* → Input = Local video file, Num sources = 3, Inference = Primary detector
> - *"detect and track objects on rtsp stream"* → Input = RTSP, Inference = Primary detector, Tracker = present
> - *"run inference on a jpg and save to png"* → Input = Local image file, Num sources = 1, Inference = Primary detector, Sink = Save PNG
> - *"4 camera streams with detection on jetson"* → Input = RTSP, Num sources = 4, Inference = Primary detector, Platform = Jetson
> - *"rotate 90 clockwise, infer on 2 videos and display"* → Input = video, Num = 2, Inference = primary, Sink = display, Extras = rotate flip-method=3 (placed after muxer, before infer)
> - *"infer on video and flip horizontally"* → Input = video, Num = 1, Inference = primary, Extras = rotate flip-method=4 (placed after muxer, before infer)

Use `AskUserQuestion` to ask **only the remaining unknown parameters** in a single call. **Never re-ask a parameter that can be inferred from the query.** Skip any question whose answer is already clear.

> **Important:** Ask all unknown questions in one call. Do NOT ask one at a time. If every parameter is already inferable from the query, skip the question call entirely and jump to Step 2. For most user queries, you should be able to resolve 3–5 parameters automatically, leaving only 2–3 questions.
>
> **Default-first ordering convention:** the first `option` in every question's `options` array is the **safe default** for that parameter (e.g. `Local video file` for input, `1` for num sources, `No tracker` for tracker, `Display on screen` for sink, `dGPU` for platform, `None` for extras). Most prompt UIs render the first option as the highlighted/initial selection, so a user who just hits Enter lands on a sensible choice. **Do not reorder** the options in the question bank below — preserving "default first" is part of the contract.
>
> **Critical — never get stuck asking.** If the user rejects/dismisses the `AskUserQuestion` call (e.g. "Tool use rejected"), or replies *"just generate"* / *"use defaults"* / *"go ahead"* / *"skip"* / no answer, **immediately fall through to Step 2** using the **first option** of each unknown question as the parameter value. Do NOT re-ask the same questions in chat — that creates a loop and frustrates the user. The user can always refine afterwards by saying *"change to NvDCF"*, *"save as mp4"*, etc., once they see the generated pipeline. The flow is **ask once, then generate** — never ask twice.

Below is the **full question bank** — include only the questions you actually need:

```json
{
  "questions": [
    {
      "id": "input_source",
      "question": "What is the input source type?",
      "header": "Input Source",
      "options": [
        {"label": "Local video file", "description": "filesrc with local .mp4/.h264/.h265 file"},
        {"label": "Local image file", "description": "filesrc with local .jpg/.png image"},
        {"label": "RTSP stream", "description": "uridecodebin/rtspsrc with rtsp:// URL"},
        {"label": "USB camera", "description": "v4l2src from /dev/video* device"},
        {"label": "Test pattern", "description": "videotestsrc for testing without real input"}
      ],
      "multiSelect": false
    },
    {
      "id": "num_sources",
      "question": "How many input sources/streams?",
      "header": "Number of Sources",
      "options": [
        {"label": "1", "description": "Single stream"},
        {"label": "2", "description": "Dual stream (tiled output)"},
        {"label": "4", "description": "Quad stream (2x2 tile)"},
        {"label": "8", "description": "8 streams (2x4 tile)"}
      ],
      "multiSelect": false
    },
    {
      "id": "inference",
      "question": "What inference/detection do you need?",
      "header": "Inference Model",
      "options": [
        {"label": "None", "description": "No inference — just decode/convert/display"},
        {"label": "Primary detector (nvinfer)", "description": "Single primary inference (e.g. object detection)"},
        {"label": "Primary + Secondary (nvinfer)", "description": "Primary detection + secondary classification (e.g. vehicle type)"},
        {"label": "Primary with preprocessor", "description": "nvdspreprocess + nvinfer for custom ROI/batching"},
        {"label": "Primary + Secondary with preprocessor", "description": "nvdspreprocess before both primary and secondary infer"},
        {"label": "Primary detector (nvinferserver/Triton)", "description": "Single primary inference via Triton Inference Server (nvinferserver)"},
        {"label": "Primary + Secondary (nvinferserver/Triton)", "description": "Primary + secondary classification via Triton (nvinferserver)"}
      ],
      "multiSelect": false
    },
    {
      "id": "tracker",
      "question": "Do you need object tracking?",
      "header": "Tracker",
      "options": [
        {"label": "No tracker", "description": "Skip tracking — inference only"},
        {"label": "NvDCF (accurate)", "description": "Discriminative Correlation Filter — best accuracy, higher compute"},
        {"label": "IOU (fast)", "description": "Intersection-over-Union tracker — lightweight, fast"},
        {"label": "NvSORT", "description": "NVIDIA SORT tracker — good balance of speed and accuracy"},
        {"label": "DeepSORT", "description": "Deep association metric — re-ID based, best for occlusion"}
      ],
      "multiSelect": false
    },
    {
      "id": "sink",
      "question": "What should happen with the output?",
      "header": "Output / Sink",
      "options": [
        {"label": "Display on screen", "description": "nveglglessink — render to display (dGPU)"},
        {"label": "Display on Jetson", "description": "nv3dsink — render on Jetson display"},
        {"label": "Save to JPG file", "description": "Encode to JPEG image via jpegenc + filesink"},
        {"label": "Save to PNG file", "description": "Encode to PNG image via pngenc + filesink"},
        {"label": "Save to MP4 file", "description": "Encode H264 + mux to .mp4 via filesink"},
        {"label": "Save to H264 file", "description": "Encode to raw .h264 bitstream file"},
        {"label": "Stream over RTSP", "description": "Encode and push to RTSP server via udpsink"},
        {"label": "Fakesink (benchmark)", "description": "fakesink — discard output, measure throughput"}
      ],
      "multiSelect": false
    },
    {
      "id": "platform",
      "question": "Which platform are you targeting?",
      "header": "Platform",
      "options": [
        {"label": "x86 dGPU", "description": "x86_64 desktop/server with discrete GPU (T4, A100, L40, RTX, etc.) — uses nveglglessink"},
        {"label": "aarch64 (Jetson / SBSA)", "description": "ARM aarch64 — Jetson Orin/Xavier/Nano and SBSA servers (e.g. Grace, GH200) — both use nv3dsink and nvv4l2* plugins"}
      ],
      "multiSelect": false
    },
    {
      "id": "extras",
      "question": "Any extra operations? (optional — select all that apply)",
      "header": "Extra Operations",
      "options": [
        {"label": "None", "description": "No extra processing needed"},
        {"label": "Resize / scale video", "description": "Change resolution via nvvideoconvert caps"},
        {"label": "Rotate / flip video", "description": "Flip method via nvvideoconvert (90/180/270/horizontal/vertical)"},
        {"label": "Crop input region", "description": "src-crop or dest-crop via nvvideoconvert"},
        {"label": "Color format conversion", "description": "Convert between NV12, RGBA, I420, BGR, etc."}
      ],
      "multiSelect": true
    }
  ]
}
```

> **OSD is automatic:** When inference is present and the sink is display or file save (MP4/JPG/PNG), `nvdsosd` is always included — do NOT ask the user about it. OSD is only omitted for raw bitstream sinks (H264), RTSP out, and fakesink.

> **Follow-up for extras:** If the user selects "Rotate / flip video", ask which rotation they want before proceeding:
>
> | Option | `flip-method` value |
> | --- | --- |
> | Rotate 90° counter-clockwise | 1 |
> | Rotate 180° | 2 |
> | Rotate 90° clockwise | 3 |
> | Horizontal flip (mirror) | 4 |
> | Vertical flip | 6 |
>
> If the user selects "Resize / scale video", ask for the target width and height (e.g., 1280x720).
> If the user selects "Crop input region", ask for the crop rectangle as X:Y:W:H.
>
> **Placement:** Rotate, resize, and crop transform the **incoming video** — insert `nvvideoconvert` with these properties **immediately after `nvstreammux`, before `nvinfer`**, so inference receives correctly oriented/scaled frames. Do NOT place them after the tiler or before the sink.

**Examples of dynamic question reduction:**

| User query | Already known | Questions to ask |
| --- | --- | --- |
| *"give me the pipeline to infer on an image"* | Input = image, Num = 1, Inference = primary | tracker, sink, platform, extras (4 questions) |
| *"detect and track on 3 videos and display on jetson"* | Input = video, Num = 3, Inference = primary, Tracker = present, Sink = display, Platform = Jetson | tracker type, extras (2 questions) |
| *"benchmark inference throughput on 8 rtsp streams"* | Input = RTSP, Num = 8, Inference = primary, Sink = fakesink | tracker, platform, extras (3 questions) |
| *"just decode and display an mp4"* | Input = video, Num = 1, Inference = none, Sink = display | platform, extras (2 questions) |
| *"infer on 3 videos, rotate 90 clockwise, save mp4"* | Input = video, Num = 3, Inference = primary, Sink = save MP4, Extras = rotate flip-method=3 | tracker, platform (2 questions) |
| *"build a pipeline"* | Nothing known | All 7 questions |
