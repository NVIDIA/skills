# § 11 — Environment Variables Reference

> All environment variables used by the SOP Inference Microservice.
> Organized by pipeline stage and subsystem.

---

## Server & API


| Variable            | Default                 | Description                                        |
| ------------------- | ----------------------- | -------------------------------------------------- |
| `API_SERVER_PORT`   | `8300`                  | FastAPI server port                                |
| `DS_SOP_VERSION`    | `1.0.0`                 | Service version string                             |
| `MEDIA_STORAGE_DIR` | `/tmp/nvds_sop_storage` | File upload storage dir                            |
| `DS_SOP_LICENSE_PATH` | `/opt/nvidia/nvds_sop/license.txt` | License file read by `/v1/metadata` (shipped via `package_file_list.txt`; METADATA_LICENSE_FROM_FILE) |
| `API_DUMMY_TEST`    | `false`                 | Enable dummy mode (no GPU needed, for API testing) |


## Stage 1: DDM / DeepStream


| Variable                     | Default                                      | Description                           |
| ---------------------------- | -------------------------------------------- | ------------------------------------- |
| `DDM_MODEL_PATH`             | `/models/gbed_models/ddm/checkpoint.pth.tar` | GEBD checkpoint path (default: DDM)   |
| `DS_ACTION_IN_RESOLUTION`    | `224`                                        | DDM input resolution (224 or 384)     |
| `DS_ACTION_IN_RESIZE_METHOD` | `nearest`                                    | Resize: nearest/bilinear/cubic/lanzos |
| `FRAMES_PER_SIDE`            | `5`                                          | DDM temporal context per side — checkpoint-bound; lower for a smaller-context ckpt |
| `SEQUENCE_BATCH`             | `8`                                          | Runtime grouping/stride; window = 2·FRAMES_PER_SIDE+SEQUENCE_BATCH (must be > 1) |
| `BOUNDARY_DELAY_FRAME`       | `8`                                          | Frame offset for boundary detection   |
| `DDM_TRT_OPTIMIZATION`       | `false`                                      | Use the optional TensorRT path for DDM (built on the fly at init, else PyTorch) (§ 5) |
| `DDM_TRT_ENGINE_OUTPUT_PATH` | `/tmp/trt_opt/ddm.engine`                    | TRT engine cache; default /tmp rebuilds per container — put under MODEL_ROOT_DIR to persist |
| `DDM_TRT_PRECISION`          | `fp32`                                       | TRT build precision: fp32/fp16/bf16 (NOT re-checked on a cached engine — delete to switch) |
| `DDM_TRT_BUILD_WORKSPACE_GB` | `4`                                          | GPU workspace (GB) for on-the-fly TRT engine build |


## Stage 2: Clip Post-Processing


| Variable          | Default | Description                               |
| ----------------- | ------- | ----------------------------------------- |
| `CV_THREAD_NUM`   | `32`    | Thread pool size for CV pipeline          |
| `CLIP_THREAD_NUM` | `32`    | Thread pool size for clip post-processing |


## Stage 3: VLM Inference


| Variable                      | Default                    | Description                                    |
| ----------------------------- | -------------------------- | ---------------------------------------------- |
| `VLLM_MODEL_PATH`             | _(none — required)_        | VLM model path or HF ID (Cosmos Reason 1 or 2). No default; startup fails fast if empty |
| `DISABLE_VLM_INFERENCE`       | `false`                    | Skip VLM inference (also disables SOP checker) |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.3`                      | Fraction of GPU memory for vLLM                |
| `VLLM_MAX_MODEL_LEN`          | `20480`                    | Max token context length                       |
| `VLLM_MAX_NUM_SEQS`           | `16`                       | Max concurrent vLLM sequences                  |
| `VLM_MAX_TOKENS`              | `256`                      | Max tokens to generate per chunk               |
| `VLM_TEMPERATURE`             | `0.2`                      | Sampling temperature                           |
| `VLM_FPS`                     | `8.0`                      | Frame sampling rate for VLM                    |
| `VLM_MAX_PIXELS`              | `0`                        | Max pixels per frame (0=no limit)              |
| `VLM_MAX_FRAMES`              | `0`                        | Max frames per chunk (0=no limit)              |
| `VLM_INFERENCE_THREAD_NUM`    | `64`                       | Thread pool size for VLM inference             |


## Stage 4: SOP Checker


| Variable              | Default                   | Description                  |
| --------------------- | ------------------------- | ---------------------------- |
| `DISABLE_SOP_CHECKER` | `false`                   | Skip SOP sequence validation |
| `ACTION_CONFIG_PATH`  | `configs/actions.json`    | Action class definitions     |
| `VLM_PROMPT_PATH`     | `configs/vlm_prompts.txt` | VLM system prompt            |


## Basler Camera (§ 8)


| Variable        | Default | Description                                          |
| --------------- | ------- | ---------------------------------------------------- |
| `CAMERA_FORMAT`        | `RGB`   | Default Basler camera format (RGB/UYVY/NV12/YUY2)     |
| `CAMERA_WIDTH`         | `1280`  | Default camera width                                  |
| `CAMERA_HEIGHT`        | `720`   | Default camera height                                 |
| `CAMERA_FPS_NUM`       | —       | Default framerate numerator (added to caps when set together with den) |
| `CAMERA_FPS_DEN`       | —       | Default framerate denominator                         |
| `CAMERA_NUM_BUFFERS`   | —       | Emit EOS after N frames; emulation only (requires `PYLON_CAMEMU=1`) |
| `PYLON_CAMEMU`         | `1`     | Set to `1` for camera emulation (serial `0815-0000`) |
| `CAMERA_EMULATION_DIR` | —       | Host dir mounted to `/opt/nvidia/nvds_sop/streams/simulation` for emulation assets |


## Kafka Messaging


| Variable               | Default               | Description                               |
| ---------------------- | --------------------- | ----------------------------------------- |
| `ENABLE_MESSAGING`     | `false`               | Kafka message publishing                  |
| `KAFKA_BROKER`         | `localhost:9092`      | Kafka broker address (default targets the bundled single-host broker; set explicitly for an external/distributed broker) |
| `SOP_MESSAGING_SCHEMA` | `JSON`                | Message format: `JSON` or `NvProtoSchema` (see § 16) |
| `DEFAULT_TOPIC`        | `mdx-vlm-captions`   | Kafka topic name (matches VSS Kibana `mdx-vlm-captions-*` index) |


## RTSP Streaming Output (§ 18) — OPTIONAL (opt-in)

> These env vars exist **only when the microservice was generated with the optional RTSP
> streaming-output feature** (the user explicitly asked for it). Omit them from the default
> build's `compose.yaml`/`.env`.

| Variable             | Default | Description                                                              |
| -------------------- | ------- | ------------------------------------------------------------------------ |
| `ENABLE_RTSP_OUTPUT` | `true`  | Auto-enable RTSP output for live (`rtsp://` / `camera://`) inputs         |
| `RTSP_PORT`          | `8554`  | RTSP server listening port; stream served at `/ds-out/{sensor_id}`        |
| `SW_ENCODER`         | `true`  | `true` → CPU `x264enc` (with `avenc_h264`/`openh264enc`/MJPEG fallback); `false` → GPU `nvv4l2h264enc` |


## Video Encoding & Alerts


| Variable                  | Default | Description                        |
| ------------------------- | ------- | ---------------------------------- |
| `ENCODE_VIDEO`            | `false` | Save chunk videos to disk          |
| `ENCODE_VIDEO_OUTPUT_DIR` | —       | Output dir for encoded chunks      |
| `ENABLE_ALERT_SOUND`      | `false` | Play alert sounds on SOP violation |


