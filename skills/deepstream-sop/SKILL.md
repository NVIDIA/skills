---
name: "deepstream-sop"
description: >
  Use this skill when building, deploying, evaluating, debugging, or measuring
  latency for the DeepStream SOP Inference Microservice — a GPU-accelerated
  FastAPI service that detects whether operators perform assembly-line steps in
  order via event boundary detection (GEBD) plus VLM classification. Trigger
  even if the user does not name it: verify operator step sequence, detect
  missing or out-of-order SOP steps, score factory/work-cell video for
  procedure compliance, run VLM-based SOP checking on industrial cameras, or
  call /v1/chat/completions with a file, RTSP, or Basler camera. Also trigger
  for its internals: SOPVideoProcessor, DeepStream GEBD model (e.g. DDM) via
  Triton CAPI, nvds_custom_postprocess, Cosmos Reason 1/2 vLLM, SSE streaming,
  Kafka NvProto/JSON output, Basler/Pylon camera + emulation, Docker compose,
  chunk-level latency. Do NOT trigger for generic DeepStream pipelines, object
  detection/tracking, NIM imports, or video summarization.
owner: "windy@nvidia.com"
service: "deepstream-sop"
version: "1.0.0"
license: "CC-BY-4.0 AND Apache-2.0"
reviewed: "2026-04-08"
metadata:
  author: "Wind Yuan <windy@nvidia.com>"
  tags:
    - deepstream
    - sop
    - vlm
    - triton
    - gpu
  languages:
    - python
  frameworks:
    - deepstream
    - triton
    - fastapi
  domain: video-analytics
---
# DeepStream SOP Inference Microservice Skill

This skill guides AI coding assistants in building, extending, and debugging the
**NVIDIA DeepStream SOP (Standard Operating Procedure) Inference Microservice** —
a GPU-accelerated pipeline for temporal action detection and VLM-based SOP compliance
monitoring on industrial video feeds.

**Reference repository**: https://github.com/NVIDIA/sop-monitoring-blueprints/tree/main/microservices/sop-inference-bp
**Local reference code**: `sop-inference-bp/` directory (from a local clone of the repository)

---

## Models

Model-agnostic at both inference stages — swap via env var (and Triton dir for GEBD).

| Stage | Role | Model class | Default | Swap via |
|------|------|-------------|---------|----------|
| Stage 1 (CV) | Per-frame boundary scoring → chunk segmentation | **Generic Event Boundary Detection (GEBD)** | **DDM** ([MCG-NJU/DDM](https://github.com/MCG-NJU/DDM)) via Triton Python backend | Replace `triton_model_repo/<model>/` + `DDM_MODEL_PATH` (§ 5) |
| Stage 3 (VLM) | Per-chunk action classification | Vision-language model via vLLM | **Cosmos Reason 1 7B** (Reason 2 also supported) | Set `VLLM_MODEL_PATH` to a different HF ID or local path |

"GEBD" = swappable Stage-1 slot; "DDM" = the default architecture (terms used interchangeably).

**Chunking is selectable per request** (§ 2): default `ddm-net` uses GEBD; `uniform` produces fixed-length chunks and **bypasses Stage-1 GEBD** (§ 3, § 6). DDM temporal window is configurable via `FRAMES_PER_SIDE` / `SEQUENCE_BATCH` (§ 4, § 5), with optional **TensorRT** (§ 5).

---

## Architecture Overview

Runs in a Docker container (`nvds-action-sop`) alongside a Kafka container. Full diagram: [`references/sop_architecture.svg`](references/sop_architecture.svg).

**Data flow through the 4-stage `SOPVideoProcessor` pipeline (per-request):**

```
Input Sources                    Docker Container: nvds-action-sop
─────────────                    ──────────────────────────────────────────────────
Video Files ──┐                  FastAPI Server (port 8300)
RTSP Streams ─┤── base64/       ├─ /v1/chat/completions → SOPProcessManager
Basler Camera ┘   file/rtsp/       │
                  camera           │ ModelInitializer: VLM first, then DDM dummy pipeline
                                   │ 4 Thread Pools: cv(32), clip(32), vlm(64), vlm_req(64)
                                   │
                                   ▼ SOPVideoProcessor (per-request)
                                   ┌────────────────────────────────────────────────┐
                                   │ Stage 1: DeepStream Pipeline (GPU)             │
                                   │   Source → nvstreammux → tee1                  │
                                   │    ├─[inference] queue1 → nvdspreprocess       │
                                   │    │  → nvinferserver (Triton CAPI + DDM)      │
                                   │    │  → InferOutputTensorParser → score_queue  │
                                   │    ├─[frames]  queue3 → nvvideoconvert         │
                                   │    │  → capsfilter → appsink                   │
                                   │    │  → DecodedFrameRetriever → frame_queue    │
                                   │    └─[RTSP out] queue → convert → H.264 enc    │  (optional, § 18)
                                   │       → rtppay → udpsink → RTSPServer (§ 18)   │  opt-in only
                                   │              │ boundary scores                 │
                                   │              ▼                                 │
                                   │ Stage 2: Clip Post-Process                     │
                                   │   Boundary detection → chunk segmentation      │
                                   │              │ video frames + timestamps        │
                                   │              ▼                                 │
                                   │ Stage 3: VLM Inference                         │
                                   │   Embedded vLLM (Cosmos Reason 1/2)            │
                                   │   Frame sampling at VLM_FPS → classification   │
                                   │              │ action labels                    │
                                   │              ▼                                 │
                                   │ Stage 4: SOP Checker                           │
                                   │   Sequence validation → missing/misordered     │
                                   │              │ chunk results                    │
                                   │              ▼                                 │
                                   │         final_queue                            │
                                   └────────────────────────────────────────────────┘
                                          │
Output                                    ▼
──────                             ┌─────────────────┐
SSE Stream (chat.completion.chunk) │ Kafka Messages   │
Non-streaming (chat.completion)    │ (JSON/Protobuf)  │
Prometheus metrics (/v1/metrics)   └────────┬────────┘
                                            ▼
                                   Docker Container: kafka
                                   (apache/kafka:3.7.0)
```

---

## Section Index

Each section is a standalone file in `references/` — load only what your task needs.

| § | File | Responsibility |
|---|------|---------------|
| 1 | [`skill_01_fastapi_endpoints.md`](references/skill_01_fastapi_endpoints.md) | FastAPI endpoints, server init, Prometheus metrics |
| 2 | [`skill_02_pydantic_schemas.md`](references/skill_02_pydantic_schemas.md) | Request/response Pydantic models (`api_types.py`) |
| 3 | [`skill_03_deepstream_pipeline.md`](references/skill_03_deepstream_pipeline.md) | DeepStream pyservicemaker pipeline, tensor parser, dummy pipeline |
| 4 | [`skill_04_config_templates.md`](references/skill_04_config_templates.md) | nvdspreprocess / nvinferserver config templates + rendering |
| 5 | [`skill_05_triton_ddm_model.md`](references/skill_05_triton_ddm_model.md) | Triton model repo, config.pbtxt, model.py, ddm_net.py |
| 5b | [`skill_05b_custom_postprocess.md`](references/skill_05b_custom_postprocess.md) | C++ postprocess plugin, Makefile, IOptions API |
| 6 | [`skill_06_sop_process_manager.md`](references/skill_06_sop_process_manager.md) | SOPProcessManager, SOPVideoProcessor, VLLMInference, Kafka |
| 6b | [`skill_06b_sop_checker.md`](references/skill_06b_sop_checker.md) | SOP sequence and checker compliance: MissingNumberDetector, SopCheckerCache, SopCheckerRequest/Response |
| 7 | [`skill_07_sse_streaming.md`](references/skill_07_sse_streaming.md) | SSE generator, stream response formatting, dummy test mode |
| 8 | [`skill_08_basler_camera.md`](references/skill_08_basler_camera.md) | Basler camera support, Pylon SDK, emulation, formats |
| 9 | [`skill_09_docker_build_deploy.md`](references/skill_09_docker_build_deploy.md) | Docker build, deploy, .env configuration |
| 10 | [`skill_10_test_suite.md`](references/skill_10_test_suite.md) | Test suite coverage, assertions, running tests |
| 11 | [`skill_11_env_variables.md`](references/skill_11_env_variables.md) | All environment variables reference |
| 12 | [`skill_12_evaluation_workflow.md`](references/skill_12_evaluation_workflow.md) | End-to-end eval workflow: static checks, build, launch, tests, API/camera/Kafka checks, report |
| 13 | [`skill_13_verification_curl.md`](references/skill_13_verification_curl.md) | Verification steps and curl examples |
| 14 | [`skill_14_implementation_checklist.md`](references/skill_14_implementation_checklist.md) | Implementation checklist: file copy list, generated files, Docker prereqs, verification |
| 15 | [`skill_15_latency_measurement.md`](references/skill_15_latency_measurement.md) | TTFC and C2C latency measurement for file input via SSE streaming |
| 16 | [`skill_16_message_schema.md`](references/skill_16_message_schema.md) | Kafka message schema selection (`JSON` default vs `NvProtoSchema`) and extending messages with custom data |
| 17 | [`skill_17_camera_latency_measurement.md`](references/skill_17_camera_latency_measurement.md) | Camera / live-stream chunk_e2e latency measurement using internal pipeline timestamps |
| 18 | [`skill_18_rtsp_streaming_output.md`](references/skill_18_rtsp_streaming_output.md) | **OPT-IN** RTSP streaming output: `tee1`-tap re-stream, `RTSPStreamingServer`, `SW_ENCODER` toggle. Generate **only** when user explicitly requests RTSP |

For end-to-end evaluation, read § 12 first; load build/test/curl/latency/camera/Kafka as needed.

**§ 18 is opt-in** — generate only when the user explicitly requests RTSP output; otherwise skip § 18 and the `RTSP_*` rules below.

---

## Key Files Map

The full source-to-target file mapping lives in
[`skill_14_implementation_checklist.md`](references/skill_14_implementation_checklist.md):

- **Files copied verbatim** from `references/` (non-trivial algorithms — cycle
  detection, qwen_vl_utils preprocessing, DeepStream `IOptions` API, protobuf
  sources) with the rationale per file.
- **Files copied as adaptable templates** (Dockerfile, compose.yaml, Triton
  config and `model.py`, `ddm_net.py`, Pylon emulation config, etc.).
- **Files generated from skill sections** — each annotated with the Critical
  Rules below that the generation must follow exactly.
- Docker build prerequisites and post-build verification checklist.

Config files (`nvds_preprocess_template.txt`, `nvds_inference_template.txt`,
`vlm_prompts.txt`) are used as-is from `configs/`.

When `skill_06b` is loaded, read `configs/actions.json` from the project root and run the
§ 6b-G generation workflow to produce `nvds_action_detector/missing_number_detector.py`.
If `configs/actions.json` is absent or invalid, fall back to copying the reference file.

---

## Critical Rules

> Each rule's full detail lives in the linked `skill_NN_*.md` reference file.

| Tag | Rule summary | Details in |
|-----|---|---|
| `MANAGER_INIT_IN_MAIN` | `SOPProcessManager` init in `main()` before `uvicorn.run()` — not inside `lifespan()` | `skill_01_fastapi_endpoints.md` |
| `NAMED_KWARGS` | `create_video_processor()` uses named kwargs; camera args as separate kwargs | `skill_06_sop_process_manager.md` |
| `LIVE_REQUIRES_STREAM_TRUE` | `stream: true` required for live inputs (RTSP / camera) | `skill_08_basler_camera.md` |
| `VLM_DISABLED_DISABLES_SOP_CHECKER` | `DISABLE_VLM_INFERENCE=true` auto-disables SOP checker at import | `skill_06_sop_process_manager.md` |
| `CHUNK_PARAMS_MAX_LENGTH` | `ChunkParams.max_length_sec` = 10s internal; 60s API default | `skill_06_sop_process_manager.md` |
| `VLM_WARMUP_BEFORE_DDM` | `ModelInitializer`: VLM warmup FIRST, then CV dummy pipeline | `skill_06_sop_process_manager.md` |
| `VLM_WARMUP_3_FRAMES` | VLM warmup needs 3 frames (`torch.zeros`) — Qwen3VL hangs on < 3 | `skill_06_sop_process_manager.md` |
| `THREAD_POOL_SIZES` | 4 thread pools: `cv`(32), `clip`(32), `vlm_inference`(64), `vlm_request`(64) | `skill_06_sop_process_manager.md` |
| `MEDIA_INFO_PYMEDIAINFO` | Media info via `pymediainfo`; live sources set fps=30/duration=inf directly | `skill_06_sop_process_manager.md` |
| `CAMERA_EMULATION_PYLON_CAMEMU` | `PYLON_CAMEMU=1` for camera emulation (serial `0815-0000`) | `skill_08_basler_camera.md` |
| `DEEPSTREAM_LIB_HIDE` | DeepStream lib hide trick: rename lib → lib.tmp during gst-plugin-pylon build | `skill_08_basler_camera.md` |
| `VLM_REAL_GPU_FRAMES` | VLM uses real GPU frames via `DecodedFrameRetriever`; never `torch.zeros` for inference | `skill_06_sop_process_manager.md` |
| `BUFFER_RETRIEVER_STATIC_BASE` | `DecodedFrameRetriever` MUST inherit `BufferRetriever` statically via `super().__init__()`; runtime `__class__.__bases__` mutation hangs `pipeline.attach()` | `skill_06_sop_process_manager.md` |
| `FRAME_RETRIEVER_PRIORITY` | `create_inference_pipeline`: `frame_retriever=` kwarg takes priority over `frame_queue` | `skill_03_deepstream_pipeline.md` |
| `MUX_ORIGINAL_RESOLUTION` | nvstreammux uses original resolution (not 224); pass `mux_width/mux_height` from `get_media_info()` (probe live RTSP for non-camera inputs; camera path unaffected) | `skill_03_deepstream_pipeline.md`, `skill_06_sop_process_manager.md` |
| `FILE_URI_NO_DOUBLE_PREFIX` | `create_inference_pipeline` file source: check `file_path.startswith("file://")` before prepending — API passes `file://` URLs directly | `skill_03_deepstream_pipeline.md` |
| `CLEANUP_ON_DISCONNECT` | Pipeline cleanup on client disconnect via `trigger_stop_processors` in `try/finally` | `skill_07_sse_streaming.md` |
| `UNIFIED_CLIP_POST_PROCESS` | Unified `clip_post_process()` for file + live; `stop()` puts `None` in `_score_queue` | `skill_06_sop_process_manager.md` |
| `ABORT_INFLIGHT_VLM` | Abort in-flight VLM requests on `stop()` via `llm.abort(req_id)` | `skill_06_sop_process_manager.md` |
| `LOGGER_EXPORT_GET_LOGGER` | `ds_logger.py` must export `get_logger` | `skill_06_sop_process_manager.md` |
| `KAFKA_USE_CREATE_PRODUCER` | Kafka: use `create_producer()` from `messager.py`; no `Messager` class | `skill_06_sop_process_manager.md` |
| `USER_PROMPT_PRIORITY` | User request text takes priority over `VLM_PROMPT_PATH` file; `{"type":"text"}` in the request overrides the config-file prompt | `skill_06_sop_process_manager.md` |
| `EVAL_USE_CONFIG_PROMPT` | Eval/latency requests omit request text by default so the VLM uses `VLM_PROMPT_PATH` | `skill_12_evaluation_workflow.md`, `skill_13_verification_curl.md`, `skill_15_latency_measurement.md`, `skill_17_camera_latency_measurement.md` |
| `CHUNK_SCHEMA_FIELD_NAMES` | Chunk schema: `chunk_idx`, `cv_boundary_score`, `checker_result`; summary `chunk_idx=-1` | `skill_06_sop_process_manager.md` |
| `SEQUENTIAL_FRAME_DRAIN` | Drain `decoded_frame_queue` (FIFO, shared across chunks) in a SINGLE thread and submit VLM per chunk incrementally; parallel drain steals frames → 0-frame chunks / wrong VLM input | `skill_06_sop_process_manager.md` |
| `WALL_CLOCK_BEFORE_GPU` | `DecodedFrameRetriever.consume()`: capture `wall_clock_entry = time.time()` BEFORE GPU dlpack; queue 3-tuple `(timestamp, wall_clock_entry, tensor)` | `skill_06_sop_process_manager.md`, `skill_17_camera_latency_measurement.md` |
| `CHUNK_E2E_PIPELINE_TIMESTAMPS` | Write `pipeline_chunk_end_timestamp` (last frame wall_clock) and `pipeline_vlm_ready_timestamp` (`tm_e2e.now()`) into `chunk_info` for camera latency (§ 17) | `skill_06_sop_process_manager.md`, `skill_17_camera_latency_measurement.md` |
| `VLM_INFERENCE_REQUIRED_KWARGS` | Every `VLLMInference.inference()` call must pass `video_fps`, `system_prompt`, `max_completion_tokens` | `skill_06_sop_process_manager.md` |
| `UNIFORM_CHUNKING_BYPASSES_DDM` | `chunking_options.algorithm="uniform"` → fixed-length chunks; `create_inference_pipeline(uniform_chunk=True)` skips DDM but keeps `tee1` fanout; Stage 2 uses `uniform_clip_post_process` | `skill_02_pydantic_schemas.md`, `skill_03_deepstream_pipeline.md`, `skill_06_sop_process_manager.md` |
| `DDM_TEMPORAL_CONFIGURABLE` | `SLIDING_WINDOWS_SIZE = 2*FRAMES_PER_SIDE + SEQUENCE_BATCH` rendered into preprocess/nvinferserver (no hard-coded 18); Triton `config.pbtxt` sequence dim `-1` | `skill_04_config_templates.md`, `skill_05_triton_ddm_model.md` |
| `DDM_TRT_OPTIONAL_PATH` | `DDM_TRT_OPTIMIZATION=true` runs DDM via TensorRT (per-thread contexts, fixed batch = SEQUENCE_BATCH); PyTorch fallback; never both. PyTorch is default | `skill_05_triton_ddm_model.md` |
| `DDM_TRT_STREAM_ORDERING` | `DDMTensorRTEngine.infer()`: `wait_stream(current)` → `execute_async_v3` → **`torch.cuda.synchronize(device)`** (NOT per-stream). Per-stream sync leaves TRT aux-stream work in flight → gst-CV SIGSEGV (NVBug 6289256) | `skill_05_triton_ddm_model.md` |
| `METADATA_LICENSE_FROM_FILE` | `/v1/metadata` reads `licenseInfo` from `DS_SOP_LICENSE_PATH` (default `/opt/nvidia/nvds_sop/license.txt`); never hard-code license text | `skill_01_fastapi_endpoints.md` |
| `CAMERA_EMULATION_FRAMES_RGB` | Pylon emulation PNGs must be explicit 3-channel RGB (matches `Emulation_0815-0000.pfs PixelFormat=RGB8Packed`); generate via `nvvideoconvert ! videoconvert ! "video/x-raw,format=RGB" ! pngenc` | `skill_08_basler_camera.md` |
| `COMPOSE_ENV_PASSTHROUGH` | `docker compose` only substitutes `${VAR}` references; every runtime env var must be explicitly listed under `environment:` to reach the container. | `skill_09_docker_build_deploy.md` |

> The four `RTSP_*` rules below apply **only when the optional RTSP streaming-output feature
> (§ 18) is requested**. They do not apply to the default build — skip them if the user did
> not ask for RTSP output.

| `RTSP_OUTPUT_TAPS_TEE1` | RTSP output branch links from the existing `tee1` (added after the main inference link) only when `rtsp_port` is present. | `skill_18_rtsp_streaming_output.md` |
| `RTSP_LEAKY_QUEUE_TINY` | RTSP branch queue must be `leaky=2` + tiny cap (`max-size-buffers=2`) to prevent backpressure and NVMM pool exhaustion. | `skill_18_rtsp_streaming_output.md` |
| `RTSP_KEYINT_MAX_30` | RTSP H.264 encoder must set `key-int-max=30` (and B-frames disabled) to allow downstream seeking. | `skill_18_rtsp_streaming_output.md` |
| `RTSP_ENCODER_FALLBACK` | Select software/hardware H.264 encoder based on `SW_ENCODER` with MJPEG fallback. | `skill_18_rtsp_streaming_output.md` |

---
