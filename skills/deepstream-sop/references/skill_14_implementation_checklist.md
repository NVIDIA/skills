# § 14 — Implementation Checklist

---

## Files copied verbatim from `references/` (do NOT regenerate)

| Reference file | Copy to | Why it can't be regenerated |
|---------------|---------|----------------------------|
| `vllm_inference_reference.py` | `nvds_action_detector/vllm_inference.py` | Complex `qwen_vl_utils` + `_preprocess_video()` tensor batching |
| `utils_reference.py` | `nvds_action_detector/utils.py` | `TimeMeasure`, `SafeThreadEventLoop`, `get_media_info_gst` |
| `missing_number_detector_reference.py` | `nvds_action_detector/missing_number_detector.py` | Cycle detection algorithm with 3 configurable thresholds (§ 6b) |
| `sop_step_checker_reference.py` | `nvds_action_detector/sop_step_checker.py` | UUID cache + regex step parser — easier to copy than regenerate (§ 6b) |
| `nvds_custom_postprocess_tensor_reference.cpp` | `nvds_action_detector/custom_postprocess/nvds_custom_postprocess_tensor.cpp` | DeepStream `IOptions` API (`getObj` vs `getValueArray`) |
| `Makefile_custom_postprocess_reference` | `nvds_action_detector/custom_postprocess/Makefile` | Exact compiler/linker flags for DS headers |
| `protos_init_reference.py` | `nvds_action_detector/protos/__init__.py` | Imports nv_pb2 + ext_pb2 |
| `nv.proto` | `nvds_action_detector/protos/nv.proto` | Protobuf source — compiled to `nv_pb2.py` at Docker build time (§ 16) |
| `ext.proto` | `nvds_action_detector/protos/ext.proto` | Protobuf source — compiled to `ext_pb2.py` at Docker build time (§ 16) |
| `copy_sources_reference.sh` | `docker/copy_sources.sh` | Reads `package_file_list.txt` manifest (not rsync) |
| `package_file_list_reference.txt` | `docker/package_file_list.txt` | Must list every file for Docker copy |
| `ddm_pytorch2.patch` | `docker/ddm_pytorch2.patch` | PyTorch 2 compatibility patch for DDM model |
| `export_ddm_to_tensorrt_reference.py` | `scripts/tensorrt/export_ddm_to_tensorrt.py` | Standalone PyTorch→ONNX→TRT export script for the optional DDM TRT path (§ 5) |
| `license.txt` | `license.txt` | NVIDIA eval-license text; ship so `/v1/metadata` reads it (METADATA_LICENSE_FROM_FILE, § 1) |

---

## Files copied from `references/` as adaptable templates

| Reference file | Copy to |
|---------------|---------|
| `Dockerfile_reference` | `docker/Docker.build` |
| `compose_reference.yaml` | `deploy/compose.yaml` |
| `test_api_endpoints_reference.py` | `tests/test_api_endpoints.py` |
| `messager_reference.py` | `nvds_action_detector/messager.py` |
| `start_server.sh` | `start_server.sh` |
| `triton_config_template.pbtxt` | `nvds_action_detector/triton_model_repo/ddm/config_template.pbtxt` |
| `triton_model_reference.py` | `nvds_action_detector/triton_model_repo/ddm/1/model.py` |
| `ddm_net_reference.py` | `nvds_action_detector/triton_model_repo/ddm/1/ddm_net.py` |
| `Emulation_0815-0000.pfs` | `configs/Emulation_0815-0000.pfs` |
| `README_reference.md` | `README.md` (microservice overview, build/run, test examples, licenses, citation; source-only — NOT in `package_file_list.txt` / Docker image) |

---

## Files generated from skill sections

Each generated file must follow the SKILL.md Critical Rules in parentheses (MANAGER_INIT_IN_MAIN, NAMED_KWARGS, CLEANUP_ON_DISCONNECT, FRAME_RETRIEVER_PRIORITY, VLM_WARMUP_BEFORE_DDM, VLM_WARMUP_3_FRAMES, VLM_REAL_GPU_FRAMES, UNIFIED_CLIP_POST_PROCESS, ABORT_INFLIGHT_VLM, KAFKA_USE_CREATE_PRODUCER, USER_PROMPT_PRIORITY, CHUNK_SCHEMA_FIELD_NAMES, VLM_INFERENCE_REQUIRED_KWARGS, LOGGER_EXPORT_GET_LOGGER) exactly:

- [ ] `api_server.py` — § 1 + § 7
  - **MUST**: init `SOPProcessManager` in `main()`, not `lifespan()` (MANAGER_INIT_IN_MAIN)
  - **MUST**: `try/finally` cleanup in SSE generator and non-streaming path (CLEANUP_ON_DISCONNECT)
  - **OPT-IN (only if RTSP output requested)**: auto-inject `rtsp_port`/`rtsp_path` for RTSP/Basler inputs when `ENABLE_RTSP_OUTPUT` (§ 18)
- [ ] `api_types.py` — § 2
- [ ] `ds_3d_action_pipeline.py` — § 3
  - **MUST**: `create_inference_pipeline(...)` — `frame_retriever=` kwarg takes priority over `frame_queue` (FRAME_RETRIEVER_PRIORITY)
  - **OPT-IN (only if RTSP output requested)**: RTSP output branch taps `tee1` with a `leaky=2` tiny queue, `key-int-max=30`, `SW_ENCODER` fallback chain (RTSP_OUTPUT_TAPS_TEE1, § 18)
- [ ] `ds_sop_process.py` — § 6
  - **MUST**: `create_video_processor(file_path, id=None, chunk_params=None, **kwargs)` (NAMED_KWARGS)
  - **OPT-IN (only if RTSP output requested)**: read + forward `rtsp_port`/`rtsp_path` (§ 18). `sensor_id` derivation + per-message stamping is kept regardless
  - **MUST**: VLM warmup BEFORE DDM dummy pipeline, with 3 zero-tensor frames (VLM_WARMUP_BEFORE_DDM, VLM_WARMUP_3_FRAMES)
  - **MUST**: VLM uses real GPU frames via `DecodedFrameRetriever`, never `torch.zeros` (VLM_REAL_GPU_FRAMES)
  - **MUST**: live stream incremental clip postprocessing (UNIFIED_CLIP_POST_PROCESS)
  - **MUST**: VLM request abort on stop (ABORT_INFLIGHT_VLM)
  - **MUST**: Kafka messaging via `create_producer()` (KAFKA_USE_CREATE_PRODUCER)
  - **MUST**: user request text takes priority over VLM_PROMPT_PATH file; file is fallback only (USER_PROMPT_PRIORITY)
  - **MUST**: chunk output schema uses `chunk_idx`, `cv_boundary_score`, `checker_result` (CHUNK_SCHEMA_FIELD_NAMES)
  - **MUST**: every `VLLMInference.inference()` call passes `video_fps`, `system_prompt`, `max_completion_tokens` (VLM_INFERENCE_REQUIRED_KWARGS)
  - **MUST**: process file chunks incrementally to avoid `decoded_frame_queue` deadlock (§ 15)
- [ ] `ds_logger.py` — **MUST** export `get_logger` (LOGGER_EXPORT_GET_LOGGER: `get_logger = setup_logger`)
- [ ] `__init__.py`, `__main__.py`

Use config files as-is from `configs/`: `nvds_preprocess_template.txt`, `nvds_inference_template.txt`, `actions.json`, `vlm_prompts.txt`.

---

## Docker build prerequisites

- [ ] Base image pulled: `docker pull nvcr.io/nvidia/blueprint/vss-engine:2.4.1`
- [ ] DS headers image pulled: `docker pull nvcr.io/nvidia/deepstream:8.0-triton-multiarch`
- [ ] Pylon SDK binary (real file, not symlink): `binaries/pylon-25.10.2_linux-x86_64_setup.tar.gz`
- [ ] `docker/ddm_pytorch2.patch` present (copy from `references/ddm_pytorch2.patch`)
- [ ] `docker/package_file_list.txt` updated with ALL source files (missing files → runtime ImportError)
- [ ] *(only if RTSP output requested)* RTSP packages in the apt block (`libgstrtspserver-1.0-dev`, `gir1.2-gst-rtsp-server-1.0`, `gstreamer1.0-plugins-{ugly,bad}`, `gstreamer1.0-libav`) + registry-cache clear (§ 9, § 18)

---

## Verification after build

- [ ] `curl /v1/ready` returns 200 — poll every 10s, model loading takes several minutes on first run (see § 9 for readiness wait script)
- [ ] File input: non-streaming chat completion returns `chat.completion` with chunks
- [ ] File input: streaming chat completion returns SSE `chat.completion.chunk` events
- [ ] File input: `stream=true` + `chunking_options` with small `max_length_sec` does not hang
- [ ] Evaluation and latency payloads omit `{"type":"text"}` by default so `VLM_PROMPT_PATH` is used (see § 13, § 15, § 17)
- [ ] Camera cleanup: after disconnecting from a camera SSE stream, the camera can be immediately reused (no "device is controlled by another application" error)
- [ ] Live stream produces real-time SSE chunks (not empty response)
- [ ] *(only if RTSP output requested)* RTSP output: `GstRtspServer` importable + `x264enc` registered in-container; a live request re-streams at `rtsp://127.0.0.1:8554/ds-out/{sensor_id}` (§ 18)
- [ ] Kafka messaging: `ENABLE_MESSAGING=true` → all chunk results published to `mdx-vlm-captions` topic (each tagged with `sensor_id`)
- [ ] `tests/test_api_endpoints.py` — all 24 tests pass (incl. 5 `TestUniformChunkingEndpoint`)
- [ ] Latency: TTFC < 1s for 2s fixed chunks (file input) — see § 15
- [ ] Uniform chunking: `stream=true` + `chunking_options={"algorithm":"uniform","chunk_length_sec":2}` returns fixed-length chunks; a payload mixing `uniform` + DDM-net fields is rejected (HTTP 422)
- [ ] Configurable DDM window: rendered `nvds_preprocess_rendered.txt` / `config.pbtxt` reflect `SLIDING_WINDOWS_SIZE = 2*FRAMES_PER_SIDE + SEQUENCE_BATCH` (e.g. FRAMES_PER_SIDE=1, SEQUENCE_BATCH=8 → 10)
- [ ] DDM TRT (optional): with `DDM_TRT_OPTIMIZATION=true`, logs show `Building DDM TRT engine on the fly` → `TRT engine ready` (or a PyTorch fallback message); default (off) uses PyTorch
