# § 6 — SOP Process Manager (`ds_sop_process.py`)

> **Generates**: `nvds_action_detector/ds_sop_process.py`, `nvds_action_detector/sop_step_checker.py`
> **RTSP streaming output (OPTIONAL, opt-in — § 18)**: only when the user requested RTSP output,
> this processor reads `rtsp_port`/`rtsp_path` kwargs, forwards them into
> `create_inference_pipeline()`, and derives a `sensor_id` for per-stream message tagging.
> For the default build, skip the `rtsp_port`/`rtsp_path` reads/forwards (the `sensor_id` derivation
> can still be kept harmlessly). See [`skill_18_rtsp_streaming_output.md`](skill_18_rtsp_streaming_output.md).
> **Critical Rules**: NAMED_KWARGS, VLM_DISABLED_DISABLES_SOP_CHECKER, CHUNK_PARAMS_MAX_LENGTH, VLM_WARMUP_BEFORE_DDM, VLM_WARMUP_3_FRAMES, THREAD_POOL_SIZES, MEDIA_INFO_PYMEDIAINFO, VLM_REAL_GPU_FRAMES, BUFFER_RETRIEVER_STATIC_BASE, VLM_INFERENCE_REQUIRED_KWARGS, UNIFIED_CLIP_POST_PROCESS, SEQUENTIAL_FRAME_DRAIN, ABORT_INFLIGHT_VLM, CHUNK_SCHEMA_FIELD_NAMES, USER_PROMPT_PRIORITY, KAFKA_USE_CREATE_PRODUCER, LOGGER_EXPORT_GET_LOGGER, UNIFORM_CHUNKING_BYPASSES_DDM, WALL_CLOCK_BEFORE_GPU, CHUNK_E2E_PIPELINE_TIMESTAMPS
> **Pipeline stage**: Orchestrates all 4 stages — creates pipelines, manages threads, coordinates VLM

---

## Critical Rules

- **`NAMED_KWARGS`**: `create_video_processor()` uses named kwargs — Signature: `(file_path, id=None, chunk_params=None, **kwargs)`. Camera args must be separate named kwargs (`camera_serial_number=...`), NOT packed into a single dict.

- **`VLM_DISABLED_DISABLES_SOP_CHECKER`**: `DISABLE_VLM_INFERENCE=true` auto-disables SOP checker — enforced at module import: `if DISABLE_VLM_INFERENCE: DISABLE_SOP_CHECKER = True`.

- **`CHUNK_PARAMS_MAX_LENGTH`**: `ChunkParams.max_length_sec` — internal default = 10.0s; API `DdmNetChunkingOptions` default = 60.0s.

- **`VLM_WARMUP_BEFORE_DDM`**: `ModelInitializer` — VLM warmup starts FIRST, then CV dummy pipeline. Never swap order.

- **`VLM_WARMUP_3_FRAMES`**: VLM warmup needs 3 frames — `inference("Say Hi", [zeros, zeros, zeros])`. Qwen3VL requires ≥ 3 frames or it hangs.

- **`THREAD_POOL_SIZES`**: 4 thread pools (all owned by `SOPProcessManager`, shared across requests):
  - `cv_process_pool(32)` — Stage 1: `_run_cv_pipeline`
  - `clip_process_pool(32)` — Stage 2: `clip_post_process`
  - `vlm_inference_pool(64)` — Stage 3a: `vlm_inference_request_process` (frame drain)
  - `vlm_request_pool(64)` — Stage 3b: `VLLMInference.inference` (GPU)
  - Plus per-processor `sop_checker_pool(1)` — Stage 4: `sop_checker_process`

- **`MEDIA_INFO_PYMEDIAINFO`**: Use `pymediainfo.MediaInfo.parse(file_path)`, iterate `tracks` for `track_type == "Video"`. For live sources (`_is_live == True`), skip media info and set `chunk_params.fps = 30.0`, `chunk_params.duration_sec = float("inf")` directly — never call `get_media_info` for live, or `chunk_params.fps` stays `None` → `TypeError: int / NoneType` in `smart_nframes()`.

- **`MUX_ORIGINAL_RESOLUTION`**: Store `self.original_width, self.original_height` from `get_media_info()` in `__init__` (probing live RTSP stream resolution for non-camera live inputs as well), then pass as `mux_width=self.original_width, mux_height=self.original_height` to `create_inference_pipeline()` in `_run_cv_pipeline` whenever known (camera path unaffected). This ensures the DeepStream mux preserves aspect ratio — nvdspreprocess handles DDM-specific downscaling internally. Without this, the mux squashes to 224×224, corrupting VLM frame inputs, degrading the RTSP output stream, VIOS recording, and VSS report clips.

- **`VLM_REAL_GPU_FRAMES`**: VLM uses real GPU frames — Create `DecodedFrameRetriever(BufferRetriever)` and pass as `frame_retriever=` to `create_inference_pipeline()`. Never use `torch.zeros` for inference (only warmup).

- **`BUFFER_RETRIEVER_STATIC_BASE`**: `DecodedFrameRetriever` MUST inherit from `pyservicemaker.BufferRetriever` as a static base class with `super().__init__()`. `pyservicemaker.Receiver.attach()` reads the MRO at attach time — runtime `__class__.__bases__` mutation hangs `pipeline.attach()` forever (no exception, no log). See the BANNED block after the `DecodedFrameRetriever` example.

- **`VLM_INFERENCE_REQUIRED_KWARGS`**: Every `VLLMInference.inference()` call MUST pass:
  ```python
  video_fps=self._chunk_params.fps,        # actual source fps — never omit
  system_prompt="Answer the questions.",
  max_completion_tokens=self._max_completion_tokens or 256,
  ```
  Omitting `video_fps` causes `_preprocess_video` to use 25.0fps → wrong `smart_nframes()` → wrong frame selection → incorrect VLM classification on every chunk.

- **`UNIFIED_CLIP_POST_PROCESS`**: Unified `clip_post_process()` — file and live use the SAME function. Reads `_score_queue` with `block=True`, puts chunk_info in `_chunk_queue`. `stop()` MUST put `None` in `_score_queue` to unblock the live loop. Do NOT separate into file/live variants.

- **`SEQUENTIAL_FRAME_DRAIN`**: Sequential frame drain in `vlm_inference_request_process` — `decoded_frame_queue` is FIFO shared across chunks. Frame drain MUST run in a single thread (the `vlm_inference_pool` thread). Parallel drain causes threads to steal each other's frames → 0-frame chunks → wrong VLM input. Both file and live inputs MUST process chunk boundaries incrementally — submit VLM as each chunk is detected so `decoded_frame_queue` drains continuously alongside boundary detection, keeping GPU memory usage bounded and preventing pipeline stalls.

- **`ABORT_INFLIGHT_VLM`**: Abort in-flight VLM requests on `stop()` — Track IDs in `self._active_vlm_request_ids`; call `llm.abort(req_id)` via `asyncio.run_coroutine_threadsafe`.

- **`CHUNK_SCHEMA_FIELD_NAMES`**: Chunk output schema — Use `chunk_idx` (not `chunk_index`), `cv_boundary_score` (not `boundary_score`), `checker_result` (not `sop_check`). No `boundary_trigger` field. SOP checker summary chunk uses `chunk_idx=-1`.

- **`USER_PROMPT_PRIORITY`**: User request text takes priority over `VLM_PROMPT_PATH` file — Resolve as `self._prompt = request_text if request_text else self._vlm_prompt`. Request text overrides the config-file prompt; the file is only used as fallback when the request contains no text.

- **`KAFKA_USE_CREATE_PRODUCER`**: Kafka messaging — Use `create_producer(kafka_broker)` from `messager.py`. There is no `Messager` class.

- **`GPU_ID_PASSED_ONCE`**: `_run_cv_pipeline()` must pass `gpu_id` exactly once to `create_inference_pipeline()`. If calling `create_inference_pipeline(..., gpu_id=self._gpu_id, **pipeline_kwargs)`, do NOT also set `pipeline_kwargs["gpu_id"]`. Passing both raises `TypeError: got multiple values for keyword argument 'gpu_id'` at runtime.

- **`LOGGER_EXPORT_GET_LOGGER`**: `ds_logger.py` must export `get_logger` — All modules call `ds_logger.get_logger(__name__)`. If `ds_logger.py` only defines `setup_logger`, the service crashes at import with `AttributeError: module has no attribute 'get_logger'`. Add the alias:
  ```python
  get_logger = setup_logger
  ```

---

## 4-Stage Pipeline Overview

```
SOPVideoProcessor (per-request)
├── Stage 1: _run_cv_pipeline()                [cv_process_pool]
│     └── _score_queue      ← (frame_id, pts, score) from DDM probe
│     └── decoded_frame_queue ← GPU frames via DecodedFrameRetriever
├── Stage 2: clip_post_process()               [clip_process_pool]  ← UNIFIED file + live
│     └── _chunk_queue      ← {chunk_idx, start_time, end_time, cv_boundary_score}
├── Stage 3a: vlm_inference_request_process()  [vlm_inference_pool, single thread]
│     └── submit_vllm_inference() — drains decoded_frame_queue (sequential)
│     └── submits VLLMInference.inference() to vlm_request_pool (async GPU)
│     └── _vlm_response_future_queue ← {chunk_info + response_future}
├── Stage 3b: vlm_inference_response_process() [daemon thread]
│     └── collects future.result() in order
│     └── _vlm_response_queue ← {chunk_info + response}
│         (→ final_queue directly when SOP checker disabled)
├── Stage 4: sop_checker_process()             [sop_checker_pool, 1 worker]
│     └── keep_alive=True per chunk; keep_alive=False at EOS → summary chunk_idx=-1
└── final_queue → SSE generator (§ 7) or non-streaming collector
```

---

## Module-Level Constants

```python
import concurrent.futures as futures
import threading
from pyservicemaker import EOSMessage, PipelineState, StateTransitionMessage

# --- Thread pool sizes for the 4 pipeline stages ---
CV_THREAD_NUM = int(os.getenv("CV_THREAD_NUM", "32"))
CLIP_THREAD_NUM = int(os.getenv("CLIP_THREAD_NUM", "32"))
VLM_INFERENCE_THREAD_NUM = int(os.getenv("VLM_INFERENCE_THREAD_NUM", "64"))

# --- Feature flags ---
DISABLE_VLM_INFERENCE = os.getenv("DISABLE_VLM_INFERENCE", "false").lower() in ["true", "1", "yes", "y"]
DISABLE_SOP_CHECKER = os.getenv("DISABLE_SOP_CHECKER", "false").lower() in ["true", "1", "yes", "y"]
# CRITICAL [VLM_DISABLED_DISABLES_SOP_CHECKER]: DISABLE_VLM_INFERENCE automatically forces DISABLE_SOP_CHECKER
# (SOP checker depends on VLM classification results)
if DISABLE_VLM_INFERENCE:
    DISABLE_SOP_CHECKER = True

ACTION_CONFIG_PATH = os.getenv("ACTION_CONFIG_PATH", "configs/actions.json")
VLM_PROMPT_PATH = os.getenv("VLM_PROMPT_PATH", "configs/vlm_prompts.txt")
# No usable default: the model must be a user-provided fine-tuned checkpoint
# (a local path under MODEL_ROOT_DIR, or a Hugging Face repo id). Validated
# non-empty in _initialize_vlm_model() before the engine is created so a missing
# value fails fast with a clear message instead of hanging ~300s/chunk.
VLLM_MODEL_PATH = os.getenv("VLLM_MODEL_PATH", "")

# --- Boundary anti-jitter delay ---
# After a boundary is detected, accumulate this many frames before committing the chunk.
# The highest-confidence boundary in the window is chosen.
BOUNDARY_DELAY_FRAME = int(os.getenv("BOUNDARY_DELAY_FRAME", "8"))
```

> **Prompt priority rule**: User request text **always takes priority** over
> `VLM_PROMPT_PATH` file content. In `SOPVideoProcessor.__init__`, the
> prompt is resolved as:
> ```python
> request_text = kwargs.get("prompt", "")
> self._prompt = request_text if request_text else self._vlm_prompt
> ```
> This means:
> - If the request includes a `{"type":"text"}` item → **request text is used** for every
>   VLM inference call, overriding the config-file prompt.
> - Only when the request contains no text does `VLM_PROMPT_PATH` act as fallback.

---

## ChunkParams

```python
@dataclass
class ChunkParams:
    """Parameters for chunk boundary detection and segmentation.

    Used by Stage 2 (clip post-processing) to determine chunk boundaries.
    `algorithm` selects "ddm-net" (DDM event-boundary detection) or "uniform"
    (fixed-length chunks of `chunk_length_sec` that bypass the DDM model — see § 3).
    Note: internal max_length_sec defaults to 10s, while API default is 60s."""
    threshold: float = 0.8
    min_length_sec: float = 1.0
    max_length_sec: float = 10.0       # Internal default = 10s (API DdmNetChunkingOptions default = 60s)
    duration_sec: Optional[float] = None  # Populated from video media info
    fps: Optional[float] = None           # Populated from video media info
    chunk_length_sec: Optional[float] = None  # Required when algorithm == "uniform"
    algorithm: str = "ddm-net"            # "ddm-net" | "uniform"

    def __post_init__(self):
        if self.algorithm == "uniform" and self.chunk_length_sec is None:
            raise ValueError("chunk_length_sec must be set when algorithm == 'uniform'")
```

## Uniform chunking (`algorithm="uniform"`)

> **CRITICAL [UNIFORM_CHUNKING_BYPASSES_DDM]**: in uniform mode the DDM model is not built or read.
> Drive Stage 2 from frame timestamps only; select `uniform_clip_post_process`; emit boundary score 1.0.

```
algorithm == "uniform":
  pipeline: create_inference_pipeline(uniform_chunk=True)   # DDM preprocess/inference branch SKIPPED (§ 3)
  Stage 2 : uniform_clip_post_process()                     # NOT clip_post_process()
            frame_queue timestamps ─► cut a chunk every chunk_length_sec
                                   ─► chunk_info{start, end, cv_boundary_score = 1.0} ─► Stage 3 VLM
  score_queue (DDM boundary scores): never produced, never read
  DecodedFrameRetriever: REQUIRED (supplies the timestamps) even if VLM is disabled
```

When `chunk_params.algorithm == "uniform"`, Stage 2 produces **fixed-length** chunks from
decoded-frame timestamps and the DDM model is **bypassed entirely** (no boundary scores):

- **Pipeline:** pass `uniform_chunk=(self._chunk_params.algorithm == "uniform")` to
  `create_inference_pipeline()` (§ 3) so the DDM preprocess/inference branch is skipped.
- **Stage-2 function selection** (replaces the single hard-wired call):
  ```python
  clip_fn = self.uniform_clip_post_process if self._chunk_params.algorithm == "uniform" else self.clip_post_process
  self._clip_process_future = submit_in_executor(self._clip_process_pool, clip_fn)
  ```
- **Frame retriever is required for uniform mode** — create `DecodedFrameRetriever` whenever
  `algorithm == "uniform"` (even if VLM is disabled), and size its queue from
  `chunk_length_sec` rather than `max_length_sec`.
- **Shared chunk-info builder** used by both clip functions:
  ```python
  def _make_chunk_info(self, chunk_idx, start_time, end_time, cv_boundary_score, cv_execute_time):
      return {
          "chunk_idx": chunk_idx, "start_time": start_time, "end_time": end_time,
          "cv_boundary_score": cv_boundary_score, "cv_execute_time": cv_execute_time,
          "first_timestamp": self.first_timestamp,
          "pipeline_starting_timestamp": self._tm_e2e.start_time,
          "pipeline_cv_ready_timestamp": self._tm_e2e.now(),
      }
  ```
- **Uniform clip post-process** — cut a new chunk every `chunk_length_sec` of frame
  timestamps; boundary score is always `1.0`; emit a final partial chunk at EOS:
  ```python
  def uniform_clip_post_process(self):
      """Generate fixed-length chunk boundaries from decoded frame timestamps."""
      self._started_event.wait()
      tm = TimeMeasure("uniform_clip_post_process")
      self.first_timestamp = self._tm_e2e.now()
      retriever = self._decoded_frame_retriever
      if retriever is None:
          logger.error("uniform_clip_post_process requires a decoded frame retriever")
          self._chunk_queue.put(None)
          return
      chunk_length_sec = self._chunk_params.chunk_length_sec
      chunk_idx, clip_start = 0, None
      while True:
          retriever.wait_for_new_frame()
          last_ts = retriever.last_timestamp()
          is_eos = retriever.is_end_of_stream()
          if clip_start is None:
              clip_start = last_ts
          while last_ts >= clip_start + chunk_length_sec:
              end = clip_start + chunk_length_sec
              self._chunk_queue.put(self._make_chunk_info(chunk_idx, clip_start, end, 1.0, tm.elapsed_time))
              chunk_idx += 1
              clip_start = end
              tm.reset()
          if is_eos:
              if last_ts > clip_start:
                  self._chunk_queue.put(self._make_chunk_info(chunk_idx, clip_start, last_ts, 1.0, tm.elapsed_time))
              break
      # Do not read _score_queue here: uniform mode skips the DDM branch, so no boundary
      # scores are produced. EOS/stop may still place a sentinel there for the ddm-net path.
      self._chunk_queue.put(None)
  ```

> `UNIFIED_CLIP_POST_PROCESS` still holds: each algorithm uses ONE function for both file
> and live; uniform simply selects `uniform_clip_post_process` instead of `clip_post_process`.

---

## ModelInitializer

```python
class ModelInitializer:
    """Warms up DDM (CV) model and embedded vLLM before serving requests.

    Warmup sequence (VLM_WARMUP_BEFORE_DDM — VLM init starts FIRST, then CV dummy pipeline):
      1. Loads VLLMInference model synchronously, starts warmup thread
      2. Starts dummy pipeline → waits for PipelineState.PLAYING (CV model loaded)
      3. Waits for EOSMessage (CV warmup frames processed)
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._started_event_cv = threading.Event()
        self._warmup_ready_event_cv = threading.Event()
        self.vlm_ready_event = threading.Event()
        self._vlm_model = None

        # 1. VLM initialization FIRST (loads model synchronously, warmup in thread)
        if not DISABLE_VLM_INFERENCE:
            self._initialize_vlm_model()
        else:
            self.vlm_ready_event.set()

        # 2. Then start dummy pipeline for CV model warmup (§ 3: create_dummy_pipeline)
        self._dummy_pipeline = create_dummy_pipeline()

        def on_message_dummy_pipeline(message):
            if isinstance(message, StateTransitionMessage):
                if message.new_state == PipelineState.PLAYING:
                    with self._lock:
                        if not self._started_event_cv.is_set():
                            self._started_event_cv.set()
            if isinstance(message, EOSMessage):
                self._warmup_ready_event_cv.set()

        self._dummy_pipeline.start(on_message_dummy_pipeline)

    def _initialize_vlm_model(self):
        from .vllm_inference import VLLMInference
        # VLLM_MODEL_PATH has no default — fail fast with a clear message instead
        # of letting the engine hang ~300s/chunk on an unset/invalid model.
        # Non-empty check only: the value may be a local path OR an HF repo id.
        if not VLLM_MODEL_PATH or not VLLM_MODEL_PATH.strip():
            raise ValueError(
                "VLLM_MODEL_PATH is unset. Set it in deploy/.env to your fine-tuned "
                "VLM checkpoint directory (under MODEL_ROOT_DIR) or a Hugging Face "
                "repo id. There is no default model path."
            )
        self._vlm_model = VLLMInference(model_path=VLLM_MODEL_PATH, device="cuda:0")

        def warmup_thread():
            # VLM_WARMUP_3_FRAMES: Qwen3VL requires at least 3 frames — 1 frame causes vLLM to hang
            self._vlm_model.inference("Say Hi", [
                torch.zeros(224, 224, 3, dtype=torch.uint8),
                torch.zeros(224, 224, 3, dtype=torch.uint8),
                torch.zeros(224, 224, 3, dtype=torch.uint8),
            ])
            self.vlm_ready_event.set()

        self.vlm_thread = threading.Thread(target=warmup_thread)
        self.vlm_thread.start()
```

---

## VLLMInference — Model Loading (AsyncLLMEngine)

**CRITICAL**: Always use `AsyncLLMEngine.from_engine_args` (not `LLM()`).
This enables async generation and concurrent request handling for best performance.

```python
import asyncio
import threading
import uuid
import vllm
import transformers
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.3"))
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "20480"))
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", "16"))
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "256"))
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.2"))


class VLLMInference:
    """Embedded vLLM engine for action classification (Stage 3).

    Uses AsyncLLMEngine (not LLM) for concurrent request handling.
    Why: LLM (synchronous) blocks the GIL and can only process one request at a time.
    AsyncLLMEngine runs continuous batching in the background event loop, enabling
    the 64-thread vlm_inference_pool to overlap requests and maximize GPU utilization.
    """
    def __init__(self, model_path, device: str = "cuda:0", **kwargs):
        # Dedicated event loop + thread — AsyncLLMEngine requires its own loop
        self._event_loop = asyncio.new_event_loop()
        self._gpu_memory_utilization = kwargs.pop(
            "gpu_memory_utilization", VLLM_GPU_MEMORY_UTILIZATION
        )

        def loop_thread():
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.run_forever()

        self._inference_thread = threading.Thread(target=loop_thread, daemon=True)
        self._inference_thread.start()
        self._llm, self._processor, self._sampling_params = self._load_model(model_path)

        # Detect model type for Qwen3VL-specific preprocessing
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as fp:
            model_config = json.load(fp)
        self._is_qwen3vl = model_config.get("model_type") == "qwen3_vl"

    def _load_model(self, model_path: str):
        engine_args = AsyncEngineArgs(
            model=model_path,
            max_model_len=VLLM_MAX_MODEL_LEN,
            limit_mm_per_prompt={"image": 8, "video": 1},
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
        )
        # AsyncLLMEngine.from_engine_args — required for concurrent async generation
        llm = AsyncLLMEngine.from_engine_args(engine_args)
        processor = transformers.AutoProcessor.from_pretrained(model_path, use_fast=True)
        sampling_params = vllm.SamplingParams(
            temperature=VLM_TEMPERATURE, max_tokens=VLM_MAX_TOKENS
        )
        return llm, processor, sampling_params

    def inference(self, prompt: str, video, **kwargs) -> dict:
        """Run VLM inference on a set of video frames.

        Called from SOPVideoProcessor._run_vlm_inference() for each chunk.
        Submits async generation to the dedicated event loop thread."""
        request_id = kwargs.pop("req_id", str(uuid.uuid4()))
        engine_prompt = {
            "prompt": ...,           # apply_chat_template result
            "multi_modal_data": ..., # {"video": video_inputs}
            "mm_processor_kwargs": ...,
        }

        async def generate_async():
            final_output = None
            async for output in self._llm.generate(
                prompt=engine_prompt,
                sampling_params=self._sampling_params,
                request_id=request_id,
            ):
                final_output = output
            return {"req_id": request_id,
                    "response": final_output.outputs[0].text if final_output else ""}

        future = asyncio.run_coroutine_threadsafe(generate_async(), self._event_loop)
        return future.result(timeout=300)

    def stop(self):
        if self._event_loop and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            self._inference_thread.join()
            self._event_loop = None
```

### ModelInitializer — Remaining Methods

```python
    @property
    def vlm_model(self):
        return self._vlm_model

    def is_model_ready(self):
        cv_ready = self._warmup_ready_event_cv.is_set()
        if not cv_ready:
            return False
        if self._vlm_model is not None and not self.vlm_ready_event.is_set():
            return False
        return True

    def wait_for_model_ready(self):
        """Block until both DDM and VLM models are warm.
        CRITICAL [MANAGER_INIT_IN_MAIN]: Must be called from main(), NOT from async lifespan()."""
        self._started_event_cv.wait()       # Wait for pipeline PLAYING (DDM loaded)
        self._warmup_ready_event_cv.wait()  # Wait for EOS (warmup complete)
        if self._vlm_model is not None:
            self.vlm_ready_event.wait()     # Wait for vLLM warmup

    def close(self):
        if self._dummy_pipeline is not None:
            self._dummy_pipeline.stop()
            self._dummy_pipeline.wait()
        if self._vlm_model is not None:
            self._vlm_model.stop()
```

---

## SOPProcessManager

```python
class SOPProcessManager:
    """Global manager. Creates 4 thread pools shared across all requests.

    Thread pool architecture:
      cv_process_pool (32)    — Stage 1: DeepStream pipeline execution
      clip_process_pool (32)  — Stage 2: Clip boundary post-processing
      vlm_inference_pool (64) — Stage 3: VLM inference dispatch
      vlm_request_pool (64)   — Stage 3: VLM request management
    """

    def __init__(self):
        self._cv_process_pool = futures.ThreadPoolExecutor(max_workers=CV_THREAD_NUM)
        self._clip_process_pool = futures.ThreadPoolExecutor(max_workers=CLIP_THREAD_NUM)
        self._vlm_inference_pool = futures.ThreadPoolExecutor(max_workers=VLM_INFERENCE_THREAD_NUM)
        self._vlm_request_pool = futures.ThreadPoolExecutor(max_workers=VLM_INFERENCE_THREAD_NUM)
        self._model_init = ModelInitializer()

    def is_model_ready(self) -> bool:
        return self._model_init.is_model_ready()

    def wait_for_model_ready(self):
        self._model_init.wait_for_model_ready()

    def create_video_processor(self, file_path, id=None, chunk_params=None, **kwargs):
        """Create a new SOPVideoProcessor for a single request.

        CRITICAL [NAMED_KWARGS]: `chunk_params` MUST be a named kwarg.
        Passing ChunkParams as positional arg 2 (the `id` slot) causes
        TypeError: unhashable type: 'ChunkParams' when used as dict key."""
        if id is None:
            id = uuid.uuid4()
        return SOPVideoProcessor(
            file_path=file_path,
            id=id,
            chunk_params=chunk_params,
            cv_process_pool=self._cv_process_pool,
            clip_process_pool=self._clip_process_pool,
            vlm_inference_pool=self._vlm_inference_pool,
            vlm_request_pool=self._vlm_request_pool,
            vlm_model=self._model_init._vlm_model,
            **kwargs
        )

    def trigger_stop_processors(self, processor, force=False):
        """Stop a processor (CLEANUP_ON_DISCONNECT: called from SSE generator try/finally)."""
        return self._cv_process_pool.submit(processor.stop, force)

    def close(self):
        for pool in [self._cv_process_pool, self._clip_process_pool,
                     self._vlm_inference_pool, self._vlm_request_pool]:
            pool.shutdown(wait=False)
```

---

## SOPVideoProcessor: RTSP params + sensor_id (§ 18)

> **OPTIONAL (opt-in):** generate the `rtsp_port`/`rtsp_path` reads only when the user requested
> RTSP output (§ 18). The `sensor_id` derivation is independently useful and may be kept either
> way (when RTSP is omitted, the `self._rtsp_path` branch is simply never taken).

Read the RTSP streaming-output kwargs alongside the camera kwargs, and derive a `sensor_id`
used to tag every published message per stream/camera (computed once in `__init__`, after
`_is_live` is known):

```python
# In SOPVideoProcessor.__init__ (alongside camera_fps_num/camera_fps_den):
self._rtsp_port = kwargs.get("rtsp_port")
self._rtsp_path = kwargs.get("rtsp_path")

# sensor_id: prefer the RTSP output path stem (/ds-out/<sensor>), then camera serial,
# then the source file stem. Lets a downstream dashboard filter messages per camera.
if self._rtsp_path:
    self._sensor_id = self._rtsp_path.rstrip("/").split("/")[-1]
elif self._camera_serial_number:
    self._sensor_id = str(self._camera_serial_number)
else:
    base = os.path.basename(file_path) if file_path else ""
    self._sensor_id = base.split(".")[0] if base else "unknown"
```

---

## SOPVideoProcessor: get_media_info() and Original Resolution

Store `original_width`/`original_height` in `__init__` for use by `_run_cv_pipeline` (MUX_ORIGINAL_RESOLUTION):

```python
# In SOPVideoProcessor.__init__:
self.original_width, self.original_height = 0, 0
if not self._is_live:
    try:
        duration, fps, self.original_width, self.original_height = self.get_media_info(file_path)
        self._chunk_params.duration_sec = duration
        self._chunk_params.fps = fps
    except Exception as e:
        logger.warning(f"Could not get media info for {file_path}: {e}")
else:
    self._chunk_params.fps = 30.0
    self._chunk_params.duration_sec = float("inf")
    # For non-camera live inputs (e.g. RTSP), probe the live source resolution and FPS if possible
    if file_path and file_path.startswith("rtsp://"):
        try:
            _, fps, self.original_width, self.original_height = self.get_media_info(file_path)
            if fps:
                self._chunk_params.fps = fps
        except Exception as e:
            logger.warning(f"Could not get live stream media info for {file_path}: {e}")
```

## SOPVideoProcessor: get_media_info()

```python
from pymediainfo import MediaInfo

@classmethod
def get_media_info(cls, file_path):
    """Extract video duration, FPS, width, height.
    Uses pymediainfo for file sources, get_media_info_gst() for RTSP.
    MEDIA_INFO_PYMEDIAINFO: Use pymediainfo.MediaInfo.parse() — do NOT use pyservicemaker.utils.MediaInfo."""
    if file_path.startswith("rtsp://"):
        return get_media_info_gst(file_path)

    if file_path.startswith("file://"):
        file_path = file_path[7:]

    media_info = MediaInfo.parse(file_path)
    for track in media_info.tracks:
        if track.track_type == "Video":
            return (
                float(track.duration) / 1e3,
                float(track.frame_rate),
                track.width,
                track.height,
            )
    raise ValueError(f"No video track found in {file_path}")
```

---

## DecodedFrameRetriever — GPU Zero-Copy Frame Sharing (CRITICAL — VLM_REAL_GPU_FRAMES)

```
DeepStream Pipeline (GPU) — Stage 1
    │
    tee1 ──► queue3 ──► nvvideoconvert ──► capsfilter(RGB) ──► appsink
                                                                   │
                                                    Receiver(DecodedFrameRetriever)
                                                                   │
                                                        buffer.extract(0).clone()
                                                                   │
                                                    torch.utils.dlpack.from_dlpack(tensor)
                                                                   │
                                                        decoded_frame_queue ──► Stage 3: VLM inference
```

```python
from pyservicemaker import BufferRetriever

class SOPVideoProcessor:

    class DecodedFrameRetriever(BufferRetriever):
        """Retrieve decoded GPU frames from DeepStream buffers into a queue.

        This is the mechanism that provides REAL decoded video frames to VLM inference
        (Stage 3). Without it, VLM receives fake torch.zeros tensors and cannot
        classify actions. Uses GPU zero-copy via dlpack buffer sharing."""

        def __init__(self, sop_video_processor: "SOPVideoProcessor"):
            super().__init__()
            self._sop_video_processor = sop_video_processor
            # Queue sized for max_length_sec of frames + headroom
            fps = sop_video_processor._chunk_params.fps if sop_video_processor._chunk_params.fps and sop_video_processor._chunk_params.fps >= 1 else 30.0
            max_length_sec = max(sop_video_processor._chunk_params.max_length_sec, 1)
            max_queue_size = max(int(fps * max_length_sec + 10), 30)
            self.decoded_frame_queue = Queue(maxsize=max_queue_size)
            self._lock = threading.Lock()
            self.new_frame_event = threading.Event()
            self._last_timestamp = 0
            self.end_of_stream = False

        def consume(self, buffer):
            """Called by pyservicemaker for each decoded frame buffer.
            Extracts GPU tensor via zero-copy dlpack and queues it for VLM.

            CRITICAL [WALL_CLOCK_BEFORE_GPU]: wall_clock_entry is captured BEFORE
            GPU extract/dlpack. This is the earliest "frame seen" wall-clock and is
            stored in the queue tuple so submit_vllm_inference() can write
            chunk_info["pipeline_chunk_end_timestamp"] for the last frame of each chunk.
            Used by § 17 camera chunk_e2e latency measurement.
            """
            wall_clock_entry = time.time()   # WALL_CLOCK_BEFORE_GPU: capture before GPU ops
            pts_ns = buffer.timestamp
            tensor = buffer.extract(0).clone()
            torch_tensor = torch.utils.dlpack.from_dlpack(tensor)
            timestamp = pts_ns / 1e9  # nanoseconds → seconds

            # File sources block (backpressure); live streams drop oldest frame
            block = not self._sop_video_processor._is_live
            try:
                self.decoded_frame_queue.put((timestamp, wall_clock_entry, torch_tensor), block=block)
            except Full:
                # Live stream: drop oldest to keep up with real-time
                self.decoded_frame_queue.get(block=False)
                self.decoded_frame_queue.put((timestamp, wall_clock_entry, torch_tensor), block=False)

            with self._lock:
                self._last_timestamp = timestamp
            self.new_frame_event.set()
            return 1

        def set_end_of_stream(self):
            self.decoded_frame_queue.put(None)
            with self._lock:
                self.end_of_stream = True
            self.new_frame_event.set()
```

**`BUFFER_RETRIEVER_STATIC_BASE` — REQUIRED: `BufferRetriever` must be the static base class.**

`pyservicemaker.Receiver.attach()` reads the MRO at attach time. Mutating the
base at runtime (`self.__class__.__bases__ = …`) leaves the C-side spinning on
an unresolvable vtable slot and `pipeline.attach()` hangs silently forever —
no exception, no log.

BANNED:

```python
class DecodedFrameRetriever:                              # no static base
    def __init__(self, ...):
        self.__class__.__bases__ = (BufferRetriever,)     # hangs attach() forever
```

REQUIRED (as shown above): `class DecodedFrameRetriever(BufferRetriever):` with `super().__init__()`.

### Initialization in SOPVideoProcessor.\_\_init\_\_

```python
# Provides decoded-frame timestamps for uniform chunking, and real GPU frames for Stage 3
# when VLM inference is enabled.
self._decoded_frame_retriever = None
if self._chunk_params.algorithm == "uniform" or (not DISABLE_VLM_INFERENCE and self._vlm_model is not None):
    self._decoded_frame_retriever = self.DecodedFrameRetriever(self)
```

### Passing to Pipeline (in \_run\_cv\_pipeline — Stage 1)

```python
# CRITICAL [MUX_ORIGINAL_RESOLUTION]: pass original video resolution so mux preserves aspect ratio
# For non-camera live inputs (e.g. RTSP), pass mux dims whenever known (camera path unaffected)
if (not self._is_live or (self._file_path and self._file_path.startswith("rtsp://"))) and self.original_width and self.original_height:
    pipeline_kwargs["mux_width"] = self.original_width
    pipeline_kwargs["mux_height"] = self.original_height

# Pass frame_retriever for GPU zero-copy decoded frames (VLM_REAL_GPU_FRAMES)
pipeline_kwargs["frame_retriever"] = self._decoded_frame_retriever
pipeline_kwargs["uniform_chunk"] = (self._chunk_params.algorithm == "uniform")

# RTSP streaming output (§ 18, OPT-IN): generate this forward only when RTSP output was
# requested. This processor does NOT pass **kwargs transparently, so the explicit forward
# is required when RTSP is enabled.
if self._rtsp_port is not None:
    pipeline_kwargs["rtsp_port"] = self._rtsp_port
if self._rtsp_path is not None:
    pipeline_kwargs["rtsp_path"] = self._rtsp_path
# CRITICAL [GPU_ID_PASSED_ONCE]: gpu_id is passed as a direct kwarg below.
# Do NOT also set pipeline_kwargs["gpu_id"], or Python raises:
# TypeError: got multiple values for keyword argument 'gpu_id'
self._pipeline = create_inference_pipeline(
    self._file_path, self._score_queue,
    gpu_id=self._gpu_id,
    frame_queue=self._frame_queue,
    **pipeline_kwargs,
)
```

### EOS Handling (in pipeline on\_message callback)

```python
def on_message(message):
    if isinstance(message, EOSMessage):
        self._score_queue.put(None)  # Signal Stage 2 that no more scores are coming
        if self._decoded_frame_retriever is not None:
            self._decoded_frame_retriever.set_end_of_stream()  # Signal Stage 3
```

### Frame Extraction for VLM (in \_extract\_decoded\_frames — Stage 3)

```python
def _extract_decoded_frames(self, start_time, end_time):
    """Drain real decoded GPU frames for the chunk time range.
    Called by Stage 3 (VLM inference) for each chunk to get actual video frames."""
    while (not self._decoded_frame_retriever.is_end_of_stream()
           and self._decoded_frame_retriever.last_timestamp() < end_time):
        self._decoded_frame_retriever.wait_for_new_frame()

    frames = []
    while not decoded_frame_queue.empty():
        frame = decoded_frame_queue.get(block=True)
        if frame is None:
            break
        timestamp, tensor = frame
        frames.append(tensor)
        if timestamp + 1e-3 >= end_time:
            break
    return frames
```

**DO NOT use fake `torch.zeros` tensors for VLM inference** — they produce meaningless classifications. Always use `DecodedFrameRetriever`.

---

## Stage 2: Unified Clip Post-Processing (UNIFIED_CLIP_POST_PROCESS, SEQUENTIAL_FRAME_DRAIN)

> Use `BOUNDARY_DELAY_FRAME` to accumulate boundary candidates before committing a chunk.
> Pick the **highest-confidence** boundary in the window via `calculate_next_chunk_boundary()`.
> **Both file and live use the same function** — `_score_queue.get(block=True)` terminates on
> `None` sentinel (EOS for file, or `stop()` puts `None` for live).

```python
def clip_post_process(self):
    """Stage 2: Unified for file and live. Puts chunk_info in _chunk_queue."""
    is_last_item = False
    delayed_frames = 0
    boundaries = []
    need_check_delayed = False
    is_ready = False
    chunk_idx = 0

    while not is_last_item:
        item = self._score_queue.get(block=True)  # block=True; None sentinel exits loop

        if item is None:
            is_last_item = True
        else:
            frame_id, pts, score = item
            self._clip_cur_sec = pts

            if score < 0:  # sentinel: first frame — init timestamps
                self.first_timestamp = pts
                self._clip_start_sec = pts
                continue

            if score >= self._chunk_params.threshold:
                boundaries.append(item)
                need_check_delayed = True
            elif pts - self._clip_start_sec >= self._chunk_params.max_length_sec:
                boundaries.append(item)
                is_ready = True

            if need_check_delayed:
                delayed_frames += 1
                if delayed_frames >= BOUNDARY_DELAY_FRAME:
                    is_ready = True

        if is_ready or is_last_item:
            end, cv_score = self.calculate_next_chunk_boundary(boundaries, is_end=is_last_item)
            if end is not None:
                self._chunk_queue.put({   # → Stage 3a vlm_inference_request_process
                    "chunk_idx": chunk_idx,
                    "start_time": self._clip_start_sec,
                    "end_time": end,
                    "cv_boundary_score": cv_score,
                })
                chunk_idx += 1
                self._clip_start_sec = end
            boundaries.clear()
            delayed_frames = 0
            need_check_delayed = False
            is_ready = False

    self._chunk_queue.put(None)  # Signal EOS to Stage 3a

def calculate_next_chunk_boundary(self, boundaries, is_end=False):
    if is_end:
        if self._is_live:
            return self._clip_cur_sec, 0
        dur = self._chunk_params.duration_sec
        return (dur if dur is not None else self._clip_cur_sec), 0

    if not boundaries:
        return None, None

    most_likely = max(boundaries, key=lambda x: x[2])
    idx, pts, score = most_likely
    if score < self._chunk_params.threshold:
        idx, pts, score = boundaries[-1]

    if pts - self._clip_start_sec <= self._chunk_params.min_length_sec:
        return None, None
    return pts, score
```

---

## Stage 3: VLM Inference — Request / Submit / Response (VLM_INFERENCE_REQUIRED_KWARGS, SEQUENTIAL_FRAME_DRAIN, VLM_REAL_GPU_FRAMES)

```python
def vlm_inference_request_process(self):
    """Stage 3a [vlm_inference_pool]: drain frames sequentially, submit GPU inference async."""
    while True:
        chunk_info = self._chunk_queue.get(block=True)
        if chunk_info is None:
            break
        start_time, end_time = chunk_info["start_time"], chunk_info["end_time"]

        if DISABLE_VLM_INFERENCE or self._vlm_model is None:
            chunk_info["response_future"] = None
        else:
            chunk_info["response_future"] = self.submit_vllm_inference(
                start_time, end_time, chunk_info
            )
        self._vlm_response_future_queue.put(chunk_info)

    self._vlm_response_future_queue.put(None)

def submit_vllm_inference(self, start_time, end_time, chunk_info):
    """Drain decoded_frame_queue up to end_time, submit to vlm_request_pool."""
    # Wait for frames
    while (
        not self._decoded_frame_retriever.is_end_of_stream()
        and self._decoded_frame_retriever.last_timestamp() < end_time
    ):
        self._decoded_frame_retriever.wait_for_new_frame(timeout=1.0)

    frames = []
    dq = self._decoded_frame_retriever.decoded_frame_queue
    while not dq.empty():
        frame = dq.get(block=True)
        if frame is None:
            break
        timestamp, wall_clock, tensor = frame   # wall_clock = WALL_CLOCK_BEFORE_GPU entry time
        frames.append(tensor)
        if timestamp + 1e-3 >= end_time:
            # Record when the last chunk frame was first seen by the frame branch.
            # Used for chunk_e2e = pipeline_vlm_ready_timestamp - pipeline_chunk_end_timestamp (§ 17).
            chunk_info["pipeline_chunk_end_timestamp"] = wall_clock
            break

    chunk_info["frame_number"] = len(frames)
    # cv_execute_time: wall-clock seconds for CV stages (frame collection) — required by Kibana dashboard
    chunk_info["cv_execute_time"] = time.time() - cv_t0
    if not frames:
        return None

    chunk_idx = chunk_info.get("chunk_idx", 0)
    req_id = f"{chunk_idx:04d}-{self._request_uuid}"
    chunk_info["req_id"] = req_id
    self._active_vlm_request_ids.append(req_id)

    return self._vlm_request_pool.submit(
        self._vlm_model.inference,
        self._prompt,
        frames,
        req_id=req_id,
        video_fps=self._chunk_params.fps,        # VLM_INFERENCE_REQUIRED_KWARGS: must pass actual fps
        system_prompt="Answer the questions.",
        max_completion_tokens=self._max_completion_tokens or 256,
    )

def vlm_inference_response_process(self):
    """Stage 3b [daemon thread]: collect futures in order → _vlm_response_queue."""
    while True:
        chunk_info = self._vlm_response_future_queue.get(block=True)
        if chunk_info is None:
            break
        future = chunk_info.pop("response_future", None)
        req_id = chunk_info.get("req_id", "")
        vlm_t0 = time.time()
        try:
            if future is not None:
                chunk_info.update(future.result())
            else:
                chunk_info.setdefault("response", "")
        except Exception as e:
            logger.error(f"VLM inference failed: {e}")
            chunk_info["response"] = ""
        finally:
            try:
                self._active_vlm_request_ids.remove(req_id)
            except ValueError:
                pass

        # vlm_execute_time: wall-clock seconds for VLM inference — required by Kibana dashboard
        chunk_info["vlm_execute_time"] = time.time() - vlm_t0
        # Record when VLM result was ready — paired with pipeline_chunk_end_timestamp
        # for camera chunk_e2e latency (§ 17).
        chunk_info["pipeline_vlm_ready_timestamp"] = self._tm_e2e.now()

        if DISABLE_SOP_CHECKER or self._sop_checker is None:
            self._publish_message(chunk_info)
            self.final_queue.put(chunk_info)  # bypass Stage 4
        else:
            self._vlm_response_queue.put(chunk_info)

    # EOS
    if DISABLE_SOP_CHECKER or self._sop_checker is None:
        self.final_queue.put(None)
    else:
        self._vlm_response_queue.put(None)
```

---

## Stage 4: SOP Checker Process (CHUNK_SCHEMA_FIELD_NAMES)

```python
def sop_checker_process(self):
    """Stage 4 [sop_checker_pool, 1 worker]: SOP sequence validation → final_queue."""
    checker_id = "*"

    def run_check(req_id, vlm_output, keep_alive):
        nonlocal checker_id
        sop_request = SopCheckerRequest(
            action_json=self.action_json_content,
            vlm_output=vlm_output,
            keep_alive=keep_alive,
            checker_id=checker_id,
            cycle_completion_threshold=0.6,
            cycle_boundary_threshold_low=0.3,
            cycle_boundary_threshold_high=0.8,
        )
        resp = self._sop_checker.process_sop_check(req_id, sop_request)
        checker_id = resp.checker_id
        return resp.asdict()

    while True:
        chunk = self._vlm_response_queue.get(block=True)
        if chunk is None:
            # EOS: final check → summary chunk_idx=-1
            if checker_id != "*":
                final_req = str(uuid.uuid4())
                checker_result = run_check(final_req, "", keep_alive=False)
                self.final_queue.put({
                    "checker_result": checker_result,
                    "req_id": final_req,
                    "chunk_idx": -1,
                    "start_time": 0,
                    "end_time": 0,
                })
            break

        req_id = chunk.get("req_id", str(uuid.uuid4()))
        checker_result = run_check(req_id, chunk.get("response", "") or "", keep_alive=True)
        chunk["checker_result"] = checker_result
        self._publish_message(chunk)
        self.final_queue.put(chunk)

    self.final_queue.put(None)
```

---

## Processor Stop / VLM Abort (VLM_REAL_GPU_FRAMES, ABORT_INFLIGHT_VLM)

```python
def stop(self, force=False):
    self._stop_event.set()
    if self._pipeline is not None:
        self._pipeline.stop()
        if not force:
            self._pipeline.wait()
    self._score_queue.put(None)   # unblock clip_post_process (block=True)
    self._abort_vlm_requests()

def _abort_vlm_requests(self):
    if self._vlm_model is None:
        return
    for req_id in list(self._active_vlm_request_ids):
        try:
            asyncio.run_coroutine_threadsafe(
                self._vlm_model._llm.abort(req_id), self._vlm_model._event_loop
            ).result(timeout=5)
        except Exception:
            pass
```

---

## Kafka Messaging (KAFKA_USE_CREATE_PRODUCER)

```python
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")

# In __init__:
self._messager = None
if ENABLE_MESSAGING:
    from .messager import create_producer   # KAFKA_USE_CREATE_PRODUCER: create_producer(), not Messager
    self._messager = create_producer(KAFKA_BROKER)

def _publish_message(self, chunk_result):
    if self._messager is None:
        return
    # RTSP/sensor tagging (§ 18): ensure every published message (chunk + EOS summary)
    # carries the sensor_id flat field so a downstream dashboard can filter per stream.
    chunk_result.setdefault("sensor_id", self._sensor_id)
    # Ensure Kibana dashboard timing fields are always present on published messages
    chunk_result.setdefault("cv_execute_time", 0.0)
    chunk_result.setdefault("vlm_execute_time", 0.0)
    try:
        self._messager.produce(chunk_result, request_id=self._request_uuid)
    except Exception as e:
        logger.error(f"Failed to publish message: {e}")
```
