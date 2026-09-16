# Configuration Schema

> Configs live under `configs/cookbook/<use-case>/`. See the [cookbook index](../../../configs/cookbook/README.md) for the folder layout.

All YAML configs are validated against a Pydantic `PipelineConfig` model (`modules/aug_utils/schema/`). Seven top-level sections:

```yaml
data:             # List of input/output sample paths (required, min 1)
endpoints:        # LIST of API endpoints (one entry per model/role)
pipeline:         # Retry, logging, evaluation settings
captioning:       # VLM and/or LLM captioning configuration
augmentation:     # Model selection, parameters, modalities
data_processing:  # Optional processors: preprocessing.resize runs BEFORE generation, alignment + transcode AFTER
evaluators:       # Ordered list of quality checks
```

## `data` — Input/Output Samples

```yaml
data:
  - inputs:
      rgb: "/path/to/input.mp4"          # Required: source video or image
      controls:                           # Optional: control modality inputs (video_transfer)
        edge: "/path/to/edge.mp4"
        depth: "/path/to/depth.mp4"
        seg: "/path/to/seg.mp4"
        vis: "/path/to/vis.mp4"
        wsm: "/path/to/wsm.mp4"           # Cosmos3 world-scenario control
      prompt_attributes:                  # Required only for VLM+template captioning
        event_type: "person_falling"
        motion_level: "natural"
        aftermath: "affected_entity_remains_visible"
    output:
      video: "/path/to/output.mp4"        # Required: generated output (a .jpg for image edit)
      caption: "/path/to/prompt.txt"       # Required: generated prompt text
      metadata: "/path/to/metadata.json"   # Required: run metadata
      evaluation: "/path/to/eval.json"     # Optional: evaluation results
```

For Cosmos3 WSM transfer, `data[].inputs.controls.wsm` is a client-readable
video path. The `openai.video.sync` and `openai.video.async` adapters upload its
bytes as `control_reference` and send `control_type=wsm`; the generation server
creates the request-scoped `control_path`. Do not also set `control` or
`control_path` under `augmentation.parameters.extra_params.wsm`. Other WSM
options, such as `control_weight`, may remain in that object. The async adapter
accepts one WSM control only on a `video_transfer` endpoint and rejects other
control combinations.

## `endpoints` — API Endpoint Registry (a LIST)

`endpoints:` is a **list** of endpoint entries (the BYOM registry). Each entry declares a `role` and, optionally, the API-contract `adapter`. The pipeline resolves the model and each captioning/evaluator consumer to an endpoint by role (and `id` when present).

```yaml
endpoints:
  - id: vlm_qwen                 # optional exact selector; required to disambiguate 2+ same-role endpoints
    role: vlm
    url: "http://localhost:8000/v1"
    model: "Qwen/Qwen3.6-27B-FP8"
    # adapter omitted -> role default (openai.chat.completions)
  - id: llm_qwen
    role: llm
    url: "http://localhost:8002/v1"
    model: "Qwen/Qwen2.5-14B-Instruct"
  - id: cosmos_transfer
    role: video_transfer
    url: "http://10.63.146.234:8000"        # CT2.5 NIM, POST /v1/infer
    model: ""                                # single-model NIM — leave empty
    adapter: nim                             # role default is also nim
    timeout: 600
```

**Endpoint fields:** `role` (required), `url` (required), `model` (wire model string, default `""`), `id` (optional handle), `adapter` (optional → role default), `api_key_env` (name of the env var holding the key — keys are never stored literally in the config), `timeout` (seconds).

For hosted NVCF NIM endpoints, `timeout` is the total budget for the invocation
and any status polling, so set it above the expected queue plus generation time.
The NIM adapter sends `NVCF-POLL-SECONDS` and follows an HTTP 202 `NVCF-REQID`
through the authenticated NVCF status endpoint. Set `NVCF_POLL_SECONDS` to
choose each long-poll window (default 300 seconds; larger values are capped at
the NVCF gateway maximum of 300). This does not replace endpoint `timeout`,
which remains the total invocation budget.

**Roles → default adapter:** `vlm`→`openai.chat.completions`, `llm`→`openai.chat.completions`, `image_edit`→`nim`, `video_transfer`→`nim`, `video_predict`→`nim`, `image2video`→`openai.video.sync`. The `evaluator` role has no default: an external Cosmos Evaluator endpoint must explicitly set `adapter: cosmos.evaluator.arbitrator`.

**Adapters (API contracts):** generation/captioning uses `openai.chat.completions`, `openai.images.edits`, `openai.video.sync`, `openai.video.async`, `nim`, or `passthrough`. `cosmos.evaluator.arbitrator` is an evaluator-only contract and intentionally does not enter the generation adapter registry.

**Model selection:** `augmentation.model.name` resolves to an endpoint by `id`, then by `role`, then by the model-name→role map (`image-edit`→`image_edit`, `cosmos-transfer2.5`→`video_transfer`, `cosmos-predict`→`video_predict`, `cosmos3-image2video`→`image2video`). If two endpoints share the matched role, disambiguate by setting `augmentation.model.name` to a specific endpoint `id`.

Cross-section validation enforces: captioning with VLM requires a `vlm`-role endpoint; captioning with LLM requires an `llm`-role endpoint (unless using text/file captioner); VLM+template samples must select defined catalog IDs; the generation model name must resolve to exactly one endpoint, and that endpoint's effective adapter must be a known contract. An enabled external Cosmos evaluator must select an endpoint by `endpoint_id` whose role is `evaluator` and whose explicit adapter is `cosmos.evaluator.arbitrator`.

## `pipeline` — Pipeline Settings

```yaml
pipeline:
  retry: 1                           # Max retries on evaluation failure (default: 1, keep low to save time)
  regenerate_caption_on_retry: true  # On evaluator failure, bump seed and rerun captioning before retrying generation
  logging:
    enabled: true
    level: "INFO"                     # DEBUG, INFO, WARNING, ERROR
  evaluation:
    strict: true                      # Fail sample on any evaluator failure
    retain_failures: true             # Keep output files on failure
```

## `captioning` — Captioning Strategies

Captioner type is **inferred from which sub-sections and fields are present** (no explicit `type` field):

| Config Pattern | Captioner | Description |
|---------------|-----------|-------------|
| `llm.text` set | TextCaptioner | Fixed prompt with `{variable}` substitution |
| `llm.file_path` set | FileCaptioner | Sample prompts from a text file |
| Both `vlm` + `template` | TemplateCaptioner | VLM describes media, fixed template assembles prompt |
| Both `vlm` + `llm` | VLMLLMCaptioner | VLM describes media, LLM generates prompt |
| Only `vlm` | VLMCaptioner | VLM-only description |
| Only `llm` (with variables) | LLMCaptioner | LLM generates prompt from variables |

**Invalid combinations**: `vlm` + `llm.text`, `vlm` + `llm.file_path`,
`template` without `vlm`, or `template` + `llm` will raise an error.

The `vlm` and `llm` captioning consumers resolve to the `vlm`-role and `llm`-role endpoints in the `endpoints:` list. For full per-strategy YAML examples and prompt-tuning guidance, read [captioning-strategy-guide.md](captioning-strategy-guide.md). A condensed VLM+LLM example:

```yaml
captioning:
  vlm:
    parser: "instruct"              # "instruct" (user_prompt only) or "reasoning" (system + user)
    system_prompt: |
      You are a vision-language model describing scene content. Plain text only.
    user_prompt: |
      Describe this footage. Focus on weather, lighting, layout, objects, people.
      Do not suggest edits; describe only what is visible.
    parameters:
      temperature: 0.3
      top_p: 0.95
      max_tokens: 4096
      stream: false
      # fps / max_pixels are DEPRECATED no-ops: still accepted so existing configs
      # validate, but never sent. Declare the server's real knobs under extra_body.
      #
      # extra_body: vendor-specific request body, forwarded VERBATIM on VIDEO input
      # only. Unset (default) -> nothing extra is sent, which is what portable /
      # hosted VLMs want. Set it when your server needs its own knobs; there is no
      # model-name auto-detection. Qwen2.5-VL / NIM want:
      # extra_body:
      #   media_io_kwargs: {video: {fps: 4.0}}
      #   mm_processor_kwargs: {videos_kwargs: {min_pixels: 1568, max_pixels: 307200}}
      # Do NOT set these for Gemma — its processor rejects the request with a 400
      # ("Failed to apply Gemma4Processor ... kwargs={'videos_kwargs': ...}").
  llm:
    system_prompt: |
      You write one concise generation prompt from a scene caption plus attribute-value
      pairs. Output only a JSON object with a single key "prompt".
    example_prompt: |
      Change the scene to a rainy night with wet roads, preserving viewpoint and motion.
    parameters:
      temperature: 0.3
      max_tokens: 512
      stream: true
    variables:
      weather_condition: ["raining"]
      lighting_condition: ["night"]
```

### VLM + deterministic template

This mode grounds the final prompt in the VLM description but does not invoke an
LLM. The client defines allowed text once and each sample selects one ID from
each of the three supported axes:

```yaml
data:
  - inputs:
      rgb: "/workspace/data/warehouse_test.png"
      prompt_attributes:
        event_type: "person_falling"
        motion_level: "natural"
        aftermath: "affected_entity_remains_visible"
    output:
      video: "/workspace/data/out/warehouse_fall.mp4"
      caption: "/workspace/data/out/warehouse_fall_prompt.txt"
      metadata: "/workspace/data/out/warehouse_fall_metadata.json"

captioning:
  vlm:
    parser: "instruct"
    user_prompt: "Describe only visible scene facts; do not invent the requested event."
  template:
    template: |
      {scene_caption}
      Requested event: {event_type_text}
      Motion: {motion_level_text}
      Aftermath: {aftermath_text}
    attributes:
      event_type:
        person_falling: "The existing foreground worker loses balance and falls."
      motion_level:
        natural: "Use physically plausible motion with realistic timing."
      aftermath:
        affected_entity_remains_visible: "Keep the affected entity visible."
```

Every template placeholder is required exactly once, and every selected ID must
exist in its catalog. See
[captioning-strategy-guide.md](captioning-strategy-guide.md) and
`config_image2video_cosmos3_vlm_template.yaml` for the full warehouse example.

### Text Captioning (fixed prompt with variable substitution)

```yaml
captioning:
  llm:
    text: "change the person top outer color to {top_outer_color}, bottom type to {bottom_type}"
    variables:
      top_outer_color: ["blue"]
      bottom_type: ["jeans"]
```

Placeholders are validated at init time — a missing variable raises an error; unused/extra variables are logged as a warning (not fatal). No endpoint required.

### File Captioning (sample from file)

```yaml
captioning:
  llm:
    file_path: "/path/to/prompts.txt"   # One prompt per line
    seed: 42                            # Optional: reproducible selection
```

## `augmentation` — Model & Generation

```yaml
augmentation:
  model:
    name: "cosmos-transfer2.5"    # free-form string; resolves to an endpoint id/role
    version: "ct2.5"              # Optional version tag
  parameters:                     # pass-through (extra="allow"): only what you set is sent
    sigma: 90                     # Cosmos Transfer noise level
    seed: null                    # null = random per run
    guidance: 3.0
    num_steps: 35
  modalities:                     # Control modality weights (cosmos-transfer only)
    edge: 1.0
    depth: 0.4
    seg: 0.3
    vis: 0.05
    seg_control_prompt: "road surface, vehicles, sidewalks"
    positive_prompt: "cinematic, photorealistic, ultra high quality..."
    negative_prompt: "cartoon, pixelated, low quality..."
```

`ModelConfig` is just `name` (free-form string — not a closed enum) + optional `version`. There is **no `executor_type`** and **no `local_parameters`** — all inference is remote, dispatched by the endpoint's adapter. `augmentation.parameters` is a permissive pass-through surface: only explicitly-set knobs are forwarded to the endpoint (so a strict server is not handed defaults meant for another model), and numeric values for string-typed fields are coerced (e.g. `resolution: 720` → `"720"`). `seed` is located wherever you place it (top-level or inside an envelope such as `extra_params`/`extra_body`) and re-rolled on retry. Model-native knobs not listed in the schema are accepted and forwarded as-is.

Per-model parameter highlights (all optional, pass-through):
- **Cosmos Transfer 2.5:** `sigma`, `guidance`, `num_steps`, `inference_name`; control weights under `modalities`.
- **Cosmos3 WSM transfer:** source video under `data[].inputs.rgb`, WSM video under `data[].inputs.controls.wsm`, and WSM tuning such as `extra_params.wsm.control_weight` plus `extra_params.control_guidance`. Use a `video_transfer` endpoint with `openai.video.sync` or `openai.video.async`; for async, the endpoint URL is the create route ending in `/v1/videos`.
- **Cosmos Predict 2.5:** `inference_type` (`text2world`/`image2world`/`video2world`), `num_output_frames`, `enable_autoregressive`, `chunk_size`, `chunk_overlap`, `resolution`, `offload_tokenizer`, `offload_text_encoder`.
- **image edit:** the `nim` `/v1/infer` contract uses its own native knob names — `steps` (5–100) and `cfg_scale` (>1.0, default 4.0), plus `negative_prompt`/`seed`; the `openai.chat.completions` and `openai.images.edits` contracts instead use `num_inference_steps`, `guidance_scale`, `negative_prompt` (often nested under an `extra_body:` envelope the server expects).
- **image-to-video:** model-native knobs pass through verbatim (Cosmos3: `size`/`num_frames`/`fps`/`num_inference_steps`/`guidance_scale`/`flow_shift`/`extra_params`; Veo: `seconds`/`size`).

## `data_processing` — Post-Processors

Optional processors, keyed by sub-key (presence enables it — no `enabled` flag). Timing depends on the processor: `preprocessing.resize` runs **before** generation (it resizes the *input*), while `alignment` and `transcode` run **after** generation. Post-processors mutate the file at `data.output.video` and write their parameters into `data.output.metadata` under the same sub-key.

| Sub-key | Purpose |
|---------|---------|
| `preprocessing.resize` | Aspect-ratio-preserving resize of the input before generation (`target_megapixels`, `grid`, `interp`); used to match an image-edit server's internal normalization for aligned output. |
| `alignment` | 5-DOF affine MI-registration warping the generated image back into the input's frame; required for Defect Image Generation where the model upscales (e.g. 158×114 → 1216×864). GPU-only (cupy). |
| `transcode` | Normalizes the generated **video** to one codec (VP9) so the output format doesn't vary with the model that produced it (Cosmos Transfer NIM emits VP9, vLLM-omni emits H.264). Knobs: `codec` (only `vp9`), `crf` (0–63, default 31), `speed` (0–8 `-cpu-used`, default 4), `pix_fmt` (default `yuv420p`), `force` (re-encode even if already VP9). Skips image outputs and skips a source already in the target codec (no generation loss). |

> **`transcode` and GPUs:** re-encoding an **H.264** source needs `--gpus` — the image ships only the hardware `h264_cuvid` decoder (software AVC decode is off for licensing). VP9 sources decode in software and need no GPU. If the re-encode can't run, the original output is kept, the run still succeeds, and the reason lands in `metadata.transcode.error` — check that field if a uniform codec is a hard requirement.

```yaml
data_processing:
  preprocessing:
    resize:
      target_megapixels: 1.048576   # = 1024*1024 px
      grid: 32                       # snap dims to /32, aspect-ratio preserved
      interp: lanczos
  alignment:
    rot_range_deg: [-1.0, 1.0, 0.1] # rotation search [start, stop, step] in degrees
    shift_step:    1                # tx/ty grid step (px)
    bins:          64               # MI histogram bins
    interp:        bilinear         # nearest | bilinear warp kernel
    no_resize:     true             # skip pre-resize of align→ref
    min_mi:        null             # null = no MI floor; else abort threshold
  transcode:
    codec: vp9                      # only supported target
    crf: 31                         # 0-63, lower = better quality
    speed: 4                        # libvpx-vp9 -cpu-used (0-8)
    force: false                    # re-encode even when already VP9
```

**Do NOT set `sx_range`, `sy_range`, `shift_range`, or `pyr_levels`** when authoring an alignment config — they are auto-derived in `run_alignment` from the generated and reference image dimensions. The schema exposes them only as escape-hatch overrides for *visibly failing* alignment runs. Derivation rules and override criteria are in the *Post-Processing: Data-Processing Alignment* section of the config decision tree reference.

## `evaluators` — Quality Checks

Evaluators run after generation. On a quality failure, the pipeline retries with an incremented seed according to `pipeline.retry`. Three executable entry shapes are accepted: `hallucination_check`, `attribute_verification`, and `cosmos_evaluator`. The first two are built in; external Cosmos evaluation delegates checker execution to Arbitrator. VLM verification is the visual half of `attribute_verification` and therefore runs only as its nested `vlm_verification` block.

```yaml
evaluators:
  - hallucination_check:
      enabled: true
      threshold: 0.682           # Optical flow similarity threshold
      params:
        grad_thresh: 10.0
        blur_ksize: 7
        morph_k: 3
        dist_tol_px: 7.0
  - attribute_verification:
      enabled: true
      generate_natural_caption_on_pass: true
      exclude_variables: []       # variables to skip in the MCQ gate
      extra_questions:            # Additional fixed MCQ checks
        - variable: "multi_view_consistency"
          question: "Is the appearance consistent across all views?"
          options: {A: "Yes, consistent", B: "No, inconsistent"}
          correct_answer: "A"
          request_reasoning: true
      question_generation:
        endpoint_id: llm_qwen     # optional; defaults to the single llm-role endpoint
        generate_options: true    # let the LLM invent distractors (correct answer stays pinned)
        system_prompt: "..."
        parameters: {temperature: 0.2, max_tokens: 2048}
      vlm_verification:           # VLM prompt + params used to answer the MCQs
        endpoint_id: vlm_qwen     # optional; defaults to the single vlm-role endpoint
        frames: 6                 # frames sampled from a video (1 = first frame only)
        system_prompt: "..."
        parameters: {temperature: 0.0, max_tokens: 10}
```

`question_generation.endpoint_id` / `vlm_verification.endpoint_id` pick a specific endpoint by `id` (otherwise the single endpoint of the matching role is used). `generate_options` lets the LLM write distractors instead of drawing from `verification_options`. `frames` controls how many evenly-spaced video frames the VLM verifier sees — use `>1` so a mid-video event is visible (the first frame of an image→video clip is the pre-event seed). A top-level `vlm_verification:` entry is rejected during validation because it supplies no questions or expected answers; put it inside `attribute_verification`.

### External Cosmos Evaluator

External Cosmos evaluation is checker-agnostic. The endpoint is the Arbitrator,
not an individual checker, and configured names are free-form strings validated
at startup against both `GET /checkers` and `GET /dependency-graph`.

```yaml
endpoints:
  - id: cosmos_evaluator
    role: evaluator
    url: "http://cosmos-evaluator-arbitrator:8000"
    adapter: cosmos.evaluator.arbitrator      # explicit; not a generation adapter
    timeout: 1800                             # total submit-and-poll budget

evaluators:
  - cosmos_evaluator:
      enabled: true
      endpoint_id: cosmos_evaluator
      poll_interval_seconds: 5
      stream_key: camera_front_wide_120fov   # optional; derived when omitted
      output_storage_prefix: "team/run-42/"  # required relative object-key prefix
      merge_captioning_selections: true
      checks:
        - name: metropolis.attribute_verification
          gate: true
          metadata_key: attribute_verification
        - name: future.checker
          gate: false                         # preserve result; do not reject output
      inputs:
        original_video_urls:                  # optional opaque Arbitrator input
          camera_front_wide_120fov: s3://bucket/original.mp4
      config:                                 # one shared dict sent to every check
        selected_variables: {weather: snowy}
        variable_options: {weather: [sunny, cloudy, rainy, snowy]}
```

Fields and behavior:

- `enabled`: disabled entries make no network requests. At most one enabled
  Cosmos entry is allowed.
- `endpoint_id`: required and must select the explicit evaluator endpoint.
- `checks`: non-empty, with unique, non-blank names. `gate: true` requires a
  literal top-level boolean `passed` in that check's result. `gate: false`
  preserves an opaque result without using it for candidate acceptance.
- `metadata_key`: optional root-level compatibility alias. It cannot collide
  with built-in evaluator metadata, the `cosmos_evaluator` provider block, or
  another alias.
- `inputs`: deep-copied pass-through input fields such as
  `original_video_urls`, `world_model_video_urls`, and `rds_hq_url`.
  `augmented_video_urls` is reserved because runtime binds it to the generated
  candidate. Per-stream companion maps must contain the resolved stream key.
- `config`: one shared opaque dictionary forwarded to every selected checker;
  per-checker configuration maps are not part of the Arbitrator contract.
- `merge_captioning_selections`: when true, runtime `selected_variables` and
  `variable_options` are merged by key into the copied shared config. Runtime
  values win for sampled keys; configured extra keys remain. The loaded YAML is
  not mutated. When false, captioning selections are not merged.
- `poll_interval_seconds`: positive delay between execution snapshots.
- `output_storage_prefix`: required for enabled entries. It is a relative
  object-storage **key prefix**, not an `s3://` URI. Surrounding whitespace is
  stripped; a trailing slash is preserved. It must have no scheme, authority,
  query, fragment, leading slash, backslash, ASCII control character, empty
  path segment, `.` segment, or `..` segment. One final empty segment from an
  optional trailing slash is allowed. Supply a concrete prefix unique to every
  live run.

When `stream_key` is present, surrounding whitespace is stripped and the
remaining non-empty value is used unchanged. Otherwise runtime derives it from
`data[].output.video` exactly once:

1. Parse the URI and remove its query and fragment.
2. Take the final non-empty POSIX path component, percent-decode it, and remove
   only its final filename suffix.
3. Normalize the stem with Unicode NFKC.
4. Replace each maximal run outside `[A-Za-z0-9._-]` with `_`, then strip
   leading/trailing `.`, `_`, and `-`.
5. If nothing remains, use `stream-<digest>`, where `<digest>` is the first 12
   lowercase hexadecimal characters of SHA-256 over the UTF-8, query-free and
   fragment-free candidate URI.

Examples: `s3://bucket/run/candidate.mp4` becomes `candidate`; and
`https://host/run/My%20Clip.final.mp4?token=secret#frame` becomes
`My_Clip.final`. The same resolved key is reused for request inputs, result
lookup, and provider metadata.

The generated candidate and configured media inputs must be remotely readable
by the checker deployment (`s3://`, `msc://`, `gs://`, `az://`, or HTTP). Augmentation
does not upload local files solely for evaluation. Results may be nested by
stream under `checker_results[checker_name][stream_key]` or stored directly at
`checker_results[checker_name]`; other checker-specific fields stay opaque.
Quality failures may spend `pipeline.retry`; at most `pipeline.retry + 1`
candidates are generated. HTTP, discovery, polling, execution,
missing/skipped-result, or gating-contract failures are service failures and do
not generate another candidate. An ambiguous POST is surfaced as a redacted
service failure with `error_code: ambiguous_post` and a machine-readable
`request_id` field (`null` when no usable ID was received). It is never
automatically resubmitted; use the operator recovery procedure in the evaluator
setup guide. Clear failures use `error_code: service_failure`.
`pipeline.evaluation.strict`
and `retain_failures` remain authoritative after evaluation.
