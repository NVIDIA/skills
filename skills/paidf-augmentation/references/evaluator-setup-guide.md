# Evaluator Setup Guide

## Overview

Evaluators run after generation to assess output quality. They are defined as a list under `evaluators:` in the config. Three executable top-level evaluator types are available:

1. **Hallucination Check** — optical-flow-based motion artifact detection
2. **Attribute Verification** — LLM generates MCQ questions, VLM answers them
   (the VLM-side config lives in a nested `vlm_verification` block)
3. **External Cosmos Evaluator** — checker-agnostic quality gates orchestrated
   by a separately deployed Cosmos Evaluator Arbitrator

`vlm_verification` is not a standalone evaluator: it has no independent
question/expected-answer contract. A top-level entry is rejected during config
validation. Nest it under `attribute_verification`, where the LLM-generated or
fixed MCQs provide that contract.

On failure, the pipeline retries with an incremented seed up to `pipeline.retry` times.

## Hallucination Check

Detects motion artifacts by comparing optical flow between the original and augmented video. A high score means the output preserves the original motion well.

**Note**: This evaluator only works for video outputs. It is not applicable to image outputs (e.g., image edit), since optical flow requires multiple frames.

```yaml
evaluators:
  - hallucination_check:
      enabled: true
      threshold: 0.682              # Score >= threshold = pass
      params:
        grad_thresh: 10.0           # Gradient threshold for motion detection
        blur_ksize: 7               # Gaussian blur kernel size
        morph_k: 3                  # Morphological operation kernel
        dist_tol_px: 7.0            # Distance tolerance in pixels
        max_frames: null            # null = all frames; set integer to limit
```

### Tuning Hallucination Check

| Symptom | Parameter to Adjust | Direction |
|---------|-------------------|-----------|
| Too many false failures (good videos rejected) | `threshold` | Lower (e.g., 0.5) |
| Artifacts not caught | `threshold` | Raise (e.g., 0.8) |
| Slow on long videos | `max_frames` | Set to 30–60 |
| Noisy input causing false motion detection | `blur_ksize` | Increase (must be odd) |
| Small artifacts missed | `grad_thresh` | Lower (e.g., 5.0) |

**Typical scores**: Good augmentations score 0.85–0.99. Hallucinated motion artifacts usually score below 0.5.

**No additional endpoints required** — runs locally using OpenCV optical flow.

## Attribute Verification

A two-stage evaluator that verifies whether the generated output matches target attributes:

1. **LLM Question Generator** — creates MCQ questions from variables (e.g., "What color is the person's shirt? A) Red B) Blue C) Green")
2. **VLM Verifier** — answers the MCQ questions by looking at the generated output

**Note**: For video outputs, `vlm_verification.frames` controls how many evenly-spaced frames the VLM verifier samples (default `1` = first frame only). Use `frames: >1` (e.g. 6) when the attribute is a **mid-video event** — the first frame of an image→video clip is the pre-event seed, so `frames: 1` cannot see a collapse/fall/etc. For steady-state attributes (e.g. clothing color, weather) the first frame is usually enough.

```yaml
evaluators:
  - attribute_verification:
      enabled: true
      generate_natural_caption_on_pass: true    # Generate natural description on success
      natural_caption:                          # Config for natural caption generation
        system_prompt: |
          Write a single natural sentence describing the person's appearance.
        user_prompt_template: |
          This image shows a person with: {attributes_text}.
          Write one sentence describing the person's appearance.
      extra_questions:                          # Custom MCQ questions (in addition to auto-generated)
        - variable: "multi_view_consistency"
          question: "Is the person's appearance consistent across all views?"
          options:
            A: "Yes, consistent across all views"
            B: "No, inconsistent across views"
          correct_answer: "A"
          request_reasoning: true
      question_generation:
        endpoint_id: llm_qwen       # optional; defaults to the single llm-role endpoint
        generate_options: false     # true = let the LLM invent distractors (correct answer stays pinned)
        system_prompt: |
          You are an expert at creating multiple choice verification questions.
          Generate a simple, direct question that verifies a specific attribute.
          The question must have 2-4 answer options.
          Output as a single JSON object.
        parameters:
          retry: 1
          temperature: 0.2          # Low temperature for consistent question format
          top_p: 0.95
          frequency_penalty: 0.0
          presence_penalty: 0.0
          max_tokens: 2048
          stream: true
      vlm_verification:             # VLM prompt + params used to answer the MCQ questions
        endpoint_id: vlm_qwen       # optional; defaults to the single vlm-role endpoint
        frames: 6                   # evenly-spaced video frames to sample (1 = first frame only)
        system_prompt: |
          You are an expert vision model. Analyze the frame(s) and select the best
          answer from the options. Respond with ONLY a single letter (A, B, C, or D).
        parameters:
          retry: 5                  # Higher retry for flaky VLM responses
          temperature: 0.0          # Deterministic for consistent answers
          top_p: 1.0
          frequency_penalty: 0.0
          max_tokens: 10            # Only need a single letter
          stream: false
```

### Endpoint Requirements

Attribute verification requires **both** an `llm`-role endpoint (question generation) and a `vlm`-role endpoint (answering questions against the generated output) in the `endpoints:` list. By default each consumer uses the single endpoint of its role; set `question_generation.endpoint_id` / `vlm_verification.endpoint_id` to target a specific endpoint by `id` when more than one shares the role. Keys resolve per endpoint via `api_key_env` → role default (`LLM_API_KEY`, `VLM_API_KEY`); unauthenticated endpoints (e.g. local vLLM) need none.

### How Variables Drive Verification

The verification questions are auto-generated from `captioning.llm.variables` (or `verification_values` if set):

```yaml
captioning:
  llm:
    variables:
      top_outer_color: ["blue"]
      shoe_type: ["boots"]
    verification_options:
      top_outer_color: ["red", "blue", "green", "black", "white"]
      shoe_type: ["sneakers", "boots", "sandals", "heels"]
```

For each variable, the LLM generates a question like:
```
What color is the person's outer top garment?
A) red
B) blue       ← correct answer (from variables/verification_values)
C) green
D) black
```

The VLM then answers by looking at the generated image/video. If it selects "B", the check passes.

### Extra Questions

Custom questions appended to the auto-generated ones. Useful for domain-specific checks:

```yaml
      extra_questions:
        - variable: "multi_view_consistency"      # Variable name for metadata
          question: "Is the appearance consistent across all views?"
          options:
            A: "Yes, consistent"
            B: "No, inconsistent"
          correct_answer: "A"
          request_reasoning: true                  # Ask VLM for reasoning
```

Fields:
- `variable` — optional, associates the question with a named attribute in metadata
- `question` — the question text
- `options` — dict of letter → answer text (2–4 options)
- `correct_answer` — the expected correct letter
- `request_reasoning` — if true, asks VLM to explain before answering

### Natural Caption Generation

When `generate_natural_caption_on_pass: true`, after all attribute checks pass, the VLM generates a natural-language description of the output:

```yaml
      generate_natural_caption_on_pass: true
      natural_caption:
        system_prompt: |
          You are an expert at describing images of people. Be concise and factual.
        user_prompt_template: |
          This image shows a person with: {attributes_text}.
          Write one natural sentence describing the person's appearance.
```

The `{attributes_text}` placeholder is replaced with the verified attribute values (e.g., "top outer color: blue; shoe type: boots"). The resulting caption is saved to `metadata.natural_caption`.

## External Cosmos Evaluator

This evaluator submits an externally readable generated candidate to a Cosmos
Evaluator Arbitrator. Augmentation selects free-form checker names through
configuration and interprets only their common top-level `passed: bool` gating
contract; checker-specific scores and fields remain opaque.

The endpoint must use the evaluator-only contract:

```yaml
endpoints:
  - id: cosmos_evaluator
    role: evaluator
    url: "http://cosmos-evaluator-arbitrator:8000"
    adapter: cosmos.evaluator.arbitrator
    timeout: 1800
```

The hostname above assumes the augmentation container joins the same Docker
bridge as Arbitrator. For a separate deployment, use its trusted URL. If that
deployment requires authentication, add `api_key_env` with the name of the
environment variable holding the key. The key is sent as
`Authorization: Bearer <key>`; never put its value in YAML.

```yaml
evaluators:
  - cosmos_evaluator:
      enabled: true
      endpoint_id: cosmos_evaluator
      poll_interval_seconds: 5
      stream_key: camera_front_wide_120fov
      output_storage_prefix: "team/evaluations/run-42/"
      merge_captioning_selections: true
      checks:
        - name: metropolis.attribute_verification
          gate: true
          metadata_key: attribute_verification
      inputs: {}
      config:
        selected_variables:
          weather: snowy
          time_of_day: night
        variable_options:
          weather: [sunny, cloudy, rainy, snowy]
          time_of_day: [morning, night]
```

### Discovery and request flow

Before captioning or generation, augmentation calls Arbitrator's `/health`,
`/checkers`, and `/dependency-graph` endpoints. Every selected checker must be
both healthy and present in the graph. For each candidate, augmentation then:

1. Deep-copies `inputs` and injects the generated URI at
   `augmented_video_urls[stream_key]`.
2. Deep-copies the one shared `config`; when
   `merge_captioning_selections: true`, sampled captioning values/options are
   merged by key and win over matching configured keys.
3. Submits once to `POST /process/async`, then polls
   `GET /executions/{request_id}` until terminal within the endpoint timeout.
4. Reads each configured result either from
   `checker_results[checker_name][stream_key]` or directly from
   `checker_results[checker_name]`. The pinned service stores `response_json`
   as a JSON string; the client also accepts an already-decoded object.

`output_storage_prefix` is required for enabled evaluation and is a relative
key prefix in storage configured for the checker services—not a storage URI.
Use a concrete prefix unique to each run. See
[configuration-schema.md](configuration-schema.md#external-cosmos-evaluator)
for its exact validation rules and the deterministic fallback stream-key
algorithm.

`inputs` can carry current or future Arbitrator fields without checker-specific
augmentation code. Common media fields are `original_video_urls` (for
Hallucination), `world_model_video_urls`, and `rds_hq_url` (for Objects).
Per-stream companion maps must contain the resolved stream key.

### Gating and failures

- `gate: true`: the selected result must contain a literal top-level boolean
  `passed`. `false` is a quality failure and can use the existing generation
  retry/seed re-roll.
- `gate: false`: the result is retained as observation metadata but does not
  reject the candidate; it need not expose `passed`.
- A terminal parent execution status of `success` means checker jobs completed;
  it is not itself a quality pass. Augmentation still inspects every gate.
- Submission/polling/discovery failures, a failed execution, missing or skipped
  results, and an invalid gating contract are service failures. Safe GETs are
  retried, but an ambiguous POST is never resubmitted because Arbitrator has no
  idempotency key. Service failures stop evaluation without generating a new
  candidate. The provider block records a redacted `service_failure`; with the
  defaults (`strict: true`, `retain_failures: true`) the generated candidate is
  retained for inspection, the sample is not counted as successful, and the
  run exits non-zero.

The persisted `cosmos_evaluator` provider block is machine-readable. It always
contains `request_id` (an integer after a valid acceptance response, otherwise
`null`). `error_code: ambiguous_post` means the POST may have reached
Arbitrator but no usable request ID reached the client; agents must enter the
recovery flow below. Other clear operational failures use
`error_code: service_failure` and must not be treated as ambiguous submissions.

For an ambiguous POST (the request may have reached Arbitrator but no
`request_id` reached the client), recovery is deliberately operator-driven:

1. Do not blindly resubmit. Check Arbitrator logs and the checker storage area
   identified by that run's unique `output_storage_prefix` to determine whether
   the original execution exists or completed.
2. If the original execution can be identified, inspect or finish handling that
   execution without posting the candidate again.
3. If its state cannot be established, preserve the retained candidate, choose
   a new unique `output_storage_prefix`, and rerun it as a new evaluation. This
   may leave an orphaned first execution, but avoids silently treating an
   uncertain submission as a quality rejection.

Automatic POST retry is unsafe until Arbitrator offers an idempotency key or a
client-supplied request identifier that can be queried after a lost response.

The generated video and every configured media input must be readable by the
external checker containers, normally through shared object storage or HTTP.
The pipeline does not upload local files solely for evaluation.
Remote evaluation adds no GPU requirement to the augmentation container.

### CT3 Omni external evaluator example

`configs/cookbook/video-data-augmentation/config_video_transfer_CT3_omni.yaml`
is the shipped external-evaluator VDA example. It has no built-in evaluator
fallback: one Cosmos entry selects gated `metropolis.hallucination` and
`metropolis.attribute_verification` and exposes their results through the
compatibility aliases `hallucination_check` and `attribute_verification`.

Hallucination requires both sides of the comparison. Augmentation injects the
generated remote URI as `augmented_video_urls[stream_key]`; the config derives
`inputs.original_video_urls[stream_key]` from `data[0].inputs.rgb`, so one input
override updates both consumers. The stream key is `camera_front_wide_120fov`.
The generated video is also remote, while `output_storage_prefix` remains a
unique relative key.

The shared flat `config` overrides the Hallucination threshold while relying on
the checker's defaults for its motion-mask settings. It defines snow/night
Attribute Verification selections and supplies checker-side LLM/VLM URL/model
values. Those service names must resolve from the checker containers, not only
from augmentation.

The deployed external Attribute Verification checker examines the first video
frame and does not generate `natural_caption`. Replace S3 and
`output_storage_prefix` placeholders before running. Both checks are gates and
`pipeline.evaluation.strict` is true, so either failure marks the sample
unsuccessful. `pipeline.retry: 0` means it does not generate a replacement.
Outputs are still retained (`retain_failures: true`).

## Evaluator Combinations

### No Evaluators (Generation Only)

```yaml
# Simply omit the evaluators section
# evaluators: null
```

### Hallucination Check Only

```yaml
evaluators:
  - hallucination_check:
      enabled: true
      threshold: 0.682
```

### Attribute Verification Only (Image Editing)

Common for image edit where there's no motion to hallucination-check:

```yaml
evaluators:
  - attribute_verification:
      enabled: true
      question_generation:
        system_prompt: "..."
      vlm_verification:
        system_prompt: "..."
```

### Full Evaluation (Video Augmentation)

Hallucination check runs first; if it passes, attribute verification runs:

```yaml
evaluators:
  - hallucination_check:
      enabled: true
      threshold: 0.682
      params:
        grad_thresh: 10.0
  - attribute_verification:
      enabled: true
      question_generation:
        system_prompt: "..."
      vlm_verification:
        system_prompt: "..."
```

### External Cosmos Gate (CT3 Omni)

Use the `config_video_transfer_CT3_omni.yaml` example. It contains one
external gate rather than duplicating the built-in attribute verifier:

```yaml
evaluators:
  - cosmos_evaluator:
      enabled: true
      endpoint_id: cosmos_evaluator
      output_storage_prefix: "<OUTPUT_PREFIX>/ct3-omni/<RUN_ID>/"
      checks:
        - name: metropolis.attribute_verification
          gate: true
          metadata_key: attribute_verification
```

## Retry Behavior

The retry mechanism is a bounded candidate loop:

1. Generation runs with initial seed
2. Evaluators run in order for that candidate:
   - If **hallucination check fails** → skip attribute verification, retry immediately
   - If **attribute verification fails** → retry
   - If an **external Cosmos gate returns `passed: false`** → retry
3. A quality failure creates a replacement only when attempts remain. On that
   retry, the seed is incremented by 1.
4. If `pipeline.regenerate_caption_on_retry: true` and a captioner is configured, captioning reruns and overwrites the prompt (saved to `output.caption`)
5. Generation reruns with the new seed (and possibly a new prompt)
6. The loop ends after a pass or after exactly `pipeline.retry + 1` candidate
   attempts. Exhaustion is an evaluation failure; there is no further cycle.

External service/contract failures are not quality failures and do not spend a
generation retry: they stop the loop on the current candidate and follow
`strict`/`retain_failures`. Observe-only Cosmos checks never trigger a retry.

**Seed progression**: If original seed is 12345, retries use 12346, 12347, etc.

**Pipeline settings** control retry behavior:
```yaml
pipeline:
  retry: 1
  regenerate_caption_on_retry: true # Rerun captioning before a retry if evaluators fail
  evaluation:
    strict: true                    # Fail sample on any evaluator failure
    retain_failures: true           # Keep output files even on failure
```

## Metadata Output

After evaluation completes, results are written to `output.metadata`:

```json
{
  "prompt": "change the person top outer color to blue...",
  "selections": {"top_outer_color": "blue", "shoe_type": "boots"},
  "output_media_path": "/workspace/data/output.png",
  "input_media_path": "/workspace/modules/input.png",
  "control_media": {},
  "hallucination_check": {
    "passed": true,
    "score": 0.9685,
    "threshold": 0.682,
    "attempt": 1,
    "seed_used": 42
  },
  "attribute_verification": {
    "passed": true,
    "details": { ... },
    "attempt": 2,
    "seed_used": 43
  },
  "natural_caption": "A person wearing a blue shirt and black jeans with brown boots."
}
```

An external run instead adds its sanitized provider block and any configured
compatibility alias. It does not imply a built-in evaluator also ran:

```json
{
  "cosmos_evaluator": {
    "request_id": 814,
    "status": "success",
    "version": "1.1.0",
    "commit_sha": "...",
    "stream_key": "camera_front_wide_120fov",
    "checks": {
      "metropolis.attribute_verification": {
        "camera_front_wide_120fov": {
          "passed": true,
          "result": {"passed": true}
        }
      }
    }
  },
  "attribute_verification": {
    "passed": true,
    "result": {"passed": true}
  }
}
```

If a Cosmos check declares `metadata_key`, the same sanitized compatibility
view is written at that root key when schema validation proves it cannot
collide. If `output.evaluation` is specified in the data section, a separate
evaluation-only JSON is also written.
