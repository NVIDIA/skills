# Evaluator Setup Guide

## Overview

Evaluators run after generation to assess output quality. They are defined as an ordered list under `evaluators:` in the config. Two top-level evaluator types are available:

1. **Hallucination Check** — optical-flow-based motion artifact detection
2. **Attribute Verification** — LLM generates MCQ questions, VLM answers them
   (the VLM-side config lives in a nested `vlm_verification` block)

The schema also accepts a standalone `vlm_verification` entry, but it is **not executed** — VLM verification only runs nested inside `attribute_verification`.

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

## Retry Behavior

The retry mechanism works as follows:

1. Generation runs with initial seed
2. Evaluators run in order:
   - If **hallucination check fails** → skip attribute verification, retry immediately
   - If **attribute verification fails** → retry
3. On retry: seed incremented by 1
4. If `pipeline.regenerate_caption_on_retry: true` and a captioner is configured, captioning reruns and overwrites the prompt (saved to `output.caption`)
5. Generation reruns with the new seed (and possibly a new prompt)
6. Max attempts = `pipeline.retry + 1`

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

If `output.evaluation` is specified in the data section, a separate evaluation-only JSON is also written.
