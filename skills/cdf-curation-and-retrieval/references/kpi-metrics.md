# KPI Metrics Reference

Detailed metric definitions, interpretation, and baseline targets for
evaluating pipeline output quality.

> **Note:** This repository does not currently ship a KPI evaluator. The
> metric definitions and diagnostic patterns below are for designing a
> custom workflow against cosmos-curator output.

## Evaluation Data Sources

Cosmos-curator split output provides the raw data for evaluation:

- **Captions**: `{output_clip_path}/metas/v0/*.json` -- dynamic window
  keys like `windows[].qwen3_vl_30b_fp8_caption`
- **Classifications**: `{output_clip_path}/metas/v0/*.json` --
  `qwen_type_classification`
- **Aggregated captions**: `{output_clip_path}/v0/all_window_captions.json`
- **Pipeline stats**: `{output_clip_path}/summary.json`

## Metric Definitions

### Overall Accuracy

```
overall_accuracy = correct_answers / total_questions
```

Counts all matched questions across all videos. Unmatched questions excluded.
Target: > 80%.

### Per-Category Accuracy

```
category_accuracy = correct_in_category / total_in_category
```

| Prefix | Category |
|--------|----------|
| `1_` | accident_detection |
| `2_` | emergency_response |
| `3_` | environmental_conditions |
| `4_` | traffic_violations |
| `5_` | traffic_congestion |

Target: > 75% per category.

### Per-Question Accuracy

```
question_accuracy = correct_for_question / total_for_question
```

Tracks each specific question ID (e.g., `1_1`, `2_3`). Target: > 70%.

### Confusion Matrix

Format: `(ground_truth_answer, generated_answer): count`.
Identifies systematic answer biases.

### Per-Video Breakdown

Per-video correct/total/accuracy with per-question detail.

## Baseline Targets (Traffic Safety)

| Metric | Baseline | Good | Excellent |
|--------|----------|------|-----------|
| Overall Accuracy | 70% | 80% | 90%+ |
| Accident Detection (1_*) | 75% | 85% | 95%+ |
| Emergency Response (2_*) | 65% | 75% | 85%+ |
| Environmental (3_*) | 70% | 80% | 90%+ |
| Violations (4_*) | 60% | 75% | 85%+ |
| Congestion (5_*) | 60% | 75% | 85%+ |

## Interpretation Guidelines

**< 60%:** Fundamental prompt issues. Check output format, event definitions,
VLM model appropriateness.

**60-75%:** Targeted improvements needed. Identify weak questions, add failure
modes, calibrate thresholds.

**75-85%:** Good baseline. Focus on edge cases, visual-grounding constraints,
consider higher `captioning_sampling_fps`.

**> 85%:** Production-ready. Monitor consistency, novel scenarios, model drift.

## Diagnostic Patterns

### High False Positive Rate

VLM says "accident" when GT says "no accident."
1. Check confusion matrix for `(No, Yes)` pattern.
2. Review failure modes -- prompt too aggressive?
3. Strengthen default-deny baseline (Element 3 in prompt framework).

### High False Negative Rate

VLM says "no accident" when GT says "accident."
1. Check confusion matrix for `(Yes, No)` pattern.
2. Review visual signatures -- specific enough?
3. Lower confidence threshold or increase `captioning_sampling_fps`.

### Category-Specific Issues

- **Environmental wrong:** Add weather/lighting vocabulary to prompt.
- **Violations wrong:** Increase `captioning_sampling_fps`, add signal detection.
- **Congestion wrong:** Align numerical thresholds with GT definitions.

## Iteration Loop

```text
Generate Config -> Run Split Pipeline -> Evaluate Output
    ^                                        |
    +------- Diagnose Weak Areas -> Improve Config/Prompts -+
```

Make ONE change per iteration. Track accuracy across iterations.

## Novel Domain KPI Setup

1. Map the 5-prefix system to domain equivalents.
2. Design 10-15 questions per video.
3. Annotate 50-100 videos with ground truth.
4. Run evaluation against cosmos-curator output.
5. Establish baseline targets from initial run.
