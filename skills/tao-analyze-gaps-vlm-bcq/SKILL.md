---
name: tao-analyze-gaps-vlm-bcq
description: Extract false-positive and false-negative gaps from VLM binary-classification-question (BCQ, yes/no) predictions.
  Use when the user asks to "analyze VLM BCQ gaps", "extract VLM false positives and false negatives", or identify failure
  cases from a predictions JSON for DEFT root-cause analysis on a binary-classification VLM workflow.
license: Apache-2.0
compatibility: Requires Docker, NVIDIA Container Toolkit, and one visible GPU.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- gap-analysis
- rcca
- vlm
- evaluation
- false-positive
- false-negative
---

# VLM Binary Classification Gap Analysis

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Reads a VLM predictions JSON, compares each model response against ground truth, and writes FP/FN failure cases to a JSONL file with a summary report. Run it with a TAO Data Services spec file; the data-services entrypoint requires `-e <spec>`.

## Purpose

After running a VLM on a binary yes/no evaluation task, the predictions need to be compared against ground truth to identify failure cases. This skill produces a structured list of FP (false positive) and FN (false negative) samples that downstream RCCA stages (e.g., cosmos generation, root cause analysis) consume to drive a DEFT iteration.

## Usage

Generate a `vlm_bcq_spec.yaml` with the bundled helper:

```bash
python3 skills/data/tao-analyze-gaps-vlm-bcq/scripts/prepare_vlm_bcq_spec.py \
  --predictions-json /path/to/results.json \
  --videos-dir /path/to/videos/root \
  --results-dir /path/to/output/gaps \
  --output-spec /path/to/output/gaps/vlm_bcq_spec.yaml
```

Omit `--videos-dir` when prediction `video_id` values are already absolute. The generated spec has this shape:

```yaml
predictions_json: /path/to/results.json
videos_dir: ""
results_dir: /path/to/output/gaps
```

Set `videos_dir` when `video_id` values in the predictions are relative paths:

```yaml
predictions_json: /path/to/results.json
videos_dir: /path/to/videos/root
results_dir: /path/to/output/gaps
```

Invoke the `vlm_bcq` action inside the TAO Toolkit data services container with `-e <spec>`:

```bash
gap_analysis vlm_bcq -e /path/to/vlm_bcq_spec.yaml
```

Request exactly one GPU from the selected platform (`compute_shape.gpus: 1`,
`compute_shape.nodes: 1`). VLM BCQ gap analysis does not perform GPU compute,
but the Data Services image always calls `nvidia-smi` and fails when no GPU is
visible. One is a GPU count, not a device ID; the platform selects the device.

After the run, surface the FP/FN counts from `kpi_gaps_report.txt` and point downstream stages at `kpi_gaps.jsonl`.

## Inputs

- **config spec**: YAML file passed with `-e`. Template: `assets/default_vlm_bcq.yaml`.
- **predictions_json**: Path to predictions JSON file. Must be a JSON array where each item has `video_id`, `response`, and `gt` fields. `response` and `gt` are parsed with word-boundary matching — `'yes'` or `'no'` anywhere in the string is recognized. Samples where both or neither are present are skipped with a warning.
- **videos_dir** (optional): Base directory for resolving relative `video_id` paths. If omitted, `video_id` values are used as absolute paths.
- **results_dir**: Output directory for gap-analysis artifacts.

**Predictions JSON format:**
```json
[
  {
    "video_id": "/path/to/video.mp4",
    "response": "Yes, there is a collision.",
    "gt": "B. No",
    "question": "Is there a collision?"
  }
]
```

## Outputs

- **kpi_gaps.jsonl**: One JSON object per line for each FP/FN case. Fields: `video_id` (absolute path), `error_type` (`FP` or `FN`), `question`, `ground_truth`, `response`.
- **kpi_gaps_report.txt**: Human-readable table with total FP/FN counts.

If no gaps are found, no files are written and a message is logged.

## Spec Fields

| Parameter | Required | Description |
|-----------|----------|-------------|
| predictions_json | Yes | Path to predictions JSON file |
| results_dir | Yes | Output directory; created if it does not exist |
| videos_dir | No | Base directory for resolving relative `video_id` paths |

Keep the spec file and every path it references under the bind-mounted workspace so they resolve inside the container. Pass `-e <spec>` even if you also add Hydra overrides; current TAO Data Services entrypoints hard-require an experiment spec file before processing overrides.

## Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError` | `predictions_json` does not exist | Check the path |
| `requires the following argument: -e/--experiment_spec_file` | The container was launched without a spec file | Write `vlm_bcq_spec.yaml` and pass `gap_analysis vlm_bcq -e <spec>` |
| `ValueError: must be a JSON array` | Predictions file is not a list | Wrap predictions in `[...]` |
| `ValueError: missing 'gt'/'response'/'video_id'` | A prediction item is missing a required field | Inspect and fix the predictions JSON |
| Samples silently skipped | `response` or `gt` contains both or neither 'yes'/'no' | Check logs for warnings; inspect those samples |
