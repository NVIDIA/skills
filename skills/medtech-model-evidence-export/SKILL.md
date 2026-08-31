---
name: medtech-model-evidence-export
description: Exports sanitized metadata, parameters, reproducibility details, quality metrics, and optional review artifacts from Medical AI inference runs or evidence packs to MLflow. Use after inference, including NV-Generate runs; not for live training tracking, model registration, or clinical use.
license: Apache-2.0
allowed-tools: Bash
permissions: [env, file_read, file_write, network, shell]
metadata:
  author: "NVIDIA MedTech <noreply@nvidia.com>"
---

# Medtech Model Evidence Export to MLflow

## Purpose

Mirror an existing medical-inference result or evidence pack into MLflow after
the run and emit the `export_result` JSON contract. Keep the original evidence
pack as the source of truth. Training skills should add MLflow inside their
training loops instead.

## Instructions

1. Run `scripts/export_evidence_pack.py` in the default `dry-run` mode.
2. Inspect `params`, `metrics`, `artifact_plan`, and `mlflow.note.content`.
3. Choose `--mode local` or `--mode databricks` only after checking the target.
4. Keep `--artifact-policy metadata` unless the target is approved for images.
5. For `preview` or `all` in a live mode, also pass
   `--confirm-medical-artifact-upload`.
6. Keep `--source-ref`, `--note`, config filenames, and artifact filenames free
   of patient or secret identifiers; always review the dry-run output first.

Hosts with a script helper can use
`run_script("scripts/export_evidence_pack.py", args=["PACK_OR_RESULT", "--mode", "dry-run"])`.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/export_evidence_pack.py` | Export post-hoc inference evidence through MLflow. | `PACK_OR_RESULT --mode dry-run --artifact-policy metadata` |

## Prerequisites

- Python 3.10+.
- `mlflow>=2.10,<4` for `local` or `databricks` mode.
- `numpy>=1.24,<3` and `nibabel>=4,<6` for NIfTI quality metrics and previews.
- `MLFLOW_TRACKING_URI` may select a caller-managed tracking server.
- Databricks mode uses the caller's `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, or
  configured Databricks profile. The declared network endpoint is
  `https://<caller-provided-mlflow-or-databricks-workspace>`; Docker and GPU
  are not required.
- Local mode may write the MLflow store under
  `<current-working-directory>/mlruns`.

## Examples

Preview the export without contacting MLflow:

```bash
python skills/medtech-model-evidence-export/scripts/export_evidence_pack.py \
  runs/inference_pack --mode dry-run --artifact-policy metadata
```

Export a direct NV-Generate result with reproducibility metadata:

```bash
python skills/medtech-model-evidence-export/scripts/export_evidence_pack.py \
  runs/nv-generate/result.json \
  --mode local \
  --experiment-name medical-ai-inference \
  --config configs/chest_lung_tumor.json \
  --seed 0 \
  --source-ref git:61c4ec709b84cad468852243c48e250bec732074
```

Log downsampled slice previews, but not raw NIfTI files:

```bash
python skills/medtech-model-evidence-export/scripts/export_evidence_pack.py \
  runs/nv-generate/result.json \
  --mode databricks \
  --experiment-name /Shared/medical-ai-inference \
  --artifact-policy preview \
  --confirm-medical-artifact-upload
```

`--artifact-policy all` additionally uploads discovered or explicitly supplied
NIfTI images and masks, subject to `--max-artifact-mb`. Use `--image` and
`--mask` when paths are not present in the result JSON.

The exporter logs:

- scalar run and quality metrics, including sampled HU mean/std/min/max for CT
  (generic intensity statistics otherwise), a documented intensity-SNR
  heuristic, mask foreground percentage, and mapped tumor volume percentage
  when a tumor label mapping is available;
- generation parameters, model/checkpoint identity, RNG seed, and recipe hash;
- source config digest or `--source-ref`, plus a prompt digest when present;
- `mlflow.note.content` with a short human-readable run summary;
- a sanitized metadata bundle by default, optional PNG slice previews, and
  raw image/mask artifacts only under the explicit `all` policy.

## Limitations

- This is post-hoc inference export, not live training-curve tracking.
- Global intensity SNR and downsampled volume statistics are engineering
  checks, not image-quality or clinical-performance claims.
- Preview and raw artifacts may contain sensitive medical information. The
  caller must approve the destination and data policy before upload.
- The exporter does not evaluate model quality, register models, or alter the
  source evidence pack.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| Evidence source not recognized | No direct result JSON or pack `manifest.json`. | Pass the result file, evidence-pack directory, or trusted-run root. |
| MLflow import fails | Live mode lacks the declared package. | Install `mlflow>=2.10,<4` or use `--mode dry-run`. |
| Preview/all confirmation error | A live image upload was not acknowledged. | Review the destination, then pass `--confirm-medical-artifact-upload`. |
| Referenced image not found | Result paths moved after inference. | Pass current paths with `--image` and `--mask`. |
