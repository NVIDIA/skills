# Eval Reference — Phases 4 and 6

Use this reference only when the parent `SKILL.md` points here for the current task. If this file conflicts with current `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current authoritative source wins.

## Contents

- Parameters
- Pre-flight
  - 1. Verify paths
  - 2. Verify SDG is complete
  - 3. Auto-detect anomaly types (if not supplied)
  - 4. Print pre-flight summary before running
- Invocation
- Output format
- Score interpretation
- Feature counts
- Error handling
- Verification checklist


Full detail for `run_eval.sh`. Read when troubleshooting unexpected scores,
interpreting output format, or running eval for the first time.

---

## Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `real_path` | Yes | — | Real anomaly images dataset. Must follow `<TEXTURE>/anomaly_image/<ANOMALY_TYPE>/` (same as `dataset_dir` used in training). |
| `generated_path` | Yes | — | SDG output directory (must contain `reconstructed_image/` and `original_mask/`). |
| `anomaly_types` | Yes | Auto-detect | `TEXTURE+TYPE` list. If omitted, auto-detect from generated filenames (see below). |
| `backbone` | No | `cradio_v3_base` | Feature extractor for **FID only**. Correspondence (`nn_score` / `mnn_score`) is locked to `dinov2-large` regardless of this flag. |
| `per_sample_csv` | No | `<generated_path>/per_sample.csv` | Always emitted. Override path with `--per-sample-csv <path>`. Columns: `anomaly_type`, `path`, `nn_score`, `mnn_score`. Consumed by `sdg-refine` and `filter_with_regen.py`. |

**Top-K averaging:** `compute_correspondence_kpi(top_k=3)` — each generated
image is scored against every real reference of its type and the three
highest-NN values are averaged. The `top_k=3` default is not exposed via CLI;
edit `cosmos_predict2/metrics/correspondence.py` if a different value is needed.

---

## Pre-flight

### 1. Verify paths

```python
import os

reconstructed_dir = os.path.join(generated_path, "reconstructed_image")
if not os.path.isdir(real_path):
    # STOP: real_path does not exist
    pass
if not os.path.isdir(reconstructed_dir):
    # STOP: no reconstructed_image/ found in generated_path
    pass
```

### 2. Verify SDG is complete

**Do not rely on `SDG_result.csv` existence alone** — it may be created before
generation finishes. Check that no SDG process is still writing to the output
directory, and count images in `reconstructed_image/` before proceeding.

### 3. Auto-detect anomaly types (if not supplied)

```bash
ls <generated_path>/reconstructed_image/ \
    | sed 's/_[0-9]*\.\(png\|jpg\)$//' \
    | sort -u
```

Extracts unique `TEXTURE+TYPE` prefixes from generated filenames.

### 4. Print pre-flight summary before running

```
=== Evaluation Pre-flight ===
Real dataset:    <real_path>
Generated:       <generated_path> (<count> images)
Anomaly types:   TEXTURE+TYPE_A, TEXTURE+TYPE_B, ...
Backbone:        cradio_v3_base
```

---

## Invocation

```bash
${ANOMALYGEN_SCRIPTS}/run_eval.sh \
    --real-path <real_path> \
    --generated-path <generated_path> \
    --anomaly-types <TEXTURE+TYPE> [<TEXTURE+TYPE> ...] \
    [--per-sample-csv <path>] \
    [--backbone cradio_v3_base]
```

---

## Output format

**Column order:** FID comes first, then correspondence scores. This surprises
people who expect `nn_score` first.

```
+-------------------+--------------------+-----------+-----------+
| Anomaly Type      | cradio_v3_base_fid | nn_score  | mnn_score |
+-------------------+--------------------+-----------+-----------+
| TEXTURE_1+TYPE_A  | 284.04             | 0.78      | 0.62      |
| TEXTURE_1+TYPE_B  | 315.64             | 0.71      | 0.51      |
| Average           | 299.84             | 0.75      | 0.57      |
+-------------------+--------------------+-----------+-----------+
```

After running, report to the user:

```
=== Evaluation Results ===
Generated:  <generated_path>/ (<N> images)
nn_score (key KPI):
  TEXTURE_1+TYPE_A:  0.78
  TEXTURE_1+TYPE_B:  0.71
  Average:           0.75
```

---

## Score interpretation

`nn_score` (DINOv2 nearest-neighbour, locked to `dinov2-large`) is the **key
KPI** — higher is better. It is **relative**: compare within the same dataset
and backbone, not across datasets. There is no fixed threshold for "good" —
use it to compare:

- same run against prior rounds (sdg-refine),
- different checkpoints of the same training run (checkpoint selection),
- different experiments on the **same** dataset with the **same** backbone.

`mnn_score` (mutual-nearest-neighbour) is a diagnostic for refinement internals.
`fid` (backbone flag) is a secondary diagnostic.

**Do not compare `nn_score` across datasets** — absolute values depend on
dataset difficulty, backbone, and sample size.

---

## Feature counts

The eval log reports feature counts that may **exceed** image count. This is
expected: `mask_crop_images` extracts individual defect instances via DBSCAN
clustering (eps=30) on mask contours. A single image with a multi-region mask
produces multiple feature crops. Feature counts also depend on how PIL resizes
masks to 512×512 at load time — small contours may merge or drop, so counts
can differ from manual mask inspection.

---

## Error handling

| Symptom | Action |
|---|---|
| Missing real or generated path | Stop; ask user to verify |
| Empty `reconstructed_image/` | Stop — SDG may not have run or output went elsewhere |
| Anomaly type not found in real data | 0 real features; check `<TEXTURE>/anomaly_image/<TYPE>/` structure |
| < 2 real or generated samples for a type | FID skipped for that type (`None`); `nn_score`/`mnn_score` still run |
| < 10 real images for a type | Warn that `nn_score` may be noisy; interpret cautiously |
| SDG still running | Wait for completion; partial output produces incorrect scores |

---

## Verification checklist

After evaluation:
1. Per-type `nn_score` (key KPI), `mnn_score`, and `fid` reported in a table.
2. Average `nn_score` and `mnn_score` printed across all types.
3. Feature counts (real vs generated) per type — flag any type with 0 real features as a path issue.
4. `per_sample.csv` written to `<generated_path>/per_sample.csv`.
