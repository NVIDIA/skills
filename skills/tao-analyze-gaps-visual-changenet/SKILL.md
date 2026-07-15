---
name: tao-analyze-gaps-visual-changenet
description: Performs gap analysis on NVIDIA TAO VCN Classify (Visual Component Net) experiments by invoking the data-services container (`tao_toolkit.data_services` from `versions.yaml`) directly via `docker run … gap_analysis vcn_aoi …` — picks the optimal decision threshold, ranks per-sample weakness, and emits a top-K weakest parquet expanded per-lighting for downstream augmentation. Use when analyzing VCN classification failures, picking SDA augmentation targets, or auditing PASS/NO_PASS boundary cases.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit and a CUDA GPU. Pulls the `tao_toolkit.data_services` image declared in `versions.yaml` at the skill bank root.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- data
- rca
- vcn
- aoi
---

# TAO VCN Classify Gap Analysis Skill

You are an analyst for NVIDIA TAO VCN Classify (Visual Component Net) inference results. Your job is to identify the **weakest samples per ground-truth label** by measuring signed distance from the decision threshold *in the wrong direction*, then surface them for downstream augmentation or relabeling.

This skill is intentionally lightweight. VCN's classify head is a single-score binary boundary (PASS vs NO_PASS by `siamese_score`), so the analysis is computational, not investigative. The whole computation lives behind one direct `docker run` invocation against the `tao_toolkit.data_services` image declared in `versions.yaml` (resolved at runtime — see Setup). The container's entrypoint takes `<category> <action> [hydra overrides...]`; we pass `gap_analysis vcn_aoi key=value …`. Each override is a bare Hydra `key=value` that selectively overrides the script's `GapAnalysisConfig` schema (defaults are baked into the container; introspect with `docker run ... gap_analysis vcn_aoi --cfg=job`). (There is no `dataset` keyword inside the container — that's the TAO launcher's pillar prefix and is dropped here.) You do **not** need delegated analysis, multi-phase image audits, or component-type clustering — VCN does not expose those dimensions. View only a small set of representative weak samples to qualify the gaps after the container returns.

CLI surface can shift between data-services container builds. If a `gap_analysis vcn_aoi` invocation fails on argument parsing, introspect the actual schema once per image with `docker run --rm "$DS_IMAGE" gap_analysis vcn_aoi --cfg=job` and reconcile any renamed keys (e.g. `inference_csv` vs `inference_results_dir`, `output_dir` vs `results_dir`) before retrying. Output parquet name is `kpi_gaps.parquet`.

---

## Inputs

1. **Experiment result directory** — contains `inference/inference.csv` from TAO VCN Classify inference. Required columns: `input_path`, `object_name`, `label`, `siamese_score`. Pass the **directory** (e.g. `inference/latest/`), not the CSV file — the container reads `inference_results_dir/inference.csv`.
2. **Training code/config directory** — contains the VCN train YAML. The container reads `dataset.classify.input_map` (lighting condition list) and `dataset.classify.image_ext` from it to expand each weak sample into one row per lighting.
3. **Dataset directory** — image root prepended to the relative `input_path` from each row (`kpi_media_path`).
4. **Schema overrides** — `min_recall`, `top_k_per_label`, and optionally a hard-pinned `threshold` are passed as Hydra overrides (defaults: `min_recall=1.0`, `top_k_per_label=50`, `threshold=-1.0` meaning sweep). **`top_k_per_label` must be a positive integer** — omitting it flips the container into "below-threshold filter" mode, which at `min_recall=1.0` returns only PASS misclassifications and zero NO_PASS rows. See Common pitfalls.

---

## Setup

The threshold sweep, weakness ranking, and per-lighting expansion all run inside the `tao_toolkit.data_services` image declared in `versions.yaml`. Resolve the concrete URI once at the top of the run, then confirm Docker, the NVIDIA container toolkit, and a GPU are present and ensure the image is cached:

```bash
# Resolve tao_toolkit.data_services → concrete nvcr.io/... URI from versions.yaml
DS_IMAGE=$(python3 -c "import yaml,os; print(yaml.safe_load(open(os.environ['TAO_SKILL_BANK_PATH']+'/versions.yaml'))['images']['tao_toolkit']['data_services'])")
echo "DS_IMAGE=$DS_IMAGE"

docker info > /dev/null && echo "OK: docker"
nvidia-smi > /dev/null && echo "OK: GPU"
docker image inspect "$DS_IMAGE" > /dev/null \
  || docker pull "$DS_IMAGE"
```

`TAO_SKILL_BANK_PATH` is usually exported by the installed skill bank. If it is unset, point it at the skill-bank repo root before resolving. A GPU is required; aborting early on a GPU-less host saves a confusing late error.

Three setup rules are load-bearing and easy to get wrong:

- **Path mounting** — every host path the container reads or writes (`inference.csv`, train YAML, dataset image root, output dir) must be bind-mounted, simplest with `-v $WORKSPACE:$WORKSPACE -w $WORKSPACE` so absolute paths resolve identically on both sides.
- **Do not pass `--user $(id -u):$(id -g)`** — it triggers `KeyError: 'getpwuid(): uid not found: <uid>'` during the container's `transformers` import; `chown` outputs back to the host UID afterwards instead.
- **`-e <spec>` is required, not optional** — current images hard-require it and exit with `ValueError: The subtask vcn_aoi requires the following argument: -e/--experiment_spec_file` before parsing CLI overrides.

See `references/container-setup.md` for the full path-mounting pattern, the `--user`/`chown` rationale and `alpine` chown command, multi-`-v` guidance, and the `-e <spec>` requirement detail.

---

## Method

The whole skill is a single `docker run` invocation followed by a small visual spot-check. The container does Steps 1–4 internally (threshold sweep, weakness scoring, top-K selection, per-lighting expansion). You handle Step 5 (visual spot-check) directly with the Read tool.

### Step 1–4 — Run the container

```bash
$DOCKER gap_analysis vcn_aoi \
    inference_results_dir=<exp_dir>/inference/<label>/ \
    train_config=<exp_dir>/train.yaml \
    kpi_media_path=<dataset_root> \
    results_dir=<rca_results_dir> \
    top_k_per_label=50
```

> **Always pass `top_k_per_label`.** This is the argument that switches the container
> from the default "samples below threshold" filter into proper top-K-per-label
> ranking. At `min_recall=1.0` the threshold is by construction at-or-below every
> NO_PASS score, so the below-threshold filter returns ONLY misclassified PASS rows
> and zero NO_PASS rows — useless as an augmentation queue. With `top_k_per_label`
> set to a positive integer (either in the spec or as a Hydra override), the
> container computes signed weakness against the threshold for every row and
> surfaces the K weakest **per ground-truth label**, which is the per-label ranked
> output downstream steps consume.

Reads `inference.csv`, sweeps every unique `siamese_score` plus one value just below the minimum, keeps the candidates with NO_PASS-class recall ≥ `min_recall` (with `1e-12` tolerance), then picks the threshold with the best F1 (tie-break: precision, then threshold value). For every row, computes signed weakness from that threshold (positive = misclassified, negative = correct, magnitude = margin). Sorts by weakness descending and takes the top `top_k_per_label` per ground-truth label, then expands each weak row into one row per lighting condition using `dataset.classify.input_map` and `dataset.classify.image_ext` from the train YAML.

If **no** candidate threshold meets the recall target, the container exits non-zero and writes `unreachable_kpi.txt` into `results_dir` explaining which recall the model can actually achieve. In that case, stop the analysis after the docker call, write a one-section report explaining the model fundamentally cannot reach the KPI at any operating point, and recommend retraining or relabeling — skip the visual spot-check.

**Container writes into `results_dir`:**

| Artifact | Contents |
|----------|----------|
| `kpi_gaps.parquet` | Top-K weakest per label, expanded per lighting. Columns: `filepath`, `label`, `siamese_score`, `weakness`. |
| `threshold.txt` | Chosen decision threshold (single float, plain text). |
| `metrics.json` | At the chosen threshold: `precision`, `recall`, `f1`, confusion matrix `{tp, fp, tn, fn}`, plus per-label `{total, mean_weakness, median_weakness, max_weakness, n_misclassified}`. |
| `weak_samples_breakdown.txt` | Per-label kept-row breakdown: `<count>` total, `<%>` of all kept rows, `N` misclassified (weakness > 0), `N` marginal (weakness ≤ 0). |
| `unreachable_kpi.txt` | Only written when the recall target is unreachable. Presence of this file means: skip Step 5, write the abridged report, recommend retrain. |

Print the container's stdout summary (chosen threshold, kept-row counts, per-label breakdown) to your own stdout so the script-check hook can verify the run produced output.

### Step 5 — Visual spot check (small, fixed)

Skip this step if `unreachable_kpi.txt` exists. Otherwise use the Read tool to **view** the 5 weakest PASS samples and the 5 weakest NO_PASS samples from `kpi_gaps.parquet` (deduplicated to one row per sample, using the FIRST-lighting `filepath`), classify each as exactly one of **mislabeled** / **edge case** / **data quality** / **systematic**, and copy each viewed image (resized to 128×128 if PIL is available, otherwise just copy) into `<results_dir>/rca_images/`. This is the only image inspection required — do not view dozens of images, run failure mode clustering, or audit goldens (VCN has no golden images).

See `references/visual-spot-check.md` for the exact sample-selection sort, the per-lighting deduplication rule, the full definition of each verdict category, and the image-copy detail.

---

## Reference invocation

Paste-and-edit the workspace, the four paths, and the two numeric knobs; this runs end-to-end. Capture stdout so the script-check hook sees row counts.

```bash
WORKSPACE=<absolute path>            # mounted identically inside the container
EXP_DIR=<experiment_result_dir>      # contains inference/inference.csv and train.yaml; must be inside $WORKSPACE
DATASET_ROOT=<dataset_root>          # image root for inference.csv input_path entries; must be inside $WORKSPACE
MIN_RECALL=1.0                       # zero-miss default; lower if KPI relaxes
TOP_K=50                             # per-label augmentation budget
OUT="$EXP_DIR/rca_results/$(date +%Y-%m-%d_%H%M%S)"
SPEC="$OUT/vcn_aoi_spec.yaml"
IMG=$(python3 -c "import yaml,os; print(yaml.safe_load(open(os.environ['TAO_SKILL_BANK_PATH']+'/versions.yaml'))['images']['tao_toolkit']['data_services'])")

mkdir -p "$OUT"

# Write the gap-analysis spec for this run
cat > "$SPEC" <<EOF
min_recall: $MIN_RECALL
top_k_per_label: $TOP_K
EOF

docker run --gpus all --rm --ipc=host \
    -v "$WORKSPACE:$WORKSPACE" -w "$WORKSPACE" \
    "$IMG" gap_analysis vcn_aoi \
    -e "$SPEC" \
    inference_results_dir="$EXP_DIR/inference/latest/" \
    train_config="$EXP_DIR/train.yaml" \
    kpi_media_path="$DATASET_ROOT" \
    results_dir="$OUT"

# Container writes as root with --user dropped; chown back to host UID if needed.
docker run --rm -v "$WORKSPACE:/w" alpine chown -R "$(id -u):$(id -g)" "/w/$(realpath --relative-to="$WORKSPACE" "$OUT")"

# Sanity print so the script-check hook sees real numbers
python3 - "$OUT" << 'PYEOF'
import json, os, sys
out = sys.argv[1]
unreachable = os.path.join(out, "unreachable_kpi.txt")
if os.path.isfile(unreachable):
    print("KPI UNREACHABLE — see", unreachable)
    sys.exit(0)
with open(os.path.join(out, "threshold.txt")) as f:
    print("threshold:", f.read().strip())
with open(os.path.join(out, "metrics.json")) as f:
    m = json.load(f)
print(f"precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}")
import pandas as pd
df = pd.read_parquet(os.path.join(out, "kpi_gaps.parquet"))
print(f"kpi_gaps.parquet: rows={len(df)}, cols={list(df.columns)}")
print(df['label'].value_counts())
PYEOF
```

---

## Outputs

Write everything into a timestamped folder under the experiment result directory. The container's outputs go straight there; the visual spot-check writes `rca_images/`; any runtime packaging hook may add session/config capture artifacts after `RCA_Report.md` is written.

```
<experiment_result_dir>/rca_results/YYYY-MM-DD_HHMMSS/
├── RCA_Report.md              # Full gap analysis report (you write this)
├── kpi_gaps.parquet           # Container: top-K weakest per label, expanded per lighting
├── threshold.txt              # Container: chosen decision threshold (single float)
├── metrics.json               # Container: confusion matrix + per-label distribution stats
├── weak_samples_breakdown.txt # Container: per-label count/misclassified/marginal counts
├── unreachable_kpi.txt        # Container: ONLY when no threshold meets min_recall
├── rca_images/                # You: thumbnails of the 10 viewed weak samples
├── rca_config/                # Auto-copied by hook
└── session log/artifacts      # Optional, runtime-dependent packaging capture
```

At the start of the run, get the real timestamp by running `date +%Y-%m-%d_%H%M%S` in Bash. Do NOT hardcode or guess. If the user specifies a custom output path, use that instead but maintain the same internal structure.

---

## Common pitfalls

The single most consequential failure mode is **forgetting `top_k_per_label` when `min_recall=1.0`**: at that recall the chosen threshold sits at or below every NO_PASS score, so without `top_k_per_label` the container falls back to a "samples below threshold" filter that returns ONLY misclassified PASS rows and zero NO_PASS rows, breaking the augmentation queue. Always include an explicit positive `top_k_per_label` (default 50) in the spec or as a Hydra override.

See `references/pitfalls.md` for the complete checklist, covering: forgetting `top_k_per_label`; passing `--user`; calling with only Hydra overrides (no `-e <spec>`); spec file outside `$WORKSPACE`; spec file with unresolved `???` sentinels; image not pulled / wrong tag; path-mount mismatch; `unreachable_kpi.txt` written; `inference.csv` missing required columns; train YAML missing `dataset.classify.input_map` or `image_ext`; `kpi_media_path` not matching `input_path` prefixes; and no GPU detected from inside the container.

---

## Report Structure

Write `RCA_Report.md` as a tight (1000–1800 word) computational gap analysis — depth comes from accurate numbers and a clear action list, not narrative. The full report template (7 sections: Verdict, Threshold Selection, Weakness Distribution, Top-K Weakest Samples, Visual Spot Check, Per-Label Breakdown, Recommended Actions — with the confusion-matrix and table layouts) is in `references/output-template.md`. When `unreachable_kpi.txt` exists, replace sections 3–6 with a single short section quoting that file's contents and collapse section 7 to one recommendation: retrain or relabel.

---

## Execution Order

1. Resolve `DS_IMAGE` from `versions.yaml` (`images.tao_toolkit.data_services`), then run `docker info`, `nvidia-smi`, and `docker image inspect "$DS_IMAGE"` (pulling if missing) once to confirm the environment. Abort with a clear message if any fail.
2. Run `date +%Y-%m-%d_%H%M%S` to get the timestamp; create `<experiment_result_dir>/rca_results/<timestamp>/`.
3. Write `vcn_aoi_spec.yaml` into the timestamped dir with `min_recall` and `top_k_per_label` filled in. Keep it under `$WORKSPACE` so the `-e` path resolves inside the container.
4. Run `docker run … "$DS_IMAGE" gap_analysis vcn_aoi -e vcn_aoi_spec.yaml inference_results_dir=… train_config=… kpi_media_path=… output_dir=…`. The container writes `kpi_gaps.parquet`, `threshold.txt`, `metrics.json`, `weak_samples_breakdown.txt` into `results_dir`. Print the chosen threshold and kept-row counts to stdout so the script-check hook can verify the run produced output.
5. If `unreachable_kpi.txt` exists, skip Step 6 and write the abridged report. Otherwise continue.
6. Pick 10 weak samples (5 weakest PASS + 5 weakest NO_PASS) from `kpi_gaps.parquet`, view each test image with Read, classify, and copy each into `rca_images/`.
7. Write `RCA_Report.md` last — writing it triggers the packaging hook, which copies session logs and skill config alongside.
