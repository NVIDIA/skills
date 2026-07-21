---
name: i4h-catheter-navigation-digital-twin
version: "0.7.0"
description: Build a patient vasculature digital twin from CT (preprocess + segment). Use when asked to preprocess CT, segment vessels, extract centerline, or prepare ct_cache for viewport/DRR.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - catheter-navigation
    - digital-twin
    - vasculature
---

# i4h Catheter Navigation - Digital Twin

## Purpose

Download or locate a CT volume, preprocess it to an attenuation cache, and segment the arterial tree into vessel mask + centerline - the vasculature digital twin required for patient-specific viewport and DRR runs.

## Base Code

```bash
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/catheter_navigation" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/catheter_navigation" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Basics

- **Output cache layout:** `--output-dir` / `--ct-dir` (e.g. `/tmp/ct_cache`) holds `mu_volume.npy`, `metadata.json`, and after segmentation vessel mask + centerline artifacts.
- Contrast-enhanced CTA subjects work best; TotalSegmentator small subset (~3.2 GB) is the documented public dataset.
- Comply with the dataset license; no patient data is committed to the repo.

## Run

Run the steps below in order. Each step is a separate bash call; variables persist in the local agent's tmux session.

### Step 1 - resolve paths

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/catheter_navigation" ] || REPO_ROOT="$HOME/i4h-workflows"
WF_ROOT="${REPO_ROOT}/workflows/catheter_navigation"
RUN_DIR="${WF_ROOT}/runs/digital_twin_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${WF_ROOT}/runs/.latest"

# User-supplied or downloaded subject directory (must contain ct.nii.gz + segmentations/)
SUBJ="${SUBJ:-}"
CACHE="${CACHE:-/tmp/ct_cache}"

if [ -z "${SUBJ}" ] || [ ! -f "${SUBJ}/ct.nii.gz" ]; then
  echo "digital-twin: set SUBJ to an extracted TotalSegmentator subject (got '${SUBJ:-<unset>}')." >&2
  echo "Example: SUBJ=/path/to/Totalsegmentator_dataset_small_v201/s0011" >&2
  exit 1
fi
```

### Step 2 - download dataset (skip if SUBJ already exists)

Only run when the user has no CT data yet.

```bash
curl -L "https://www.dropbox.com/scl/fi/pee5yxebfxrhz007cbuy5/Totalsegmentator_dataset_small_v201.zip?rlkey=osvfk02jc4lw5gr6uhrldtb9e&dl=1" \
  -o "${RUN_DIR}/Totalsegmentator_dataset_small_v201.zip"
unzip "${RUN_DIR}/Totalsegmentator_dataset_small_v201.zip" -d "${RUN_DIR}/Totalsegmentator_dataset_small_v201"
ls "${RUN_DIR}/Totalsegmentator_dataset_small_v201"
# Then set SUBJ to one extracted subject before continuing.
```

### Step 3 - preprocess CT

```bash
"${REPO_ROOT}/i4h" run catheter_navigation preprocess_ct --local \
  --run-args="--nifti ${SUBJ}/ct.nii.gz --output-dir ${CACHE} --save-hu" \
  2>&1 | tee "${RUN_DIR}/logs/preprocess_ct.log"
```

### Step 4 - segment vessels

```bash
"${REPO_ROOT}/i4h" run catheter_navigation segment_vessels --local \
  --run-args="--ct-dir ${CACHE} --ts-gt-dir ${SUBJ}/segmentations" \
  2>&1 | tee "${RUN_DIR}/logs/segment_vessels.log"
```

## Verify

```bash
test -f "${CACHE}/mu_volume.npy"
test -f "${CACHE}/metadata.json"
ls -la "${CACHE}"
```

## Notes

- `SUBJ` must point at one extracted subject with `ct.nii.gz` and `segmentations/` (TotalSegmentator layout).
- `CACHE` is reused by [[i4h-catheter-navigation-viewport]] and cache-based [[i4h-catheter-navigation-render-drr]].
- Segmentation is CPU/GPU mixed and may take several minutes depending on volume size.

## Prerequisites

- [[i4h-catheter-navigation-setup]] completed (imports and CLI work).
- A CT NIfTI and matching vessel segmentations (or TotalSegmentator subject).
- >= 32 GB RAM recommended for large volumes.

## Limitations

- Does not ship data; user must download or provide their own CT.
- Zenodo mirror is throttled; prefer the Dropbox URL in Step 2.

## Troubleshooting

- **Error:** `SUBJ` unset or missing `ct.nii.gz` - Fix: download Step 2 dataset or set `SUBJ` to an existing subject path.
- **Error:** segment_vessels fails on `--ts-gt-dir` - Fix: confirm `${SUBJ}/segmentations` exists (TotalSegmentator ground truth).
- **Error:** out of memory during preprocess - Fix: use a smaller subject or increase swap; close other GPU/CPU workloads.

## Final Response

Report `CACHE` path, key artifacts present, log paths under `RUN_DIR`, and recommend [[i4h-catheter-navigation-viewport]] or [[i4h-catheter-navigation-render-drr]] next.
