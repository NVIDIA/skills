# Prep-Testcase Reference — Phase 2 JSONL Preparation

Use this reference only when the parent `SKILL.md` points here for the current task. If this file conflicts with current `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current authoritative source wins.

## Contents

- What it produces
- Pairing strategy and n_seeds
- Parameters
- Invocation
- Helper scripts (each supports `--help`)
- Submask handling
- Verification
- Error handling


Full detail for `prep_testcase.sh`. Read when debugging AMP failures,
understanding allocation, or customising clean-image pairing.

---

## What it produces

A JSONL that pairs clean images with AMP-placed masks. Every defect routes to
one of three AMP branches based on `spatial_dependency` in `defect_spec`:

| spatial_dependency | AMP branch | Extra inputs |
|---|---|---|
| `free` | whole-image ROI | none |
| `text` | text2roi (Qwen VL + SAM2) | `roi_prompt_defect_location` (text, required) |
| `cad` | cad2roi | `<dataset>/<TEXTURE>/cad_mask/<stem>.png` + `<dataset>/semantic_segmentation_labels.json` |

`run_auto_roi_amp.py` defaults unrecognized values to `text`; auto-cad
routing fires whenever the per-sample record has a non-null `cad_mask`
regardless of `spatial_dependency`.

The JSONL contains no mask-augmentation fields — AMP handles placement.

---

## Pairing strategy and n_seeds

Pair budget per defect = `num_submasks[d] × num_cleans[texture]` (every
submask × clean combination). Pairing iterates every combination once
(deterministic shuffle per defect) before any pair repeats. n_seeds > 1
only when allocation exceeds the budget:

```
n_seeds = max_d ⌈allocation[d] / (num_submasks[d] × num_cleans[texture])⌉
```

**Pipeline invariant:**

```
num_SDG → allocation (proportional to mask counts)
       → ⌈alloc[d]/n_seeds⌉ AMP records per defect
         (iterating every submask × clean combination once before any repeat)
       → ≥num_SDG AMP masks (n_seeds placements per record once all pairs used)
       → N JSONL rows (first alloc[d] per defect)
```

**Two modes (selected via `--mode`):**

| | `--mode validation` | `--mode inference` (default) |
|---|---|---|
| Allocation across defect types | **Proportional** to training mask counts (largest-remainder rounding) | **Uniform** — `base = num_sdg // N`; first `num_sdg % N` defects get +1 |
| KPI floor (per-defect ≥ 1) | **Enforced** (raises if any defect would get 0) | **Skipped** (allows 0 per defect, e.g., when `--per-defect-counts` excludes a type) |
| `--per-defect-counts <JSON>` | **Rejected** (would break KPI balance) | **Accepted** — bypasses uniform with explicit counts |
| Typical use | Phase 1 validation JSONL — `num_SDG` = total training mask count → every training submask appears once | Phase 2 SDG inference — `num_SDG` = user target |

**Inference + `--per-defect-counts`** lets the skill translate natural-language
intent like "generate 5 IC+bridge and 10 passive_component+missing" into:

```bash
--num-sdg 15 --per-defect-counts '{"IC+bridge":5,"passive_component+missing":10}'
```

Sum mismatch (override sum ≠ `--num-sdg`) prints a warning to stderr and uses
the override sum as effective `num_sdg`. The skill should reconcile with the
user before invoking.

---

## Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `name` | yes | — | Experiment label. |
| `num_SDG` | yes | — | Total SDG entries across all defect types. `0` → stop. |
| `dataset_dir` | yes | — | Training dataset root. Drives allocation, submask source, cad_mask lookup (`<dataset>/<TEXTURE>/cad_mask/<stem>.png`), and cad labels (`<dataset>/semantic_segmentation_labels.json`). |
| `clean_dir` | no | `dataset_dir` | Clean images. Layouts probed in order: `<clean_dir>/<TEXTURE>/clean_image/*`, `<clean_dir>/<TEXTURE>/*`, flat `<clean_dir>/*`. Omit when clean images are at `<dataset_dir>/<TEXTURE>/clean_image/`. |
| `defect_spec` | yes | — | JSONL tagging each defect `free`/`text`/`cad`. `text` entries need `roi_prompt_defect_location`. Template at `.agents/skills/anomalygen/assets/defect_spec_template.jsonl`. |
| `mode` | no | `inference` | `inference` (uniform allocation, no KPI floor) or `validation` (proportional + ≥1-per-defect floor). |
| `per_defect_counts` | no | — | JSON dict overriding uniform allocation in inference mode, e.g. `'{"IC+bridge":5,"passive_component+missing":10}'`. Not allowed with `--mode validation`. |
| `guidance` | no | `7.0` | Default guidance written to each JSONL entry (overridden per-sample in Phase 5). |
| `crop_ratio` | no | `2.0` | Default crop ratio. Matches `cosmos_predict2/data/anomaly_gen/anomaly_dataset.py` fallback. |
| `seed` | no | `42` | Base random seed for `run_auto_roi_amp.py`. |

Defect types are derived from `defect_spec` — no separate `--defect-types` arg.

---

## Invocation

```bash
${ANOMALYGEN_SCRIPTS}/prep_testcase.sh \
    --name <name> \
    --num-sdg <N> \
    --dataset-dir <dataset_dir> \
    --clean-dir <path> \
    --defect-spec <path> \
    --amp-output-dir ag_inference/<name>/amp \
    --output-jsonl ag_inference/<name>/testcase.jsonl \
    [--mode inference|validation]            # default: inference
    [--per-defect-counts '{"t":N,...}']      # inference only
    [--guidance 7.0] [--crop-ratio 2.0] [--seed 42]
```

**Do NOT pass `--seeds`** — it is not a recognized flag and the script will
halt with `unknown arg`. n_seeds is auto-computed internally.

---

## Helper scripts (each supports `--help`)

| Script | Role |
|---|---|
| `validate_amp_inputs.py` | Pre-flight: cross-check dataset layout, clean pool, cad masks, cad labels, and `roi_prompt_defect_location`. Runs automatically as step 1. |
| `allocate_samples.py` | Allocates `num_SDG` across defect types. `--mode inference` (default): uniform; accepts `--per-defect-counts`. `--mode validation`: proportional to mask counts, ≥1-per-defect floor. |
| `build_amp_samples.py` | Emit exactly `allocation[defect]` AMP input records per defect. |
| `build_jsonl.py` | Scan AMP output, pair with clean images, honor allocation ceiling. |
| `verify_jsonl.py` | Resize mismatched masks into `resized_masks/` cache; validate all paths. |

The AMP branching lives in `scripts/anomaly_gen/run_auto_roi_amp.py` (repo root).

---

## Submask handling

`build_amp_samples.py` does not override `preprocess_submask`'s
`submask_split_largest=True` default — a multi-component training submask
(e.g., two scratches on one image) becomes a single-component mask.
Set `submask_split_largest: false` on an individual record in
`amp_samples.json` before `build_jsonl.py` runs if preserving multiple
components matters for that defect.

---

## Verification

After `prep_testcase.sh` completes:
- Output JSONL has `num_SDG` entries (minus logged AMP skips).
- `allocation.json` sums to `num_SDG`.
- `amp_samples.json` has exactly `num_SDG` records.
- `<amp-output-dir>/<clean_stem>__<submask_stem>/<TEXTURE>+<ANOMALY>/seed0.png` exists per AMP record.

---

## Error handling

| Symptom | Action |
|---|---|
| Validator failure | Stop with itemised report (missing submask/clean/cad/prompt) |
| AMP output count < allocation for some defect | `build_jsonl.py` warns and writes what's available; JSONL is short by that delta. Check `run_auto_roi_amp.py` logs for `NO_DETECTION` / `FAILED` |
| AMP produced 0 outputs for some defect | `build_jsonl.py` warns, defect is dropped from JSONL |
| AMP produced 0 outputs for **every** defect | `build_jsonl.py` halts with `error: 0 entries written` — SDG cannot run on empty JSONL |
| Mask-size mismatch | `verify_jsonl.py` auto-resizes into `resized_masks/` cache |
| `num_SDG = 0` | Stop — nothing to generate |
