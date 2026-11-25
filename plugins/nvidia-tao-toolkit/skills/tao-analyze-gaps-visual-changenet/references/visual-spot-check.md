# Visual Spot Check

Skip this step if `unreachable_kpi.txt` exists in `results_dir` — there is nothing meaningful to spot-check when the model can't reach the KPI at any threshold.

Otherwise, use the Read tool to **view** the test images for:

- The 5 weakest PASS samples (the top of the "PASS misclassified as NO_PASS" pile) — pick by sorting `kpi_gaps.parquet` rows where `label == 'PASS'` by `weakness` descending.
- The 5 weakest NO_PASS samples (the top of the "NO_PASS misclassified as PASS" pile) — same, with `label != 'PASS'`.

`kpi_gaps.parquet` is already expanded per-lighting (multiple rows per sample). For the spot check, deduplicate to one row per (input_path, object_name) — pick the row whose `filepath` uses the FIRST lighting from the train YAML (one image per sample is enough — VCN's classify head sees all lightings stacked, but for human spot-check one is representative).

Classify each viewed sample as exactly one of:
- **mislabeled** — visual content disagrees with the CSV label
- **edge case** — genuinely ambiguous boundary case
- **data quality** — corrupted, dark, wrong crop, bad framing
- **systematic** — model has learned the wrong feature (the image looks "obviously PASS/NO_PASS" but the model disagrees)

Copy each viewed image (resized to 128×128 if PIL is available, otherwise just copy) into `<results_dir>/rca_images/` so it can be embedded inline in the report.

This is the **only** image inspection required. Do not view dozens of images, do not run failure mode clustering, do not audit goldens — VCN does not have golden images.
