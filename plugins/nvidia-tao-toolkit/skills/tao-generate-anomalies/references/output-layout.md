# Output Layout

Every bucket that gets eval'd carries the same triad of files:
`SDG_result.csv` (generation params + `nn_score` + `guardrail_pass`),
`per_sample.csv` (per-sample nn + mnn), and `eval.log` (aggregate FID /
per-defect avg).

Each bucket's images live in four sister dirs: `reconstructed_image/`,
`original_mask/`, `annotated_image/`, and `original_image/`. All but
`annotated_image/` hold exactly one file per sample; `annotated_image/` is
written once per *anomaly instance* as `<stem>_<j>.png`, so a multi-instance
sample contributes several files there.

```
results/<name>/
├── original/                              # Phase 3 + Phase 4
│   ├── reconstructed_image/ + 3 sister dirs
│   ├── SDG_result.csv                      # with nn_score + guardrail_pass
│   ├── per_sample.csv
│   └── eval.log
├── searched/                              # final SDG bucket (Phase 6 stitch + Phase 7 filter+regen+eval)
│   ├── reconstructed_image/ + 3 sister dirs
│   ├── SDG_result.csv                      # with nn_score + guardrail_pass + source
│   ├── per_sample.csv                      # bucket-evaluated (Phase 7)
│   └── eval.log                            # canonical post-pipeline aggregate (Phase 7)
├── rounds/                                # Phase 5
│   ├── round_001/
│   │   ├── draws.json
│   │   ├── testcase.jsonl
│   │   └── sdg/{images, SDG_result.csv, per_sample.csv, eval.log}
│   ├── round_002/
│   ├── ...
│   └── search_summary.csv                  # per-sample best-of-round audit
└── regens/                                # Phase 7
    ├── regen_001/
    │   ├── allocation.json
    │   ├── amp_samples.json
    │   ├── amp/
    │   ├── testcase.jsonl
    │   └── sdg/{images, SDG_result.csv, per_sample.csv, eval.log}
    ├── regen_002/
    ├── ...
    └── regen_summary.csv                   # per-sample source + prev_nn + nn audit
```

## Image content guardrail (`guardrail_pass`)

Since 1.0.1 a SigLIP content-safety guardrail runs on **every** generated
image, in both training validation and inference. It is on by default and adds
~2% per-image cost with the classifier resident on GPU.

- Each `SDG_result.csv` row carries `guardrail_pass`: `1` = safe, `0` = blocked.
- A blocked image is **replaced with an all-black image** on disk. The file
  still exists and still counts toward the bucket total, so a count-only check
  cannot detect it.
- Blacked-out samples score near-zero `nn_score`, so Phase 7 filtering treats
  them as failures and regenerates them like any other sub-threshold sample.
- Downstream consumers must filter on `guardrail_pass=1` rather than trusting
  file presence. A CSV without the column predates 1.0.1 — treat every row as
  passing for resumability.
- Toggle with `ANOMALYGEN_IMAGE_GUARDRAIL=0|1`, which overrides the config's
  `guardrail_config.image_enabled`. If the guardrail fails to initialize it is
  disabled and logged at error level — images then run unscreened.

## Verification

1. `${ORIGINAL}/reconstructed_image/` has up to `num_SDG` images.
2. `${SEARCHED}/reconstructed_image/` count == `num_SDG` (Phase 7 fills with regen + best-per-defect fallback if needed).
3. `${ROUNDS}/search_summary.csv` has one row per sample.
4. `original/eval.log`, each `rounds/round_NN/sdg/eval.log`, and `searched/eval.log` contain per-type `nn_score`, `mnn_score`, and `fid`.
5. `${REGENS}/regen_summary.csv` exists when Phase 7 ran; `passed_threshold` column reports per-sample status, `prev_nn` vs `nn_score` reveals which samples regen actually improved.
6. No `guardrail_pass=0` rows remain in `${SEARCHED}/SDG_result.csv` — any that survive Phase 7 are all-black images and must not be handed downstream.
