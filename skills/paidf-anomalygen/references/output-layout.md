# Output Layout

Every bucket that gets eval'd carries the same triad of files:
`SDG_result.csv` (generation params + `nn_score`), `per_sample.csv`
(per-sample nn + mnn), and `eval.log` (aggregate FID / per-defect avg).

```
results/<name>/
├── original/                              # Phase 3 + Phase 4
│   ├── reconstructed_image/ + 3 sister dirs
│   ├── SDG_result.csv                      # with nn_score
│   ├── per_sample.csv
│   └── eval.log
├── searched/                              # final SDG bucket (Phase 6 stitch + Phase 7 filter+regen+eval)
│   ├── reconstructed_image/ + 3 sister dirs
│   ├── SDG_result.csv                      # with nn_score + source
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

## Verification

1. `${ORIGINAL}/reconstructed_image/` has up to `num_SDG` images.
2. `${SEARCHED}/reconstructed_image/` count == `num_SDG` (Phase 7 fills with regen + best-per-defect fallback if needed).
3. `${ROUNDS}/search_summary.csv` has one row per sample.
4. `original/eval.log`, each `rounds/round_NN/sdg/eval.log`, and `searched/eval.log` contain per-type `nn_score`, `mnn_score`, and `fid`.
5. `${REGENS}/regen_summary.csv` exists when Phase 7 ran; `passed_threshold` column reports per-sample status, `prev_nn` vs `nn_score` reveals which samples regen actually improved.
