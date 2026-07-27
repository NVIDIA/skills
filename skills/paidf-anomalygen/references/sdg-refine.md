# SDG Refine Reference — Phase 5 Per-Sample Search

Use this reference only when the parent `SKILL.md` points here for the current task. If this file conflicts with current `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current authoritative source wins.

## Contents

- Inputs
- Agent search loop
- draws.json format
- Output
- Re-rolling AMP augmentation (optional)
- Notes


Full detail for `run_round.sh` and `assemble_searched.py`. Read when debugging
per-sample search, draws.json alignment, or re-AMP augmentation.

---

## Inputs

| Parameter | Description |
|---|---|
| `base_jsonl` | JSONL used to produce `original/`. |
| `original_dir` | Existing SDG output bucket. |
| `original_csv` | `per_sample.csv` from eval on `original_dir`. Carries `nn_score` (key KPI) and `mnn_score`; the search loop consumes `nn_score`. |
| `real_path` | Real-image reference for per-round eval. |
| `anomaly_types` | Full `TEXTURE+TYPE` names. |
| `searched_dir` | Final output bucket (same layout as SDG). |
| `rounds_dir` | Working dir for per-round artifacts + `search_summary.csv`. |
| `checkpoint_dir` / `step` | Model checkpoint used for rounds. |
| `num_search_run` | Number of rounds (default `3` in the pipeline). `0` → skip search; assemble still runs and clones `original/` into `searched/`, so the downstream contract holds. |

---

## Agent search loop

For `r` in `1..num_search_run`:

1. **Decide draws.** Read `original_csv` (r=1) or `rounds_dir/round_<r-1>/per_sample.csv`
   plus the running best-seen map. Pick new `(guidance, crop_ratio)` per sample index.
   Skip samples whose current best is already high to save inference time.
   Write to `rounds_dir/round_<r:03d>/draws.json`.

2. **Run the round:**

   ```bash
   ${ANOMALYGEN_SCRIPTS}/run_round.sh \
       --base-jsonl ${JSONL} \
       --draws ${ROUNDS}/round_${r}/draws.json \
       --output-dir ${ROUNDS}/round_${r} \
       --real-path ${DATASET_DIR} \
       --anomaly-types ${DEFECTS[@]} \
       --checkpoint-dir ${CKPT} --step ${STEP} \
       [--model-size 2b|14b]
   ```

   Produces `rounds/round_r/testcase.jsonl`, `sdg/`, `per_sample.csv`.

3. **Inspect `per_sample.csv`** and update your mental model before the next round.

After all rounds:

```bash
python3 -m scripts.utilities.assemble_searched \
    --original-dir ${ORIGINAL} \
    --original-csv ${ORIGINAL}/per_sample.csv \
    --rounds-dir ${ROUNDS} \
    --searched-dir ${SEARCHED}
```

Copies each sample's best-seen attempt into `searched_dir/` and writes
`rounds_dir/search_summary.csv`.

---

## draws.json format

```json
{
  "<sample_index>": {"guidance": <float>, "crop_ratio": <float>}
}
```

- Sample indices are 0-based JSONL line numbers.
- Omitted samples are not retried this round (remain at current best-seen).
- Each round's `sdg/SDG_result.csv` `.index` column carries the base-JSONL
  `sample_index` (not the reduced-JSONL 0..N-1 position) — this is what lets
  `assemble_searched.py` align best-seen attempts across rounds correctly.
- Ranges: `guidance ∈ [1.5, 10.0]`, `crop_ratio ∈ [1.5, 10.0]`. Narrow or
  widen based on prior-round feedback.

---

## Output

- `searched_dir/` — same layout as SDG (`reconstructed_image/`, `original_mask/`,
  `overlay_image/`, `original_image/`, `SDG_result.csv`). One entry per input sample.
- `rounds_dir/round_<k>/` — per-round testcase JSONL, SDG output, and `per_sample.csv` (for audit).
- `rounds_dir/search_summary.csv` — columns: `best_round`, `best_guidance`,
  `best_crop_ratio`, `best_nn_score`, `attempts`.

---

## Re-rolling AMP augmentation (optional)

By default the `(clean_image, mask)` pair per sample index is fixed across all
rounds — only `(guidance, crop_ratio)` change. If `nn_score` stays flat for a
sample across ≥ 2 rounds of `(g, c)` variation, the mask placement itself may
be the blocker. Add `--reamp-seed` to re-run AMP with a fresh base seed on the
same `(clean, submask)` records:

```bash
${ANOMALYGEN_SCRIPTS}/run_round.sh \
    ...standard args... \
    --reamp-seed $((1000 + r)) \
    --defect-spec ${DEFECT_DESC}
```

The text2roi ROI cache is reused — Qwen VL + SAM2 are not re-run. New
augmentations (rotation/shift/morph) replace the old masks for that round's
SDG pass; sample indices and per-sample CSV tracking stay aligned.

Use sparingly: re-roll only after ≥ 2 `(g, c)` rounds without improvement.
Each re-AMP is one extra `run_auto_roi_amp.py` pass.

---

## Notes

- `num_search_run = 0` is valid: skip `run_round.sh` entirely; still run
  `assemble_searched.py` with an empty rounds dir — it clones `original/`
  into `searched/`, preserving the "final is always `searched/`" invariant.
- You can retry any subset of samples per round — the draws JSON is the selector.
- `run_round.sh` intentionally runs SDG single-GPU: per-round sample count is
  small, and `torchrun` init overhead would dominate. Bulk generation in Phase 3
  uses the orchestrator's `num_gpus`.
