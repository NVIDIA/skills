# Pipeline, State, and Runtime Behavior

## Pipeline

All stages run inline in the parent context. For SKILL stages, read the matching `references/*.md` first, then invoke the underlying `tao-skill-bank:*` skill via the Skill tool. INLINE stages have no underlying skill — the parent does the work directly.

Baseline runs once before the loop: `train` → `inference` → `evaluate` (skill: `tao-skill-bank:tao-train-visual-changenet`), then `rca` (skill: `tao-skill-bank:tao-analyze-gaps-visual-changenet`). The `train` sub-step is **skipped** when `deft_state.json` arrives with `iterations.baseline.stage_completed == "train"` and a `best_ckpt_path` pointing at an existing file — the `automl-deft-pipeline` main skill pre-seeds these from its Phase 1 AutoML winner so DEFT doesn't retrain at the same HPs. In that case, baseline picks up at `inference` against the pre-seeded checkpoint, then `evaluate`, then `rca`. Then each iteration:

1. **[SKILL — `tao-skill-bank:tao-analyze-gaps-visual-changenet`] RCA** on the previous inference result. Output: `rca_results/`. Write `iterations.<iter>.rca_target_defects` and `rca_gaps_parquet` into `deft_state.json` before advancing. See `references/tao-analyze-gaps-visual-changenet.md`.

2. **[SKILL — `tao-skill-bank:tao-route-visual-changenet-samples`] Route weak samples.** Split `rca_gaps_parquet` into `routing_mining_parquet` and `routing_anomalygen_parquet` in `deft_state.json`. Downstream mining and AnomalyGen stages read those paths from disk. See `references/tao-route-visual-changenet-samples.md`.

3. **[SKILL — `tao-skill-bank:paidf-anomalygen`] Run AMP + SDG.** Pass `dataset_dir` verbatim — no pool-staging, no parallel cache. Pre-create only `${RESULTS_DIR}/iter${N}/anomalygen/sdg/`. The four invariants that actually gate the run (cad_mask RGB preserved, `text` entries have prompts, clean+cad pairs by stem, `semantic_segmentation_labels.json` present) and the full parameter mapping live in `references/paidf-anomalygen.md`. Read it before invoking. Set `num_search_run=0` and `nn_threshold=0` to skip the SDG-quality phases (4–7) — the DEFT loop only needs the NG/OK pairs from Phase 3.

   **SDG training contribution (INLINE).** Convert returned AnomalyGen outputs into ChangeNet paired training rows. Stage NG/OK image pairs under `results/iter${N}/dataset/images/synthetic_iter${N}_{ng,ok}/`, run `scripts/changenet_data_pair_prepare.py` with `--input-dir ${RESULTS_DIR}/iter${N}/anomalygen/sdg/reconstructed_image`, `--golden-dir ${RESULTS_DIR}/iter${N}/anomalygen/sdg/original_image`, `--images-dir`, `--subdir synthetic_iter${N}`. Rewrite the script's bare `synthetic_iter${N}_ng/` paths to workspace-root-relative form (`results/run_<TS>/iter${N}/dataset/images/synthetic_iter${N}_ng`) before appending into `mining_filter/mining_pool.csv`, since the per-iter training spec sets `images_dir=/data/workspace`. SDG rows skip k-NN filtering; only real-image mining applies the cosine threshold.

4. **[SKILL — `tao-skill-bank:tao-mine-aoi-images`] Mining pool — real-image contribution.** Mine real images from `augmentation/mining_pool/mining_pool.csv` against the current iteration's weak samples (`routing_mining_parquet` from `deft_state.json`) using SigLIP k-NN embeddings. **Retain only entries with cosine similarity ≥ `state.config.mining_filter.min_similarity`** (default `0.9` when unset). Lower-similarity candidates are rejected. Append the retained rows into `mining_filter/mining_pool.csv` (same file as the SDG contribution above). When converting mined filepaths into ChangeNet rows, follow the path-form rule in `references/tao-mine-aoi-images.md` → *Mined rows → ChangeNet CSV*. Output: updated `mining_filter/mining_pool.csv` and `mining_filter/knn_summary.csv` (`candidate_count`, `kept_count`, `rejected_count`, `similarity_threshold=<value>`). See `references/tao-mine-aoi-images.md`.

   **Mid-iteration leakage check.** Right after the mining stage finishes — before any further CSV assembly — diff `mining_filter/mining_pool.csv` against `train/base/validation_set.csv` on `(input_path, golden_path, label, object_name, boardname)` (use `scripts/validate_training_csv.py --csv <mining_pool.csv> --workspace-root <ws> --validation-csv <validation_set.csv>`). Hard-stop on any hit. Catching leakage here, with only the new rows in scope, is cheap and isolates the offending source. The post-assembly leakage check in step 6b stays as a defence-in-depth backstop.

5. **[INLINE] Assemble training CSV** with monotonic growth:
   - Iter 1: `train/base/training_set.csv` + `mining_filter/mining_pool.csv`.
   - Iter N/resume: previous `train_combined_iter${N-1}.csv` + current `mining_filter/mining_pool.csv`. Never re-add `base_train` when using a previous combined CSV.
   - Write a sibling `_provenance.csv` for every output row; `source ∈ {base_train, previous_iter_train, mining_pool}`.
   - **`images_dir` for the iteration training spec** must be set to the workspace root (e.g. `/data/workspace/`), not `kpi/images/`. SDG rows already carry workspace-root-relative paths. Base training rows carry paths relative to `kpi/images/` — prepend `kpi/images/` to their `input_path` and `golden_path` so all rows share the same coordinate space.
   - **Normalize `label` case — preserve `PASS` uppercase, lowercase+strip everything else.** See `references/visual-changenet.md` for the dataloader rule and the failure mode if you violate it.

6. **[INLINE] Pre-train CSV validation** — run **both** checks below; hard stop on either failure. Both must pass before launching the training container; an invalid CSV burns a full GPU run before the container surfaces the root cause.

   a. **Existence check.** Run `scripts/validate_training_csv.py --csv ${RESULTS_DIR}/iter${ITER}/dataset/train_combined_iter${ITER}.csv --workspace-root <workspace>`. It hard-stops if any `input_path` / `golden_path` refers to a file missing on disk or if a required column is missing.

   b. **Train/validation leakage check.** `scripts/validate_training_csv.py` accepts `--validation-csv`; pass `train/base/validation_set.csv` so the diff on `(input_path, golden_path, label, object_name, boardname)` runs as part of the single validation pass. Hard stop on any validation row appearing in training. (Step 4 already runs the mid-iteration variant on `mining_filter/mining_pool.csv`; this check is the defence-in-depth backstop against leakage introduced by base-CSV reassembly.)

7. **[SKILL — `tao-skill-bank:tao-train-visual-changenet`] Fine-tune + evaluate.** Invoke the skill for the `train` and `evaluate` tasks. For the train task, pass `automl_policy: off` as a **workflow argument** (to the Skill tool call or SDK runner), **not** as a spec field — see `## Train AutoML Policy` in SKILL.md for the failure mode if you put it in the YAML. For direct inline `docker run visual_changenet train -e <spec>`, the argument is implicit (plain training is the default entrypoint) and no spec edit is needed. The skill owns TAO training, checkpoint discovery, inference, KPI analysis, and best-checkpoint selection. Write the selected checkpoint and KPI metrics into `deft_state.json`. Stop the loop if KPI met or `max_iterations` reached. See `references/visual-changenet.md`.

## State & Logging

Two artifacts persist loop state:

- `results/deft_state.json` — current resume snapshot. Schema: `references/deft_state.json`. **Initialize once on a fresh run via `scripts/init_deft_state.py`** — the script builds the dict with literal-once keys so duplicates are impossible. After initialization, update with Python/jq (never `echo`) after every step; never re-init on resume.
- `results/loop_log.jsonl` — append-only event stream, one JSON line per stage:

```json
{
  "seq":            <int, monotonically increasing from 1>,
  "ts":             "<ISO-8601 UTC; stage end time>",
  "iter":           "baseline|iter1|iter2|...",
  "stage":          "evaluate|rca|routing|anomalygen|data_mining|train|loop_stop",
  "status":         "ok|error",
  "summary":        "<one-line outcome, e.g. 'FAR=52.0% threshold=0.31'>",
  "duration_sec":   <int seconds from stage start to end>,
  "context_tokens": <0 at write time; backfilled at loop end by align_token_usage.py>,
  "tokens":         <object added at loop end: input, output, cache_read, cache_create, n_messages, models>
}
```

`context_tokens` is a placeholder written as 0 by `scripts/log_stage.py` (the bash caller cannot measure LLM context size in-flight). The loop-end sequence runs `scripts/align_token_usage.py` to read the Claude Code transcript at `~/.claude/projects/<slug>/<session-id>.jsonl`, attribute each assistant message to the stage whose timestamp window it falls in, and rewrite the file with real `context_tokens` plus a per-stage `tokens` object.

**Disk is the source of truth.** Before every stage, *unconditionally* re-read the last line of `loop_log.jsonl` and the full `deft_state.json` from disk; overwrite any in-memory state. Compaction is invisible — there is nothing to detect. `seq` is always `last_seq + 1` from disk; `seq = 1` if the file does not exist.

Use `scripts/log_stage.py` to write entries (guarantees valid JSON and computes `seq` from disk). Pass `log_path` as `pathlib.Path`, not `str` — `append_stage()` calls `.exists()` on it directly. **Never emit JSON via `echo` or inline jq** — the `seq` invariant requires reading the live tail through `next_seq()`.

**On startup / resume:** Print the last 5 entries of `loop_log.jsonl` so the user can see recent progress, then proceed using the disk-loaded state.

## Stage Execution

Every stage runs in the parent's context. The disk contracts
(`deft_state.json` + `loop_log.jsonl` + `results/iter${ITER}/`) are the
canonical interface between stages — never assume in-memory state survives.

Three stage types:

- **SKILL** — read `references/<stage>.md` first, then invoke the matching `tao-skill-bank:*` skill via the Skill tool. Stage→skill mapping is the **Stage Reference Modules** table in `references/scripts-and-agents.md`.
- **INLINE** — parent does the work directly (pre-flight, CSV assembly, leakage check).
- **AGENT** — parent spawns a subagent. The only AGENT stage is `agents/reporter.md` for HTML rendering.

For `tao-skill-bank:tao-train-visual-changenet`, pass a separate task name (`train`, `inference`, or `evaluate`); the `stage` value in `loop_log.jsonl` is still only `train` or `evaluate`.

If the matching `references/*.md` file is missing, stop. Do not replace it with generic shell commands. Artifacts must stay under the stage-specific output directory defined by the reference file.

### Post-stage check

After every stage finishes, before advancing:

1. Re-read the last line of `loop_log.jsonl` and the full `deft_state.json` from disk. Trust disk over in-memory.
2. If `status=error` — halt, surface the disk evidence verbatim, **do not auto-retry**.
3. If `status=ok` — print one status line in the standard format `[iter <N>/<max> · <stage>] <key metric> · <duration> · next: <stage>` (e.g. `[iter 2/3 · train] FAR 6.34% → 3.11% (target <0.5%) · 11m · next: evaluate`; show FAR-vs-target whenever a new FAR is available), then advance. Render `DEFT_Loop_Report.html` only at iteration end (`trigger="after-iteration"`) and at loop end (`trigger="loop-end"`); never inline.

## Reports

- `results/iter${ITER}_summary.md` — ≤300 words; readable after context compaction.
- `results/iter${ITER}/report.html` — RCA targets, branch outputs, filter decision, metric delta.
- `results/DEFT_Loop_Report.html` — re-rendered **after every stage** and at loop end by the `reporter` subagent (`agents/reporter.md`). The agent owns the entire render: it reads the template, the rendering protocol (`references/REPORT_RENDERING.md`), and disk state, then writes atomically. The parent's only responsibility is to spawn the agent — never render inline.

## Runtime Behavior

Run without pausing. Between stages, follow `## Stage Execution`: re-read `loop_log.jsonl` tail + `deft_state.json` from disk, print a one-line status from the disk-loaded summary, then spawn the `reporter` subagent (`agents/reporter.md`, `trigger="after-stage"`) to re-render `DEFT_Loop_Report.html`. Append exactly one `loop_log.jsonl` entry per stage — never both before and after a skill invocation.

**Loop-end sequence** (run in order, each step depends on the previous):

1. Append the final `loop_stop` entry via `scripts/log_stage.py`.
2. Backfill real per-stage token usage into `loop_log.jsonl` from the Claude Code transcript:

   ```bash
   python ${TAO_SKILL_BANK_PATH}/skills/tao-run-deft-aoi/scripts/align_token_usage.py \
       --log-path ${RESULTS_DIR}/loop_log.jsonl \
       --project-dir ~/.claude/projects/$(pwd | sed 's|/|-|g')
   ```

   This rewrites every entry's `context_tokens` field with the real context size at stage end and adds a `tokens` object (`input`, `output`, `cache_read`, `cache_create`, `n_messages`, `models`). The next step's report includes the numbers.
3. Spawn `reporter` with `trigger="loop-end"` to re-render `DEFT_Loop_Report.html` against the now-aligned log.
4. Run `scripts/prepare_inference_spec.py` (see below).

**Stop conditions:**

- KPI met → run the loop-end sequence.
- `max_iterations` reached → run the loop-end sequence with the best-iteration report + final RCA on the best checkpoint.
- Unrecoverable gate failure → halt and report the exact missing artifact. Do not run a reduced loop. Do not fabricate CSVs. Skip prepare-for-inference (no valid checkpoint to hand off); steps 1–3 of the loop-end sequence still apply.

**Prepare-for-inference (final step).** Run `scripts/prepare_inference_spec.py` to emit the inference handoff:

```bash
python scripts/prepare_inference_spec.py --results-dir ${RESULTS_DIR}
```

This writes two artifacts under `${RESULTS_DIR}/`:

- `best_model.json` — handoff metadata (checkpoint, threshold, far_pct, backbone, images_dir, training_spec)
- `best_model_inference_spec.yaml` — runnable TAO inference spec built from the training config so model architecture, lighting layout, image size, and difference module match the checkpoint exactly

Downstream inference skills consume these — they should never read `deft_state.json` or the training spec directly. Full contract, consumer workflow, and silent-failure modes are documented in `references/prepare-for-inference.md`.

If a partial `${RESULTS_DIR}/` is missing iteration artifacts or fails the leakage check, restart from the last valid checkpoint instead of resuming. Starting a fresh run always creates a new timestamped `results/run_<YYYYMMDD_HHMMSS>/` — prior runs are preserved under their own directories.
