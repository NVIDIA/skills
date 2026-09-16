# Pipeline, State, and Runtime Behavior

## Pipeline

All stages run inline in the parent context. For SKILL stages, read the matching `references/*.md` first, then invoke the underlying `tao-skill-bank:*` skill via the Skill tool or the documented direct-container fallback in `references/scripts-and-agents.md`. INLINE stages have no underlying skill — the parent does the work directly.

Baseline runs once before the loop: `train` → `inference` → `evaluate` (skill: `tao-skill-bank:tao-train-visual-changenet` plus the approved evaluator in `references/metric-contract.md`), then `rca` (skill: `tao-skill-bank:tao-analyze-gaps-visual-changenet`). The `train` sub-step is **skipped** when `deft_state.json` arrives with `iterations.baseline.stage_completed == "train"` and a `best_ckpt_path` pointing at an existing file — the `automl-deft-pipeline` main skill pre-seeds these from its Phase 1 AutoML winner so DEFT doesn't retrain at the same HPs. In that case, baseline picks up at `inference` against the pre-seeded checkpoint, then evaluates the configured customer metric, then runs RCA.

Every RCA stage must complete the gaps skill's Steps 5–7 before its commit:
spot-check and save `rca_images/`, then write the seven-section
`RCA_Report.md`; only `unreachable_kpi.txt` permits the abridged report with no
image directory.

The RCA that feeds iteration N is already committed before iteration N starts:
`baseline.rca_gaps_parquet` feeds iter1; `iterN.rca_gaps_parquet` feeds iterN+1.
Do not run baseline RCA a second time under `iter1`. After an iteration's
evaluate stage, run RCA only when another iteration will follow. Then each
iteration executes:

1. **[INLINE] Resolve prior RCA input from disk.** For iter1 read
   `state["iterations"]["baseline"]["rca_gaps_parquet"]`; for iterN (N>1)
   read `state["iterations"][f"iter{N-1}"]["rca_gaps_parquet"]`. The path
   recorded in state must exist. `commit_stage.py` sets
   `state["current_iteration"] = N` when it commits the first `iterN/routing`
   event; never edit this field directly.

2. **[SKILL — `tao-skill-bank:tao-route-visual-changenet-samples`] Route weak samples.** Split the prior phase's `rca_gaps_parquet` into `routing_mining_parquet` and `routing_anomalygen_parquet` under the current `iterN` object. Downstream mining and AnomalyGen stages read those paths from disk. See `references/tao-route-visual-changenet-samples.md`.

3. **[SKILL — `tao-skill-bank:tao-generate-anomalies`] Run AMP + SDG.** Pass `dataset_dir` verbatim — no pool-staging, no parallel cache. Pre-create only `${RESULTS_DIR}/iter${N}/anomalygen/sdg/`. Commit both `SDG_result.csv` and Phase 2's defect-to-count `allocation.json`; the latter is the canonical proof for AMP allocated. The four invariants that actually gate the run (cad_mask RGB preserved, `text` entries have prompts, clean+cad pairs by stem, `semantic_segmentation_labels.json` present) and the full parameter mapping live in `references/tao-generate-anomalies.md`. Read it before invoking. Set `num_search_run=0` and `nn_threshold=0` to skip the SDG-quality phases (4–7) — the DEFT loop only needs the NG/OK pairs from Phase 3.

   If the committed `routing_anomalygen_parquet` has zero rows, do not launch
   the GPU generator. Set `anomalygen_skipped=true`, advance
   `stage_completed` to `anomalygen`, and commit an
   `anomalygen/status=skipped` event whose `skip_reason` says that routing
   produced zero rows. This is a
   documented branch skip, not a fabricated artifact.

   **SDG training contribution (INLINE).** Convert returned AnomalyGen outputs into ChangeNet paired training rows. Pre-create `${RESULTS_DIR}/iter${N}/mining_filter/` and `${RESULTS_DIR}/iter${N}/dataset/images/` before running the row-prep script. Stage NG/OK image pairs under `results/iter${N}/dataset/images/synthetic_iter${N}_{ng,ok}/`, run `"$PYTHON" scripts/changenet_data_pair_prepare.py` with `--input-dir ${RESULTS_DIR}/iter${N}/anomalygen/sdg/reconstructed_image`, `--golden-dir ${RESULTS_DIR}/iter${N}/anomalygen/sdg/original_image`, `--images-dir`, `--subdir synthetic_iter${N}`, `--light SolderLight`, `--image-ext .jpg` (or the exact `dataset.classify.image_ext` from the training spec), and `--output-csv ${RESULTS_DIR}/iter${N}/mining_filter/sdg_rows_raw.csv`. Rewrite the script's bare `synthetic_iter${N}_ng/` paths to workspace-root-relative form (`results/run_<TS>/iter${N}/dataset/images/synthetic_iter${N}_ng`) before appending into `mining_filter/mining_pool.csv`, since the per-iter training spec sets `images_dir=/data/workspace`. SDG rows skip k-NN filtering; only real-image mining applies the cosine threshold.

4. **[SKILL — `tao-skill-bank:tao-mine-aoi-images`] Mining pool — real-image contribution.** First run `"$PYTHON" scripts/resolve_mining_pool.py` with `state.config.mining_pool_csv`, `state.config.mining_images_root`, and `state.config.resolved_mining_pool_csv`; use only the resolved output. It fails on every missing or ambiguous file, so never guess an `augmentation/mining_pool/images` directory. Mine the resolved rows against the current iteration's weak samples (`routing_mining_parquet` from state) using SigLIP k-NN embeddings. **Retain only entries with cosine similarity ≥ `state.config.mining_filter.min_similarity`**, then apply history-aware filtering. Append only final novel rows into `mining_filter/mining_pool.csv`; never append raw/candidate parquets. `commit_stage.py` verifies the reported count against the mined parquet and rejects placeholder logs. See `references/tao-mine-aoi-images.md`.

   If history filtering leaves zero novel real rows, surface its recommendation
   to increase `topn` or expand the source pool. Continue only when AnomalyGen
   contributed new synthetic rows to this iteration; otherwise hard-stop
   instead of retraining the unchanged cumulative dataset.

   If the committed `routing_mining_parquet` has zero rows, do not run the
   embedding/mining containers. Set `data_mining_skipped=true`, advance
   `stage_completed` to `data_mining`, and commit a
   `data_mining/status=skipped` event whose `skip_reason` records the zero-row
   routing result.
   This is the only legal mining skip: an invalid source CSV or failed mining
   command with routed rows remaining is a hard stop.

   **Mid-iteration leakage check.** Right after the mining stage finishes — before any further CSV assembly — diff `mining_filter/mining_pool.csv` against `train/base/validation_set.csv` on `(input_path, golden_path, label, object_name, boardname)` (use `"$PYTHON" scripts/validate_training_csv.py --csv <mining_pool.csv> --workspace-root <ws> --validation-csv <validation_set.csv>`). Hard-stop on any hit. Catching leakage here, with only the new rows in scope, is cheap and isolates the offending source. The post-assembly leakage check in step 6b stays as a defence-in-depth backstop.

5. **[INLINE] Assemble training CSV** with monotonic growth:
   - Iter 1: `train/base/training_set.csv` + `mining_filter/mining_pool.csv`.
   - Iter N/resume: previous `train_combined_iter${N-1}.csv` + current `mining_filter/mining_pool.csv`. Never re-add `base_train` when using a previous combined CSV.
   - Write a sibling `_provenance.csv` with one aligned row per combined Train
     row. For iter1, `source ∈ {base_train, mining_pool}`. For N>1,
     `source ∈ {previous_iter_train, mining_pool}`; every row from the preceding
     combined CSV must appear unchanged with `source=previous_iter_train`.
     Check the row count, source vocabulary, and exact multiset retention
     before committing so every iteration grows monotonically.
   - **`images_dir` for the iteration training spec** must be set to the workspace root (e.g. `/data/workspace/`), not the real-image directory. SDG rows already carry workspace-root-relative paths. Resolve `state.config.images_dir` relative to the workspace (`images/` for the canonical layout; `kpi/images/` only for a legacy layout), then prepend that relative prefix exactly once to every iter1 base row's relative `input_path` and `golden_path`; do not merely change the spec. For iter N>1, the previous combined CSV is already in workspace coordinates and must not be prefixed again. If validation shows base files exist only after adding the resolved image-root prefix, fix the CSV and rerun validation; never bypass the FATAL.
   - **Normalize `label` case on every source before concatenation — base_train, previous_iter_train, SDG rows, and mined rows.** Preserve `PASS` uppercase and lowercase+strip everything else; write the normalized combined CSV before running `validate_training_csv.py`. See `references/visual-changenet.md` for the dataloader rule and the failure mode if you violate it.

6. **[INLINE] Pre-train CSV validation** — run **both** checks below; hard stop on either failure. Both must pass before launching the training container; an invalid CSV burns a full GPU run before the container surfaces the root cause.

   a. **Existence check.** Run `"$PYTHON" scripts/validate_training_csv.py --csv ${RESULTS_DIR}/iter${ITER}/dataset/train_combined_iter${ITER}.csv --workspace-root <workspace> --validation-csv <workspace>/train/base/validation_set.csv --report-json ${RESULTS_DIR}/iter${ITER}/dataset/merge_validation.json`. It hard-stops if any `input_path` / `golden_path` refers to a file missing on disk or if a required column is missing.

   b. **Train/validation leakage check.** `scripts/validate_training_csv.py` accepts `--validation-csv`; pass `train/base/validation_set.csv` so the diff on `(input_path, golden_path, label, object_name, boardname)` runs as part of the single validation pass. Hard stop on any validation row appearing in training. (Step 4 already runs the mid-iteration variant on `mining_filter/mining_pool.csv`; this check is the defence-in-depth backstop against leakage introduced by base-CSV reassembly.)

   The same successful invocation must write `dataset/merge_validation.json`
   with `rows_checked`, `missing_file_count=0`, and
   `train_val_leakage_overlap_count=0`; this artifact is mandatory proof of the
   merge checks, not a summary string substitute.

   After both checks pass, commit one `data_merge` stage event and write these
   fields under `state["iterations"][f"iter{ITER}"]`:

   ```python
   phase["combined_training_csv"] = "<abs_path>/train_combined_iterN.csv"
   phase["provenance_csv"] = "<abs_path>/train_combined_iterN_provenance.csv"
   phase["merge_validation_report"] = "<abs_path>/merge_validation.json"
   phase["stage_completed"] = "data_merge"
   ```

   The snippet documents the schema only. Commit it with:

   ```bash
   PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
   "$PYTHON" <skill_root>/scripts/commit_stage.py \
     --results-dir "${RESULTS_DIR}" --iter-label iter${ITER} \
     --stage data_merge \
     --combined-csv "${RESULTS_DIR}/iter${ITER}/dataset/train_combined_iter${ITER}.csv" \
     --provenance-csv "${RESULTS_DIR}/iter${ITER}/dataset/train_combined_iter${ITER}_provenance.csv" \
     --merge-validation-report "${RESULTS_DIR}/iter${ITER}/dataset/merge_validation.json" \
     --duration-sec "${STAGE_DURATION_SEC}" \
     --summary "validated combined training CSV"
   ```

   Do not launch training while the last committed stage is still
   `data_mining`; this commit is the proof that all path/leakage checks passed.

7. **[SKILL — `tao-skill-bank:tao-train-visual-changenet`] Fine-tune + evaluate.** Invoke the skill for separate `train`, `inference`, and `evaluate` tasks, committing `train` through `"$PYTHON" scripts/commit_stage.py` **before** evaluate; preserve the documented `data_merge → train → evaluate` order. Do not fold train into the evaluate commit. Iter N's committed checkpoint must be a newly emitted file under `${RESULTS_DIR}/iter${N}/train/`; reusing or copying the baseline/previous checkpoint is not an iteration train. For the train task, pass `automl_policy: off` as a **workflow argument** (to the Skill tool call or SDK runner), **not** as a spec field — see `## Train AutoML Policy` in SKILL.md. For direct `docker run visual_changenet train -e <spec>`, plain training is already the default. A nonzero train exit or a TAO status `FAILURE`, or a zero-step `PASS` that emits no new iteration checkpoint, is a hard stop; never evaluate a checkpoint written before that failure. Run the configured evaluator for every candidate checkpoint, select by its operator, then pass the winner and evaluator JSON to `commit_stage.py`; it records the metric and `evaluate` event transactionally. See `references/visual-changenet.md` and `references/metric-contract.md`.

## State

`results/deft_state.json` is the only persistent loop record. Initialize it
once with `init_deft_state.py`, then mutate it only through `commit_stage.py`;
never hand-edit, reinitialize, or write it with inline Python/jq/heredocs. It
contains the immutable execution policy, resume snapshot, resolved mining
paths, optional terminal artifacts, and an `events` array with one object per commit:

```json
{
  "seq":            <int, monotonically increasing from 1>,
  "ts":             "<ISO-8601 UTC; stage end time>",
  "iter":           "baseline|iter1|iter2|...",
  "stage":          "evaluate|rca|routing|anomalygen_finetune|anomalygen|data_mining|data_merge|train|loop_stop",
  "status":         "ok|error|skipped",
  "summary":        "<one-line outcome, e.g. 'weighted_escape_cost=0.018 target<=0.02'>",
  "skip_reason":    "<present only for status=skipped; copied from summary>",
  "duration_sec":   <positive int for executed stages; non-negative int for skips>,
  "context_tokens": <0 at write time; backfilled at loop end by align_token_usage.py>,
  "tokens":         <object added at loop end: input, output, cache_read, cache_create, n_messages, models>
}
```

`context_tokens` starts at 0 because the stage caller cannot measure LLM
context size in-flight. The loop-end sequence runs `align_token_usage.py` to
attribute transcript usage to the state events and atomically refresh the same
file.

**Disk is the source of truth.** Before every stage, re-read
`deft_state.json`. Use the latest event plus `iterations.<label>.status` and
`stage_completed` to decide where to resume. Do not print the full state into
context. `commit_stage.py` re-reads it and computes the next event sequence
from disk. A failed stage may be retried after its cause is fixed; the retry is
recorded as a new event and the current snapshot is updated.

**On startup / resume:** Show the last five `events` entries so the user can
see recent progress, then proceed from the disk-loaded snapshot.

## Stage Execution

Every stage runs in the parent's context. The disk contracts
(`deft_state.json` + `results/iter${ITER}/`) are the
canonical interface between stages — never assume in-memory state survives.

Three stage types:

- **SKILL** — read `references/<stage>.md` first, then invoke the matching `tao-skill-bank:*` skill via the Skill tool or its documented direct-container fallback. Stage→skill mapping and fallback contract are in `references/scripts-and-agents.md`.
- **INLINE** — parent does the work directly (pre-flight, CSV assembly, leakage check).
- **HOOK** — after each valid commit, `commit_stage.py` starts
  `scripts/render_report.py` with its current Python interpreter. Report
  generation is deterministic and never delegated.

For `tao-skill-bank:tao-train-visual-changenet`, pass a separate task name (`train`, `inference`, or `evaluate`); the state event's `stage` value is still only `train` or `evaluate`.

If the matching `references/*.md` file is missing, stop. Do not replace it with generic shell commands. Artifacts must stay under the stage-specific output directory defined by the reference file.

### Post-stage check

After every stage finishes, before advancing:

1. Verify the documented required artifacts exist.
2. Invoke `commit_stage.py` once with the documented artifact flags. For an error use `--status error`; it sets the iteration failed. Never repair a rejected commit by editing JSON.
3. Re-read `deft_state.json` and confirm the latest event matches the stage.
4. If the committed event status is `error` — halt and surface the recorded error. Retry only after repairing its cause.
5. If the committed event status is `ok` or `skipped` — confirm that
   `commit_stage.py` did
   not print `report hook failed`, then print one status line in the standard
   format `[iter <N>/<max> · <stage>] <primary metric> · <duration> · next:
   <stage>` (e.g. `[iter 2/3 · evaluate] weighted escape cost 0.024 → 0.018
   (target <=0.02) · 2m · next: loop_stop`). Use the contract's display name,
   direction, target, and unit. The post-commit hook has already refreshed
   `DEFT_Loop_Report.html`; do not spawn a reporting agent.

## Reports

- `results/iter${ITER}_summary.md` — ≤300 words; readable after context compaction.
- `results/iter${ITER}/report.html` — RCA targets, branch outputs, filter decision, metric delta.
- `results/DEFT_Loop_Report.html` — initialized by `init_deft_state.py` and
  re-rendered after every committed stage by `commit_stage.py`; the hook starts
  `scripts/render_report.py` with its current Python interpreter. The script reads the source template and disk
  evidence, validates the full output, and writes atomically.

## Runtime Behavior

Run without pausing. Between stages, re-read state and print only a one-line
status/next action. `commit_stage.py` appends one event per commit and
refreshes the report through its post-commit hook. For background
Docker work, redirect both streams, save its PID, poll one line or `tail -40`
at intervals no longer than 30s, and always `wait`; never poll a Skill-tool call.

**Loop-end sequence** (run in order, each step depends on the previous):

1. Run `"$PYTHON" scripts/deft_context.py --stage finalize`, then call
   `"$PYTHON" scripts/finalize_run.py` with
   `--stop-reason metric_met` or `max_iterations`. It first writes
   `best_model.json` and `best_model_inference_spec.yaml`, then commits
   `loop_stop`; terminal state cannot exist without those artifacts.
2. Optionally backfill real per-stage token usage into `deft_state.json` from the Claude Code transcript:

   ```bash
   PYTHON=$(bash <this-skill-dir>/scripts/deft_python.sh)
   "$PYTHON" <this-skill-dir>/scripts/align_token_usage.py \
       --state-path ${RESULTS_DIR}/deft_state.json \
       --project-dir ~/.claude/projects/$(pwd | sed 's|/|-|g')
   ```

   This rewrites every entry's `context_tokens` field with the real context size at stage end and adds a `tokens` object (`input`, `output`, `cache_read`, `cache_create`, `n_messages`, `models`). The next step's report includes the numbers.
3. Run the renderer once after token alignment so the final report includes
   the backfilled usage fields:

   ```bash
   PYTHON=$(bash <this-skill-dir>/scripts/deft_python.sh)
   "$PYTHON" \
     <this-skill-dir>/scripts/render_report.py \
     --results-dir "${RESULTS_DIR}" --require-terminal
   ```

   This is an explicit deterministic script call, not delegated work; the
   preceding `loop_stop` commit already produced a valid report before token
   alignment.

Before telling the user the loop is complete, re-read `deft_state.json` and
require `status == "complete"`, a complete baseline, and a complete final
iteration. For a hard-stop path, require `status == "failed"` and report the
run as `FAILED`; never relabel it as a completed loop. Once the state check
passes, reap any saved run-owned PIDs, return the user/harness completion token
if one was requested, and end the session.

**Stop conditions:**

- Primary metric and all configured constraints pass → run the loop-end sequence.
- `max_iterations` reached → run the loop-end sequence with the best-iteration report. Do not add a post-loop RCA event; the terminal evaluate transitions directly to `loop_stop`.
- Unrecoverable gate failure → commit the failing stage with `--status error`, halt, and report the exact missing artifact. Do not run `loop_stop`, a reduced loop, or fabricate CSVs.

**Prepare-for-inference.** `scripts/finalize_run.py` invokes
`scripts/prepare_inference_spec.py` before the terminal commit. For inspection
or recovery before terminal state, the lower-level command is:

```bash
PYTHON=$(bash <this-skill-dir>/scripts/deft_python.sh)
"$PYTHON" <this-skill-dir>/scripts/prepare_inference_spec.py --results-dir ${RESULTS_DIR}
```

This writes two artifacts under `${RESULTS_DIR}/`:

- `best_model.json` — handoff metadata (checkpoint, optional operating threshold, metric contract/result, backbone, images_dir, training_spec)
- `best_model_inference_spec.yaml` — runnable TAO inference spec built from the training config so model architecture, lighting layout, image size, and difference module match the checkpoint exactly

Downstream inference skills consume these — they should never read `deft_state.json` or the training spec directly. Full contract, consumer workflow, and silent-failure modes are documented in `references/prepare-for-inference.md`.

If a partial `${RESULTS_DIR}/` is missing iteration artifacts or fails the leakage check, restart from the last valid checkpoint instead of resuming. Starting a fresh run always creates a new timestamped `results/run_<YYYYMMDD_HHMMSS>/` — prior runs are preserved under their own directories.
