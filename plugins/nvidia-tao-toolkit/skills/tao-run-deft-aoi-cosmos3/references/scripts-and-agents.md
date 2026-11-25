# Cosmos3 DEFT Scripts and Agents

Set `PYTHON=$(bash scripts/deft_python.sh)`, then run host Python scripts with
`"$PYTHON"`. Resolve all paths to absolute paths before invoking a script.

## Available scripts

| Script | Purpose |
|---|---|
| `init_deft_state.py` | Initialize version-6 Cosmos3 Framework state once; freeze Benchmark hash, bare mode, media root, and immutable execution policy. |
| `deft_context.py` | Re-read state and print the deterministic next stage plus network/Python policy; reject a requested-stage mismatch. |
| `deft_exec.py` | Enforce state-backed offline/no-install/no-pull policy for external commands. |
| `commit_stage.py` | Validate one stage's inputs and atomically update state. AnomalyGen skip requires an empty recorded false-accept array; terminal commits require reason/report evidence. |
| `metric_contract.py` | Validate/compare the Benchmark KPI contract. |
| `record_metric_result.py` | Bind `benchmark_metrics/metric_result.json` to an iteration. |
| `validate_sharegpt.py` | Enforce two-image, exact bare OK/NG ShareGPT records. |
| `validate_split_contract.py` | Enforce split isolation, monotonic generated-Train lineage, and Benchmark hash. |
| `check_annotations.py` | Per-role field-contract check over all three workspace annotation files. `ROLE_CONTRACT` is the authoritative field list. |
| `render_cfw_sft.py` | Render native Framework SFT TOML with an optional previous DCP. |
| `submit_cfw_train.py` | Compose and execute recorded Framework Train submission. |
| `render_cfw_evaluate.py` | Render native bare-label Framework evaluation TOML. |
| `submit_cfw_evaluate.py` | Compose and execute recorded Framework Evaluate submission. |
| `submit_cfw_inference.py` | Compose and execute recorded Framework Inference submission. |
| `analyze_gaps.py` | Proxy RCCA artifacts or Benchmark aggregate metric evidence. |
| `emit_sdg_sharegpt.py` | Resolve documented or PAIDF 1.0.1 SDG path forms and emit bare `NG` ShareGPT records. |
| `filter_mined_by_cosine.py` | Recompute max cosine to Proxy targets and enforce the floor. |
| `emit_mined_sharegpt.py` | Align filtered paths to Mining prompts, golden images, and labels. |
| `assemble_training_json.py` | Monotonic bare training-data merge with dedupe/leakage checks. |
| `align_token_usage.py` | Backfill stage token accounting after a run when a transcript is available. |
| `render_report.py` | Deterministically render the self-contained NVIDIA-styled HTML report from state and recorded artifacts, including escaped annotation prompt examples; validate required sections/placeholders and replace atomically. |
| `finalize_run.py` | Render final evidence, validate the explicit stop reason, commit `loop_stop`, and record the report path. |

`init_deft_state.py` requires `--gpu-model` with the exact model string from
the selected platform's Preflight. `commit_stage.py` requires a positive
`--duration-sec` on every executed success, error, and `loop_stop` commit. Use
the backend's measured elapsed wall time for submitted jobs and a directly
measured host duration for inline stages; round a measured sub-second executed
stage up to `1`. A documented `--skip` records `status=skipped`, copies the
summary into `skip_reason`, and accepts a non-negative duration including `0`.
An omitted duration or any negative duration is rejected.

Before every stage, call `deft_context.py --state <deft_state.json> --stage
<stage>`. Wrap local external/container commands with `deft_exec.py --state
<deft_state.json> -- <command>`; the selected remote platform must enforce the
same policy when it constructs a job.

Train, Proxy evaluate, and Benchmark evaluate use the application-owned native
Framework render and submit scripts. Mining reuses
`tao-mine-aoi-images`. The application owns only the DEFT-specific state,
isolation, OK/NG analysis, filtering, and assembly scripts.

## Stage references

| Stage | Producer | Read first |
|---|---|---|
| Train | `tao-finetune-cosmos-reason` train, `automl_policy: off` | `cosmos-reason.md`, `cosmos_framework_sft_full.toml` |
| Proxy / Benchmark evaluate | `tao-finetune-cosmos-reason` evaluate | `cosmos-reason.md` |
| Proxy RCCA / Benchmark metric | bundled `analyze_gaps.py` | `gap-analysis.md` |
| Routing / mining | Proxy gaps + `tao-mine-aoi-images` | `tao-mine-aoi-images.md` |
| AnomalyGen | `tao-generate-anomalies`, `mode=inference_only` | `tao-generate-anomalies.md` |
| Assemble / validate | bundled bare ShareGPT scripts | `aoi-annotation.md` |
| State/report | bundled state commit + deterministic report hook | this file |

## Automatic report hook

`init_deft_state.py` invokes `render_report.py` after writing canonical state.
`commit_stage.py` invokes it again after every valid state commit,
including error and `loop_stop` commits. The hook is outside the state
transaction: it reports `report hook failed` without rolling back a valid
stage result.

Normally do not render separately. After optional loop-end token alignment, or
to recover from a reported presentation error, run:

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/render_report.py \
  --results-dir "${RESULTS_DIR}" --require-terminal
```

The legacy `agents/reporter.md` is a compatibility wrapper around this exact
command and contains no HTML-generation logic.

## Path invariants

- Every recorded stage artifact is absolute.
- Every iteration artifact is under `${RESULTS_DIR}/baseline` or
  `${RESULTS_DIR}/iterN`.
- Train checkpoints are under `${RESULTS_DIR}/<label>/train`.
- Proxy artifacts never contain Benchmark paths.
- Mining, routing, and assembly never consume Benchmark per-sample errors.
- User/model identity paths are mounted read-only; only the stage result and
  action-model directories are writable.

Framework submit scripts reject stale packaged specs and use only public console
entrypoints. Check credential presence only (`SET`/`UNSET`); never read or print
credential values.
