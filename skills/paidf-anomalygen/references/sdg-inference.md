# SDG Inference Reference — Phase 3

Use this reference only when the parent `SKILL.md` points here for the current task. If this file conflicts with current `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current authoritative source wins.

## Contents

- Parameters
- Steps
  - Step 1 — Validate the checkpoint
  - Step 2 — Prepare the inference JSONL (skip if `input_jsonl` given)
  - Step 3 — Validate the JSONL against the checkpoint
  - Step 4 — Launch SDG
  - Step 5 — Verify completion before eval
  - Step 6 — Evaluate
- Multi-GPU notes
  - NCCL hang controls
- Verification checklist
- Error handling


Full detail for running SDG. Read when troubleshooting checkpoint validation,
multi-GPU NCCL issues, or SDG output completeness.

---

## Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `checkpoint_dir` | Yes | — | Trained checkpoint. Must contain `ag_config.yaml`. |
| `step` | Yes | — | Checkpoint iteration to load. Must be on a `save_iter` boundary. |
| `input_jsonl` | Yes (or auto) | Auto via prep-testcase | Pre-built inference JSONL. |
| `output_dir` | Yes | — | Where to write generated images + `SDG_result.csv`. |
| `model_size` | No | `2b` | `2b` or `14b` — must match checkpoint. |
| `seed` | No | `0` | Random seed. |
| `num_gpus` | No | `1` | DDP-style multi-GPU. |
| `defect_spec` | Yes (if no JSONL) | — | JSONL tagging each defect `free`/`text`/`cad`. Template at `.agents/skills/anomalygen/assets/defect_spec_template.jsonl`. |
| `real_path` | No | `dataset_dir` | Real anomaly images for downstream eval. |

---

## Steps

### Step 1 — Validate the checkpoint

```bash
python3 -m scripts.utilities.validate_checkpoint <checkpoint_dir> --step <step>
```

Exits non-zero if `ag_config.yaml` is missing or malformed. On success, prints
the supported `<TEXTURE>+<ANOMALY>` set — show this to the user.

### Step 2 — Prepare the inference JSONL (skip if `input_jsonl` given)

See `references/prep-testcase.md`. Output path becomes `input_jsonl`.

### Step 3 — Validate the JSONL against the checkpoint

```bash
python3 -m scripts.utilities.validate_jsonl <checkpoint_dir> <input_jsonl>
```

Exits non-zero if the JSONL contains anomaly types the checkpoint cannot
generate. Missing-file paths are reported as warnings.
If many paths are missing → stop; ask the user to verify before burning GPU time.

### Step 4 — Launch SDG

```bash
${ANOMALYGEN_SCRIPTS}/run_sdg.sh \
    --checkpoint_dir <checkpoint_dir> \
    --step <step> \
    --input_jsonl <input_jsonl> \
    --output_dir <output_dir> \
    --model_size <model_size> \
    --seed <seed> \
    [--num_gpus N]
```

### Step 5 — Verify completion before eval

```bash
${ANOMALYGEN_SCRIPTS}/verify_output.sh <input_jsonl> <output_dir>
```

Non-zero if row or image counts don't match the JSONL. **Do not skip** — eval
on an incomplete output produces incorrect scores.

### Step 6 — Evaluate

See `references/eval.md`. Use `real_path` as `--real-path` and `output_dir`
as `--generated-path`.

---

## Multi-GPU notes

- Rank 0 merges `SDG_result.csv` and `timing_summary.json`; per-rank shards
  are not surfaced.
- 14B at inference: FSDP auto-disabled when `world_size > 1`
  (`synthetic_dataset_generation.py`). Each rank holds the full 14B model —
  ~80 GB VRAM per GPU (H100 fits; A100 40 GB does not). 14B single-GPU
  still uses FSDP and fits on smaller GPUs.

### NCCL hang controls

| Env var | Default | Effect |
|---|---|---|
| `ANOMALY_GEN_FINALIZE_BACKEND` | `gloo` | Set to `nccl` only if gloo finalization hangs |
| `TORCH_DISTRIBUTED_TIMEOUT_SEC` | `1800` | Process-group timeout, seconds |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | (alias) | Same effect as `TORCH_DISTRIBUTED_TIMEOUT_SEC`; set whichever is present in your env |
| `MASTER_PORT` | `12341` | Override if port is occupied |

---

## Verification checklist

After the full pipeline:
- `<output_dir>/SDG_result.csv` exists.
- `verify_output.sh` exits 0.
- `nn_score` and `mnn_score` reported per anomaly type via eval.
- Next steps: pipeline default runs Phase 5 search (`num_search_run=3`) and
  Phase 7 filter+regen (`nn_threshold=0.4`). Tune these to skip either.

---

## Error handling

| Symptom | Action |
|---|---|
| `ag_config.yaml` missing | Stop; verify `checkpoint_dir` |
| `validate_jsonl.py` non-zero (unsupported types) | Stop; show unsupported list + supported set; options: retrain, isolated model, trim spec |
| `step` not on a `save_iter` boundary | `torch.load` FileNotFoundError; run `ls ${CKPT}/checkpoints/model/iter_*.pt` and pick a valid step |
| OOM | Reduce `num_gpus` or switch to `model_size=2b` |
| Non-binary masks | Pipeline raises `ValueError`; masks must be 0 or 255 only |
| `verify_output.sh` non-zero | SDG interrupted; re-run SDG before eval |
| Port conflict | Increment `MASTER_PORT` |
