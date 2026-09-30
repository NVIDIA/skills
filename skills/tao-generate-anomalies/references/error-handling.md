# Error Handling

Pipeline-level failure modes and how the scripts respond.

- `dataset_dir` missing per-type mask dir → allocation scans zero and errors.
- `validate_dataset.py` reports issues → it now exits `1` on *any* structure or image/mask pairing problem (not only on "no anomaly types detected"). Fix the pairs or remove the offending files; the training dataloader asserts on these and would crash mid-iteration.
- `check.sh` passes but a `14b` run then fails to find its base checkpoint → both `check.sh` and `download_checkpoints.sh` default to `2B` only. Pass `--model-sizes ${MODEL_SIZE^^}` to each so a `14b` run actually verifies and fetches the 14B base.
- Generated images come out all black with `guardrail_pass=0` in `SDG_result.csv` → the image content guardrail blocked them; they are replaced with a black image by design. See `references/output-layout.md §Image content guardrail`.
- Guardrail fails to initialize → logged at **error** level and generation continues *unscreened*. Treat a run with that log line as unvalidated rather than safe; the usual cause is a missing `checkpoints/nvidia/Cosmos-Guardrail1/` (re-run `check.sh`).
- AMP output short of allocation for some defect → `build_jsonl.py` warns and writes what's available; JSONL is shorter than `num_SDG` by that delta. Check `run_auto_roi_amp.py` logs for `NO_DETECTION` / `FAILED`. If a defect produces **zero** AMP outputs, that defect is dropped (warn-only). If **every** defect produces zero, `build_jsonl.py` halts with `error: 0 entries written` since SDG cannot run on an empty JSONL.
- SDG failure mid-round in Phase 5 → halts; re-run resumes from the next round (rounds are append-only).
- `mode=inference_only` with a `step` not on a `save_iter` boundary → `torch.load` FileNotFoundError; `ls ${CKPT}/checkpoints/model/iter_*.pt` to find valid steps.
- See `references/finetune.md` and `references/inference.md` for phase-specific error handling.
