# Error Handling

Pipeline-level failure modes and how the scripts respond.

- `dataset_dir` missing per-type mask dir → allocation scans zero and errors.
- AMP output short of allocation for some defect → `build_jsonl.py` warns and writes what's available; JSONL is shorter than `num_SDG` by that delta. Check `run_auto_roi_amp.py` logs for `NO_DETECTION` / `FAILED`. If a defect produces **zero** AMP outputs, that defect is dropped (warn-only). If **every** defect produces zero, `build_jsonl.py` halts with `error: 0 entries written` since SDG cannot run on an empty JSONL.
- SDG failure mid-round in Phase 5 → halts; re-run resumes from the next round (rounds are append-only).
- `mode=inference_only` with a `step` not on a `save_iter` boundary → `torch.load` FileNotFoundError; `ls ${CKPT}/checkpoints/model/iter_*.pt` to find valid steps.
- See `references/finetune.md` and `references/inference.md` for phase-specific error handling.
