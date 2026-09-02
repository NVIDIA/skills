# WTS on a Single GB300

Load this reference for Woven Traffic Safety (WTS) LoRA SFT or evaluation on a
single NVIDIA GB300 with a Cosmos-RL image whose native BF16 Conv3D path cannot
select a cuDNN engine.

## Packaged runtime helpers

- `scripts/train_wts_gb300.py` maps WTS LLaVA records to Qwen3-VL messages,
  caches repeated video decodes, and replaces the non-overlapping Qwen3-VL
  patch-embedding Conv3D with equivalent linear math.
- `scripts/evaluate_wts_gb300.py` applies the corresponding vLLM Conv3D
  fallback and evaluates video inputs in bounded chunks.

Mount the selected helper read-only into the user-selected Cosmos-RL image and
invoke it with the same TOML config that would otherwise be passed to the
standard train or evaluate entry point. Do not copy credentials into the image,
config, command log, or output directory.

## WTS record contract

The training annotation is a JSON array. Each record must contain:

```json
{
  "video": "relative/path.mp4",
  "conversations": [
    {"from": "human", "value": "<video> question"},
    {"from": "gpt", "value": "answer"}
  ]
}
```

Set `custom.train_dataset.annotation_path` and
`custom.val_dataset.annotation_path` to the JSON files. Set each section's
`media_path` or `media_root` to the directory against which `video` resolves.

## Single-GPU config guards

Apply these guards before the first launch:

- Set `policy.parallelism.dp_shard_size=1` and
  `policy.parallelism.dp_replicate_size=1`.
- Disable unused rollout and distillation replicas for SFT with
  `rollout.parallelism.n_init_replicas=0` and
  `distillation.parallelism.n_init_replicas=0`.
- Set both train and validation `dataloader_num_workers=0`. Omit their
  `dataloader_prefetch_factor`; a positive prefetch factor is invalid with zero
  workers. Forked CUDA video decoding can otherwise return zero frames.
- Keep exactly one of `custom.vision.nframes` and `custom.vision.fps`; the
  validated WTS run used `nframes=8`.
- Keep checkpointing epoch-based by default with
  `train.ckpt.save_freq_in_epoch=1`. WTS and GB300 selection do not justify a
  step-based override. Only when the user explicitly requests step-based
  checkpointing, set `train.ckpt.save_freq` and omit
  `train.ckpt.save_freq_in_epoch` (or set it to `0`).
- Keep `train.ckpt.enable_checkpoint=true`, `export_safetensors=true`, and
  `save_ckpt_at_exit=true` so both the resumable policy and LoRA adapter are
  recoverable.

Keep validation epoch-based by default with `validation.freq_in_epoch=1`.
WTS and GB300 selection do not justify a step-based override. If the user
explicitly requests `validation.freq=20`, it means a complete validation-set
pass every 20 optimizer steps; on 2,676 WTS validation records, 37 passes
consumed about 5 hours 48 minutes.

## Checkpoint handoff

For the default epoch cadence, treat the concrete `safetensors/epoch_N`
directory as the extracted LoRA adapter and `checkpoints/epoch_N/policy` as the
resumable full training state. If the user explicitly selected step cadence,
use the corresponding `step_N` paths instead. Verify both artifacts exist
before cleanup or evaluation, and retain the concrete best validated cadence
point instead of silently choosing the final one.
