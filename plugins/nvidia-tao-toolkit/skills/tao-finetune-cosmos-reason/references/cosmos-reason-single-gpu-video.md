# Conversation-style video SFT on one high-memory GPU

Load this reference only for a conversation-style video LoRA/PEFT run or
evaluation on a single high-memory NVIDIA GPU with a Cosmos-RL
image whose native BF16 Conv3D path cannot select a cuDNN engine. Dense SFT
does not use this profile and must not emit a `policy.lora` table.

## Packaged runtime helpers

- `scripts/train_video_conversation_single_gpu.py` maps ShareGPT/LLaVA-style records to Qwen3-VL messages,
  caches repeated video decodes, and replaces the non-overlapping Qwen3-VL
  patch-embedding Conv3D with equivalent linear math.
- `scripts/evaluate_video_conversation_single_gpu.py` applies the corresponding vLLM Conv3D
  fallback and evaluates video inputs in bounded chunks.

Mount the selected helper read-only into the user-selected Cosmos-RL image and
invoke it with the same TOML config that would otherwise be passed to the
standard train or evaluate entry point. Do not copy credentials into the image,
config, command log, or output directory.

## Conversation record contract

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
  use the frame count selected from the model and dataset profile.
- Keep checkpointing epoch-based by default with
  `train.ckpt.save_freq_in_epoch=1`. Dataset or GPU selection does not justify a
  step-based override. Only when the user explicitly requests step-based
  checkpointing, set `train.ckpt.save_freq` and omit
  `train.ckpt.save_freq_in_epoch` (or set it to `0`).
- Keep `train.ckpt.enable_checkpoint=true`, `export_safetensors=true`, and
  `save_ckpt_at_exit=true` so both the resumable policy and LoRA adapter are
  recoverable.

Keep validation epoch-based by default with `validation.freq_in_epoch=1`.
Dataset or GPU selection does not justify a step-based override. If the user
explicitly requests a step frequency, calculate the number and cost of complete
validation passes from that customer's validation record count before launch.

## Checkpoint handoff

For the default epoch cadence, treat the concrete `safetensors/epoch_N`
directory as the extracted LoRA adapter and `checkpoints/epoch_N/policy` as the
resumable full training state. If the user explicitly selected step cadence,
use the corresponding `step_N` paths instead. Verify both artifacts exist
before cleanup or evaluation, and retain the concrete best validated cadence
point instead of silently choosing the final one.

## Detached train-to-evaluate gate

For a long local Docker workflow that must launch evaluation only after a
successful training container and a validated adapter checkpoint, use
`scripts/gate_docker_train_evaluate.py`. Run it under a durable process
supervisor (or `setsid`/`nohup`) and supply the training job and container,
training results root, expected adapter metadata JSON, evaluation TOML, image,
mounts, and state/output paths. The gate:

- polls Docker rather than inferring live state from job records;
- uses an advisory lock and atomic state files so the same command can safely
  resume after interruption;
- validates a unique epoch adapter directory, requested PEFT metadata, and the
  safetensors header before opening the child evaluation record;
- preserves record-before-launch ordering and terminal job-record states; and
- accepts both metric-object and row-oriented Cosmos evaluation results.

Container identity defaults are derived from the invoking account. Pass
`--group-add GID` once per required host video/render group; never copy numeric
UIDs or group IDs from another machine. See the script's `--help` output for
the complete argument contract.
