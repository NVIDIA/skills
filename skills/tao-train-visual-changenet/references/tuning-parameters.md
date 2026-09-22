# Tuning Parameters

## Important Parameters

- **train.validation_interval**: Default 50. Run validation every N epochs. **IMPORTANT: must be ≤ num_epochs**, otherwise no validation runs and training may fail or produce no metrics. For short runs (e.g., 10 epochs), set to 5.
- **train.checkpoint_interval**: Default 200. Save checkpoint every N epochs when periodic checkpointing is active. **IMPORTANT: must be ≤ num_epochs**, otherwise no periodic checkpoint is saved. For short direct-training runs, set to match num_epochs or lower.
- **train.num_epochs**: Default 100. Defect detection datasets are typically small, so training may converge in 50-100 epochs. Monitor validation metrics to avoid overfitting.
- **model.classify.train_margin_euclid**: Margin for the Euclidean distance loss during training (default 2.0). Larger values push embeddings further apart. Increase if the model struggles to separate defective from non-defective.
- **model.classify.eval_margin**: Classification threshold during evaluation (default 0.3). Samples with embedding distance below this margin are classified as non-defective; above as defective. This is the primary knob for precision/recall tradeoff -- lower values increase recall (catch more defects), higher values increase precision (fewer false alarms).
- **model.classify.embedding_vectors**: Number of embedding dimensions (default 5). Increase for more complex defect patterns; decrease for simpler binary tasks.
- **dataset.classify.batch_size**: Default 16. Training uses the Optical Inspection dataloader and requires this value to be greater than 1; use 2 as the minimum smoke-test value. Can be increased for small images (224x224) on GPUs with sufficient VRAM.
- **dataset.classify.fpratio_sampling**: False positive ratio for balanced sampling during training (default 0.25). Controls the ratio of non-defective to defective samples in each batch.
- **train.classify.cls_weight**: Class weights for cross-entropy loss (default [1.0, 10.0]). The higher weight on class 1 (defective) compensates for class imbalance typical in defect detection datasets.

## Checkpoint Retention

The classification and segmentation train templates preserve direct-training behavior with
`train.checkpointer.enable_topk: false` and
`train.checkpointer.replace_periodic: false`, so `train.checkpoint_interval`
controls periodic checkpoint saves. When AutoML checkpoint retention is
enabled, orchestration sets both flags to `true`, uses `monitor: val_loss`,
`mode: min`, and `save_top_k: 1`, and replaces the periodic checkpoint series
with the single best checkpoint. When retention is disabled, periodic saving
remains in effect. Checkpointer fields control artifact lifecycle and must not
be included in the AutoML search space.

### Retain the best FAR checkpoint

`train.optim.monitor_name` is scheduler metadata; it does not configure
checkpoint selection. `val_far` is FAR at 100% defect recall. If validation is
missing either PASS or defect samples, it reports the explicit worst-case
`100.0` sentinel; use a two-class validation split for meaningful retention.
To retain the lowest validation false-alarm rate, set the checkpointer
explicitly:

```yaml
train:
  checkpointer:
    enable_topk: true
    monitor: val_far
    mode: min
    save_top_k: 1
    replace_periodic: false
```

With `replace_periodic: false`, metric-best retention is additive to periodic
checkpoints. Set it to `true` only when the single best-FAR checkpoint should
replace periodic history and drive the latest symlink.

## Hardware

- **Minimum**: 1 GPU with 16GB+ VRAM (V100 or A100). Single-GPU training works for small datasets (<10k images).
- **Recommended**: 8 GPUs for production training on larger datasets. Visual ChangeNet uses DDP (DistributedDataParallel) across GPUs.
- GPU count is managed internally by TAO -- do not set `gpu_spec_key` in the spec. The `num_nodes` field (default 1) controls multi-node training.
