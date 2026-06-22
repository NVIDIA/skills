# Tuning Parameters

## Important Parameters

- **train.validation_interval**: Default 50. Run validation every N epochs. **IMPORTANT: must be ≤ num_epochs**, otherwise no validation runs and training may fail or produce no metrics. For short runs (e.g., 10 epochs), set to 5.
- **train.checkpoint_interval**: Default 200. Save checkpoint every N epochs. **IMPORTANT: must be ≤ num_epochs**, otherwise no checkpoint is saved and the training output is lost. For short runs, set to match num_epochs or lower.
- **train.num_epochs**: Default 100. Defect detection datasets are typically small, so training may converge in 50-100 epochs. Monitor validation metrics to avoid overfitting.
- **model.classify.train_margin_euclid**: Margin for the Euclidean distance loss during training (default 2.0). Larger values push embeddings further apart. Increase if the model struggles to separate defective from non-defective.
- **model.classify.eval_margin**: Classification threshold during evaluation (default 0.3). Samples with embedding distance below this margin are classified as non-defective; above as defective. This is the primary knob for precision/recall tradeoff -- lower values increase recall (catch more defects), higher values increase precision (fewer false alarms).
- **model.classify.embedding_vectors**: Number of embedding dimensions (default 5). Increase for more complex defect patterns; decrease for simpler binary tasks.
- **dataset.classify.batch_size**: Default 16. Training uses the Optical Inspection dataloader and requires this value to be greater than 1; use 2 as the minimum smoke-test value. Can be increased for small images (224x224) on GPUs with sufficient VRAM.
- **dataset.classify.fpratio_sampling**: False positive ratio for balanced sampling during training (default 0.25). Controls the ratio of non-defective to defective samples in each batch.
- **train.classify.cls_weight**: Class weights for cross-entropy loss (default [1.0, 10.0]). The higher weight on class 1 (defective) compensates for class imbalance typical in defect detection datasets.

## Hardware

- **Minimum**: 1 GPU with 16GB+ VRAM (V100 or A100). Single-GPU training works for small datasets (<10k images).
- **Recommended**: 8 GPUs for production training on larger datasets. Visual ChangeNet uses DDP (DistributedDataParallel) across GPUs.
- GPU count is managed internally by TAO -- do not set `gpu_spec_key` in the spec. The `num_nodes` field (default 1) controls multi-node training.
