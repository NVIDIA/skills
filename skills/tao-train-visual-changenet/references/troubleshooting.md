# Troubleshooting

## Error Patterns

**Checkpoint not found**: The evaluate, inference, export, and quantize actions
require a valid checkpoint path. Current TAO 6.25.10 Visual ChangeNet training
emits epoch/step checkpoint files such as `model_epoch_000_step_00012.pth`;
it does not necessarily write `changenet_model_classify_latest.pth` or
`changenet_model_segment_latest.pth`. Use the model-skill `parent_model`
resolver for downstream actions and `resume_model` for resume, or pass the exact
epoch/step checkpoint when running local Docker directly.

**CSV format mismatch**: The classify CSV must have exactly four columns:
`input_path`, `golden_path`, `label`, and `object_name`. Missing columns or
extra headers cause a silent failure or KeyError. Verify the CSV has no BOM
characters and uses comma delimiters (not semicolons or tabs).

**Image extension mismatch**: If `dataset.classify.image_ext` is `.jpg` but the actual images are `.png` (or vice versa), the data loader will find zero samples and training will fail with an empty dataset error. Always verify the extension matches your data.

**OOM during training**: Reduce `dataset.classify.batch_size` (16 -> 8 -> 4). With the default image size of 224x224, batch_size=16 typically fits on a 16GB GPU. If using larger images via `image_width`/`image_height`, reduce batch size proportionally.

**Low evaluation accuracy with correct training loss**: The `eval_margin` threshold may be miscalibrated for your data. After training, run inference on a validation set and inspect the embedding distance distribution to pick an appropriate threshold. The default 0.3 is tuned for the reference dataset and may not generalize.

**`AssertionError: Contrastive loss only supports Euclidean distance module`** at evaluate/inference: the spec dropped the `train` subtree. Model `__init__` reads `train.classify.loss` regardless of action; omitting it falls back to contrastive loss, which then conflicts with non-default `model.classify.difference_module` (e.g. `learnable`) saved in the checkpoint. Keep `train.classify.loss` (and `train.classify.cls_weight`) in the spec for evaluate and inference too.

**Checkpoint load key mismatch at evaluate/inference**: Keep the classify model
architecture fields aligned with the train spec. C-RADIO classify checkpoints
require `model.backbone.type: c_radio_v2_vit_base_patch16_224`,
`model.classify.difference_module: learnable`, `model.classify.embed_dec: 30`,
`model.classify.eval_margin: 0.3`, `dataset.classify.num_input: 1`, and
`dataset.classify.input_map: {SolderLight: 0}` unless the training run used a
different override set.

**Training does not converge**: Check that `train.classify.cls_weight` is appropriate for your class distribution. If defects are very rare (<1% of samples), increase the defective class weight. Also verify that `fpratio_sampling` is not too low, which would under-sample the majority class.

**Backbone dimension mismatch** (segment only): If the log shows size mismatch
errors while loading the backbone, such as a checkpoint tensor with shape
`[1024, 1024]` being copied into a model tensor with shape `[384, 384]`, the
checkpoint does not match `model.backbone.type`. Keep the packaged
`vit_large_nvdinov2` segment templates when using `NV_DINOV2_518_16_256.ckpt`,
or clear `model.backbone.pretrained_backbone_path` to use default
initialization.

**OSError: Could not load MultiScaleDeformableAttention...so** (segment only): CUDA ops not compiled. The ViT adapter backbone requires custom CUDA kernels that must be compiled on first run. Run `python setup.py develop` inside the container (~5 min compilation). This only applies to the segmentation task.

**MisconfigurationException: current_epoch=N, but max_epochs=M**: Old checkpoints in results directory. PyTorch Lightning auto-resumes from checkpoints and crashes if the new `max_epochs` is lower than a previous run's epoch. Fix: use a fresh results directory or unique run name.

**PYTHONPATH / ModuleNotFoundError: nvidia_tao_pytorch**: The TAO entrypoint spawns subprocesses that don't source `.bashrc`. Pass `PYTHONPATH` explicitly via environment variables, not shell init files. The TAO pyt container resolved from `versions.yaml::images.tao_toolkit.pyt` has PYTHONPATH pre-configured.

**Epoch defaults**: Classify training typically uses 100-2000 epochs depending on dataset size. Segmentation uses 200 epochs by default. For small datasets (<1k images), 100 epochs may suffice. For large production datasets, 2000 epochs with early stopping is common. Monitor validation metrics to determine convergence.
