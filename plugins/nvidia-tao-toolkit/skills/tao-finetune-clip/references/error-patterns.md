# CLIP Error Patterns

**Invalid LoRA numeric settings**: Check both `peft.vision` and `peft.text`
before running: `rank` and `alpha` must be integers at least 1, and `dropout`
must be finite and satisfy `0 <= dropout < 1`. `num_last_blocks` must be
between 0 (all blocks) and the tower's block count. Older runtimes may report
a bare `ZeroDivisionError` for rank 0, accept alpha 0 or dropout 1 while
silently disabling the adapter contribution, or defer invalid dropout to
PyTorch's dropout error. Correct the named tower settings; the standard
defaults are rank 8, alpha 16, dropout 0.05, and num_last_blocks 3. Under the
newer tower contract, select `mode: frozen` to freeze a tower.

**PEFT tower `enabled` ConfigKeyError after upgrading to 7.3**: If OmegaConf
reports `Key 'enabled' not in 'CLIPLoRATargetConfig'` at
`peft.vision.enabled` or `peft.text.enabled`, migrate each tower independently:
`enabled: true` becomes `mode: lora`, and `enabled: false` becomes
`mode: frozen`. Remove the old tower keys, preserving top-level
`peft.enabled` and `peft.method` and the tower's adapter settings. The new
`peft.train_logit_calibration` defaults to `true`; set it to `false` to freeze
logit calibration parameters. This error occurs before model construction;
see the [migration example](clip-peft.md#migrating-peft-specs-from-72-to-73).

**CUDA out of memory**: Reduce `dataset.train.batch_size`, `dataset.val.batch_size`, or the TensorRT opt/max batch sizes. For export/deploy, check `export.input_height` and `export.input_width` against the selected fixed-resolution backbone.

**NaN loss**: Learning rate is too high for fine-tuning. Reduce `train.optim.vision_lr` and `train.optim.text_lr`, increase `train.optim.warmup_steps`, and verify that captions are valid non-empty text.

**Zero retrieval or classification quality**: Check that captions and prompts match the target label vocabulary. CLIP compares image and text embeddings, so prompt wording matters.

**No CLIP-compatible dataset in a bucket prefix**: Training requires either
custom-format `images.tar.gz` + `captions.tar.gz` + `image_list.txt`, or WDS
`.tar` shards plus a shard list/root. Image-classification archives with
`classes.txt` are not sufficient unless the user explicitly asks to generate
caption files from labels as a separate data-preparation step.

**Dataset size smaller than total batch size**: The total batch size is `batch_size * num_gpus`. If the dataset, especially validation, has fewer samples than this, reduce `dataset.val.batch_size` or `dataset.train.batch_size`.

**Radio-CLIP config validation error**: Set `model.adaptor_name` explicitly to `siglip` or `clip`.

**Unsupported model identifier or transform error**: Use a TAO-registered CLIP model ID supported by the current container. For minimal AutoML validation, prefer `ViT-L-14-SigLIP-CLIPA-224`. Generic OpenCLIP built-ins that are not in TAO's CLIP registry can bypass TAO's augmentation adapter and fail on transform keys.

**ONNX external data missing**: Models larger than 2 GB export an ONNX file plus an external data file. Keep both files in the same directory and do not rename the external data file before `gen_trt_engine`.

**TensorRT shape mismatch**: When using dynamic batch export, provide min/opt/max shape profiles for every input. Text sequence length must match the tokenizer length, commonly 77 for CLIP tokenizers and 64 for SigLIP2 tokenizers.

**PyTorch 2.6 checkpoint load failure**: If a trusted TAO CLIP Lightning checkpoint fails with a `Weights only load failed` / `numpy.dtypes.Float64DType` unpickling error, rerun checkpoint-dependent PyTorch actions with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. For known parent outputs, set this environment variable up front on checkpoint-backed `evaluate`, `inference`, `export`, and resume train actions. Do not use this override for untrusted checkpoints.

**PyTorch inference with null checkpoint**: If `clip inference` fails with `TypeError: expected str, bytes or os.PathLike object, not NoneType`, the spec did not provide `inference.checkpoint`. Use the exact resolved checkpoint from a parent train job for PyTorch inference, or use the export plus TensorRT image-only deploy path when no training checkpoint exists.

**CLIP TensorRT text or retrieval failure**: Full TensorRT retrieval needs both `_vision.engine` and `_text.engine` artifacts in the same directory, or a combined engine. If `clip evaluate` fails with `Text engine not loaded`, build the matching text ONNX with `clip gen_trt_engine` before rerunning evaluation. If an older deploy image fails while parsing `input_ids` or `attention_mask`, fall back to image-only TensorRT inference with the `_vision.engine` and document text/retrieval deployment as blocked for that image.

**attention_mask warning**: `attention_mask` is currently accepted by exported graphs for compatibility, but TAO ignores its values and may remove it in a future release. Do not build new direct-ONNX inference code that depends on mask values.

**Error merging spec.yaml with schema**: A Hydra/OmegaConf config validation error. Common causes are putting `num_epochs` or `num_gpus` at the spec root instead of under `train.*`, or mixing up training image size (`model.image_size`) with export dimensions (`export.input_height` and `export.input_width`).
