# Scripts

Inference and evaluation scripts for FastGen.

## Inference

Generate images and videos using pretrained or distilled models.

| Script | Modality | Modes |
|--------|----------|-------|
| [`image_model_inference.py`](inference/image_model_inference.py) | Image | Unconditional, class-conditional, T2I, image-editing (I2I) |
| [`video_model_inference.py`](inference/video_model_inference.py) | Video, Video+Audio | T2V, I2V, V2V, Video2World |
| [`flashdreams_inference.py`](inference/flashdreams_inference.py) | Video | FlashDreams Self-Forcing Wan 2.1 T2V example |
| [`vlm_infer.py`](inference/vlm_infer.py) | Image, Video | VLM-as-judge scores for existing media listed in a JSONL file, using the endpoint and token of [`VLMLogger`](../fastgen/callbacks/README.md#log-modules) |

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--do_student_sampling` | Run configured student (default: `True`) |
| `--do_teacher_sampling` | Run the teacher (default: `False`). For methods without one, such as SFT, the script builds it from `model.teacher` (or `model.net`) and `model.pretrained_model_path` |
| `--ckpt_path` | Path ending in `.pth` for either a monolithic checkpoint or a DCP completion marker |
| `--sample_steps` | Teacher sampling steps; does not set the distilled student's grid |
| `model.student_grid_cfg.steps` | Student grid steps (config override); verify the selected method's effective network evaluations |
| `--prompt_file` | File with one prompt per line, or one integer class label per line with `--classes`, resolved from the working directory |
| `--classes N` | Class-conditional with N classes |
| `--unconditional` | Unconditional generation |
| `--neg_prompt_file` | Negative prompts for CFG, one per `--prompt_file` prompt (a short file repeats its last, extras are dropped). Omit it for no negative conditioning; `scripts/inference/prompts/empty_prompt.txt` requests the empty negative prompt. Class-conditional image runs ignore it. See [Guidance](#guidance) |
| `--input_image_file` | Input images for I2V, or source/reference images for image-editing (I2I) |
| `--source_video_file` | Source videos for V2V |
| `model.guidance_cfg.cfg.scale` | CFG scale (config override). See [`fastgen/utils/guidance.py`](../fastgen/utils/guidance.py) for the full `GuidanceConfig` schema (mix, rescale) |
| `trainer.seed` | Random seed for reproducibility (config override) |

#### Checkpoint selection

`--ckpt_path` loads student `net` and configured EMA weights, not the teacher. Student sampling uses the first
configured EMA when present; teacher sampling uses its constructor/pretrained configuration. To evaluate a
fine-tuned SFT checkpoint, select student sampling. Without `--ckpt_path`, the scripts use constructor/pretrained
weights and do not discover the latest training checkpoint. Check startup's loaded keys and selected weight source.

#### Guidance

`model.guidance_cfg` decides which guidance terms run, and `--neg_prompt_file`
supplies their negative prompts. Classifier-free guidance (`cfg.scale`) and mix
guidance (`mix.scale`) evaluate a negative branch, so with either of them active,
pass a negative prompt file, or `scripts/inference/prompts/empty_prompt.txt` for
the empty negative prompt; omitting the file does not disable them. Configs whose
guidance runs neither term, such as those of guidance-distilled networks, need no
negative prompt. Class-conditional image runs (`--classes`) ignore the file and
use the all-zero class vector as the negative condition.

### Example commands

#### Unconditional
```bash
python scripts/inference/image_model_inference.py \
    --config fastgen/configs/experiments/EDM/config_sft_edm_cifar10.py \
    --unconditional --num_samples 16
```

#### Class-Conditional
```bash
python scripts/inference/image_model_inference.py \
    --config fastgen/configs/experiments/DiT/config_sft_sit_xl.py \
    --classes 1000 \
    --prompt_file scripts/inference/prompts/classes.txt
```

#### Text-to-Image (T2I)
```bash
python scripts/inference/image_model_inference.py \
    --config fastgen/configs/experiments/Flux/config_sft.py
```

#### Image Editing (I2I)
```bash
python scripts/inference/image_model_inference.py \
    --config fastgen/configs/experiments/Flux2/config_sft_image_edit.py \
    --prompt_file scripts/inference/prompts/edit_prompts.txt \
    --input_image_file scripts/inference/prompts/source_image_paths.txt
```

#### Text-to-Video (T2V)
```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanT2V/config_sft.py \
    --neg_prompt_file scripts/inference/prompts/negative_prompt.txt
```

#### Image-to-Video (I2V)
```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanI2V/config_sft_wan22_5b.py \
    --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \
    --input_image_file scripts/inference/prompts/source_image_paths.txt
```

#### Camera-Conditioned I2V
```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanI2V/config_sft_lingbot_world.py \
    --neg_prompt_file scripts/inference/prompts/negative_prompt_lingbot_world.txt \
    --input_image_file scripts/inference/prompts/source_image_paths.txt \
    --camera_poses_file scripts/inference/prompts/camera_poses_paths.txt
```


#### Video-to-Video (V2V)
```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanV2V/config_sft.py \
    --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \
    --source_video_file scripts/inference/prompts/source_video_paths.txt
```

#### Text-to-Video+Audio (LTX-2)
```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/LTX2/config_sft.py \
    --neg_prompt_file scripts/inference/prompts/negative_prompt_ltx.txt
```

#### Video2World (Cosmos)
```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/CosmosPredict2/config_sft_v2w.py \
    --neg_prompt_file scripts/inference/prompts/negative_prompt_cosmos.txt \
    --input_image_file scripts/inference/prompts/source_image_paths.txt
```

#### Causal / Autoregressive
Causal configs (e.g., `config_sft_causal_wan22_5b.py`) generate autoregressively, chunk by chunk with a KV cache.

```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanI2V/config_sft_causal_wan22_5b.py \
    --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \
    --input_image_file scripts/inference/prompts/source_image_paths.txt
```

Generating longer videos via extrapolation needs a CausVid or Self-Forcing config, such as
[`WanT2V/config_sf.py`](../fastgen/configs/experiments/WanT2V/config_sf.py); other methods, including CausalSFT
configs like the one above, raise an error:
- `--num_segments N`: Generate N consecutive video segments autoregressively (default: 1)
- `--overlap_frames K`: Overlap K latent frames between segments for temporal consistency (default: 0); K must end on
  a chunk boundary

```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanT2V/config_sf.py \
    --num_segments 3 --overlap_frames 3
```

### FlashDreams Example: Self-Forcing Wan 2.1 T2V

[FlashDreams](https://github.com/NVIDIA/flashdreams) provides an optimized
streaming runtime with `torch.compile`, CUDA Graph replay, and multi-GPU context
parallelism. This example uses it to run FastGen-distilled Self-Forcing Wan 2.1
T2V checkpoints. Other model families require a compatible FlashDreams
integration and checkpoint mapping.

Install the `self_forcing` integration from a local FlashDreams clone. The script targets FlashDreams v0.1.0; later versions moved this integration and changed its API:

```bash
git clone --branch v0.1.0 https://github.com/NVIDIA/flashdreams.git
cd flashdreams
uv sync --project integrations/self_forcing
```

Then run inference from the FastGen repo using that environment:

```bash
cd /path/to/FastGen
uv run --project /path/to/flashdreams/integrations/self_forcing \
    --with-editable . --with pandas \
    python scripts/inference/flashdreams_inference.py \
    --checkpoint /path/to/checkpoint.pth \
    --config fastgen/configs/experiments/WanT2V/config_sf.py
```

For DCP checkpoints, consolidate first with [`scripts/fsdp_convert.py`](fsdp_convert.py), converting the network that inference uses: the first EMA when the config enables one (e.g. `--model-type ema_model` for `use_ema=True`), otherwise the default `net_model`. Monolithic checkpoints select the EMA automatically.

The inference settings are loaded from the FastGen config, and overrides use the same trailing `-` syntax as other FastGen scripts.
The rollout uses the first prompt of `--prompt-file`, which defaults to the same validation prompts as the other inference scripts.
For an approximately 30-second rollout at 16 FPS with a 1.3B chunk-3 config, use `--sink --total-blocks 40`. 
For 14B, pass `--config fastgen/configs/experiments/WanT2V/config_sf_14b.py`.
Videos are written to `$FASTGEN_OUTPUT_ROOT/<project>/<group>/<name>/flashdreams` by default.

---

## FID Evaluation

Compute Fréchet Inception Distance (FID) for image models using [`fid/compute_fid_from_ckpts.py`](fid/compute_fid_from_ckpts.py).

### Usage

```bash
torchrun --nproc_per_node=8 scripts/fid/compute_fid_from_ckpts.py \
    --config fastgen/configs/experiments/EDM/config_dmd2_cifar10.py
```

For each checkpoint in `trainer.checkpointer.save_dir`, the script samples
`eval.num_samples` images from the student and records its FID against the
[reference statistics](#reference-statistics) in
`{log_config.save_path}/{eval.samples_dir}/fid.json`. See the
[script docstring](fid/compute_fid_from_ckpts.py) for checkpoint selection,
reruns, and storage requirements.


### Config Options

| Parameter | Description |
|-----------|-------------|
| `eval.num_samples` | Number of samples to generate (default: 50000) |
| `eval.min_ckpt` | Minimum checkpoint iteration to evaluate |
| `eval.max_ckpt` | Maximum checkpoint iteration to evaluate |
| `eval.samples_dir` | Subdirectory name for generated samples |
| `eval.save_images` | Save visualization grid instead of computing FID |

### Reference Statistics

FID reference statistics can be computed with the following commands:
```bash
# Prepare FID refs for CIFAR-10 (uses EDM's fid.py, outputs .npz)
python scripts/download_data.py --dataset cifar10 --compute-fid-refs
# Prepare all FID refs (uses EDM's fid.py for EDM format, EDM2's calculate_metrics.py for EDM2 format)
python scripts/download_data.py --dataset all --imagenet-source /path/to/imagenet --compute-fid-refs
```
This computes the FID reference statistics according to the [EDM](https://github.com/NVlabs/edm) and [EDM2](https://github.com/NVlabs/edm2) repositories and stores them in `$DATA_ROOT_DIR/fid-refs/`:

| Dataset | Reference File | Format |
|---------|----------------|--------|
| CIFAR-10 | `fid-refs/cifar10-32x32.npz` | EDM (`.npz`) |
| ImageNet-64 (EDM) | `fid-refs/imagenet-64x64.npz` | EDM (`.npz`) |
| ImageNet-64 (EDM2) | `fid-refs/imagenet-64x64-edmv2.pkl` | EDM2 (`.pkl`) |
| ImageNet-256 | `fid-refs/imagenet_256.pkl` | EDM2 (`.pkl`) |
