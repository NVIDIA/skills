# Setup and Run

## bp2 distilled width overrides (Step 3)

FFS requires 15 model-section width override fields whose values match the bp2 commercial checkpoint exactly. Omitting any field falls back to TAO defaults that do **not** match the bp2 ckpt and produce shape-mismatch errors at forward time.

```yaml
model:
  model_type: FastFoundationStereo
  encoder: vitl
  hidden_dims: [128]                    # 1-layer GRU; NOT [128,128,128]
  n_gru_layers: 1                       # bp2 single-GRU
  corr_radius: 4
  corr_levels: 2
  n_downsample: 2
  valid_iters: 8
  max_disparity: 192                    # bp2 commercial; NOT 416 (full FS default)
  volume_dim: 28                       # bp2 ckpt invariant; NOT 32 (full FS default)
  mixed_precision: false                # see "Important Parameters"
  gwc_feature_normalize: true           # see "Important Parameters"

  # 15 bp2 distilled width overrides — copy as-is
  motion_encoder_widths: [56, 96, 16, 12]
  motion_encoder_final: 48
  gru_hidden: 60
  gru_gating_conv_widths: [100, 168]
  disp_head_input_dim: 60
  disp_head_intermediate: 36
  disp_head_pwconv1_widths: [212, 244]
  mask_widths: [32, 16]
  stem_2_widths: [12, 16]
  spx_2_gru_widths: [16, 12, 16, 24]
  spx_gru_out: 9
  classifier_mid: 14
  cnet_conv04_widths: [60, 48]
  cam_mid_channels: 8
  cost_agg_conv_patch_padding: [0, 0, 0]
```

The spec templates at `references/spec_template_*.yaml` carry this block as the canonical source.

## Chained train → next action checkpoint path (Step 4)

For local Docker chaining (no SDK runner), Lightning `ModelCheckpoint` nests under the task name. Example: `train.results_dir: /workspace/results/finetune/train` produces checkpoints under `/workspace/results/finetune/train/train/`. The exact checkpoint pattern is `model_epoch_<epoch>_step_<step>.pth`, plus a `dn_model_latest.pth` symlink. Use the model-specific or SDK-provided checkpoint resolver to select the intended exact epoch/step checkpoint for `evaluate`, `inference`, `export`, resume, and deploy handoff. Use `dn_model_latest.pth` only when the user explicitly asks for latest. SDK-runner deploys resolve this automatically via `parent_job_id` — see `references/parent-model-inference.md`.

Shape consistency: `crop_size` in `dataset.test_dataset.augmentation.crop_size` should match `export.input_height` / `input_width` for end-to-end pyt-vs-deploy comparability — see `references/tao-deploy-fast-foundation-stereo.md`'s shape table.

## Run (Step 5)

Create writable home/cache directories inside the mounted output path before using `--user`. Some TAO containers do not have an `/etc/passwd` entry for the host UID, and PyTorch / matplotlib need writable cache paths when running as that UID.

```bash
mkdir -p <output_dir>/home \
         <output_dir>/.cache/matplotlib \
         <output_dir>/.cache/torchinductor \
         <output_dir>/.cache/xdg
```

```
docker run --gpus 'device=0' --shm-size 16G --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" \
  -e LOGNAME="$(id -un)" \
  -e HOME=<output_dir>/home \
  -e MPLCONFIGDIR=<output_dir>/.cache/matplotlib \
  -e TORCHINDUCTOR_CACHE_DIR=<output_dir>/.cache/torchinductor \
  -e XDG_CACHE_HOME=<output_dir>/.cache/xdg \
  -v <data_root>:<data_root>:ro \
  -v <output_dir>:<output_dir> \
  -v <bp2_ckpt_dir>:<bp2_ckpt_dir>:ro \
  <container> \
  depth_net <action> -e <spec.yaml>
```

Without `--user "$(id -u):$(id -g)"` the container writes outputs as `nobody:nogroup`, blocking host-side cleanup / retry.

**Local bind-mount tip (QA / development only)**: When bind-mounting a modified TAO repo (`tao-pytorch`, `tao-core`, `tao-deploy`) into the container, stale `__pycache__/*.pyc` files from a previous container run can shadow your patched `.py` source. The symptom is a cryptic TRT-side error (e.g., `IOptimizationProfile::setDimensions Error Code 3`) when the new code path should have produced something different. Clear the caches before launching the container:

```bash
find /path/to/tao-pytorch -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find /path/to/tao-core    -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find /path/to/tao-deploy  -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

SDK-runner production deployments are not affected — the runner copies sources fresh per job.

## Verify (Step 6)

- Container exit code 0
- `status.json` `kpi` block populated
- For `train`: inspect per-step `train_loss` directly (the entrypoint reports `Execution status: PASS` even when loss is NaN)
- For `evaluate`: rely on `epe` / `bp1` / `bp2` / `bp3` / `d1` / `rmse` (the evaluator also emits `abs_rel` / `sq_rel` / `rmse_log` which are non-meaningful for stereo)
- For `inference`: artifacts under `results_dir`
- **KPI namespace difference between pyt and deploy**: pyt `evaluate` writes the metric set under `kpi.val/epe`, `kpi.val/bp1`, etc. (namespaced by Lightning's `val/` prefix). Deploy `evaluate` (TRT engine path) writes the same metric set under `kpi.epe`, `kpi.bp1`, etc. (no `val/` prefix). Downstream verification scripts that read `status.json` need to handle both shapes.
- **Validate drift on your own dataset**: if you compare TAO FFS deploy (`gen_trt_engine` + TRT `evaluate`) against the upstream FFS deploy path on the same input, expect a small residual mean_abs disparity drift (TAO export graph + TRT 10.13 interaction; not improvable at the source-code level). The exact magnitude is dataset and hardware dependent — measure on your own data and decide whether the drift is acceptable for your downstream task.
