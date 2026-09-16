# Local Docker Invocation

When running without the TAO SDK (local docker), use the pinned TAO pyt image and invoke directly:

```bash
# Pinned TAO pyt container URI (stamped from the release manifest).
TAO_PYT_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt  # versions-key: images.tao_toolkit.pyt

set -a; source /path/to/.env; set +a   # omit if already exported
docker run --rm --gpus all --shm-size=8g \
    -e NGC_API_KEY \
    -v <workspace>:/data/workspace \
    -v <workspace>/results:/results \
    -v <images_dir>:/data/datasets/NV_PCB_Siamese/images \
    -v <workspace>/train/base:/data/datasets/NV_PCB_Siamese/csv \
    -v <workspace>/kpi:/data/datasets/NV_PCB_Siamese/kpi \
    -v <workspace>/augmentation/backbone/c_radio_v2_b.safetensors:/data/pretrained_models/C-RADIOv2_B.safetensors \
    "$TAO_PYT_IMAGE" \
    visual_changenet <train|evaluate|inference|export|quantize> -e /data/workspace/specs/<spec>.yaml \
    [key=value overrides...]
```

`<images_dir>` is the host directory against which the CSV `input_path` and
`golden_path` values resolve. DEFT workspaces record it in
`deft_state.json::config.images_dir`; their canonical layout uses
`<workspace>/images`.

**`--shm-size=8g` is required** — without it, dataloader workers crash with `Unexpected bus error encountered in worker` due to insufficient shared memory.

**Backbone mount**: mount the C-RADIO `.safetensors` file directly as a single
file or mount its parent directory, and set
`model.backbone.pretrained_backbone_path` to the container path
`/data/pretrained_models/C-RADIOv2_B.safetensors`.

Select an actual `model_epoch_*_step_*.pth` checkpoint or the
`changenet_model_classify_latest.pth` symlink produced by training. Override it
with an explicit container path; `${results_dir}` is action-local and must not
be used to cross from inference/evaluate/export back into the train directory:
```bash
visual_changenet inference -e /data/workspace/specs/spec.yaml \
    inference.checkpoint=/results/<iter>/train/model_epoch_<EEE>_step_<SSS>.pth \
    inference.results_dir=/results/<iter>/inference/<label>
```
