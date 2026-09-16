# Local Docker Invocation

When running without the TAO SDK (local docker), use the pinned TAO pyt image and invoke directly:

```bash
set -a; source <workspace>/.env; set +a

# Pinned TAO pyt container URI (stamped from the release manifest).
TAO_PYT_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt  # versions-key: images.tao_toolkit.pyt

docker run --rm --gpus all --shm-size=8g \
    -e NGC_API_KEY="${NGC_API_KEY}" \
    -v <workspace>:/data/workspace \
    -v <workspace>/results:/results \
    -v <workspace>/kpi/images:/data/datasets/NV_PCB_Siamese/images \
    -v <workspace>/train/base:/data/datasets/NV_PCB_Siamese/csv \
    -v <workspace>/kpi:/data/datasets/NV_PCB_Siamese/kpi \
    -v <workspace>/augmentation/backbone/c_radio_v2_b.safetensors:/data/pretrained_models/C-RADIOv2_B.safetensors \
    "$TAO_PYT_IMAGE" \
    visual_changenet <train|evaluate|inference|export|quantize> -e /data/workspace/specs/<spec>.yaml \
    [key=value overrides...]
```

**`--shm-size=8g` is required** — without it, dataloader workers crash with `Unexpected bus error encountered in worker` due to insufficient shared memory.

**Backbone mount**: mount the C-RADIO `.safetensors` file directly as a single
file or mount its parent directory, and set
`model.backbone.pretrained_backbone_path` to the container path
`/data/pretrained_models/C-RADIOv2_B.safetensors`.

Override checkpoint and results_dir on the command line to avoid editing the spec:
```bash
visual_changenet inference -e /data/workspace/specs/spec.yaml \
    inference.checkpoint=/results/<iter>/train/model_epoch_<EEE>_step_<SSS>.pth \
    inference.results_dir=/results/<iter>/inference/<label>
```
