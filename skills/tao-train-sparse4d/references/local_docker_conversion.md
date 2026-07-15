# Local-Docker Conversion and Smoke-Run Guidance

For local-docker runs, keep Sparse4D conversion rooted at `/data/aicity_root` and train/eval data roots at the converted split folder, for example `/data/aicity_root/train`. Mount the same host directory at `/data/aicity_root` for dataset_convert, train, evaluate, and inference; the converter extracts RGB frames there and writes those absolute frame paths into the pickle files. The annotations converter writes absolute RGB paths under the conversion root and relative depth paths under the split, so both mounts must stay stable across conversion and training. When `aicity.depth_format: h5`, normalize the converted pickle depth tuples before training if the H5 files live under `depth_maps/`:

```bash
models/sparse4d/scripts/normalize_depth_paths.py \
  --data-root /path/to/aicity_root/train \
  /path/to/results/<dataset_convert_job_id>/results_dir/train
```

Use the actual converted annotation filename emitted by Data Services. For the
packaged AICity smoke dataset the basename is
`subsetscene+bev-sensor-random-0`, so the train file is
`train/subsetscene+bev-sensor-random-0_infos_train.pkl`; do not strip the
BEV-sensor suffix back to `subsetscene_infos_train.pkl`.

For small local smoke runs with fewer camera streams than the production
default, keep `model.head.deformable_model.max_num_cams: 20` if the resulting
checkpoint will be exported. The current Sparse4D ONNX exporter constructs a
20-camera dummy input, so checkpoints trained with `max_num_cams` reduced to the
dataset camera count can load for evaluate/inference but fail export with a
deformable-attention reshape error. It is safe to keep `max_num_cams: 20` while
training/evaluating on fewer real cameras because the runtime projection matrix
controls the active camera count. Only reduce `num_cams` for smoke data when
needed; leave `max_num_cams` at 20 for export-compatible checkpoints.

For export-compatible smoke checkpoints, also keep the default anchor contract:
`model.head.instance_bank.num_anchor: 900`,
`model.head.instance_bank.num_temp_instances: 600`, and
`model.head.num_output: 900`. Sparse4D export currently creates cached feature
and anchor tensors sized for the default 600 temporal instances. If a tiny
dataset conversion produces fewer anchors, for example a 3-frame conversion that
emits a 72-row `anchor_init.npy`, evaluate/inference can still run with matching
reduced config values but export will fail during memory-bank update. Prefer
rerunning `dataset_convert` with enough real frames to initialize 900 anchors
instead of padding or inventing anchors.

When reusing a previous dataset conversion for AutoML or repeated training,
copy or mount the conversion output by the explicit `dataset_convert_job_id`,
not by the first `results_dir` found under a results root. Before launching
train/evaluate/inference, verify all required converted artifacts exist:

```bash
CONVERTED="/path/to/results/${dataset_convert_job_id}/results_dir"
test -f "${CONVERTED}/anchor_init.npy"
test -f "${CONVERTED}/train/subsetscene+bev-sensor-random-0_infos_train.pkl"
test -f "${CONVERTED}/train/subsetscene+bev-sensor-random-1_infos_train.pkl"
test -f "${CONVERTED}/train/subsetscene+bev-sensor-random-2_infos_train.pkl"
```

If any check fails, rerun `dataset_convert` with the `tao_toolkit.data_services`
image instead of launching train. A wrong conversion artifact often surfaces as
`FileNotFoundError: .../anchor_init.npy` during model construction.

The AICity converter may extract the full camera videos to RGB frames even when
`aicity.num_frames` is set to a small value for the converted pickle. Plan for
the raw-data mount to hold the extracted frames as well as the H5 depth maps.
