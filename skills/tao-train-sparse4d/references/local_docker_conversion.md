# Local-Docker Conversion and Smoke-Run Guidance

Set `TAO_SKILL_BANK_PATH` to the installed bank root. For local-docker runs, keep Sparse4D conversion rooted at `/data/aicity_root` and train/eval data roots at the converted split folder, for example `/data/aicity_root/train`. Mount the same host directory at `/data/aicity_root` for dataset_convert, train, evaluate, and inference; the converter extracts RGB frames there and writes those absolute frame paths into the pickle files. The annotations converter writes absolute RGB paths under the conversion root and relative depth paths under the split, so both mounts must stay stable across conversion and training. For a legacy labeled conversion with directory-valued `depth_map_path` tuples, inspect the trusted PKLs and use the helper below only if the files actually live under `depth_maps/`. Already-correct `.h5` paths are left unchanged. Current annotation-free conversion has no depth paths. Run a dry check first, then omit `--dry-run` to apply the verified changes:

```bash
python3 "${TAO_SKILL_BANK_PATH}/skills/models/tao-train-sparse4d/scripts/normalize_depth_paths.py" \
  --dry-run \
  --data-root /path/to/aicity_root/train \
  "/path/to/results/${dataset_convert_job_id}/results_dir/train"
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

For a newly initialized smoke model intended for the current exporter, retain the default `model.head.instance_bank.num_anchor: 900` and `model.head.instance_bank.num_temp_instances: 600`. Preserve `model.head.num_output` from the selected model specification (the schema default is 300); 900 output queries are not an export requirement. For an existing checkpoint, preserve its dimensions and check exporter compatibility instead of rewriting its architecture. Sparse4D export currently creates cached feature
and anchor tensors sized for the default 600 temporal instances. If a tiny
dataset conversion produces fewer anchors, for example a 3-frame conversion that
emits a 72-row `anchor_init.npy`, evaluate/inference can still run with matching
reduced config values but export will fail during memory-bank update. For fine-tuning, reuse the selected pretrained model's matching anchors. For a labeled conversion that needs new anchors, use enough annotated frames to generate the configured count; do not pad or invent anchors. Annotation-free conversion intentionally produces no anchor file.

The current exporter reads image dimensions from `model.input_shape` in `[width, height]` order, not the generic `export.input_width` / `export.input_height` fields. Preserve the selected checkpoint's paired preprocessing and verify the emitted ONNX input signature before building an engine. For example, `model.input_shape: [960, 540]` produces an image input shaped `[batch_size, num_cams, 3, 540, 960]`; setting `export.input_height: 544` alone does not change it.

When reusing a previous dataset conversion for AutoML or repeated training,
copy or mount the conversion output by the explicit `dataset_convert_job_id`,
not by the first `results_dir` found under a results root. The following checks are specific to the labeled AICity smoke fixture, not required filenames for every dataset. Before launch, verify the selected action's actual artifacts. Standalone evaluation reads `dataset.test_dataset.ann_file`, even when evaluating a held-out `val` PKL:

```bash
CONVERTED="/path/to/results/${dataset_convert_job_id}/results_dir"
test -f "${CONVERTED}/anchor_init.npy"
test -f "${CONVERTED}/train/subsetscene+bev-sensor-random-0_infos_train.pkl"
test -f "${CONVERTED}/train/subsetscene+bev-sensor-random-1_infos_train.pkl"
test -f "${CONVERTED}/train/subsetscene+bev-sensor-random-2_infos_train.pkl"
```

If a required artifact is missing, resolve its path or rerun the relevant labeled conversion with the action-level Data Services image. For unlabeled adaptation, use the existing pretrained anchors. A wrong artifact path often surfaces as
`FileNotFoundError: .../anchor_init.npy` during model construction.

The AICity converter may extract the full camera videos to RGB frames even when
`aicity.num_frames` is set to a small value for the converted pickle. Plan for
the raw-data mount to hold the extracted frames as well as the H5 depth maps.

## Local Docker command shape

After platform preflight, specification staging, and job-record creation, the local Docker platform can submit this command shape. Set `TAO_PYT_IMAGE` to the resolved model image, `TAO_GPU_REQUEST` to explicit allocated GPU IDs (for example `device=0`), `TAO_DATA_DIR` to the absolute host staging root (including `aicity_root`, `specs`, and model/artifact files), and `TAO_RESULTS_DIR` to the current job record's results directory. `TAO_JOB_ID` is the ID returned by `tao_job_record.py open`. Use the Docker platform's `HOST_IDENTITY_ARGS` array to preserve the submitting UID, GID, and supplementary groups. All specification paths must match these mounts.

```bash
docker run -d --name "${TAO_JOB_ID}" --label "tao-job=${TAO_JOB_ID}" \
  --gpus "${TAO_GPU_REQUEST}" --shm-size=16g \
  "${HOST_IDENTITY_ARGS[@]}" \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache \
  -v "${TAO_DATA_DIR}:/data:ro" \
  -v "${TAO_RESULTS_DIR}:/results" \
  "${TAO_PYT_IMAGE}" \
  sparse4d train -e /data/specs/train.yaml
```

Record the returned container ID as `RUNNING`, then use the platform's `status` and `logs` verbs. Retain the exited container until its logs and terminal state have been recorded. Results persist in the bound results directory; the example's cache directory is ephemeral.

Conversion uses the action-level Data Services image and `annotations convert`; its raw-data mount must be writable if video decoding will extract frames. For quantization that needs calibration, supply `dataset.quant_calibration_dataset.images_dir` and the matching calibration PKL at `dataset.test_dataset.ann_file`. Weight-only quantization can omit calibration data when supported by the selected backend.
