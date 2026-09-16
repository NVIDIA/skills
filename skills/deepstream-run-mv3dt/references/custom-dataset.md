# Custom Dataset Runs

Load this reference when the user wants to run MV3DT on custom synchronized MP4 files through the DeepStream Container path.

## Dataset Contract

The auto-configurator expects:

```text
your_dataset/
|-- videos/
|   |-- cam_00.mp4
|   |-- cam_01.mp4
|   `-- ...
|-- camInfo/
|   |-- cam_00.yml
|   |-- cam_01.yml
|   `-- ...
|-- map.png          # optional, needed for BEV map rendering
|-- transforms.yml   # optional, needed for BEV map rendering
`-- calibration.json # optional, AMC export/provenance for downstream consumers
```

Requirements:

- Videos are synchronized and have the same resolution.
- `camInfo/*.yml` contains DeepStream camera projection data. The current auto-configurator scans `.yml`, so create `.yml` copies for `.yaml` exports when needed.
- Video and calibration names should match by basename or camera number. If they do not, the auto-configurator falls back to sorted order.

## Validate Inputs

```bash
DATASET_DIR="${DATASET_DIR:?set to your dataset directory}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/experiments/deepstream/$(basename "${DATASET_DIR}")}"
MODEL_REPO="${MODEL_REPO:-${REPO_ROOT}/models}"
RUN_STARTED_AT="${RUN_STARTED_AT:-$(date +%s)}"

test -d "${DATASET_DIR}/videos" || { echo "ERROR: ${DATASET_DIR}/videos missing"; exit 1; }
find "${DATASET_DIR}/videos" -maxdepth 1 -type f -name '*.mp4' | sort
find "${DATASET_DIR}/camInfo" -maxdepth 1 -type f -name '*.yml' | sort 2>/dev/null || true
```

If no `.yml` files exist but `.yaml` files do, do not rename user files silently. Prefer non-destructive `.yml` copies after showing the planned source and destination paths and asking for confirmation:

```bash
find "${DATASET_DIR}/camInfo" -maxdepth 1 -type f -name '*.yaml' -print0 \
  | while IFS= read -r -d '' f; do cp -n "$f" "${f%.yaml}.yml"; done
```

If exact renames are required, create a backup or dry-run list first and ask before modifying dataset files.

If calibration is missing, resolve the MV3DT detector first so the calibration handoff can normalize `camInfo` for the selected model. Use the detector requested by the user, or set `DETECTOR_MODEL=PeopleNetTransformer` when the user did not choose one. Then load `amc-calibration-handoff.md` and return here.

## Select Detector And Tracker

Ask the user to confirm detector choice before a long custom-data run. If they do not choose, use `PeopleNetTransformer`, set `DETECTOR_MODEL=PeopleNetTransformer`, and include that default in the run summary.

Detector config:

| Detector | `DETECTOR_MODEL` | Config |
|---|---|---|
| Default people detector | `PeopleNetTransformer` | `config_pgie.txt` |
| Warehouse multi-class detector | `RTDETR` | `config_pgie_rt_detr.txt` |
| PeopleNet v2.6.3 | `PeopleNet2.6.3` | `config_pgie_peoplenet.txt` |

Tracker config:

- Default MV3DT tracker: `config_tracker.yml`.
- Existing 2D tracker conversion: pass the user's tracker config template, for example `config_tracker_2d.yml`. The auto-configurator injects MV3DT sections.

## Display Or Headless Routing

Check display access before generating configs. The display path enables OSD and starts live BEV when map files are available. The OSD window does not save MP4 by default; if the user explicitly asks to save output in display mode, also enable file output. The headless path saves a DeepStream MP4, keeps Kafka metadata enabled, and starts offline BEV capture when map files are available.

```bash
if [ -n "${DISPLAY:-}" ] && python3 - <<'PY' >/dev/null 2>&1
import tkinter as tk
root = tk.Tk(); root.withdraw(); root.destroy()
PY
then
  MV3DT_RUN_MODE=display
else
  MV3DT_RUN_MODE=headless
fi
echo "MV3DT_RUN_MODE=${MV3DT_RUN_MODE}"
```

If `MV3DT_RUN_MODE=headless`, explain that OSD/BEV windows will not appear and use saved outputs by default. If `MV3DT_RUN_MODE=display` and the user asked to save video or output, set `MV3DT_SAVE_OUTPUTS=1` before generating configs so display and MP4 output are both enabled.

## Generate Configs

Load `references/generate-configs.md` and run its `Generate Configs` block after the dataset, detector, tracker, run mode, and output directory are selected. That shared reference owns the configurator command, venv `PATH` setup, detector config mapping, output-directory creation, and file-output/message-broker flags.

For a 4-camera dataset that should use the sample tuning override:

```bash
export CONFIG_OVERRIDES=override_tracker_4cam.yml
```

## BEV Visualizer And Capture

BEV requires `map.png` and `transforms.yml` in the dataset root. For display mode, start live BEV before launching DeepStream when those files exist:

```bash
python utils/kafka_bev_visualizer.py \
  --dataset-path="${DATASET_DIR}" \
  --msgconv-config="${EXPERIMENT_DIR}/config_msgconv.txt" \
  --average-multi-cam \
  --show-ids &
MV3DT_BEV_PID=$!
echo "Started current-run BEV visualizer PID: ${MV3DT_BEV_PID}"
```

For agent-managed display runs, stop `MV3DT_BEV_PID` after the DeepStream command exits. Users can still press `q` in the BEV window to close it early. Do not kill BEV visualizer processes that were not started for the current run.

For headless runs, use the offline `--from-end` capture flow in `visualization-metadata.md` by default. Start it immediately before the Docker launch attempt expected to run, after config/security/startup blockers are resolved, and wait until it has subscribed before launching DeepStream. If Docker must be retried after capture starts, stop that capture and start a fresh one for the retry. If `map.png` or `transforms.yml` is missing, report that BEV video cannot be produced and continue with the tiled DeepStream MP4.

## Docker Security Gate

Before presenting or executing the Docker launch command, state that this run uses `--privileged` and `--net=host`. These flags give the container broad host device, network, mounted-repo, and mounted-dataset access.

Require an explicit user confirmation such as: "I approve running DeepStream with `--privileged --net=host` for this MV3DT run." Do not execute the launch command on vague or unrelated approval. After approval, set `MV3DT_DOCKER_SECURITY_APPROVED=I_APPROVE_PRIVILEGED_HOST_NETWORK` for the launch shell.

Before using those flags, discuss least-privilege alternatives with the user: prefer `${GPU_FLAG}`/`--gpus all` without `--privileged` when it works, add only specific `--device` mounts required by observed failures, use narrow `--cap-add` values only when needed, and prefer explicit port mappings for Kafka/MQTT/display over `--net=host` when the deployment allows it. Use the command below only after the user accepts the compatibility/security tradeoff.

## Launch DeepStream

```bash
DOCKER_CMD="${DOCKER_CMD:-docker}"
if $DOCKER_CMD info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  GPU_FLAG="--runtime=nvidia"
elif $DOCKER_CMD run --help 2>/dev/null | grep -q -- "--gpus"; then
  GPU_FLAG="--gpus all"
else
  echo "ERROR: Docker GPU support was not detected. Fix NVIDIA Container Toolkit before running MV3DT."
  exit 1
fi

DOCKER_DISPLAY_ARGS=()
if [ "${MV3DT_RUN_MODE:-display}" != "headless" ]; then
  DOCKER_DISPLAY_ARGS=(-v /tmp/.X11-unix/:/tmp/.X11-unix -e DISPLAY="${DISPLAY}")
fi

case "${MV3DT_DOCKER_SECURITY_APPROVED:-}" in
  I_APPROVE_PRIVILEGED_HOST_NETWORK) ;;
  *)
    echo "ERROR: explicit approval required before running Docker with --privileged --net=host."
    echo "Set MV3DT_DOCKER_SECURITY_APPROVED=I_APPROVE_PRIVILEGED_HOST_NETWORK only after the user approves this run."
    exit 1
    ;;
esac

# If saved BEV is expected, start the offline BEV capture from
# visualization-metadata.md now, in the same launch shell or background task.
# Keep MV3DT_OFFLINE_BEV_PID, MV3DT_OFFLINE_BEV_LOG, and
# MV3DT_OFFLINE_BEV_STARTED_AT for retry cleanup and verification.
# Run only after the Docker Security Gate confirmation above. Use DOCKER_CMD="sudo docker" only if Docker needs sudo.
MV3DT_DOCKER_LAUNCH_AT="$(date +%s)"
MV3DT_DEEPSTREAM_STATUS=0
$DOCKER_CMD run -t --privileged --rm --net=host ${GPU_FLAG} \
  -v "${MODEL_REPO}:/workspace/models" \
  -v "${DATASET_DIR}:/workspace/inputs" \
  -v "${EXPERIMENT_DIR}:/workspace/experiments" \
  "${DOCKER_DISPLAY_ARGS[@]}" \
  -w /workspace/experiments \
  "${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}" \
  deepstream-test5-app -c config_deepstream.txt || MV3DT_DEEPSTREAM_STATUS=$?

if [ "${MV3DT_DEEPSTREAM_STATUS}" -ne 0 ] && [ -n "${MV3DT_OFFLINE_BEV_PID:-}" ] && kill -0 "${MV3DT_OFFLINE_BEV_PID}" 2>/dev/null; then
  echo "Stopping offline BEV capture for failed DeepStream attempt: ${MV3DT_OFFLINE_BEV_PID}"
  kill "${MV3DT_OFFLINE_BEV_PID}" 2>/dev/null || true
  wait "${MV3DT_OFFLINE_BEV_PID}" 2>/dev/null || true
fi
if [ -n "${MV3DT_BEV_PID:-}" ] && kill -0 "${MV3DT_BEV_PID}" 2>/dev/null; then
  echo "Stopping current-run BEV visualizer PID: ${MV3DT_BEV_PID}"
  kill "${MV3DT_BEV_PID}" 2>/dev/null || true
  wait "${MV3DT_BEV_PID}" 2>/dev/null || true
fi
if [ "${MV3DT_DEEPSTREAM_STATUS}" -ne 0 ]; then
  exit "${MV3DT_DEEPSTREAM_STATUS}"
fi
if [ -n "${MV3DT_OFFLINE_BEV_PID:-}" ]; then
  wait "${MV3DT_OFFLINE_BEV_PID}" 2>/dev/null || true
fi
```

When file output is enabled, saved DeepStream video should land at `${EXPERIMENT_DIR}/outVideos/tiled_display_raw.mp4`; BEV video should land at `${EXPERIMENT_DIR}/bev_outputs/trajectory_video_<timestamp>.mp4` when required BEV map files exist and offline capture receives complete frames. Do not report old artifacts as success; verify files were modified after `RUN_STARTED_AT` or clearly label them as pre-existing. If offline BEV was used, summarize `MV3DT_OFFLINE_BEV_LOG`, message/frame counts, and the capture-start to Docker-launch gap; do not report BEV success when the log shows zero messages, zero complete frames, or no saved video.

Generated dumps or videos may be root-owned because DeepStream runs inside the container as root. If the user approves a fix, keep it scoped to generated output directories only:

```bash
: "${EXPERIMENT_DIR:?set EXPERIMENT_DIR before ownership fixes}"
if [ "${EXPERIMENT_DIR}" = "/" ]; then
  echo "ERROR: EXPERIMENT_DIR must not be /"
  exit 1
fi

sudo chown -R "$(id -u):$(id -g)" \
  "${EXPERIMENT_DIR}/infer-kitti-dump" \
  "${EXPERIMENT_DIR}/tracker-kitti-dump" \
  "${EXPERIMENT_DIR}/outVideos" \
  "${EXPERIMENT_DIR}/bev_outputs"
```

## Success Criteria

- Generated files include `config_deepstream.txt`, `config_tracker.yml`, `config_msgconv.txt`, `config_mqtt.txt`, and `pub_sub_info_config_0.yml`.
- `deepstream-test5-app` starts and eventually prints `App run successful`; during long runs, provide progress every 30-60 seconds with elapsed time and the latest useful log line or artifact check.
- OSD shows the camera grid when display is enabled, or `${EXPERIMENT_DIR}/outVideos/tiled_display_raw.mp4` exists in headless mode.
- Kafka topic `mv3dt` receives tracking metadata when `--enable-msg-broker` is used. In headless mode, the BEV capture log reports nonzero parsed messages and frames from the current run when BEV map files are present.
- For `DETECTOR_MODEL=RTDETR`, `App run successful` alone is not enough; require nonzero non-empty inference and tracker dump counts, plus Kafka and BEV activity when those outputs are enabled. If RTDETR produces zero detections on a Blackwell-based GPU, rerun config generation so `strongly-typed=1` is applied to the RTDETR detector template and the stale RTDETR TensorRT engine cache is rebuilt.

## Troubleshooting

| Symptom | First check |
|---|---|
| `No MP4 files found` | Confirm videos are directly under `${DATASET_DIR}/videos` |
| Empty calibration list | Confirm files are under `${DATASET_DIR}/camInfo` and use `.yml` |
| Wrong camera pairing | Rename videos and calibration files to matching basenames |
| BEV visualizer fails | Confirm `map.png`, `transforms.yml`, and `config_msgconv.txt` exist |
| Docker app cannot show OSD | Confirm `DISPLAY` is set, then use a scoped X11 entry such as `xhost +local:docker` or `xhost +SI:localuser:root`; remove it after the run |
| Parser or model load error | Load `setup.md` and rebuild parsers |
