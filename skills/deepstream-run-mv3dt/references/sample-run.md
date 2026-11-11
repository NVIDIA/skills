# Sample Runs

Load this reference when the user wants to run the shipped 4-camera or 12-camera MV3DT sample datasets through the DeepStream Container pipeline.

## Prerequisite Gate

If `datasets/mtmc_4cam`, `datasets/mtmc_12cam`, or `mv3dt_venv` is missing, follow `setup.md` first.

## Choose Sample And Detector

| Choice | Default | Command |
|---|---|---|
| 4-camera sample | Yes | `./scripts/test_4cam_ds.sh` |
| 12-camera sample | No | `./scripts/test_12cam_ds.sh` |

Before launching, state the selected sample, detector, run mode, and output directory. If the user did not choose a sample, run the default 4-camera sample with `PeopleNetTransformer` without stopping to ask. In the final report, state that the repo also includes a 12-camera sample and ask whether to run it as a follow-up.

Detector values:

| User intent | `DETECTOR_MODEL` |
|---|---|
| default people detector | `PeopleNetTransformer` |
| warehouse multi-class detector | `RTDETR` |
| PeopleNet v2.6.3 | `PeopleNet2.6.3` |

Set `DETECTOR_MODEL` before running repo scripts when the user requests a non-default detector. Keep `PeopleNetTransformer` as the default.

## Display Or Headless Routing

The repo quick-start scripts are display-oriented: they enable OSD, start the BEV visualizer, and mount X11 into the container. Check display access before selecting the run path.

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

For `display`, ask before changing host display access, then use the config-generation and Docker launch below with `MV3DT_RUN_MODE=display`. The shipped quick-start scripts use broad container permissions; do not invoke them from this skill.

Before changing X11 access, warn that disabling all X11 access control can let other clients capture the screen, inject input, or interfere with GUI apps. Do not use an allow-all X11 command for routine runs. Prefer a scoped local entry and remove it after the run:

```bash
xhost +local:docker || xhost +SI:localuser:root
# run the display workflow
xhost -local:docker 2>/dev/null || true
xhost -SI:localuser:root 2>/dev/null || true
```

For `headless`, explain that OSD/BEV windows will not appear and use saved outputs by default. Generate configs with `--enable-file-output` and `--enable-msg-broker`. The DeepStream MP4 is the tiled camera output only; BEV is a separate Kafka-derived artifact. Start the headless BEV capture from `visualization-metadata.md` immediately before the Docker launch attempt expected to run, after config/security/startup blockers are resolved. If Docker must be retried after capture starts, stop that capture and start a fresh one for the retry.

## Launch Inputs (Display And Headless)

Set these inputs for either mode, then read [Generate Configs](generate-configs.md) and run its command block. Select the dataset once; both modes use the Docker launch below.

```bash
cd "${REPO_ROOT}"
export SAMPLE_CAMERAS="${SAMPLE_CAMERAS:-4}"
export DETECTOR_MODEL="${DETECTOR_MODEL:-PeopleNetTransformer}"
export MODEL_REPO="${REPO_ROOT}/models"
export RUN_STARTED_AT="$(date +%s)"
export MV3DT_RUN_MODE="${MV3DT_RUN_MODE:-headless}"
case "$SAMPLE_CAMERAS" in
  4)
    export DATASET_DIR="${REPO_ROOT}/datasets/mtmc_4cam"
    export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/experiments/deepstream/4cam}"
    export TRACKER_CONFIG=config_tracker.yml
    export CONFIG_OVERRIDES=override_tracker_4cam.yml
    ;;
  12)
    export DATASET_DIR="${REPO_ROOT}/datasets/mtmc_12cam"
    export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/experiments/deepstream/12cam}"
    unset CONFIG_OVERRIDES
    case "$DETECTOR_MODEL" in
      PeopleNetTransformer) export TRACKER_CONFIG=config_tracker_tuned_12cam.yml ;;
      RTDETR) export TRACKER_CONFIG=config_tracker_tuned_12cam_rt_detr.yml ;;
      PeopleNet2.6.3) export TRACKER_CONFIG=config_tracker.yml ;;
      *) echo "ERROR: unsupported detector"; exit 1 ;;
    esac
    ;;
  *) echo "ERROR: SAMPLE_CAMERAS must be 4 or 12"; exit 1 ;;
esac
```

Display mode enables OSD. Set `MV3DT_SAVE_OUTPUTS=1` when the user also requests tiled MP4 output. Start the live BEV visualizer from [Visualization and Metadata](visualization-metadata.md) in display mode; keep only that process's PID for cleanup. In headless mode start the offline BEV capture immediately before Docker instead.

## Docker Access

Use NVIDIA GPU access on Docker bridge networking. Before generating configs, follow [Container Networking](container-network.md) to verify an advertised Kafka listener reachable from the bridge. Retain the host localhost listener for BEV tools. The launch exposes only the dataset, model cache, run output directory, and (in display mode) the X11 socket. Do not add capabilities or arbitrary host devices to bypass an error.

After the shared config block completes, launch DeepStream:

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

: "${MV3DT_KAFKA_HOST:?prepare the bridge-reachable Kafka listener}"
: "${MV3DT_KAFKA_PORT:?prepare the bridge-reachable Kafka listener}"

DRIVER_CAPABILITIES=compute,utility,video
DOCKER_DISPLAY_ARGS=()
if [ "${MV3DT_RUN_MODE:-headless}" != "headless" ]; then
  DRIVER_CAPABILITIES+=,graphics,display
  DOCKER_DISPLAY_ARGS=(-v /tmp/.X11-unix/:/tmp/.X11-unix -e DISPLAY="${DISPLAY}")
fi

# If saved BEV is expected, start the offline BEV capture from
# visualization-metadata.md now, in the same launch shell or background task.
# Keep MV3DT_OFFLINE_BEV_PID, MV3DT_OFFLINE_BEV_LOG, and
# MV3DT_OFFLINE_BEV_STARTED_AT for retry cleanup and verification.
# Run after the broker endpoint checks above. Use DOCKER_CMD="sudo docker" only if Docker needs sudo.
MODEL_REPO="$(realpath -e "${MODEL_REPO:?set MODEL_REPO}")" || exit 1
DATASET_DIR="$(realpath -e "${DATASET_DIR:?set DATASET_DIR}")" || exit 1
EXPERIMENT_DIR="$(realpath -e "${EXPERIMENT_DIR:?set EXPERIMENT_DIR}")" || exit 1
for mount_dir in "$MODEL_REPO" "$DATASET_DIR" "$EXPERIMENT_DIR"; do
  test -d "$mount_dir" && test "$mount_dir" != / || { echo "ERROR: invalid mount directory"; exit 1; }
done
MV3DT_DOCKER_LAUNCH_AT="$(date +%s)"
MV3DT_DEEPSTREAM_STATUS=0
$DOCKER_CMD run -t --rm --network bridge ${GPU_FLAG} \
  -e NVIDIA_DRIVER_CAPABILITIES="${DRIVER_CAPABILITIES}" \
  -v "${MODEL_REPO}:/workspace/models" \
  -v "${DATASET_DIR}:/workspace/inputs:ro" \
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

Saved DeepStream video should land at `${EXPERIMENT_DIR}/outVideos/tiled_display_raw.mp4` whenever `--enable-file-output` is used, including display runs where the user explicitly asked to save output. BEV video should land at `${EXPERIMENT_DIR}/bev_outputs/trajectory_video_<timestamp>.mp4` when the offline capture process receives complete frames. Kafka metadata is available on topic `mv3dt`. If offline BEV was used, summarize `MV3DT_OFFLINE_BEV_LOG`, message/frame counts, and the capture-start to Docker-launch gap; do not report BEV success when the log shows zero messages, zero complete frames, or no saved video.

Do not report stale files from a previous run as success. Track `RUN_STARTED_AT` before launch and verify that reported MP4 artifacts are newer than the run start, or clearly label them as pre-existing.

DeepStream runs inside the container as root on many hosts, so generated dumps or videos may be root-owned. Report this if it happens and ask before changing ownership or deleting generated artifacts. If the user approves a fix, keep it scoped to generated output directories only:

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

When verifying artifacts, avoid listing the full per-frame dump directories. Report file sizes, modification times, and counts instead:

```bash
find "${EXPERIMENT_DIR}/outVideos" -maxdepth 1 -type f -newermt "@${RUN_STARTED_AT:-0}" -printf '%p %s bytes %TY-%Tm-%Td %TH:%TM:%TS\n'
printf 'non-empty infer dump files: '
find "${EXPERIMENT_DIR}/infer-kitti-dump" -type f -size +0c | wc -l
printf 'non-empty tracker dump files: '
find "${EXPERIMENT_DIR}/tracker-kitti-dump" -type f -size +0c | wc -l
```

For the 12-camera sample, use `DATASET_DIR="${REPO_ROOT}/datasets/mtmc_12cam"`, `EXPERIMENT_DIR="${REPO_ROOT}/experiments/deepstream/12cam"`, and unset `CONFIG_OVERRIDES` unless intentionally applying a custom override. Mirror the DeepStream repo `scripts/test_12cam_ds.sh` detector-specific tracker defaults: `PeopleNetTransformer` uses `TRACKER_CONFIG=config_tracker_tuned_12cam.yml`; `RTDETR` uses `TRACKER_CONFIG=config_tracker_tuned_12cam_rt_detr.yml`; `PeopleNet2.6.3` uses `TRACKER_CONFIG=config_tracker.yml`. Do not run the 12-camera sample automatically after the default 4-camera run; offer it as the next run in the final report.

## Display Expected Output

Two windows should appear:

- `Bird-Eye View of Multi-View 3D Tracking`: BEV visualization from Kafka messages. It may be blank until tracking messages arrive. Press `q` in the window to close it.
- `DeepStreamTest5App`: DeepStream OSD grid. Left-click a camera tile for single-camera view; right-click to return to the grid. Press `q` to stop early.

First run can spend about 15 minutes building TensorRT engines. Warnings such as "Load engine failed. Create engine again", "INT8 calibration file not specified. Trying FP16 mode", and GStreamer warnings are expected during startup. During long runs, provide a short progress update every 30-60 seconds with elapsed time and the latest meaningful log line or artifact check.

## Success Criteria

- The app eventually prints `App run successful`.
- OSD shows a camera grid when display is enabled.
- BEV shows live tracks after messages arrive.
- Kafka topic `mv3dt` receives protobuf frame metadata; MQTT communicator logs alone are not proof that BEV Kafka messages were written.

Headless success indicators:

- `deepstream-test5-app` starts and eventually prints `App run successful`.
- `${EXPERIMENT_DIR}/outVideos/tiled_display_raw.mp4` exists when `--enable-file-output` is used.
- `timeout "${KAFKA_CLIENT_TIMEOUT:-15s}" python utils/kafka_client.py --broker localhost:9092 --topic mv3dt` prints decoded JSON messages, or Kafka offsets advance during the run.
- `${EXPERIMENT_DIR}/bev_outputs/trajectory_video_<timestamp>.mp4` exists and the BEV capture log reports nonzero parsed messages and frames from the current run.

For `DETECTOR_MODEL=RTDETR`, `App run successful` alone is not enough. Require nonzero non-empty inference and tracker dump counts, and require Kafka offsets/messages plus BEV messages/frames when message broker and BEV capture are expected. If RTDETR produces zero detections on a Blackwell-based GPU, rerun the shared config-generation step so `strongly-typed=1` is applied to the RTDETR detector template and the stale RTDETR TensorRT engine cache is rebuilt.

Use `visualization-metadata.md` to inspect Kafka, run headless BEV capture, or debug blank BEV/OSD output. If the sample was not specified and the default 4-camera sample was run, close by saying the shipped 12-camera sample is also available and can be run next.
