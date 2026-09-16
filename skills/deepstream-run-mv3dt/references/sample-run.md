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

For `display`, ask before changing host display access, then use the quick-start scripts. These scripts show OSD/BEV windows but do not save the tiled DeepStream MP4 because they do not enable the file-output sink.

Before changing X11 access, warn that disabling all X11 access control can let other clients capture the screen, inject input, or interfere with GUI apps. Do not use an allow-all X11 command for routine runs. Prefer a scoped local entry and remove it after the run:

```bash
xhost +local:docker || xhost +SI:localuser:root
# run the display workflow
xhost -local:docker 2>/dev/null || true
xhost -SI:localuser:root 2>/dev/null || true
```

For `headless`, explain that OSD/BEV windows will not appear and use saved outputs by default. Generate configs with `--enable-file-output` and `--enable-msg-broker`. The DeepStream MP4 is the tiled camera output only; BEV is a separate Kafka-derived artifact. Start the headless BEV capture from `visualization-metadata.md` immediately before the Docker launch attempt expected to run, after config/security/startup blockers are resolved. If Docker must be retried after capture starts, stop that capture and start a fresh one for the retry.

## Display Launch Commands

For agent-managed display runs, wrap the repo quick-start script so BEV visualizer processes created by that script are stopped after DeepStream exits. This keeps user-started BEV windows untouched while cleaning up current-run windows.

```bash
cd "${REPO_ROOT}"
# Defaults to the 4-camera sample. Set SAMPLE_SCRIPT=./scripts/test_12cam_ds.sh for the 12-camera sample.
SAMPLE_SCRIPT="${SAMPLE_SCRIPT:-./scripts/test_4cam_ds.sh}"
BEV_PIDS_BEFORE="$(pgrep -f 'kafka_bev_visualizer.py' || true)"
MV3DT_DEEPSTREAM_STATUS=0
"${SAMPLE_SCRIPT}" || MV3DT_DEEPSTREAM_STATUS=$?
BEV_PIDS_AFTER="$(pgrep -f 'kafka_bev_visualizer.py' || true)"
comm -13 \
  <(printf '%s\n' "${BEV_PIDS_BEFORE}" | sed '/^$/d' | sort -u) \
  <(printf '%s\n' "${BEV_PIDS_AFTER}" | sed '/^$/d' | sort -u) \
  | while IFS= read -r pid; do
      if kill -0 "${pid}" 2>/dev/null; then
        echo "Stopping current-run BEV visualizer PID: ${pid}"
        kill "${pid}" 2>/dev/null || true
      fi
    done
if [ "${MV3DT_DEEPSTREAM_STATUS}" -ne 0 ]; then
  exit "${MV3DT_DEEPSTREAM_STATUS}"
fi
```

Set these before the wrapper when needed:

```bash
export SAMPLE_SCRIPT=./scripts/test_12cam_ds.sh
export DETECTOR_MODEL=RTDETR
```

If the user has display access and explicitly asks to save video or output, do not use the quick-start script unchanged. Generate configs with `--enable-osd`, `--enable-file-output`, and `--enable-msg-broker`, then launch Docker with the normal X11 mounts. The OSD window remains live, and the tiled DeepStream MP4 is saved to `${EXPERIMENT_DIR}/outVideos/tiled_display_raw.mp4`. For an unattended saved BEV MP4, use the offline BEV capture command from `visualization-metadata.md` immediately before the Docker launch attempt; interactive BEV recording also works by pressing `r` in the BEV window.

## Headless Launch Commands

Use the same sample dataset choices, but regenerate configs explicitly instead of using the quick-start scripts so file output and Kafka broker output are enabled. Set run-specific inputs first, then load `references/generate-configs.md` and run its `Generate Configs` block.

Example inputs for the 4-camera sample with the default detector:

```bash
cd "${REPO_ROOT}"
export DATASET_DIR="${REPO_ROOT}/datasets/mtmc_4cam"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/experiments/deepstream/4cam}"
export RUN_STARTED_AT="$(date +%s)"
export MODEL_REPO="${REPO_ROOT}/models"
export MV3DT_RUN_MODE=headless
export DETECTOR_MODEL="${DETECTOR_MODEL:-PeopleNetTransformer}"
export CONFIG_OVERRIDES=override_tracker_4cam.yml
```

## Docker Security Gate

Before presenting or executing the Docker launch command, state that this run uses `--privileged` and `--net=host`. These flags give the container broad host kernel capability, device, network, mounted-repo, and mounted-dataset access.

Require an explicit user confirmation such as: "I approve running DeepStream with `--privileged --net=host` for this MV3DT sample run." Do not execute the launch command on vague or unrelated approval. After approval, set `MV3DT_DOCKER_SECURITY_APPROVED=I_APPROVE_PRIVILEGED_HOST_NETWORK` for the launch shell.

Before using those flags, discuss least-privilege alternatives with the user: prefer `${GPU_FLAG}`/`--gpus all` without `--privileged` when it works, add only specific `--device` mounts required by observed failures, use narrow `--cap-add` values only when needed, and prefer explicit port mappings for Kafka/MQTT/display over `--net=host` when the deployment allows it. Use the command below only after the user accepts the compatibility/security tradeoff.

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
  -w /workspace/experiments \
  "${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}" \
  deepstream-test5-app -c config_deepstream.txt || MV3DT_DEEPSTREAM_STATUS=$?

if [ "${MV3DT_DEEPSTREAM_STATUS}" -ne 0 ] && [ -n "${MV3DT_OFFLINE_BEV_PID:-}" ] && kill -0 "${MV3DT_OFFLINE_BEV_PID}" 2>/dev/null; then
  echo "Stopping offline BEV capture for failed DeepStream attempt: ${MV3DT_OFFLINE_BEV_PID}"
  kill "${MV3DT_OFFLINE_BEV_PID}" 2>/dev/null || true
  wait "${MV3DT_OFFLINE_BEV_PID}" 2>/dev/null || true
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
