# Visualization And Metadata

Load this reference when the user wants to view MV3DT outputs, debug blank visualization windows, save BEV artifacts, disable visualization, or consume Kafka metadata.

## Output Surfaces

| Surface | Source | Tool |
|---|---|---|
| DeepStream OSD grid | Direct app output | `deepstream-test5-app` window |
| DeepStream tiled MP4 | DeepStream file sink, not the OSD window itself | `--enable-file-output`, writes `outVideos/tiled_display_raw.mp4` |
| BEV map | Kafka messages plus dataset map files | `utils/kafka_bev_visualizer.py` |
| Metadata stream | Kafka protobuf messages | `utils/kafka_client.py` |
| Saved images/video | BEV visualizer controls | `s` screenshot, `r` recording |

## Resolve Paths

```bash
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/datasets/mtmc_4cam}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/experiments/deepstream/4cam}"
cd "${REPO_ROOT}"
```

## OSD Behavior

The OSD window is named `DeepStreamTest5App`. It does not save video by default. To save the tiled DeepStream MP4 while keeping the window, regenerate configs with both `--enable-osd` and `--enable-file-output`.

- Left-click a camera tile to view one camera.
- Right-click to return to the multi-camera grid.
- Press `q` in the window to stop early.

If OSD is missing:

```bash
echo "${DISPLAY}"
ls /tmp/.X11-unix
```

Ask before changing host display access. Warn that disabling all X11 access control can let other clients capture the screen, inject input, or interfere with GUI apps. Do not use an allow-all X11 command for routine runs. Prefer a scoped local entry or XAUTHORITY cookie sharing, and remove the entry after the run:

```bash
xhost +local:docker || xhost +SI:localuser:root
# run the display or BEV workflow
xhost -local:docker 2>/dev/null || true
xhost -SI:localuser:root 2>/dev/null || true
```

## BEV Visualizer

Start before the DeepStream app when live BEV output is desired:

```bash
source "${REPO_ROOT}/mv3dt_venv/bin/activate"
python utils/kafka_bev_visualizer.py \
  --dataset-path="${DATASET_DIR}" \
  --msgconv-config="${EXPERIMENT_DIR}/config_msgconv.txt" \
  --average-multi-cam \
  --show-ids
```

Controls:

- `q`: quit
- `s`: save screenshot
- `c`: clear trajectories
- `r`: start or stop video recording

Remove `--average-multi-cam` to show per-camera trajectories instead of one fused point per object.

For agent-managed display runs, start live BEV in the background, capture its PID, and stop only that PID after the DeepStream command exits. Users can still press `q` to close the window early.

```bash
python utils/kafka_bev_visualizer.py \
  --dataset-path="${DATASET_DIR}" \
  --msgconv-config="${EXPERIMENT_DIR}/config_msgconv.txt" \
  --average-multi-cam \
  --show-ids &
MV3DT_BEV_PID=$!
echo "Started current-run BEV visualizer PID: ${MV3DT_BEV_PID}"

# After the DeepStream run command returns:
if [ -n "${MV3DT_BEV_PID:-}" ] && kill -0 "${MV3DT_BEV_PID}" 2>/dev/null; then
  kill "${MV3DT_BEV_PID}" 2>/dev/null || true
  wait "${MV3DT_BEV_PID}" 2>/dev/null || true
fi
```

Save BEV artifacts under a chosen output path during an interactive run by pressing `r` to record or `s` to screenshot:

```bash
python utils/kafka_bev_visualizer.py \
  --dataset-path="${DATASET_DIR}" \
  --msgconv-config="${EXPERIMENT_DIR}/config_msgconv.txt" \
  --output-path="${EXPERIMENT_DIR}/bev_outputs" \
  --average-multi-cam \
  --show-ids
```

## Offline BEV Video Capture

BEV is not embedded in the DeepStream tiled MP4. It is rendered from Kafka protobuf messages plus the dataset map files. On headless systems, or display systems where the user explicitly wants an unattended saved BEV MP4, start offline BEV capture immediately before the Docker launch attempt expected to run, after known startup blockers such as X11 access, Docker/GPU access, missing configs, and obvious model/cache issues are resolved. Do not keep one offline capture process alive across Docker retries; if a launch attempt fails after capture starts, stop that capture and start a fresh one before retrying.

Start this in the launch shell or a background task immediately before the DeepStream Docker command, and keep the PID, log path, and start time for cleanup and verification:

```bash
cd "${REPO_ROOT}"
mkdir -p "${EXPERIMENT_DIR}/bev_outputs"
source "${REPO_ROOT}/mv3dt_venv/bin/activate"
export PATH="${REPO_ROOT}/mv3dt_venv/bin:${PATH}"
export MV3DT_BEV_WIDTH="${MV3DT_BEV_WIDTH:-1280}"
export MV3DT_BEV_HEIGHT="${MV3DT_BEV_HEIGHT:-720}"
if ! find "${MODEL_REPO:-${REPO_ROOT}/models}" -name '*.engine' -print -quit 2>/dev/null | grep -q .; then
  export BEV_FIRST_MESSAGE_TIMEOUT="${BEV_FIRST_MESSAGE_TIMEOUT:-900}"
else
  export BEV_FIRST_MESSAGE_TIMEOUT="${BEV_FIRST_MESSAGE_TIMEOUT:-180}"
fi
MV3DT_OFFLINE_BEV_LOG="${MV3DT_OFFLINE_BEV_LOG:-${EXPERIMENT_DIR}/bev_outputs/offline_bev_$(date +%Y%m%d_%H%M%S).log}"
MV3DT_OFFLINE_BEV_STARTED_AT="$(date +%s)"
PYTHONUNBUFFERED=1 python utils/kafka_bev_visualizer.py \
  --offline \
  --from-end \
  --first-message-timeout="${BEV_FIRST_MESSAGE_TIMEOUT}" \
  --idle-timeout="${BEV_IDLE_TIMEOUT:-12}" \
  --max-timeout="${BEV_MAX_TIMEOUT:-600}" \
  --dataset-path="${DATASET_DIR}" \
  --msgconv-config="${EXPERIMENT_DIR}/config_msgconv.txt" \
  --output-path="${EXPERIMENT_DIR}/bev_outputs" \
  --average-multi-cam \
  --show-ids \
  --verbose >"${MV3DT_OFFLINE_BEV_LOG}" 2>&1 &
MV3DT_OFFLINE_BEV_PID=$!
echo "Started offline BEV capture PID: ${MV3DT_OFFLINE_BEV_PID}"
echo "Offline BEV log: ${MV3DT_OFFLINE_BEV_LOG}"
```

Wait until the log shows that capture connected to Kafka and sought to current end offsets before launching DeepStream. Keep `PYTHONUNBUFFERED=1` so readiness lines are written promptly. `--from-end` starts at the current Kafka topic tail and waits for the next run's messages. `--first-message-timeout` must be long enough for DeepStream startup and TensorRT engine creation after the actual Docker launch; the snippet uses 900 seconds when no `.engine` cache is detected. `--idle-timeout` controls when capture ends after the stream stops. `--verbose` prints the complete-frame count used to verify the BEV MP4. The offline renderer falls back to a fixed render size if Tk cannot read a screen; set `MV3DT_BEV_WIDTH` and `MV3DT_BEV_HEIGHT` to override it.

After launching Docker, record `MV3DT_DOCKER_LAUNCH_AT="$(date +%s)"`. If the Docker attempt fails, stop `MV3DT_OFFLINE_BEV_PID` before retrying and start a new offline capture for the next attempt. On success, wait for `MV3DT_OFFLINE_BEV_PID` to exit so the MP4 writer can finish.

Expected BEV output path: `${EXPERIMENT_DIR}/bev_outputs/trajectory_video_<timestamp>.mp4`. Headless MV3DT runs should start this capture by default immediately before DeepStream. Success output should include `Collected <N> messages`, `Complete frames ready for rendering: <M>`, and `Video saved: ...` with nonzero `N` and `M`. If the log reports zero messages, zero complete frames, or no saved video, report that no BEV video was generated rather than treating the run as successful. Also report the gap between `MV3DT_OFFLINE_BEV_STARTED_AT` and `MV3DT_DOCKER_LAUNCH_AT`; if it is large relative to `BEV_FIRST_MESSAGE_TIMEOUT`, flag a likely capture/Docker resync issue.

Use this post-run check when offline BEV capture was enabled:

```bash
if [ -n "${MV3DT_OFFLINE_BEV_LOG:-}" ] && [ -f "${MV3DT_OFFLINE_BEV_LOG}" ]; then
  grep -E 'Collected [0-9]+ messages|Complete frames ready for rendering|Video saved' "${MV3DT_OFFLINE_BEV_LOG}" || true
  if grep -Eq 'Collected 0 messages|Complete frames ready for rendering: 0' "${MV3DT_OFFLINE_BEV_LOG}"; then
    echo "WARNING: offline BEV capture produced no usable current-run video. Restart capture immediately before retrying Docker."
  fi
fi
if [ -n "${MV3DT_OFFLINE_BEV_STARTED_AT:-}" ] && [ -n "${MV3DT_DOCKER_LAUNCH_AT:-}" ]; then
  echo "Offline BEV capture started $((MV3DT_DOCKER_LAUNCH_AT - MV3DT_OFFLINE_BEV_STARTED_AT))s before Docker launch"
fi
```

## Headless Success Summary

For a headless run, report these artifacts and counts when available:

```text
Generated configs: config_deepstream.txt, config_tracker.yml, config_msgconv.txt
DeepStream MP4: ${EXPERIMENT_DIR}/outVideos/tiled_display_raw.mp4
BEV MP4: ${EXPERIMENT_DIR}/bev_outputs/trajectory_video_<timestamp>.mp4, if offline BEV succeeded
Kafka topic: mv3dt
Kafka messages: <Collected N messages from BEV capture or bounded kafka_client.py output>
Latest frame ID: <from decoded JSON, when sampled>
Sensors seen: <sensor IDs from decoded JSON, when sampled>
Objects in latest frame: <objects length from decoded JSON, when sampled>
```

## Kafka Metadata

DeepStream may also print MQTT communicator status for internal tracker communication. That is separate from BEV metadata. Verify Kafka topic `mv3dt` directly when debugging BEV.

Default broker and topic:

```bash
source "${REPO_ROOT}/mv3dt_venv/bin/activate"
timeout "${KAFKA_CLIENT_TIMEOUT:-15s}" python utils/kafka_client.py
```

Explicit broker and topic:

```bash
timeout "${KAFKA_CLIENT_TIMEOUT:-15s}" python utils/kafka_client.py --broker localhost:9092 --topic mv3dt
```

`kafka_client.py` streams until interrupted, so use a bounded `timeout` for agent-run inspections. A timeout exit after messages are printed is not a failure by itself; use the offset check below to verify current-run activity without depending on an unbounded consumer.

Confirm topic availability:

```bash
KAFKA_DIR="${BASE_DIR:-$HOME}/kafka_2.13-4.2.0"
"${KAFKA_DIR}/bin/kafka-topics.sh" --bootstrap-server localhost:9092 --list | grep '^mv3dt$'
```

Confirm offsets moved during a run without depending on Kafka CLI tools. Capture offsets before and after DeepStream to avoid mistaking retained old messages for current-run activity:

```bash
source "${REPO_ROOT}/mv3dt_venv/bin/activate"
PYTHONPATH=utils python - <<'PYCODE'
from kafka import KafkaConsumer, TopicPartition
c = KafkaConsumer(bootstrap_servers='localhost:9092', consumer_timeout_ms=1000)
tp = TopicPartition('mv3dt', 0)
c.assign([tp])
print('beginning:', c.beginning_offsets([tp])[tp])
print('end:', c.end_offsets([tp])[tp])
c.close()
PYCODE
```

## Disable Visualization

For DeepStream Container runs, omit `--enable-osd` from the `utils/deepstream_auto_configurator.py` command and regenerate configs.

For BEV, do not launch `utils/kafka_bev_visualizer.py`.

Keep `--enable-msg-broker` if downstream metadata consumers need Kafka messages.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| BEV window opens but remains blank | No Kafka messages yet, wrong topic, or app not running | Start the DeepStream app and inspect `python utils/kafka_client.py` |
| Headless BEV MP4 has zero messages | Capture started after DeepStream, started too early before a delayed/retried Docker launch, or Kafka offsets did not advance | Start a fresh offline capture immediately before the Docker attempt with `--from-end`, increase `--first-message-timeout`, and verify Kafka offsets advance |
| Offline BEV video fails on SSH | Tk could not read display size or video writer failed | Set `MV3DT_BEV_WIDTH`/`MV3DT_BEV_HEIGHT`, verify the output path is writable, and check the visualizer error output |
| BEV map missing or distorted | Missing or mismatched `map.png` or `transforms.yml` | Verify both files are in the dataset root and match calibration |
| `kafka_client.py` shows no messages | Configs were generated without message broker sink | Regenerate with `--enable-msg-broker` |
| OSD window does not appear | No X11 display access | Check `DISPLAY`, `/tmp/.X11-unix`, and display access |
| Object IDs not visible in grid | DeepStream OSD behavior | Left-click a camera tile for single-camera view |
