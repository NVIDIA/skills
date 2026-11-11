# Generate DeepStream Configs

Load this reference only when a sample or custom dataset workflow needs to regenerate MV3DT DeepStream configs. This is the canonical config-generation flow shared by `sample-run.md` and `custom-dataset.md`; do not duplicate its shell logic in those references.

## Inputs

Set these before running the command block:

- `REPO_ROOT`: MV3DT app directory.
- `DATASET_DIR`: dataset root containing `videos/` and `camInfo/`.
- `EXPERIMENT_DIR`: output directory for generated configs and artifacts.
- `MODEL_REPO`: model directory, usually `${REPO_ROOT}/models`.
- `RUN_STARTED_AT`: epoch seconds for fresh-artifact checks.
- `MV3DT_RUN_MODE`: `display` or `headless`.
- `MV3DT_SAVE_OUTPUTS`: set to `1` when display mode should also save tiled MP4 output.
- `DETECTOR_MODEL`: optional; defaults to `PeopleNetTransformer`. Supported values: `PeopleNetTransformer`, `RTDETR`, `PeopleNet2.6.3`.
- `TRACKER_CONFIG`: optional; defaults to `config_tracker.yml`. For the DeepStream 12-camera sample, callers should set `config_tracker_tuned_12cam.yml` with `PeopleNetTransformer`, `config_tracker_tuned_12cam_rt_detr.yml` with `RTDETR`, and `config_tracker.yml` with `PeopleNet2.6.3`.
- `CONFIG_OVERRIDES`: optional; set to a config override filename such as `override_tracker_4cam.yml` when needed.
- `MV3DT_KAFKA_HOST`, `MV3DT_KAFKA_PORT`: verified bridge-reachable broker endpoint from [Container Networking](container-network.md).
- `MV3DT_KAFKA_CONFIG_FILE`: protected Kafka adapter config when the broker requires TLS/authentication; copied locally with mode `0600` and never printed.
- `MV3DT_MQTT_HOST`, `MV3DT_MQTT_PORT`, `MV3DT_MQTT_CONFIG_FILE`: verified peer broker and protected adapter config from the same networking reference.
- `MV3DT_FORCE_RTDETR_STRONGLY_TYPED`: optional; set to `1` to force the RTDETR strongly typed guard when GPU detection is unavailable or incomplete.

## Generate Configs

```bash
mkdir -p "${EXPERIMENT_DIR}/infer-kitti-dump" \
         "${EXPERIMENT_DIR}/tracker-kitti-dump" \
         "${EXPERIMENT_DIR}/outVideos" \
         "${EXPERIMENT_DIR}/bev_outputs"

source "${REPO_ROOT}/mv3dt_venv/bin/activate"
export PATH="${REPO_ROOT}/mv3dt_venv/bin:${PATH}"
export DETECTOR_MODEL="${DETECTOR_MODEL:-PeopleNetTransformer}"
case "${DETECTOR_MODEL}" in
  PeopleNetTransformer)
    DETECTOR_CONFIG="${DETECTOR_CONFIG:-config_pgie.txt}"
    ;;
  RTDETR)
    DETECTOR_CONFIG="${DETECTOR_CONFIG:-config_pgie_rt_detr.txt}"
    ;;
  PeopleNet2.6.3)
    DETECTOR_CONFIG="${DETECTOR_CONFIG:-config_pgie_peoplenet.txt}"
    ;;
  *)
    echo "ERROR: unsupported DETECTOR_MODEL=${DETECTOR_MODEL}. Use PeopleNetTransformer, RTDETR, or PeopleNet2.6.3."
    exit 1
    ;;
esac
TRACKER_CONFIG="${TRACKER_CONFIG:-config_tracker.yml}"
echo "Using detector model: ${DETECTOR_MODEL} (detector=${DETECTOR_CONFIG}, tracker=${TRACKER_CONFIG})"

CONFIG_FLAGS=(--enable-msg-broker)
if [ "${MV3DT_RUN_MODE:-display}" = "headless" ]; then
  CONFIG_FLAGS+=(--enable-file-output)
else
  CONFIG_FLAGS+=(--enable-osd)
  if [ "${MV3DT_SAVE_OUTPUTS:-0}" = "1" ]; then
    CONFIG_FLAGS+=(--enable-file-output)
  fi
fi

CONFIG_OVERRIDE_ARGS=()
if [ -n "${CONFIG_OVERRIDES:-}" ]; then
  CONFIG_OVERRIDE_ARGS+=(--config-overrides="${CONFIG_OVERRIDES}")
fi

python "${REPO_ROOT}/utils/deepstream_auto_configurator.py" \
  --dataset-dir="${DATASET_DIR}" \
  "${CONFIG_FLAGS[@]}" \
  --detector-config="${DETECTOR_CONFIG}" \
  --tracker-config="${TRACKER_CONFIG}" \
  "${CONFIG_OVERRIDE_ARGS[@]}" \
  --output-dir="${EXPERIMENT_DIR}"

# Change only generated run files. Source templates and shared engine caches stay intact.
export MV3DT_CONFIG_OUTPUT_DIR="${EXPERIMENT_DIR}"
export MV3DT_ENABLE_STRONGLY_TYPED=0
if [ "${DETECTOR_MODEL}" = "RTDETR" ]; then
  GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  BLACKWELL_GPU_PATTERN="${MV3DT_BLACKWELL_GPU_PATTERN:-Blackwell|RTX PRO (2000|4000|4500|5000|6000)|GB10|GB200|B200|B300|DGX Spark}"
  if [ "${MV3DT_FORCE_RTDETR_STRONGLY_TYPED:-0}" = "1" ] || printf '%s\n' "$GPU_NAMES" | grep -Eiq "$BLACKWELL_GPU_PATTERN"; then
    export MV3DT_ENABLE_STRONGLY_TYPED=1
  fi
fi
python3 - <<'PYCONFIG'
import os
from pathlib import Path
import re
import yaml

output = Path(os.environ['MV3DT_CONFIG_OUTPUT_DIR']).resolve(strict=True)
main_path = output / 'config_deepstream.txt'
detector_path = output / 'config_pgie.txt'
peer_paths = sorted(output.glob('pub_sub_info_config_*.yml'))
if not peer_paths:
    raise SystemExit('No generated production MQTT peer configs were found')
mqtt_path = output / 'config_mqtt.txt'
host = os.environ.get('MV3DT_KAFKA_HOST', '')
def tcp_port(name):
    try:
        value = int(os.environ.get(name, '0'))
    except ValueError:
        raise SystemExit(f'Set {name} to a valid TCP port') from None
    if not 1 <= value <= 65535:
        raise SystemExit(f'Set {name} to a valid TCP port')
    return value
port = tcp_port('MV3DT_KAFKA_PORT')
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', host) or host.lower() == 'localhost' or host.startswith('127.'):
    raise SystemExit('Set a verified bridge-reachable MV3DT_KAFKA_HOST before config generation')
mqtt_host = os.environ.get('MV3DT_MQTT_HOST', '')
mqtt_port = tcp_port('MV3DT_MQTT_PORT')
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', mqtt_host) or mqtt_host.lower() == 'localhost' or mqtt_host.startswith('127.'):
    raise SystemExit('Set a bridge-reachable MV3DT_MQTT_HOST')
if not os.environ.get('MV3DT_MQTT_CONFIG_FILE'):
    raise SystemExit('Set MV3DT_MQTT_CONFIG_FILE to the protected adapter config')
credential_source = Path(os.environ['MV3DT_MQTT_CONFIG_FILE']).expanduser()
if not credential_source.is_file() or credential_source.stat().st_mode & 0o077:
    raise SystemExit('MQTT adapter config must be an owner-only readable file')
mqtt_config = credential_source.read_text()
if '[message-broker]' not in mqtt_config:
    raise SystemExit('MQTT adapter config is missing its message-broker section')
for path in (main_path, detector_path, mqtt_path, *peer_paths):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f'Refusing missing/symlink generated config: {path}')

def set_property(text, section, key, value):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == f'[{section}]'), None)
    if start is None:
        raise SystemExit(f'Missing generated section {section}')
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith('[')), len(lines))
    key_pattern = re.compile(r'^\s*' + re.escape(key) + r'\s*=')
    hits = [i for i in range(start + 1, end) if key_pattern.match(lines[i])]
    if len(hits) > 1:
        raise SystemExit(f'Ambiguous generated property {key}')
    if hits:
        lines[hits[0]] = f'{key}={value}'
    else:
        lines.insert(start + 1, f'{key}={value}')
    return '\n'.join(lines) + '\n'

main = set_property(main_path.read_text(), 'sink3', 'msg-broker-conn-str', f'{host};{port};mv3dt')
kafka_config = None
kafka_path = output / 'config_kafka.txt'
if os.environ.get('MV3DT_KAFKA_CONFIG_FILE'):
    source = Path(os.environ['MV3DT_KAFKA_CONFIG_FILE']).expanduser()
    if not source.is_file() or source.stat().st_mode & 0o077 or kafka_path.is_symlink():
        raise SystemExit('Kafka adapter config must be an owner-only file and its generated target must not be a symlink')
    kafka_config = source.read_text()
    main = set_property(main, 'sink3', 'msg-broker-config', '/workspace/experiments/config_kafka.txt')
detector = detector_path.read_text()
def mqtt_address(value):
    if not isinstance(value, str) or ';' not in value:
        raise SystemExit('Unexpected generated MQTT peer address')
    return f'{mqtt_host}:{mqtt_port};{value.split(";", 1)[1]}'
peers = {}
for peer_path in peer_paths:
    peer = yaml.safe_load(peer_path.read_text())
    if not isinstance(peer, dict) or not peer.get('pubBrokerTopicStr') or 'subPeerBrokerTopicStrs' not in peer:
        raise SystemExit('Expected production MQTT peer config; do not use the debug communicator here')
    if not isinstance(peer['pubBrokerTopicStr'], list) or not isinstance(peer['subPeerBrokerTopicStrs'], list):
        raise SystemExit('Expected lists of generated MQTT peer topics')
    if any(not isinstance(group, list) for group in peer['subPeerBrokerTopicStrs']):
        raise SystemExit('Expected nested lists of generated MQTT subscriptions')
    peer['pubBrokerTopicStr'] = [mqtt_address(value) for value in peer['pubBrokerTopicStr']]
    peer['subPeerBrokerTopicStrs'] = [[mqtt_address(value) for value in group] for group in peer['subPeerBrokerTopicStrs']]
    peers[peer_path] = yaml.safe_dump(peer, sort_keys=False)
if os.environ['MV3DT_ENABLE_STRONGLY_TYPED'] == '1':
    detector = set_property(detector, 'property', 'strongly-typed', '1')
    engine = re.search(r'^\s*model-engine-file\s*=\s*(.+?)\s*$', main, re.M)
    if engine is None:
        raise SystemExit('Missing generated engine path')
    cache_name = Path(engine.group(1)).stem + '-strongly-typed.engine'
    if (output / 'engines').is_symlink():
        raise SystemExit('Refusing a symlink engine cache directory')
    (output / 'engines').mkdir(exist_ok=True)
    cache = f'/workspace/experiments/engines/{cache_name}'
    main = set_property(main, 'primary-gie', 'model-engine-file', cache)
    detector = set_property(detector, 'property', 'model-engine-file', cache)
main_path.write_text(main)
detector_path.write_text(detector)
for peer_path, peer_text in peers.items():
    peer_path.write_text(peer_text)
os.chmod(mqtt_path, 0o600)
mqtt_path.write_text(mqtt_config)
if kafka_config is not None:
    descriptor = os.open(kafka_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(kafka_path, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(kafka_config)
print('Prepared run-local Kafka/MQTT endpoints and detector settings; credentials withheld')
PYCONFIG
```

For RTDETR on Blackwell, the generated `${EXPERIMENT_DIR}/config_pgie.txt` uses `strongly-typed=1` and a separate engine path under the run's `engines/` directory. The source detector template and shared model engine caches remain unchanged. Set `MV3DT_FORCE_RTDETR_STRONGLY_TYPED=1` when GPU detection is incomplete. Kafka settings apply only to generated `config_deepstream.txt`; all generated `pub_sub_info_config_*.yml` files receive the MQTT endpoint. Those peer files are machine-generated YAML; do not edit them manually, as regeneration replaces their formatting and comments.

The automatic GPU check reads the host's visible GPUs. If container device restrictions or `NVIDIA_VISIBLE_DEVICES` change that set, verify GPU names inside the selected DeepStream image using the same GPU options as the intended launch before generating configs. For an RTDETR run with any visible Blackwell GPU, set `MV3DT_FORCE_RTDETR_STRONGLY_TYPED=1` explicitly. If device visibility cannot be established, stop rather than trusting the host-only result.

Immediately before each launch or retry, repeat the container-side Kafka and MQTT reachability checks in [Container Networking](container-network.md), using the exact endpoints saved in the generated configs. Recheck Kafka advertised listeners and MQTT authentication, not just host TCP access. If the bridge gateway or broker endpoint changed after generation, regenerate and verify configs before launching; do not run with stale endpoints.

Report the selected `DETECTOR_MODEL`, `DETECTOR_CONFIG`, `TRACKER_CONFIG`, `CONFIG_FLAGS`, optional `CONFIG_OVERRIDES`, whether the RTDETR strongly typed guard ran, and `EXPERIMENT_DIR` before launching DeepStream.
