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

if [ "${DETECTOR_MODEL}" = "RTDETR" ]; then
  BLACKWELL_GPU_PATTERN="${MV3DT_BLACKWELL_GPU_PATTERN:-Blackwell|RTX PRO (2000|4000|4500|5000|6000)|GB10|GB200|B200|B300|DGX Spark}"
  GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  if [ "${MV3DT_FORCE_RTDETR_STRONGLY_TYPED:-0}" = "1" ] || printf '%s\n' "${GPU_NAMES}" | grep -Eiq "${BLACKWELL_GPU_PATTERN}"; then
    echo "RTDETR on Blackwell-class GPU detected or forced; ensuring strongly-typed=1 in ${REPO_ROOT}/config_templates/${DETECTOR_CONFIG}"
    export RTDETR_TEMPLATE_PATH="${REPO_ROOT}/config_templates/${DETECTOR_CONFIG}"
    python - <<'PY'
from pathlib import Path
import os

path = Path(os.environ["RTDETR_TEMPLATE_PATH"])
text = path.read_text()
lines = text.splitlines()
property_start = next((idx for idx, line in enumerate(lines) if line.strip() == "[property]"), None)
if property_start is None:
    raise SystemExit(f"ERROR: [property] section not found in {path}")
property_end = len(lines)
for idx in range(property_start + 1, len(lines)):
    stripped = lines[idx].strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        property_end = idx
        break
for idx in range(property_start + 1, property_end):
    stripped = lines[idx].strip()
    if not stripped or stripped.startswith("#"):
        continue
    if stripped.split("=", 1)[0].strip() == "strongly-typed":
        if stripped == "strongly-typed=1":
            print(f"{path} already has strongly-typed=1")
        else:
            lines[idx] = "strongly-typed=1"
            path.write_text("\n".join(lines) + "\n")
            print(f"updated strongly-typed=1 in {path}")
        break
else:
    lines.insert(property_start + 1, "strongly-typed=1")
    path.write_text("\n".join(lines) + "\n")
    print(f"added strongly-typed=1 to {path}")
PY
    : "${MODEL_REPO:?set MODEL_REPO before RTDETR engine cache handling}"
    RTDETR_ENGINE="${MODEL_REPO}/RTDETR/rtdetr_warehouse_v1.0.fp16.onnx_b$(find "${DATASET_DIR}/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)_gpu0_fp16.engine"
    if [ -f "${RTDETR_ENGINE}" ]; then
      echo "Removing stale RTDETR TensorRT engine so strongly typed mode is used: ${RTDETR_ENGINE}"
      rm -f "${RTDETR_ENGINE}"
    fi
  fi
fi

python "${REPO_ROOT}/utils/deepstream_auto_configurator.py" \
  --dataset-dir="${DATASET_DIR}" \
  "${CONFIG_FLAGS[@]}" \
  --detector-config="${DETECTOR_CONFIG}" \
  --tracker-config="${TRACKER_CONFIG}" \
  "${CONFIG_OVERRIDE_ARGS[@]}" \
  --output-dir="${EXPERIMENT_DIR}"
```

For RTDETR on Blackwell-based GPUs such as RTX PRO 2000-6000 Blackwell, GB10, GB200, DGX Spark, and related platforms, the detector template `${REPO_ROOT}/config_templates/config_pgie_rt_detr.txt` must contain `strongly-typed=1` before config generation. The guard above patches the template and removes only the matching generated RTDETR TensorRT engine cache so DeepStream rebuilds it with strongly typed mode. Set `MV3DT_FORCE_RTDETR_STRONGLY_TYPED=1` if `nvidia-smi` does not expose a recognizable Blackwell GPU name.

Report the selected `DETECTOR_MODEL`, `DETECTOR_CONFIG`, `TRACKER_CONFIG`, `CONFIG_FLAGS`, optional `CONFIG_OVERRIDES`, whether the RTDETR strongly typed guard ran, and `EXPERIMENT_DIR` before launching DeepStream.
