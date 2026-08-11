# AutoMagicCalib Handoff

Load this reference only when a custom synchronized MP4 dataset lacks `camInfo/*.yml` calibration files.

## Principle

Do not duplicate AutoMagicCalib setup or calibration logic here. Delegate to the standalone AutoMagicCalib skills, then return to `custom-dataset.md` after the MV3DT-format export is placed in the dataset.

## Handoff Inputs

Before delegating, ask the user to choose the calibration detector (`resnet` or `transformer`) and whether they have a calibration settings JSON to apply. Settings files have no required filename convention; use an explicit user-provided path when available, and otherwise run AMC without a settings file or use the AMC UI Step 3 fallback. Do not silently calibrate with a default detector for custom datasets.

AMC's video calibration workflow uploads files matching `cam_*.mp4` sorted alphabetically. If the custom dataset videos are not already named that way, do not rename user files. Create a generated staging directory with ordered `cam_NN.mp4` symlinks or copies, then pass that staging directory as the AMC videos directory:

```bash
AMC_VIDEO_DIR="${AMC_VIDEO_DIR:-${EXPERIMENT_DIR}/amc_video_staging/videos_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${AMC_VIDEO_DIR}"
i=0
while IFS= read -r src; do
  src_abs="$(realpath "$src")"
  dst="$(printf "%s/cam_%02d.mp4" "${AMC_VIDEO_DIR}" "$i")"
  ln -s "$src_abs" "$dst" 2>/dev/null || cp -n "$src_abs" "$dst"
  printf '%s -> %s\n' "$dst" "$src_abs"
  i=$((i + 1))
done < <(find "${DATASET_DIR}/videos" -maxdepth 1 -type f -name '*.mp4' | sort)
test "$i" -gt 0 || { echo "ERROR: no MP4 files found under ${DATASET_DIR}/videos"; exit 1; }
```

Pass these details to the AutoMagicCalib video calibration workflow:

- Videos directory: `${DATASET_DIR}/videos` when files already match `cam_*.mp4`, otherwise `${AMC_VIDEO_DIR}`
- Project name: a short dataset slug
- Selected detector: the user-selected `resnet` or `transformer` value
- Calibration settings: optional explicit JSON path from the user; pass it as `CONFIG_FILE` only when the user confirms
- Optional alignment files: `alignment_data.json` and `layout.png` from the dataset directory or its parent

## AMC Compose Image Guard

Treat the latest checked-out AutoMagicCalib compose file as authoritative. If the AMC checkout is not already available, delegate repo clone, checkout, and setup to `amc-setup-calibration-stack`; do not clone or substitute AMC backend images from this MV3DT skill.

Before running calibration, resolve `AMC_COMPOSE_DIR`, normally `${AMC_REPO_ROOT}/compose`, from the AMC setup skill or an explicit user-provided AMC checkout. Record the compose images and refuse silent overrides:

```bash
AMC_COMPOSE_DIR="${AMC_COMPOSE_DIR:-${AMC_REPO_ROOT:?set AMC_REPO_ROOT or AMC_COMPOSE_DIR before AMC handoff}/compose}"
test -d "${AMC_COMPOSE_DIR}" || { echo "ERROR: AMC compose directory not found: ${AMC_COMPOSE_DIR}"; exit 1; }
(
  cd "${AMC_COMPOSE_DIR}"
  docker compose config --images | sort -u
) | tee /tmp/mv3dt_amc_compose_images.txt
test -s /tmp/mv3dt_amc_compose_images.txt || { echo "ERROR: AMC compose image list is empty"; exit 1; }
```

Do not create temporary compose override files, set image override environment variables, or use locally cached alternate AMC images unless the user explicitly approves the alternate image list. If the recorded compose images cannot be pulled or started, stop and report the exact registry/auth/image failure before accepting any calibration result.

After AMC starts, verify the running AMC containers use only images from `/tmp/mv3dt_amc_compose_images.txt`; if not, stop and ask whether to proceed with the observed alternate image list:

```bash
(
  cd "${AMC_COMPOSE_DIR}"
  docker compose images --format '{{.Repository}}:{{.Tag}}' | sort -u
) > /tmp/mv3dt_amc_running_images.txt
comm -23 /tmp/mv3dt_amc_running_images.txt /tmp/mv3dt_amc_compose_images.txt > /tmp/mv3dt_amc_unapproved_images.txt
if [ -s /tmp/mv3dt_amc_unapproved_images.txt ]; then
  echo "ERROR: AMC is running images not declared by the checked-out compose file:"
  cat /tmp/mv3dt_amc_unapproved_images.txt
  exit 1
fi
```

Use these skills in order when needed:

1. `amc-setup-calibration-stack` if the service and UI are not already running.
2. `amc-run-video-calibration` for local MP4 calibration.

Wait until the project reaches `project_state == COMPLETED` before fetching the MV3DT export. During the delegated calibration run, expect the AMC skill to report project ID, UI/log review locations, progress updates, completion state, and the calibration result review artifact before returning to MV3DT.

## Fetch MV3DT Export

```bash
AMC_BASE_URL="${AMC_BASE_URL:-${BASE_URL:-http://localhost:8000}}"
AMC_BASE_URL="${AMC_BASE_URL%/}"
AMC_BASE_URL="${AMC_BASE_URL%/v1}"
PROJECT_ID="${PROJECT_ID:?AMC project id}"
RESULT_TYPE="${RESULT_TYPE:-amc}"

curl -sfL "${AMC_BASE_URL}/v1/result/${PROJECT_ID}/mv3dt_result?result_type=${RESULT_TYPE}" \
  -o /tmp/mv3dt_output.zip
unzip -l /tmp/mv3dt_output.zip
```

Prefer `RESULT_TYPE=vggt` only when the AutoMagicCalib run completed VGGT refinement and the user wants the refined result; otherwise use `amc`. If carrying a URL from the AMC skills, normalize it to the service root here; both `http://host:8000` and `http://host:8000/v1` are accepted by the snippet above.

## Land Export Into Dataset

Show the planned extracted filenames before modifying the dataset. By default, extract into a temporary directory and copy only missing files into the dataset; overwrite existing calibration only after explicit user approval.

```bash
TMP_MV3DT_EXPORT="$(mktemp -d)"
unzip -q /tmp/mv3dt_output.zip -d "${TMP_MV3DT_EXPORT}"
find "${TMP_MV3DT_EXPORT}" -maxdepth 3 -type f | sort

mkdir -p "${DATASET_DIR}/camInfo"
find "${TMP_MV3DT_EXPORT}" -path '*/camInfo/*' -type f \
  \( -name '*.yml' -o -name '*.yaml' \) -print0 \
  | while IFS= read -r -d '' f; do cp -n "$f" "${DATASET_DIR}/camInfo/$(basename "$f")"; done

for artifact in transforms.yml map.png calibration.json; do
  src="$(find "${TMP_MV3DT_EXPORT}" -type f -name "$artifact" | sort | head -n 1)"
  if [ -n "$src" ]; then
    cp -n "$src" "${DATASET_DIR}/$artifact"
  else
    echo "WARNING: AMC MV3DT export did not contain $artifact"
  fi
done

find "${DATASET_DIR}/camInfo" -maxdepth 1 -type f -name '*.yaml' -print0 \
  | while IFS= read -r -d '' f; do cp -n "$f" "${f%.yaml}.yml"; done
```

## Normalize `modelInfo` For MV3DT

AMC exports may contain a single-class `modelInfo` section. Before generating MV3DT configs, update the landed `camInfo/*.yml` files with the same detector selected for the MV3DT run. Default `DETECTOR_MODEL` to `PeopleNetTransformer` if the user did not request a detector.

Detector-specific `modelInfo` mapping:

| `DETECTOR_MODEL` | `modelInfo` class IDs |
|---|---|
| `PeopleNetTransformer` | `1` person |
| `PeopleNet2.6.3` | `0` person |
| `RTDETR` | `0` person, `1` agility_digit, `2` gr1_t2, `3` nova_carter, `4` transporter, `5` forklift, `6` pallet |

```bash
export DATASET_DIR="${DATASET_DIR:?DATASET_DIR must be set before normalizing modelInfo}"
export DETECTOR_MODEL="${DETECTOR_MODEL:-PeopleNetTransformer}"

python3 - <<'PY'
from pathlib import Path
import os

model_info_by_detector = {
    "PeopleNetTransformer": """# PeopleNetTransformer detector
modelInfo:
  - classID: 1 # person
    height: 1.7
    radius: 0.3""",
    "PeopleNet2.6.3": """# PeopleNet v2.6.3 detector
modelInfo:
  - classID: 0 # person
    height: 1.7
    radius: 0.3""",
    "RTDETR": """# RTDETR detector
modelInfo:
  - classID: 0 # person
    height: 1.7
    radius: 0.3
  - classID: 1 # agility_digit (not shown in sample videos)
    height: 1.7
    radius: 0.3
  - classID: 2 # gr1_t2 (not shown in sample videos)
    height: 1.7
    radius: 0.3
  - classID: 3 # nova_carter (not shown in sample videos)
    height: 0.48
    radius: 0.3
  - classID: 4 # transporter (not shown in sample videos)
    height: 0.2
    radius: 0.52
  - classID: 5 # forklift
    height: 2.2
    radius: 0.9
  - classID: 6 # pallet (filtered out in sample config)
    height: 0.48
    radius: 0.3""",
}

detector_model = os.environ.get("DETECTOR_MODEL", "PeopleNetTransformer")
if detector_model not in model_info_by_detector:
    expected = ", ".join(sorted(model_info_by_detector))
    raise SystemExit(f"ERROR: unsupported DETECTOR_MODEL={detector_model!r}; use one of: {expected}")
model_info = model_info_by_detector[detector_model]

cam_info_dir = Path(os.environ["DATASET_DIR"]) / "camInfo"
cam_info_files = sorted(cam_info_dir.glob("*.yml"))
if not cam_info_files:
    raise SystemExit(f"ERROR: no calibration files found under {cam_info_dir}")
for path in cam_info_files:
    text = path.read_text()
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "modelInfo:" and not line.startswith((" ", "\t"))), None)
    if start is None:
        updated = text.rstrip() + "\n\n" + model_info + "\n"
    else:
        end = start + 1
        while end < len(lines):
            line = lines[end]
            if line and not line.startswith((" ", "\t", "-")) and not line.lstrip().startswith("#"):
                break
            end += 1
        updated = "\n".join(lines[:start] + model_info.splitlines() + lines[end:]) + "\n"
    path.write_text(updated)
    print(f"updated {detector_model} modelInfo: {path}")
PY
```

Verify the normalized files before returning to the MV3DT run flow:

```bash
export DATASET_DIR="${DATASET_DIR:?DATASET_DIR must be set before verifying modelInfo}"

python3 - <<'PY'
from pathlib import Path
import os
import yaml

expected_by_detector = {
    "PeopleNetTransformer": {1},
    "PeopleNet2.6.3": {0},
    "RTDETR": {0, 1, 2, 3, 4, 5, 6},
}
detector_model = os.environ.get("DETECTOR_MODEL", "PeopleNetTransformer")
if detector_model not in expected_by_detector:
    expected = ", ".join(sorted(expected_by_detector))
    raise SystemExit(f"ERROR: unsupported DETECTOR_MODEL={detector_model!r}; use one of: {expected}")
expected = expected_by_detector[detector_model]
cam_info_dir = Path(os.environ["DATASET_DIR"]) / "camInfo"
cam_info_files = sorted(cam_info_dir.glob("*.yml"))
if not cam_info_files:
    raise SystemExit(f"ERROR: no calibration files found under {cam_info_dir}")
for path in cam_info_files:
    data = yaml.safe_load(path.read_text()) or {}
    class_ids = {item.get("classID") for item in data.get("modelInfo", [])}
    if class_ids != expected:
        raise SystemExit(f"ERROR: {path} modelInfo class IDs are {sorted(class_ids)}, expected {sorted(expected)} for {detector_model}")
    print(f"verified {detector_model} modelInfo: {path}")
PY
```

If replacing existing `modelInfo`, `camInfo`, `transforms.yml`, `map.png`, or `calibration.json` is needed outside the newly landed AMC export, back up the current files first and ask before overwriting. `map.png` and `transforms.yml` are required for BEV map rendering; `calibration.json` should be kept with the dataset for AMC export provenance and downstream consumers, even though the DeepStream auto-configurator does not consume it directly.

## Return Point

After the export is landed and `modelInfo` is normalized, return to `custom-dataset.md` at the input validation step. Confirm that `find "${DATASET_DIR}/camInfo" -name '*.yml'` now returns one calibration file per camera, verify the expected `modelInfo` class IDs, verify whether `map.png`, `transforms.yml`, and `calibration.json` were landed, and include the selected calibration detector, settings-file decision, and AMC compose images used in the final run summary.
