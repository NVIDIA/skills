---
name: rtvi-cv-customize-model
description: How to swap the DeepStream CV detection model in the VSS Alerts Blueprint verification (2d_cv) mode - covers ONNX export, custom bbox parsers, compose mount gotchas, nvinfer config, runtime TRT engine build, deployment, and a segmentation-capable model addendum handoff.

metadata:
  team: accelerated-microservices
  author: "NVIDIA CORPORATION <info@nvidia.com>"
  tags:
    - vss
    - cv-customization
    - rtvi-cv-customize-model
    - alerts-blueprint
  languages:
    - en
  domain: computer-vision
---

# CV Detection Model Customization — VSS Alerts Blueprint (`2d_cv` mode only)

The RT-CV perception container (`vss-rt-cv`) runs a DeepStream pipeline with a configurable primary GIE (GPU Inference Engine). By default it uses GDINO or RTDETR. This guide covers replacing it with any ONNX-format model, using YOLOv11 COCO 80 as the worked example.

This only applies to `--mode verification` (`2d_cv`). Real-time alerts mode (`2d_vlm`) has no CV detector.

For instance-segmentation or detection-plus-mask models, complete [references/segmentation-model-contract.md](references/segmentation-model-contract.md) before writing parser or handoff code.

---

## When to use

Use this skill when the user wants to:

- swap the stock `vss-rt-cv` detector for another ONNX model in verification mode,
- debug a broken ONNX staging path, ghost-directory bind mount, or missing runtime TRT engine build,
- fix a DeepStream parser load failure such as `dlsym failed` on the bbox parser symbol.

Do **not** use this skill for:

- `2d_vlm` real-time alerts mode,
- scaffolding a brand-new standalone RTVI CV microservice (use `rtvi-cv-scaffold-vss-service`).

## Instructions

- Keep the answer scoped to VSS Alerts Blueprint verification mode (`2d_cv`) unless the user explicitly asks to compare modes.
- Paths beginning with `deploy/docker/` are relative to the [VSS Blueprint repository](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization), not the DeepStream repository. Clone or reuse a VSS checkout that is **v3.2.1 or compatible**, then run these commands from that repository root (see [VSS Quickstart](https://docs.nvidia.com/vss/latest/quickstart.html#download-the-deployment-package)).
- If the user asks about ONNX staging or a compose mount, explicitly say that mounting a missing file path is wrong: use the stock parent-directory mount, keep the ONNX under `${VSS_DATA_DIR}/models/yolo`, and re-stage the ONNX file after any `dev-profile.sh up` that recreates the models directory.
- If the user asks about parser load failures, line up `parse-bbox-func-name`, the `extern "C"` function symbol, and `CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(...)`, then rebuild the `.so` with both DeepStream and CUDA include paths present.
- If the model is detection-plus-mask or instance segmentation, route the mask-contract decision first through [references/segmentation-model-contract.md](references/segmentation-model-contract.md) before editing parser or handoff logic.

## Examples

- "Replace the VSS Alerts Blueprint verification detector with a YOLO11 ONNX model and redeploy `perception-alerts`."
- "DeepStream says `dlsym failed` for the bbox parser after loading the custom parser library. What should I check?"
- "I mounted the ONNX file path directly in compose before the host file existed. Is that okay?"

---

## Architecture

```
NVStreamer (RTSP) → SDR (port 9010) → DeepStream (perception-alerts)
                                            └─ primary-gie (nvinfer)
                                                  ├─ ONNX → TRT engine (built once, then cached)
                                                  ├─ custom bbox parser (.so)
                                                  └─ label file (.txt)
                                            └─ Kafka → mdx-raw
                                                  └─ vss-behavior-analytics
                                                        └─ mdx-incidents
                                                              └─ vlm-as-verifier
```

## VSS source location

This skill is documentation-only and does not ship the VSS deployment sources. Use a VSS **v3.2.1 or compatible** checkout; run all paths and commands from that repository root. Clone/LFS setup, stock Alerts profile files, and the YOLOv11 customization file tree are in [references/vss-source-layout.md](references/vss-source-layout.md).

---

## Adapting to a Different Model

YOLOv11 COCO 80 is the worked example throughout all steps below. The same pattern applies to any ONNX-format detector — substitute at these points:

| Step | What to change |
|------|----------------|
| **Step 1** | Replace the export procedure with whatever your model's training library requires. Confirm the resulting `.onnx` exists on the host before continuing. |
| **Steps 2–3** | Inspect your model's actual output tensor name, shape, layout, and whether NMS is applied in-graph. Do not assume it matches YOLOv11. The `deepstream-dev` skill ([`skills/deepstream-dev/`](../deepstream-dev/)) has a generation-by-generation YOLO output format table and `references/nvinfer_config.md` for full property reference. |
| **Step 4** | Update `output-blob-names` to your tensor name, `infer-dims` to your input shape, and `cluster-mode` to match whether NMS is in-graph (`4`) or not (`2`). |
| **Step 5** | Update the `--onnx` path, the input tensor name in `--minShapes`/`--optShapes`/`--maxShapes` (not `images` for non-YOLO models), and the `sed` commands to reference your nvinfer `.txt` file. Add a new `if [[ $MODEL_NAME_2D == "YOURMODEL" ]]` block rather than editing the YOLO block. |

---

## Step 1: Stage Your ONNX Model (Host, One-Time)

Obtain a TensorRT-compatible ONNX model and place it at `${VSS_DATA_DIR}/models/yolo/yolo11s.onnx` before continuing.

See [references/yolov11-onnx-export.md](references/yolov11-onnx-export.md) for the export settings and tensor layout used in this reference. Adapt the parser and nvinfer config in later steps to match your model's actual output.

After every full `dev-profile.sh up`, restore ownership of the recreated `models/yolo` directory and re-stage your ONNX file:

```bash
sudo mkdir -p "${VSS_DATA_DIR}/models/yolo"
sudo chown -R $(id -u):$(id -g) "${VSS_DATA_DIR}/models/yolo"
# Copy your ONNX model to ${VSS_DATA_DIR}/models/yolo/yolo11s.onnx
```

---

## Steps 2–3: Inspect the Model Output and Build the Parser


Read and follow
[references/yolov11-parser.md](references/yolov11-parser.md) before creating
`${VSS_PROFILE_DIR}/deepstream/custom_parser/nvdsparseyolov11.cpp` or editing
the parser build in `Dockerfiles/perception.Dockerfile`.

Before writing any parser code, confirm these four things for your model:

- **Output tensor name and shape** — drives `output-blob-names` and `infer-dims` in Step 4
- **Pre-NMS or post-NMS** — drives `cluster-mode` in Step 4; see `deepstream-dev` Rule 13 for the generation-by-generation breakdown
- **Coordinate semantics** — `cx/cy/w/h` center-format or `x1/y1/x2/y2` corner-format; the YOLOv11 example uses center-format (see [references/yolov11-parser.md](references/yolov11-parser.md))
- **Parser symbol name** — the `extern "C"` function name must match `parse-bbox-func-name` in Step 4 and `CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(...)` exactly

Confirm the model output tensor layout instead of assuming it matches
YOLOv11. Keep the parser symbol identical in the exported C++
function, `CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(...)`, and the Step 4
`parse-bbox-func-name` setting.

For instance segmentation or detection-plus-mask models, complete the contract decisions table in [references/segmentation-model-contract.md](references/segmentation-model-contract.md) before writing parser or handoff code.

---

## Step 4: nvinfer Config

`${VSS_PROFILE_DIR}/deepstream/configs/yolov11.txt` — the primary GIE sub-config referenced by the DeepStream run config.

**`model-engine-file` and `batch-size` are patched at container startup by the `ds-start.sh` block in Step 5.** Copy the values below as-is; Step 5 will overwrite both with the correct engine path and sensor count before DeepStream reads the file.

```ini
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373    # 1/255 input normalization
model-engine-file=/opt/storage/yolo11s_fp16.engine
onnx-file=/opt/storage/yolo/yolo11s.onnx
batch-size=1                             # patched at startup from NUM_SENSORS
network-mode=2                           # 0=FP32, 1=INT8, 2=FP16
network-type=0                           # 0=Detector
num-detected-classes=80
interval=0
gie-unique-id=1
output-blob-names=output0               # must match your model's output tensor name
infer-dims=3;640;640                    # C;H;W
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseCustomYoloE  # must match extern "C" function name in .so
custom-lib-path=/opt/deepstream-yolo/libnvdsparseyolov11.so
labelfile-path=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/metropolis_perception_app/mounted-configs/yolo-coco-labels.txt
cluster-mode=2                   # 2=NMS — required: Ultralytics ONNX has no NMS in-graph

[class-attrs-all]
pre-cluster-threshold=0.25
topk=300
nms-iou-threshold=0.45
```

---

## Step 5: Runtime TRT Engine Build (`ds-start.sh`)

TRT engines are GPU-architecture-specific. Build the engine on first startup when it is missing, then reuse it on subsequent restarts. Rebuild an existing engine only when `FORCE_REBUILD=true`. The stock Alerts compose file bind-mounts `${VSS_DATA_DIR}/models/` at `/opt/storage/`, so the generated engine persists on the host.

**Which `ds-start.sh`?** Stock Alerts runs `services/rtvi/rtvi-cv/ds-start.sh` (via `extends`), not `${VSS_PROFILE_DIR}/deepstream/init-scripts/ds-start.sh`. This skill edits the profile copy, then remounts it in Step 6 so those edits actually run. Full map of the three paths, remount trade-offs, and `DS_CONFIG_FILE` pairing: [references/ds-start-entrypoint.md](references/ds-start-entrypoint.md).

Edit `${VSS_PROFILE_DIR}/deepstream/init-scripts/ds-start.sh`, then override the stock rtvi-cv `ds-start.sh` bind mount in the Alerts profile compose file as shown in Step 6. Without that override, profile-script edits never execute.

**Critical — honor `DS_CONFIG_FILE`.** The profile script sets `CONFIG_FILE=${1:-...}` and never reads `DS_CONFIG_FILE`. The Alerts compose `command` invokes `ds-start.sh` with no positional args, while Step 6 sets `DS_CONFIG_FILE` to the absolute **mounted-configs** run config (not the stock `.../configs/...` path). Replace the script's existing `CONFIG_FILE=...` assignment with:

```bash
# Prefer $1 when provided; otherwise use DS_CONFIG_FILE from compose (absolute path).
# Without this, YOLO_CONFIG resolves to ./yolov11.txt and the stock detector keeps running.
CONFIG_FILE="${1:-${DS_CONFIG_FILE:-/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/metropolis_perception_app/mounted-configs/run_config-api-rtdetr-protobuf.txt}}"
```

Then add the YOLO block (before the GDINO/RT-DETR model branches is fine):

```bash
if [[ $MODEL_NAME_2D == "YOLO" ]]; then
    YOLO_ONNX=/opt/storage/yolo/yolo11s.onnx
    YOLO_ENGINE=/opt/storage/yolo11s_fp16.engine
    YOLO_ENGINE_TMP="${YOLO_ENGINE}.building"
    FORCE_REBUILD=${FORCE_REBUILD:-false}

    if [[ ! -f "$YOLO_ONNX" ]]; then
        echo "ERROR: ONNX not found at ${YOLO_ONNX}. Stage your model there before starting."
        exit 1
    fi

    if [[ ! -s "$YOLO_ENGINE" || "${FORCE_REBUILD,,}" == "true" ]]; then
        echo "Building TensorRT engine: ${YOLO_ENGINE}"
        rm -f "$YOLO_ENGINE_TMP"
        if /usr/src/tensorrt/bin/trtexec \
            --onnx=${YOLO_ONNX} \
            --minShapes=images:1x3x640x640 \
            --optShapes=images:${NUM_SENSORS}x3x640x640 \
            --maxShapes=images:${NUM_SENSORS}x3x640x640 \
            --fp16 --saveEngine=${YOLO_ENGINE_TMP}; then
            mv "$YOLO_ENGINE_TMP" "$YOLO_ENGINE"
        else
            rm -f "$YOLO_ENGINE_TMP"
            echo "ERROR: TensorRT engine build failed; existing engine was preserved."
            exit 1
        fi
    else
        echo "Reusing cached TensorRT engine: ${YOLO_ENGINE}"
    fi

    # Patch the run config to use yolov11.txt for the primary GIE
    # (relative name is intentional: DeepStream resolves it against CONFIG_FILE's dir)
    sed -i '/^\[primary-gie\]/,/^\[/{s/config-file=.*/config-file=yolov11.txt/;}' "$CONFIG_FILE"
    # Patch engine path and batch size into yolov11.txt (same dir as CONFIG_FILE)
    YOLO_CONFIG="$(dirname "${CONFIG_FILE}")/yolov11.txt"
    sed -i "s|model-engine-file=.*|model-engine-file=${YOLO_ENGINE}|" "${YOLO_CONFIG}"
    sed -i "/^\[property\]/,/^\[/{s/^batch-size=.*/batch-size=${NUM_SENSORS}/;}" "${YOLO_CONFIG}"
fi
```

To add a different model, add a new `if [[ $MODEL_NAME_2D == "YOURMODEL" ]]` block. Update:
- `--onnx` path and `--saveEngine` output path
- `--minShapes`/`--optShapes`/`--maxShapes` input tensor name and dimensions for your model
- The `sed` commands to point at your nvinfer `.txt` file

---

## Step 6: Compose + Env

In the existing `${VSS_PROFILE_DIR}/compose.yml`, update the `perception-alerts` service:

```yaml
perception-alerts:
  build:
    context: $VSS_APPS_DIR/developer-profiles/dev-profile-alerts
    dockerfile: Dockerfiles/perception.Dockerfile
  volumes:
    # Keep the stock parent-directory mount. Do not replace it with a direct
    # mount of yolo11s.onnx: Docker creates a ghost directory if the host
    # file does not exist.
    - $VSS_DATA_DIR/models/:/opt/storage/
    - $VSS_APPS_DIR/developer-profiles/dev-profile-alerts/deepstream/configs/:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/metropolis_perception_app/mounted-configs/
    # Override stock rtvi-cv ds-start.sh (extends bind). Without this,
    # edits under deepstream/init-scripts/ never run — see
    # references/ds-start-entrypoint.md.
    - $VSS_APPS_DIR/developer-profiles/dev-profile-alerts/deepstream/init-scripts/ds-start.sh:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/metropolis_perception_app/ds-start.sh:ro
  environment:
    MODEL_NAME_2D: ${MODEL_NAME_2D}
    NUM_SENSORS: ${NUM_SENSORS}
    FORCE_REBUILD: ${FORCE_REBUILD:-false}
    # Use mounted-configs (where yolov11.txt lives), not stock .../configs/...
    DS_CONFIG_FILE: /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/metropolis_perception_app/mounted-configs/run_config-api-rtdetr-protobuf.txt
```

Set in `${VSS_PROFILE_DIR}/.env`:
```bash
MODEL_NAME_2D="YOLO"
NUM_SENSORS=1
FORCE_REBUILD=false
```

Keep `FORCE_REBUILD=false` for normal starts. Compose's `--force-recreate` option recreates the container but does **not** rebuild the TensorRT engine unless `FORCE_REBUILD=true`.

**Gotcha — `generated.env` wins at restart.** Step 7's redeploy uses `--env-file .../generated.env`, which is produced by `dev-profile.sh` from `.env`. Editing `.env` alone does **not** change `FORCE_REBUILD` for that compose command. For a one-shot engine rebuild, pass the override on the shell (see Step 7). To persist the value for later `dev-profile.sh` runs, edit `.env` and re-run `dev-profile.sh` so it regenerates `generated.env`.

---

## Step 7: Deploy

Run from the video-search-and-summarization repository root
(`VSS_ROOT` / `VSS_DEPLOY_DIR` / `VSS_PROFILE_DIR` from VSS source location):

```bash
: "${VSS_ROOT:=$PWD}"
: "${VSS_DEPLOY_DIR:=${VSS_ROOT}/deploy/docker}"
: "${VSS_PROFILE_DIR:=${VSS_DEPLOY_DIR}/developer-profiles/dev-profile-alerts}"
```

**1. Single-GPU host (only GPU 0):** edit `${VSS_PROFILE_DIR}/.env` so these
keys match the required final values below. Edit the source profile env file,
not `generated.env` (`dev-profile.sh` regenerates that). Stock Alerts reserves
GPU 0 and puts RT-VLM / LLM / VLM on GPU 1; `dev-profile.sh` CLI flags cannot
clear `RESERVED_DEVICE_IDS` or set `FIXED_SHARED_DEVICE_IDS`. Confirm device
IDs exist with `nvidia-smi --query-gpu=index --format=csv,noheader,nounits`
before deploying.

```text
RESERVED_DEVICE_IDS=''
FIXED_SHARED_DEVICE_IDS='0'
RT_CV_DEVICE_ID='0'
RT_VLM_DEVICE_ID='0'
LLM_DEVICE_ID='0'
VLM_DEVICE_ID='0'
```

**2. Deploy** the official Alerts verification profile. Select the hardware
profile supported by your host, as documented by the VSS Quickstart.
Do not present the remaining commands as a copy-paste handoff unless execution
is blocked.

```bash
"${VSS_DEPLOY_DIR}/scripts/dev-profile.sh" up \
  -p alerts \
  -m verification \
  -H <H100|L40S|RTXPRO4500BW|RTXPRO6000BW|DGX-SPARK|IGX-THOR|AGX-THOR|OTHER>

# 3. dev-profile.sh recreates the model directory. Restore ownership and
# re-stage your ONNX model.
export VSS_DATA_DIR="${VSS_DEPLOY_DIR}/data-dir"
sudo mkdir -p "${VSS_DATA_DIR}/models/yolo"
sudo chown -R $(id -u):$(id -g) "${VSS_DATA_DIR}/models/yolo"
# Copy your ONNX model to ${VSS_DATA_DIR}/models/yolo/yolo11s.onnx

# 4. Rebuild and recreate the customized perception service using the compose
# project generated by dev-profile.sh.
cd "${VSS_DEPLOY_DIR}"
docker compose \
  --env-file developer-profiles/dev-profile-alerts/generated.env \
  up -d --build --force-recreate perception-alerts

# One-shot TensorRT rebuild (after ONNX / shape / GPU / TRT changes).
# Inline FORCE_REBUILD overrides generated.env for this invocation only:
# FORCE_REBUILD=true docker compose \
#   --env-file developer-profiles/dev-profile-alerts/generated.env \
#   up -d --build --force-recreate perception-alerts
```

TRT engine build takes approximately 15–30 seconds on Blackwell hardware.
Subsequent restarts reuse the engine persisted under
`${VSS_DATA_DIR}/models/`.

---

## Common Gotchas

See [references/common-gotchas.md](references/common-gotchas.md) for single-GPU reservation failures, wipe/restage after `dev-profile.sh up`, ghost file mounts, `dlsym` parser mismatches, and Redis stream contamination.
