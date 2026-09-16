# Model prerequisite: bring your own ONNX

This skill does not install Ultralytics software or download YOLO weights. Place a TensorRT-compatible ONNX model at the path below before continuing.

> **YOLO11 is referenced throughout this skill as a configuration example only** — tensor names, shapes, and parser code illustrate the pattern. Adapt them to your supplied model.

```bash
# Required before starting perception-alerts.
# Replace yolo11s.onnx with your model filename if different.
ls "${VSS_DATA_DIR}/models/yolo/yolo11s.onnx"
```

If the file is missing, place your ONNX model at that path before proceeding.

## Export settings used in this reference

The YOLO11 configuration examples in this skill assume an ONNX export with:

- `dynamic=True` — variable batch sizes for multiple cameras
- `opset=12` — compatible with TensorRT 10.x in the target container
- `simplify=True` — reduced graph complexity

The configuration examples in this skill were produced using `YOLO("yolo11s.pt")` as the reference model. If you export your own model, match the settings above for compatibility with the nvinfer config and parser in Steps 2–4.

## Restaging after VSS deployment

`dev-profile.sh up` recreates `${VSS_DATA_DIR}/models/`. After each full VSS
deployment, restore the model directory and replace your ONNX file:

```bash
sudo mkdir -p "${VSS_DATA_DIR}/models/yolo"
sudo chown -R "$(id -u):$(id -g)" "${VSS_DATA_DIR}/models/yolo"
# Copy or re-export your ONNX model to:
# ${VSS_DATA_DIR}/models/yolo/yolo11s.onnx
```

The ONNX file must exist at `${VSS_DATA_DIR}/models/yolo/yolo11s.onnx` before
starting `perception-alerts`.
