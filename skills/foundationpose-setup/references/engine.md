# FoundationStereo engine construction

Use the TAO `deployable_*` ONNX from the
[Hugging Face model page](https://huggingface.co/nvidia/c-foundationstereo-s).
Its model-page terms apply separately from the pipeline's code license. No FoundationStereo
source checkout or second Python environment is used.

## Obtain the export

Reuse a supplied ONNX path. For a new download, let the user choose an export from the
[Hugging Face model files](https://huggingface.co/nvidia/c-foundationstereo-s/tree/main/onnx).

For the repository's FP32 baseline, use
[`deployable_foundation_stereo_s_dynamic.onnx`](https://huggingface.co/nvidia/c-foundationstereo-s/blob/main/onnx/deployable_foundation_stereo_s_dynamic.onnx).
Build a static engine profile from this dynamic ONNX at the rig's measured shape.

The model card labels the dynamic export ONNX Runtime-only but cites an FP16 TensorRT conversion
failure; do not interpret that as proof that this repository's FP32 path fails. Keep FP32 for
the dynamic export. For FP16, select `deployable_foundationstereo_small_320x736_v2.0.onnx` with
`--shape 320x736` or `deployable_foundationstereo_small_576x960_v2.0.onnx` with `--shape 576x960`.
Fixed exports require those exact dimensions and may resample or crop the rectified pair.
Identify the export before selecting build arguments.

## Measure, build, and configure

For the dynamic ONNX, the scene must first be adapted to the pipeline's rig layout.
`--shape-from-scene` measures pair selection and rectification; raw image dimensions are not
the TensorRT input dimensions.
Use the dataset profile's split (shipped profiles use `test`):

```bash
./.venv/bin/python tools/build_tao_engine.py \
  --config <profile> \
  --onnx <actual-downloaded-onnx> \
  --shape-from-scene <dataset-root>/<dataset>/<split>/<scene> \
  --max-width <profile-depth-foundation_stereo_max_width> \
  --precision fp32
```

FP32 and a static min=opt=max profile are the defaults. Keep them for accuracy comparisons.
TAO allocates at the profile's maximum shape; a broad dynamic engine wastes GPU memory. For a
dynamic ONNX, if data has not arrived, defer the shape-dependent build. A user-requested
provisional engine must be labelled provisional and rebuilt from an adapted scene before
accuracy evaluation.

Record the generated filename and its JSON sidecar. The engine depends on GPU architecture,
TensorRT version, precision, input shape, and source ONNX hash. Rebuild when these change;
never commit engines or bypass stale-sidecar checks just to get a run to start.

Edit the dataset profile's existing commented engine entry, resolving relative paths from that
profile's directory:

```yaml
overrides:
  depth:
    engine: ../../models/<generated-engine>.engine
```

Keep this machine-specific setting local. `depth.engine` is deliberately unset in shipped
profiles. The runtime model path selects the backend; `--depth-backend commercial` asserts the
expected path and is not a model selector or proof of license approval.

Run `tools/verify_foundationstereo.py` and `test/check_engine_depth_smoke.py` as in SKILL.md.
Require `backend=tao`, `normalization=imagenet`, and no cropping warning. Static engines can
rescale and crop instead of failing on mismatched dimensions, so changing
`foundation_stereo_max_width` requires rebuilding the engine from the dynamic ONNX for that
width and regenerating cached depth. A fixed ONNX cannot change dimensions; investigate
cropping and retained object coverage rather than trying to override its shape.
