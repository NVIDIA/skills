# FoundationStereo engine construction

Use the TAO `deployable_*` ONNX from the
[NGC model page](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/foundationstereo).
Its model-page terms apply separately from the pipeline's code license. No FoundationStereo
source checkout or second Python environment is used.

## Obtain the export

Reuse a supplied ONNX path. For a new download, the documented dynamic export is
`nvidia/tao/foundationstereo:deployable_foundation_stereo_s_dynamic_v2.0`.
The artifact is public. An NGC CLI recipe is:

```bash
PIPELINE_ROOT="$PWD"
PIPELINE_PARENT="$(dirname "$PIPELINE_ROOT")"
curl -fLsS -o ngccli.zip \
  https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.34.10/files/ngccli_linux.zip
python3 -m zipfile -e ngccli.zip .
chmod +x ngc-cli/ngc
mkdir -p "$PIPELINE_PARENT/models"
(
  cd "$PIPELINE_PARENT/models"
  env -u NGC_CLI_ORG -u NGC_CLI_TEAM "$PIPELINE_ROOT/ngc-cli/ngc" registry model download-version \
    nvidia/tao/foundationstereo:deployable_foundation_stereo_s_dynamic_v2.0
)
```

The subshell leaves the caller in the product checkout. Do not configure org/team identity for
an anonymous download: the fully qualified model name already specifies them. Reuse an NGC CLI
already installed if available. If the public artifact becomes inaccessible, inspect the HTTP or
CLI error and model page; do not invent a credential requirement or silently change the model.

Prefer a dynamic ONNX export for measuring this rig's shape. A fixed-shape export requires the
exact baked-in dimensions, such as 320x736, and may require resampling. Identify which export is
present before selecting build arguments.

## Measure, build, and configure

The scene must first be adapted to the pipeline's rig layout. `--shape-from-scene` measures pair
selection and rectification; raw image dimensions are not the TensorRT input dimensions.
Use the dataset profile's split (shipped profiles use `test`):

```bash
./.venv/bin/python tools/build_tao_engine.py \
  --config <profile> \
  --onnx <actual-downloaded-onnx> \
  --shape-from-scene <dataset-root>/<dataset>/<split>/<scene> \
  --max-width <profile-depth-foundation_stereo_max_width>
```

FP32 and a static min=opt=max profile are the defaults. Keep them for accuracy comparisons.
TAO allocates at the profile's maximum shape; a broad dynamic engine wastes GPU memory. If data
has not arrived, defer the shape-dependent build. A user-requested provisional engine must be
labelled provisional and rebuilt from an adapted scene before accuracy evaluation.

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
`foundation_stereo_max_width` requires rebuilding for that width and regenerating cached depth.
