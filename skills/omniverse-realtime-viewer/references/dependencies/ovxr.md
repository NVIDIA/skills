# OVXR library routing

Read [dependency selection](nvidia-runtime.md#ovxr) before setup. Use
[NVIDIA-Omniverse/ovxr](https://github.com/NVIDIA-Omniverse/ovxr) as the canonical
SDK source. Select documentation and artifacts for the application's resolved
release; no Kit checkout or implementation source is required.

## Acquisition

Python applications use [ovxr on PyPI](https://pypi.org/project/ovxr/) in an
isolated environment, preserving an existing compatible manifest:

```bash
python3 -m pip install ovxr
```

Honor its declared OVRTX/OVStage requirements. Native C/C++ applications use
the matching upstream SDK. Follow current package/platform metadata and the
selected runtime's own installation guidance rather than copying artifact
names, loader procedures or platform lists here.

## Upstream documentation and samples

Resolve these paths within that upstream revision:

- `README.md`: public packaging, behavior and integration authority.
- `skills/README.md`: choose the relevant library skills for setup, lifecycle,
  context, publication, frames, readback, anchors, native input, display,
  passthrough, composition, runtime adapters and SDK validation.
- `examples/README.md`: select the runnable Python or C/C++ minimal XR sample.
  Follow its language-specific README and source for bootstrap, population,
  runtime probing, first frame and shutdown. Its minimal lifecycle does not
  implement a viewer's editing or co-writing policy.
- `VERSION.md` and release metadata: artifact identity and compatibility.
- `include/ovxr/ovxr.h` and `python/ovxr/`: exact exported signatures.

Discover current agent guidance if present. If a library skill or example
contradicts the release README or package metadata, record the conflict and
check matching exports; do not perpetuate it in another local bootstrap.

## Viewer-owned integration

Return to [the viewer recipe](../xr-viewer-recipe/README.md) for source-safe
composition, navigation/gesture policy, UI/compositor algorithms, tool ownership,
simulation handoff, mirror delivery and integrated client evidence. This bundle
selects and connects those features; upstream owns SDK mechanics and sample
startup. Runtime vendor service/extension contracts route through the upstream
runtime-adapters skill. Additional layers or borrowed graphics handles alone
do not supply the custom primary-depth compositor described by this bundle.
