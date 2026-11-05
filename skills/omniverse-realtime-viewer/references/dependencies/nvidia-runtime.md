# NVIDIA Runtime Dependencies

Use this reference to find the libraries selected for the viewer. Package
metadata and the corresponding upstream release guidance own compatibility and
installation; the viewer references own application integration.

## Select A Compatible Set

For an existing app, start from its manifest/lockfile. For a new app or requested
upgrade, resolve the current public supported package set for its OS,
architecture, Python/native runtime and delivery path from upstream release and
package metadata. Do not select an older cohort just because its examples are
easier; report an unavailable or incompatible current set rather than silently
downgrading. Resolve OVRTX and OVStage together. Add UI, transport,
storage, or simulation packages only when the app uses them.

Check the selected release's metadata, migration notes, and relevant upstream
examples before implementation. Record resolved versions in the generated app's
manifest. A successful import or experimental ABI probe does not establish a
vendor-supported combination.

## Package And Documentation Sources

| Library | Package source | Upstream guidance |
|---|---|---|
| OVRTX | [PyPI](https://pypi.org/project/ovrtx/) | [OVRTX](https://github.com/nvidia-omniverse/ovrtx): renderer configuration, outputs, picking, native API, release notes |
| OVStage | [PyPI](https://pypi.org/project/ovstage/) | [OVStage](https://github.com/NVIDIA-Omniverse/ovstage): population, schemas, data plane, publication, attachment |
| ovstream | [PyPI](https://pypi.org/project/ovstream/) | [ovstream](https://github.com/NVIDIA-Omniverse/ovstream): server/client skills, native artifacts, frame submission, examples |
| ovui | [PyPI](https://pypi.org/project/ovui/) | [ovui](https://github.com/NVIDIA-Omniverse/ovui): supported Python/platforms, companion packages, widgets and presentation |
| ovstorage | [PyPI](https://pypi.org/project/ovstorage/) | [ovstorage](https://github.com/NVIDIA-Omniverse/ovstorage): `AGENTS.md`, skills, storage adapters, authentication, resolver/cache integration |
| OVPhysX | [PyPI](https://pypi.org/project/ovphysx/) | [OVPhysX](https://github.com/NVIDIA-Omniverse/PhysX/tree/main/ovphysx): supported OVStage pair, schemas, attachment, simulation and tensor APIs |
| ovnewton | [PyPI](https://pypi.org/project/ovnewton/) | Selected wheel's metadata and examples: bridge compatibility, lifecycle and output types |
| Newton | [PyPI](https://pypi.org/project/newton/) | [Newton](https://github.com/newton-physics/newton) and [documentation](https://newton-physics.github.io/newton/stable/): model building, importers, solvers and its agent skills |
| OVXR | [PyPI](https://pypi.org/project/ovxr/) for Python; matching native SDK through `ovxr.md` | [OVXR](https://github.com/NVIDIA-Omniverse/ovxr): README, library skills, release manifest, headers, bindings and minimal examples |
| CloudXR runtime SDK | Separately supplied runtime SDK | [CloudXR](https://docs.nvidia.com/cloudxr-sdk/): service hosting and opaque-channel contracts; see `ovxr.md` for the runtime-adapter route |
| Warp | `warp-lang` Python package | [Warp](https://nvidia.github.io/warp/stable/): arrays, CUDA streams, DLPack and kernels |

Use the upstream guidance matching the package being installed. Read its agent
instructions and relevant skills when available; fetch only the API/build detail
needed for the chosen integration. Reuse a known matching local checkout rather
than cloning the same dependency for every feature.

## OVRTX And OVStage

Create the renderer and attached stage with a compatible release pair and
register required schemas before population. Follow
`references/ovstage-ovrtx-integration/README.md` for the viewer-owned loop and
`references/ovstage-population/README.md` for loading and schema handling.

For native apps, match headers and libraries to the selected SDK. Keep USD
metadata/query workers isolated unless the selected in-process OpenUSD import
path is supported; choose the worker's USD package from its own requirements.

For an existing OVRTX application upgrade, discover the selected release's
upstream migration skills, including
[the 0.5-to-0.6 source/data upgrade guide](https://github.com/NVIDIA-Omniverse/ovrtx/blob/main/skills/update-0-5-to-0-6/SKILL.md)
when that transition applies. Use the matching tag/package guidance and its
linked OVStage contracts; upstream owns callback ABI, sensor, and raw-data
migrations. Review renderer-setting changes separately.

## ovstream

Use the selected release's Python installation path for Python producers and its
native SDK for C/C++/Rust consumers. Build native readers against matching headers
and runtime libraries. Refer to `ovstream.md` for the viewer's transport and frame
ownership contract; upstream owns packaging and loader setup.

### ov-web-rtc client

For a browser connected directly to a standalone ovstream server, use
`@nvidia/ov-web-rtc` and the compatible client version identified by the upstream
[WebRTC client example](https://github.com/NVIDIA-Omniverse/ovstream/tree/main/examples/webrtc_client).
Use the package's supported registry configuration:

```text
registry=https://registry.npmjs.org/
@nvidia:registry=https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/
```

Resolve and lock the client dependency in the generated app. After orchestration
provides an endpoint, the client uses the standalone server's Direct connection
configuration. For NVCF self-hosted deployment, read
`references/cloud-deployment/nvcf-self-hosted.md` before selecting the connection
model.

## ovui

Choose the Python interpreter and companion UI packages from the selected ovui
release's supported platform set before creating the environment. If no matching
wheel is available, check Python, OS, architecture, and companion package
requirements before changing application UI code. Follow `local-openusd-gpu.md`
and the chosen native delivery reference for viewer integration.

For a new app, begin at the upstream [ovui repository](https://github.com/NVIDIA-Omniverse/ovui).
Read its current `AGENTS.md` and discover the application, widget, UI API, and
styling skills it routes to. Consult the latest public release and package
metadata to resolve a supported set for the chosen platform. Use documentation
matching that resolved set when implementing; development-branch APIs may not
be present in published packages.

Resolve the UI, widget, and adapter distributions as one compatible set using
upstream dependency constraints. Check current public package indexes and
release assets for availability. Record the resolved set in the generated app's
manifest and verify it with `python -m pip check`.

The application and provider packages currently use `ovui-widgets-*` and
`ovui-data-adapters-*` distribution names, with `ovui_widgets` and
`ovui_data_adapters` Python namespaces; core UI imports use `omni.ui` and
`omni.ui_scene`. Confirm names, extras, supported interpreters, and provider
requirements in current upstream metadata before installing. For an OpenUSD
editor, use the provider's supported OpenUSD runtime and choose its standalone
extra only when that runtime is not already supplied. Resolve OVRTX compatibility
from that provider's contract. Resolve the native OVStage provider's runtime
cohort separately; support for one provider does not establish the other's ABI
compatibility. Follow the selected provider's bootstrap and import order.

Discover bundled agent guidance through the installed package's documented skill
entry points when available. Before adding panels or editing features, run the
upstream minimal application with a real USD scene and capture a nonblank OVRTX
frame through the intended delivery path.

For native C++ UI integration, discover the selected package's SDK contents,
build manifests, and CMake exports through the upstream native SDK documentation.
Confirm header/library identity, compiler and Python development requirements,
and ImGui ABI before linking. Preserve the documented package layout. Continue
using the OVRTX SDK for USD rendering and the viewer runtime for stage ownership.

## ovstorage

Follow its upstream `AGENTS.md` and selected storage skills for connection,
authentication, plugins, resolver behavior, and transfers. The viewer-side
contract is in `references/cloud-assets/README.md`: background preparation,
cache identity/freshness, progress, and handoff to the runtime owner. Storage
provider installation and SDK source builds belong to their upstream projects.

## ovxr

Begin at [NVIDIA-Omniverse/ovxr](https://github.com/NVIDIA-Omniverse/ovxr) and
follow [the OVXR library route](ovxr.md) for its current README, skill index,
release artifacts, examples and manifest. Resolve OVXR together with its matching
OVRTX and OVStage packages. Use matching distributed headers, bindings and native
artifacts, record build identity, and prove native, runtime and client readiness
separately. For Python, use the public [ovxr PyPI package](https://pypi.org/project/ovxr/)
and honor its declared OVRTX/OVStage requirements instead of upgrading those
libraries independently. Select supported Python/platform tags from the current
wheel metadata. CloudXR is a separately licensed OpenXR runtime.

## ovphysx

Read the selected release's upstream `AGENTS.md`, `SKILLS.md`, and
[basic workflow](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/ovphysx/skills/basic-workflow/SKILL.md).
For Python first-run setup, install both `physx-usd-schemas` and
`newton-usd-schemas` from public PyPI alongside OVPhysX, as required by the
upstream basic workflow:

```bash
python -m pip install ovphysx physx-usd-schemas newton-usd-schemas
```

The OVPhysX wheel and SDK do not bundle these schema resources. Follow upstream
[Physics Schemas](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/ovphysx/docs/physics_schemas.md)
for metadata-based resource discovery, native provisioning, and registration of
both PhysX and deformable resource directories before the first population or
export. Avoid importing the provider's stock-USD bindings in the viewer process.
The upstream `hello_world` sample always calls `ovphysx.newton_schema_root()`,
so it needs the Newton provider even when the application scene has no
`newton:*` attributes. After resolving the compatible package set, run:

```bash
python -m ovphysx.samples.python_samples.hello_world
```

Require successful simulation stepping and cleanup. For application code,
include Newton resources in the same registration call when the scene authors
`newton:*`; installing the provider alone does not register it. Record the
resolved provider versions with the app's runtime packages; resource discovery
alone does not establish semantic compatibility.

For an existing application, inspect the selected release's
[migration guide](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/ovphysx/docs/migrating_0_6_to_0_7.md)
before reusing lifecycle, session, configuration, or joint-authoring code. Rebuild
native consumers against the matching headers and SDK.

Resolve OVPhysX against its declared OVStage requirements. For in-process shared
stages, confirm the supported attachment and schema initialization sequence from
the selected release before attaching either consumer. If the required package
sets cannot share a process, use the explicit worker architecture in
`references/physics-simulation/README.md`, with independent environments and
copied pose/state messages. An ABI error is a compatibility problem to diagnose,
not a reason to guess another population API in the renderer process.

## ovnewton and Newton

Newton owns simulation; ovnewton bridges model state into OVStage. Begin with
its current [support guide](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/docs/site/support.md)
and [using-ovnewton skill](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/skills/using-ovnewton/SKILL.md),
then use the selected wheel's metadata and bundled examples. Resolve the bridge's
OVStage requirement and the examples' OVRTX requirements together with the app's
other consumers. Include the selected example's extras, schema providers, and
assets; account for USD dependencies introduced by optional Newton importers
when choosing process boundaries.

Recheck public releases before declaring the bridge unavailable for a runtime
cohort. If no compatible set exists, keep simulation controls unavailable and
explain the missing bridge; do not bypass requirements with `--no-deps` or widen
them locally. A thread does not isolate native dependencies. A worker with a
copied-state adapter needs its own implementation and validation.

Run the selected release's bounded simulation example, then validate an OVRTX
frame and the intended viewer interactions separately. Package compatibility or
a CPU-only simulation does not qualify GPU, viewer, or XR behavior. Use
`references/newton-physics/README.md` and
`references/realtime-sim-integration/README.md` for the viewer-owned simulation
clock, publication, and interaction path.

## Warp and CUDA compatibility

Resolve two separate constraints: the packages must accept each other's versions,
and each GPU library build must support the host GPU and NVIDIA driver. A
successful dependency resolution or import establishes neither GPU execution nor
cross-library interoperability.

Distinguish the host driver, a locally installed CUDA Toolkit, and the CUDA
runtime/toolkit used to build each wheel or SDK. The CUDA version displayed by
`nvidia-smi` describes driver capability; it does not identify the installed
Toolkit or the build used by Warp. Prebuilt GPU packages can bring their own
runtime libraries, so installing another local Toolkit does not repair an
unsupported wheel/driver combination. Follow each selected library's upstream
compatibility requirements; matching every CUDA version string is not a policy.

Select a Warp build compatible with the GPU and driver as well as every selected
package's declared version constraints. The default PyPI build can require a
newer CUDA driver than OVPhysX itself. Use the selected Warp release's
[compatibility guidance](https://nvidia.github.io/warp/stable/user_guide/compatibility.html)
for supported CUDA builds. Resolve optional extras and test tools with the app's
constraints; do not bypass incompatible bounds or add another USD runtime to the
viewer process. Record package/build identities and the driver in app evidence.

Verify CUDA device discovery, an allocation and small kernel with synchronization,
then the app's actual consumer path, such as physics reads or streaming buffer
conversion. Keep readiness evidence for renderer, simulation, and transport
separate. A passing CPU simulation or successful import does not validate GPU
execution, and a Warp kernel alone does not validate shared buffers or streams.

## Verify Setup

Run the selected upstream minimal example, then render a real frame through the
app's intended delivery path before adding features. Capture exact versions and
build identities in the app's validation evidence. See
`environment-validation.md` for environment checks and diagnostics.
