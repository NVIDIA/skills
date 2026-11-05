# Dependencies

## Triggers

Use for viewer setup, a library upgrade, native SDK builds, or an installation
or compatibility failure. Choose the delivery path first so setup includes only
the libraries the app needs.

## Resolve Once For The App

Read `nvidia-runtime.md` for package sources, supported combination selection,
and upstream API/build guidance. Preserve an existing manifest unless an upgrade
is requested; write newly resolved versions into the app's manifest/lockfile.
Keep package locations and acquisition instructions here rather than repeating
them in feature references.

| Need | Read next |
|---|---|
| Baseline environment setup | `quick-setup.md` |
| Renderer setup | `ovrtx.md` |
| Streaming/native transport contract | `ovstream.md` |
| Browser UI dependencies | `frontend.md` |
| Electron SHM setup | `electron-shm.md` |
| OVXR SDK setup and upstream library guidance | `ovxr.md` |
| Skia and Yoga for spatial UI | `skia-yoga.md` |
| Python UI, OpenUSD workers, GPU arrays | `local-openusd-gpu.md` |
| Environment checks and diagnostics | `environment-validation.md` |
| Windows-specific setup | `references/windows-native-setup/README.md` |

Use the selected release's upstream guidance for exact package/API requirements.
An installed package or cached checkout is useful when it matches that release;
it is not evidence of latest compatibility. Revisit upstream when upgrading or
when an API fails, rather than refreshing every dependency on every prompt.
When an API remains uncertain, inspect the matching installed signatures or
type declarations and retain a focused probe result with the app's evidence.

Before expanding the app, verify the selected renderer/stage combination and
one displayed or decoded frame in its real shell. Keep setup diagnostics and
resolved package identities in application evidence, not in feature skills.

## CAE/CFD Visualization Libraries

This is the **single acquisition source** for the two domain libraries the
`cae-cfd-visualization` reference family depends on. Every doc in that family
points here by role instead of repeating package locations inline. Refer to them
by role, not by URL, so the prose survives a rename or a repository move:

- **the `warp-simdata` GPU operator library** — NVIDIA Warp kernels that build an
  in-memory `Dataset` from renderable arrays and run the visualization operators
  (`element_faces` / `iso_surface` / `slice` / `streamlines` / points /
  voxelization), with CPU and CUDA execution. The CAE family uses its dataset,
  field and operator APIs.
- **the `cae-openusd-plugins` USD reader/schema library** — a read-only
  `SdfFileFormat` plugin plus OmniSci (`Cae*`/`OmniSci*`) schemas that expose
  CGNS / OpenFOAM / VTK / EnSight / FLASH files as composed USD prims with
  data-on-demand field arrays. Use it with Warp SimData's USD adapters for native
  file ingestion; raw-array workflows can use Warp SimData without OpenUSD.

> **These libraries are public on GitHub.** This section is the single source of
> truth for their acquisition — the family docs point here rather than repeating URLs.

| Library (role) | Acquisition | Guidance |
|---|---|---|
| `warp-simdata` (GPU operator library) | [NVIDIA/warp-simdata](https://github.com/NVIDIA/warp-simdata) | Discover agent guidance and skills if present; otherwise start with its README. Follow the selected release's installation guidance, changelog, and task-specific API docs. For input ownership and cache lifetime, read [datasets and fields](https://github.com/NVIDIA/warp-simdata/blob/main/docs/DatasetsAndFields.md); for native-file adapters, read the [OpenUSD API](https://github.com/NVIDIA/warp-simdata/blob/main/docs/USDAPI.md). Verify these default-branch entry points against the app's resolved release. |
| `cae-openusd-plugins` (USD reader/schema library) | [NVIDIA-Omniverse/cae-openusd-plugins](https://github.com/NVIDIA-Omniverse/cae-openusd-plugins) | Before setup, read its current `AGENTS.md`, `skills/`, README, build/install, plugin-registration, and compatibility guidance. Use its [usage guide](https://github.com/NVIDIA-Omniverse/cae-openusd-plugins/blob/main/docs/usage.md) and [time mapping](https://github.com/NVIDIA-Omniverse/cae-openusd-plugins/blob/main/docs/file_formats/time.md) for registration and native-file reads; see `cae-cfd-visualization/cae-data-ingestion.md` for viewer integration. |

Rules:

- Do not duplicate these acquisition URLs or version pins in the
  `cae-cfd-visualization` family docs. Point back to this section.
- This section is the single route to current upstream setup guidance. Do not
  duplicate acquisition commands, version pins, build flags, or plugin paths in
  this package.
- Let each upstream repository define its supported CUDA/Warp or USD/runtime
  compatibility set; validate the selected combination before building an app.

## USD BRep: Public Solid Modeling Dependency

Use [NVIDIA-Omniverse/usd-brep](https://github.com/NVIDIA-Omniverse/usd-brep)
for exact BRep modeling, tessellation, and BrepArray USD interchange. Before
setup or implementation, read the selected revision's `AGENTS.md`, discover its
repo-local skills (currently `.agents/skills/`), and follow its operation router
and docs. Start with its `README.md` and `tools/pyproject/README.md` for public
artifacts, platform support, native prerequisites, and compatibility. Upstream
owns installation and build procedures; do not reuse early-access archives,
private solidmodeling checkouts, Packman paths, or fixed prerelease versions.

The public distribution is [`usd-brep` on PyPI](https://pypi.org/project/usd-brep/),
imported as `usd_brep`; operation groups
include `primitives`, `booleans`, `fillets`, and `usd`. `_omni_solid` is the
source-checkout API, not a drop-in spelling for the packaged API. Resolve the
artifact against current upstream guidance and record its version and platform
in the app manifest. Verify exports against that artifact's stubs before coding.

The reviewed package requires CPython 3.12 and bundles OpenUSD without importing
`pxr`. It exchanges stages as files. Loading a second build of the same bundled
OpenUSD version is unsupported; different-version coexistence demonstrated by
upstream is not proof of compatibility with OVRTX/OVStage. Use a separate worker
when interpreter or native runtime compatibility is unverified. Do not add a
separate OpenUSD install to the modeling environment merely to import this wheel.
Follow upstream for validation extras and Windows runtime prerequisites.

For styles with explicit feature/hidden lines, use the OVRTX SPG path in
[SPG processing](../spg-processing/README.md) and
[line compositor delivery](../stylized-visualization/line-compositor-delivery.md).
This path needs no ModernGL, EGL context, or host framebuffer readback. Only
when intentionally retaining an existing legacy ModernGL adapter, read the
optional dependency and preflight in `local-openusd-gpu.md`.
