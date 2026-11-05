# Local Desktop, OpenUSD, And GPU Utilities

## Local Desktop Viewer Note

The base streaming Omniverse Realtime Viewer does not require `ovui`:
rendering is server-side `ovrtx` and browser delivery is `ovstream`. Browser DOM
overlays and native selection do not require it. When the user specifically
requests server-side Vulkan ovui overlays baked into the delivered pixels,
select [viewport overlays](../viewport-overlays/README.md) and verify transparent
headless export in the chosen ovui package.

For a selected local desktop Omniverse Realtime Viewer, read
`nvidia-runtime.md` for the latest-version `ovui` PyPI package guidance.
Keep all local UI packages from the same compatible package set. For ovui-owned skills,
widget samples, `ovui_widgets`, or headless overlay examples, also use the current
ovui repository pointer in `nvidia-runtime.md` and inspect that
repo's `skills/`, samples, and widget code. If local validation cannot complete
in the current environment, document the runtime requirement and continue
scaffolding the expected local Omniverse Realtime Viewer integration rather than
adding local install instructions here.

Minimal verification when `ovui` is already installed:

```bash
python3 -c "import omni.ui as ui; print('ovui OK')"
```

Common local-only failures:

- `ModuleNotFoundError: omni.ui`: local desktop UI packages are not installed in the active environment.
- `ImportError: libcudart.so.12`: ovui is installed, but the CUDA 12 runtime is not on the native library path; discover it beneath `$OVRTX_BIN_PATH/plugins` and retry the verification command with that directory prepended.
- Window does not open in CI or remote shells: no real display is available; use a configured X display or desktop session.
- UI frame does not resize with the OS window: app code did not configure the local window shell correctly; read `references/local-viewer`.

## usd-core / pxr

Purpose: direct USD queries for hierarchy, properties, variants, bounds, authored cameras, and metadata.

Package: PyPI package `usd-core`.

Install `usd-core` only in the selected USD query worker or a short-lived validation environment. Resolve a release compatible with the selected runtime package set by checking the current upstream runtime documentation, release notes, and package metadata:

```bash
python3 -m pip install --upgrade usd-core
```

Why process isolation is required:

- `ovrtx` bundles its own USD C++ libraries.
- `usd-core` ships a separate USD runtime.
- Loading both USD runtimes in one process can produce linker-level conflicts, duplicate registry state, duplicate debug symbols, and plugin/type alias errors.

Default process contract:

1. In the main renderer process, set `OVRTX_SKIP_USD_CHECK=1` before imports.
2. Construct the viewer runtime that owns `ovrtx.Renderer` and any live OVStage
   state.
3. Do not import `pxr` in that renderer/runtime process unless the exact wheel
   set and import order have been verified for the target platform.
4. Run normal `pxr` work in an app-generated subprocess, such as
   `server/pxr_worker.py`; implement it using the
   [fallback worker protocol](../stage-hierarchy/fallback-worker-protocol.md).
5. Communicate with the subprocess through JSON, files, pipes, or another explicit IPC boundary.

Risk boundary:

- Treat `pxr`/`usd-core` as a second USD runtime until proven otherwise.
- Treat OVPhysX USD population similarly: bounded OVPhysX worker path belongs in a
  bounded physics worker unless `PhysX.attach_ovstage(stage, read_ordinal=...)` is verified against
  the exact installed OVStage/OVPhysX ABI.
- The parent viewer should own OVRTX, OVStage stage lifetime, ordinals, and
  renderer publication; workers should return DTOs such as hierarchy rows,
  properties, pose matrices, and diagnostics.

Verify `pxr` only in the intended query process or subprocess:

```bash
python3 -c "from pxr import Usd, UsdGeom, Sdf, Gf; print('pxr OK')"
```

Verify package metadata:

```bash
python3 -m pip show usd-core
```

Common failure modes:

- `_tf` import failure: Python version, wheel tag, platform, or shared library resolution mismatch.
- `TfType::AddAlias` schema conflict: the installed `usd-core` release is not compatible with the selected runtime packages, or conflicting USD runtimes are loaded.
- Duplicate USD registry or debug symbol errors with ovrtx: `pxr` was imported
  in the renderer process, or another USD-populating runtime entered the same
  process; move queries or physics population to a subprocess.
- Slow hierarchy queries: app logic is traversing too much USD on the UI/render path; read `references/stage-hierarchy`.

## warp-lang

Purpose: CUDA buffer operations and GPU utility kernels, especially RGBA to BGRA conversion before streaming.

Resolve the `warp-lang` package and its CUDA build through
[Warp and CUDA compatibility](nvidia-runtime.md#warp-and-cuda-compatibility)
before installation. Use the selected release's supported artifacts and the
combined application constraints.

Verify the selected build in the intended environment:

```bash
python3 -c "import warp as wp; wp.init(); print('warp', wp.__version__); print(wp.get_devices()); assert wp.get_cuda_devices(), 'Warp has no usable CUDA device'"
```

Then run an allocation and small synchronized CUDA kernel using that release's
upstream example, followed by the app's actual conversion or physics path.

Common failure modes:

- `ModuleNotFoundError: warp`: package name is `warp-lang`, import name is `warp`.
- No CUDA device: inspect initialization warnings, the selected wheel's driver/GPU requirements, and container GPU access. CPU fallback can leave import and CPU checks passing.
- CUDA conversion code fails after a working kernel: verify the app passes valid CUDA buffers, coordinates streams, and keeps lifetimes stable.

## numpy

Purpose: matrices, camera math, CPU frame copies, serialized values, and general numeric utilities.

Package: PyPI package.

Install:

```bash
python3 -m pip install numpy
```

Verify import:

```bash
python3 -c "import numpy as np; print('numpy OK', np.__version__)"
```

Common failure modes:

- ABI errors after upgrading packages: recreate the virtual environment or reinstall compiled packages.
- Camera math produces invalid transforms: this is usually app logic, not install; validate finite arrays and matrix layout through `references/camera-controls`.

## Legacy ModernGL Headless Line Adapter

Use this section only when intentionally retaining an existing ModernGL
readback/upload adapter as a migration oracle or separately measured
compatibility implementation. The default explicit feature/hidden-line path is
the OVRTX SPG graph in
[line compositor delivery](../stylized-visualization/line-compositor-delivery.md),
which requires no GL context or host framebuffer readback.

Purpose: optional public Python binding for that legacy standalone OpenGL
feature-line pass. The installation, preflight and fallback guidance below
applies only to the selected legacy adapter.

Package: PyPI extra `moderngl[headless]`, which includes the headless context
provider. Use the app manifest or lockfile for its selected version; do not add
an untracked runtime install step.

Before enabling the selected legacy adapter, validate all of the following in
the target server environment:

- a standalone EGL context can be created without an X display;
- its reported version is OpenGL 3.3 or newer;
- an RGBA8 framebuffer with depth attachment can be created;
- four-component framebuffer readback returns the expected byte count and
  bottom-left origin;
- context, framebuffer, program, and buffer cleanup succeeds on the owner
  render-loop thread.

Use the current [ModernGL context documentation](https://moderngl.readthedocs.io/en/stable/reference/context.html)
and [framebuffer documentation](https://moderngl.readthedocs.io/en/stable/reference/framebuffer.html)
for the installed API. Do not assume context sharing with OVRTX; this legacy
Python adapter uses explicit host readback and CUDA upload.

Common failure modes:

- EGL context creation fails: the container or host lacks a compatible EGL/OpenGL
  runtime; keep the OVRTX stream live and report that the legacy line adapter
  is unavailable. Follow the selected style
  [fallback contract](../stylized-visualization/line-compositor-delivery.md#revisions-fallback-and-acceptance);
  an EGL failure does not establish an SPG limitation.
- A context is created through X11 in a headless deployment: select the EGL
  backend explicitly and repeat validation without a display.
- Readback is vertically inverted: flip the overlay exactly once before
  compositing into the top-left-origin stream buffer.
- Frame rate collapses during camera motion: measure line refresh separately,
  lower its cadence or disable explicit lines, and retain base video throughput.
