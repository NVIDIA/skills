# ovrtx Dependency

## ovrtx

Purpose: NVIDIA RTX renderer used by local and streaming Omniverse Realtime Viewers.

Read `nvidia-runtime.md` for the latest-version acquisition command. This file
documents renderer environment and validation behavior.

For ovrtx-owned skills, renderer samples, Python/C API examples, stage
composition examples, render-var/AOV behavior, picking/selection examples, or
release-specific behavior, read `nvidia-runtime.md` for the current
ovrtx repository pointer and inspect that repo's `skills/`, samples, and
release notes.

For current OVStage-backed viewer architectures, verify the OVRTX/OVStage attach
or update path from the supplemental docs before copying older renderer-owned
stage/query/write examples. The parent viewer runtime should own OVRTX, OVStage
stage lifetime, ordinals, write floors, and renderer publication.

Install through the project server requirements when available:

```bash
python3 -m pip install -r server/requirements.txt
```

If a project manifest pins an exact `ovrtx` version, keep that pin. Otherwise,
use the latest available package from `nvidia-runtime.md`.

Verify import:

```bash
OVRTX_SKIP_USD_CHECK=1 python3 -c "import ovrtx; print('ovrtx OK', getattr(ovrtx, '__version__', 'version unavailable'))"
```

Resolve the ovrtx `bin` directory:

```bash
OVRTX_SKIP_USD_CHECK=1 python3 -c "import ovrtx, os; print(os.path.join(os.path.dirname(ovrtx.__file__), 'bin'))"
```

Set renderer environment variables:

```bash
export OVRTX_SKIP_USD_CHECK=1
export OVRTX_BIN_PATH="$(python3 -c 'import ovrtx, os; print(os.path.join(os.path.dirname(ovrtx.__file__), "bin"))')"
export OVRTX_CUDA_RUNTIME_DIR="$(find "$OVRTX_BIN_PATH/plugins" -type f -name 'libcudart.so.12' -printf '%h\n' -quit)"
export LD_LIBRARY_PATH="${OVRTX_CUDA_RUNTIME_DIR:+$OVRTX_CUDA_RUNTIME_DIR:}$OVRTX_BIN_PATH/plugins${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Verify renderer construction:

```bash
OVRTX_SKIP_USD_CHECK=1 python3 -c "from ovrtx import Renderer, RendererConfig; r=Renderer(config=RendererConfig(sync_mode=True, active_cuda_gpus='0')); print('renderer OK', getattr(r, 'version', 'version unavailable'))"
```

Common failure modes:

- `No matching distribution found for ovrtx`: wrong package index, unsupported platform, unsupported Python version, or no wheel for the environment.
- `usd-core detected`: set `OVRTX_SKIP_USD_CHECK=1` before any ovrtx import and follow the `pxr` subprocess contract.
- `CRenderApi not found`: set `OVRTX_BIN_PATH` and put ovrtx plugin libraries on the dynamic library path.
- Magenta materials: `OVRTX_BIN_PATH` or plugin library path is missing, so MDL libraries cannot resolve.
- Duplicate `SDF_ASSET` debug symbol errors: two USD builds are being loaded; isolate `pxr` queries in a subprocess.
- OVStage/OVPhysX bridge or attach errors: verify the selected package set and
  keep physics USD population in a bounded worker unless the exact ABI is known
  good.
- Stale renderer hangs after a crash: inspect `nvidia-smi` and terminate only stale Python Omniverse Realtime Viewer processes that still hold GPU state.
