# Windows Native Setup

## Triggers

Use this skill for Windows, native Windows, WSL2, ERROR_INCOMPATIBLE_DRIVER, NVML, DLL load failed, or OpenUSD import compatibility issues.

Run natively on Windows 10/11. Do not use WSL2; ovrtx needs direct Vulkan/NVML GPU access and WSL2 commonly fails with `ERROR_INCOMPATIBLE_DRIVER` or `NVML_ERROR_DRIVER_NOT_LOADED`.

## Prerequisites

- NVIDIA RTX GPU, Turing or newer.
- NVIDIA RTX-capable GPU with a compatible driver.
- A CUDA 12 runtime is required only when the selected ovui package or capability requires it; follow the current upstream ovui guidance rather than interpreting the driver version as a CUDA toolkit version.
- Python version matching the latest selected runtime wheels. Check the current
  `ovui` package files from `references/dependencies` and create the virtual
  environment with a supported Python version unless the project manifest pins a
  different compatible package set.
- Node.js 20+, npm 10+, Git.

Additional prerequisites may be required by the selected local desktop `ovui`
package or dependency build instructions:

- Visual Studio Build Tools with the MSVC C++ x64 toolchain.
- `vswhere.exe` available from Visual Studio Installer.
- Ninja installed in the active venv or visible to pip build isolation.
- Vulkan SDK when required by the current `ovui` package or dependency
  instructions.

## Install

Read `references/dependencies` first. Its `nvidia-runtime.md` file owns
current acquisition details for `ovrtx`, `ovstream`, `ovui`, and the
`ov-web-rtc` browser client; this Windows guide should not repeat release URLs, wheel
names, or artifact locations.

Start from the root of the generated viewer project. First resolve the current
`ovui` Python requirement from the `NVIDIA-Omniverse/ovui` `README.md`,
`AGENTS.md`, relevant `skills/omniverse-ui-*` references, and package metadata;
then substitute that interpreter selector in the venv command:

```powershell
cd C:\path\to\generated-viewer
py -<ovui-supported-python> -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `pip install ovui` reports `No matching distribution found`, re-check the
current upstream `ovui` docs and package files before changing install indexes
or generated app code.

Install NVIDIA runtimes using `references/dependencies`, then install supporting
packages:

```powershell
pip install warp-lang
pip install --upgrade usd-core
if (Test-Path server\requirements.txt) { pip install -r server\requirements.txt }
```

Resolve the `usd-core` release from the current upstream runtime documentation, release notes, and package metadata; do not copy a fixed version into a generated app. Keep `pxr` queries in `pxr_worker.py` unless the selected package set and in-process import path have been verified on Windows. If `TfType::AddAlias` or related USD registry conflicts occur, re-check the selected package set and retain process isolation.

## App-Local Desktop Preflight

Use this preflight for an Electron viewer that starts a selected Python sidecar,
or any generated desktop app that launches its project-local Python runtime. Run
it from the app root before starting the desktop shell; it validates the actual
interpreter the launcher will use, not whichever `python` happens to be on
`PATH`.

```powershell
$python = Join-Path $PWD '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Missing app-local Python: $python" }
& $python -c "import sys; print(sys.executable); print(sys.version)"
& $python -m pip check
if ($LASTEXITCODE) { throw "Python dependency check failed" }
if (-not (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue)) {
  throw 'Microsoft Visual C++ x64 runtime is not detected'
}
```

Then run the sidecar through that exact `$python` path and preserve its startup
output with the app's validation evidence. This preflight does not claim that a
particular SHM addon, DLL layout, or plugin configuration is portable across
Windows packaging methods; verify those against the selected runtime and app
before packaging.


## Local ovui On Windows

Use this section for native local desktop viewers. Do not follow the WebRTC run
steps below unless the generated app is a browser-streamed viewer.

Use `references/dependencies` for the current `ovui` PyPI package guidance.
Resolve the interpreter from the current upstream `ovui` repo docs and package
metadata before creating the virtual environment. Keep the base `ovui` package
and companion packages on one compatible package set.

Install `ovrtx` and `ovui` with the commands from `references/dependencies`,
then install local support packages selected by the app:

```powershell
pip install --upgrade usd-core numpy
```

Verify the UI import path before building app logic:

```powershell
python -c "import omni.ui as ui; import omni.ui_scene; print('ok')"
```

If the installed ovui wheel reports `DLL load failed while importing _ui`, add the
CUDA 12 runtime directory before importing `omni.ui`. Keep the returned handle alive
for the lifetime of the process:

```python
import os
from pathlib import Path

_dll_handles = []

def prepare_ovui_native_runtime():
    roots = []
    if os.environ.get("OVRTX_BIN_PATH"):
        roots.append(Path(os.environ["OVRTX_BIN_PATH"]) / "plugins")
    import ovrtx
    roots.append(Path(ovrtx.__file__).resolve().parent / "bin" / "plugins")
    try:
        import ovstream
        roots.append(Path(ovstream.__file__).resolve().parent)
    except ImportError:
        pass
    for root in roots:
        for dll in root.rglob("cudart64_12.dll"):
            _dll_handles.append(os.add_dll_directory(str(dll.parent)))
            return
    raise RuntimeError("CUDA 12 runtime DLL not found for ovui")

prepare_ovui_native_runtime()
import omni.ui as ui
```

Run local-only apps through their local module entry point, for example:

```powershell
$env:OVRTX_SKIP_USD_CHECK = "1"
python -m local_app
```

If the local app needs USD hierarchy, variants, or property queries and direct
`pxr` imports conflict with ovrtx, use the same `pxr_worker.py` subprocess
strategy documented below. Local `ovui` apps do not need `ovstream`, Vite,
`@nvidia/ov-web-rtc`, media ports, or browser WebRTC config.

If `ovui-data-adapters` reports that no `setup.py` or `pyproject.toml` exists,
use a package set that includes matching package metadata. Do not patch
packaging metadata from this skill.

If PowerShell launches a `.bat` file that uses `for /f "usebackq" ... in (\`...\`)` around `python -c "..."`, quoting can be mangled before `cmd.exe` receives it. Use a small helper `.py` script for Python probes inside batch loops.

## Stage Syntax Check

If using `samples/stage01.usda`, `clippingRange` must be:

```usda
float2 clippingRange = (0.1, 10000)
```

Not separate `float clippingRange.near` and `.far` attributes.

## Streaming Viewers Only: Run

Use this section only for browser-streamed viewers backed by `ovstream` and
`@nvidia/ov-web-rtc`. Local `ovui` viewers use `python -m local_app` and skip the
frontend/server split.

```powershell
cd frontend
npm install
npm run dev
```

```powershell
cd server
python ov_web_viewer_server.py --stage ..\samples\stage01.usda --port 49100
```

Open:

```text
http://localhost:3000?server=127.0.0.1&signalingport=49100
```

First launch can spend 5 to 10 minutes compiling RTX shaders after `Stage loaded successfully`, with many UJITSO material warnings. Do not kill it; cached shaders make later launches faster.

## Architecture

Windows keeps `pxr` and `ovrtx` in separate processes:

```text
ov_web_viewer_server.py
  ovrtx.Renderer
  ovstream.Server
  PxrWorkerClient -> pxr_worker.py -> pxr.Usd
```

The main process never imports `pxr`; USD hierarchy, variants, and property queries use newline-delimited JSON over stdin/stdout.


ovrtx also needs inline root/session data with a camera, RenderProduct, RenderVar, and RenderSettings. Missing this causes `Unable to find RenderProduct prim`.
## Streaming Viewers Only: WebRTC Rules

Server:

```python
config.webrtc_signal_port = 49100
config.webrtc_public_ip = "127.0.0.1"
```


Frontend:
```typescript
return { server, signalingPort }; // no mediaServer/mediaPort
```

The client must use `server`, not `signalingServer`, and must not set media fields. Callback registration must happen before `server.start()`. Data-channel messages may arrive wrapped in `{messageType,messageRecipient,data}` and must be unwrapped. On connect, push `openStageResult` and root `getChildrenResult` after about 300 ms so the frontend sees already-loaded state.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR_INCOMPATIBLE_DRIVER` / `NVML_ERROR_DRIVER_NOT_LOADED` | run native Windows, not WSL2 |
| `No matching distribution found for ovui` | re-check the current `ovui` repo docs/package files and recreate the venv with a supported interpreter or manifest-pinned compatible package set |
| `DLL load failed while importing _ui` | keep the `os.add_dll_directory()` handle alive and add the discovered CUDA 12 runtime directory before importing `omni.ui`; do not rely on `LD_LIBRARY_PATH` or `PATH` on Windows |
| `_tf` DLL import failure | keep pxr in worker subprocess; rerun the app-local desktop preflight before changing imports |
| Electron sidecar starts a different Python or fails before `shmReady` | launch the project `.venv\Scripts\python.exe`, run App-Local Desktop Preflight, then use `electron-shm-viewer` lifecycle diagnostics |
| `TfType::AddAlias` conflict | re-check the selected compatible runtime package set and keep `pxr` isolated in its worker |
| `OSError: cannot load library ovstream` | remove wrong `OVSTREAM_LIB_PATH`; use the current `ovstream` package from `references/dependencies` |
| `cannot import name 'VIEWPORT_CAMERA_POSE_SOURCE'` | install local UI packages from the same package set |
| `Neither 'setup.py' nor 'pyproject.toml' found` under `ovui-data-adapters` | use an `ovui` package set that includes matching package metadata |
| Native UI package requires a compiler toolchain | follow the current `ovui` package/build instructions |
| `TypeError: a coroutine was expected` from `ui.run` | pass an async render loop coroutine, not a plain callback |
| stuck "Loading stage..." | remove `mediaServer`/`mediaPort` |
| `Previous session is already running` | reduce reconnects, add delay |
| `VideoEncoder was not deinitialized` | non-fatal shutdown-order warning |
| `Unable to find RenderProduct prim` | use `stage-loading` wrapper/session stage |
| red/blue swapped | apply RGBA-to-BGRA warp swap before streaming |

See also: `electron-shm-viewer` for selected Python sidecar lifecycle,
`troubleshooting` for boundary triage, `streaming-server`, `streaming-client`,
`streaming-lifecycle`, `stage-hierarchy`, and `stage-loading`.
