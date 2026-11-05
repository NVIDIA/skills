# Environment And Validation

## Environment Variable Summary

Set before any ovrtx work:

```bash
export OVRTX_SKIP_USD_CHECK=1
```

Set when renderer plugins or MDL libraries do not resolve:

```bash
export OVRTX_BIN_PATH="$(python3 -c 'import ovrtx, os; print(os.path.join(os.path.dirname(ovrtx.__file__), "bin"))')"
export OVRTX_CUDA_RUNTIME_DIR="$(find "$OVRTX_BIN_PATH/plugins" -type f -name 'libcudart.so.12' -printf '%h\n' -quit)"
export LD_LIBRARY_PATH="${OVRTX_CUDA_RUNTIME_DIR:+$OVRTX_CUDA_RUNTIME_DIR:}$OVRTX_BIN_PATH/plugins${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Use `nvidia-runtime.md` for current NVIDIA runtime acquisition
locations. Do not keep separate `ovstream` source URLs or cache controls here.

Set only when an ovstream install cannot locate SDK libraries automatically:

```bash
export OVSTREAM_LIB_PATH=/absolute/path/to/ovstream/native/libs
```

Set for a selected desktop UI under a known display; headless ovui overlays
follow their own upstream initialization and export contract:

```bash
export DISPLAY=${DISPLAY:-:0}
```

Set when the default home cache is not writable:

```bash
export XDG_CACHE_HOME="$PWD/.cache"
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export __GL_SHADER_DISK_CACHE_PATH="$PWD/.cache/gl"
export npm_config_cache="$PWD/.cache/npm"
```

## Verification Checklist

Run these checks after installation:

1. Confirm GPU and driver are visible with `nvidia-smi`.
2. Confirm Python is the expected interpreter inside the virtual environment.
3. Confirm `OVRTX_SKIP_USD_CHECK=1` is set before renderer imports.
4. Confirm `ovrtx` imports and renderer construction succeeds on the target GPU.
5. Confirm `OVRTX_BIN_PATH` resolves to the installed ovrtx `bin` directory when plugin resolution fails.
6. Confirm `pxr` imports only in the selected query subprocess unless the exact
   renderer-process import path has been verified for the selected wheels.
7. If a `pxr` worker is generated, confirm its `usd-core` release is compatible with the selected runtime package set and that it imports in the worker environment.
8. If OVStage is selected, confirm `ovstage` imports and the app can create or
   attach the runtime stage path required by the selected viewer architecture.
9. If OVPhysX is selected, follow `nvidia-runtime.md` → ovphysx for both schema
   providers and the unchanged `hello_world` smoke test in its intended
   environment. Then validate the selected shared-stage or worker integration;
   a passing upstream CPU example does not establish GPU or viewer readiness.
10. Confirm `ovstream.initialize()` and `ovstream.shutdown()` work for streaming Omniverse Realtime Viewers.
11. For Warp GPU paths, follow `nvidia-runtime.md` → Warp and CUDA compatibility:
    verify a usable CUDA device, a small allocation/kernel with synchronization,
    and the intended conversion or physics path. Record build/driver identity;
    import or CPU-only success is insufficient.
12. Confirm `@nvidia/ov-web-rtc` appears in `npm ls` when building a browser
    streaming frontend.
13. If the app imports local viewer UI components, confirm
    `frontend/src/viewer-ui/` exists and exports the referenced components and
    `ViewerBackend` types.
14. If the app uses ovui, follow its selected package's bootstrap and import
    checks for desktop UI or server-side headless overlays. For overlays, also
    verify transparent headless export through `viewport-overlays`; an import
    alone does not validate composition. On Linux, if the import reports
    `libcudart.so.12`, discover the CUDA 12 runtime beneath
    `$OVRTX_BIN_PATH/plugins` and retry with that directory on the subprocess
    library path.
15. Confirm `.cache/` is writable when running in containers, CI, or shared environments.

For generated browser-streamed viewers, add these readiness checks before
reporting completion:

1. Start the Python server from the generated run wrapper or equivalent command.
2. Wait for `/healthz` to return `200 ok`; if it does not, capture the server log.
3. Confirm the log contains a first-frame message after a valid BGRA8 CUDA
   mapping or completed required conversion into the app-owned BGRA stream
   buffer. RGBA8 outputs require RGBA-to-BGRA conversion; already-BGRA8
   outputs must not be swizzled again.
4. Run the frontend build and browser smoke check only after the server runtime
   proof above has passed or has produced a concrete failure report.

## Failure Mode Index

- Wrong Python package index: a runtime install fails or finds no matching
  distribution. Use the selected package source and compatible wheel set in
  [runtime dependencies](nvidia-runtime.md); check interpreter/platform tags
  and resolver output before changing indexes.
- Incorrect ovstream acquisition: use the current PyPI package source in `nvidia-runtime.md`.
- Platform mismatch: wheel is unavailable or native import fails. Confirm OS, architecture, Python version, GPU driver, and package wheel tags.
- Import order issue: USD registry, `_tf`, duplicate debug symbol, or MDL
  resolver errors. Set `OVRTX_SKIP_USD_CHECK` and follow the selected
  [rendering import discipline](../ovrtx-rendering/README.md#import-order).
  Attached OVStage apps register OVRTX schema paths before constructing or
  populating the stage; isolate unverified `pxr` consumers in a subprocess.
- OVStage/OVPhysX bridge mismatch: missing `ovstage_*` bridge symbols during
  `PhysX.attach_ovstage(stage, read_ordinal=...)`. Use the bounded OVPhysX worker path instead of
  calling OVPhysX in the parent viewer process.
- Native library path issue: `CRenderApi`, MDL, ovstream native, or plugin load failures. Set `OVRTX_BIN_PATH`, dynamic library path, or `OVSTREAM_LIB_PATH` as appropriate.
- GPU access issue: renderer or streaming initialization fails despite successful imports. Verify `nvidia-smi`, container GPU devices, driver support, RTX cores, and NVENC.
- Display issue: local desktop UI cannot open a window. Verify `DISPLAY` and X server availability.
- Cache permission issue: Warp, CUDA shader caching, GL shader caching, or npm fails under `~/.cache`. Set project-local cache paths and ignore `.cache/`.
- Frontend registry issue: npm cannot resolve the NVIDIA package. Check package
  spelling, the `@nvidia` registry in your npm configuration, lockfile, and proxy
  configuration.
- Local viewer UI issue: TypeScript cannot resolve `ViewerBackend`, `StageTree`,
  `Inspector`, or related local viewer UI imports. Generate
  `frontend/src/viewer-ui/` from `viewer-backend-interface`, or update imports to the
  app's actual local module path.

See also: `ovrtx-rendering`, `local-viewer`, `streaming-server`, `streaming-client`, `stage-hierarchy`, and `windows-native-setup`.
