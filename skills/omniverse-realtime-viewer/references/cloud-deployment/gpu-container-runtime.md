# GPU Container Runtime Preflight

## Triggers

Use this reference when an OVRTX/ovstream viewer is packaged for Docker,
Kubernetes, NVCF, CI, or another GPU container runtime, especially when it
imports successfully but fails to render, encode, or become ready.

## Runtime Requirements

The image and host must agree on the supported OS, NVIDIA driver, CUDA runtime,
Python version, and OVRTX/ovstream package set. Select versions from current
dependency guidance and upstream deployment requirements, not this document.

For a streamed viewer, verify that the platform provides an NVIDIA GPU,
compute and utility access, graphics access for headless rendering, and
video/NVENC access for encoding. The exact runtime class, device plugin,
security context, display strategy, and capability declaration belong to the
target platform; `nvidia-smi` alone does not prove graphics or encoder access.

## Preflight Sequence

Run these checks in the same image and runtime class used for deployment:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
python3 -c "import ovrtx; print('ovrtx import OK')"
python3 -c "import ovstream; print('ovstream import OK')"
```

Then construct the renderer and initialize ovstream using the selected viewer
recipe. Confirm that native libraries resolve, ovstream uses its full native
library, the renderer produces a frame without a browser, the encoder
initializes, readiness is reported, and shutdown leaves no GPU-held process.

Keep `OVRTX_SKIP_USD_CHECK=1` and native library setup in centralized dependency
guidance or the generated launch wrapper, not scattered manifests.

## USD And Cache Isolation

Keep `pxr`/`usd-core` work in the worker process prescribed by the selected
viewer references unless the renderer-process path is verified. Do not add an
incompatible USD runtime just to make a hierarchy query work.

Treat shader and CUDA caches as disposable, GPU-family-specific runtime data.
Persist or prewarm them only when the platform supports that safely; never use
a cache from an incompatible GPU family.

## Diagnostic Table

| Symptom | First checks |
| --- | --- |
| GPU is not visible | Device allocation, runtime class/device plugin, node selection, host driver |
| OVRTX imports but rendering fails | Graphics libraries, display strategy, native plugin paths, renderer logs |
| Stream starts but no video | Video/NVENC capability, encoder, ICE/media path, browser diagnostics |
| Native import or symbol error | Package provenance, full native library, architecture, library paths |
| First frame never arrives | Stage path, shader warmup, render-loop ownership, readiness gate |
| Shutdown crashes or hangs | Earlier native failure, concurrent mutation, stale process, incompatible cache |

Do not paper over a container failure with a browser-renderer fallback. Report
the missing GPU or runtime capability explicitly.
