<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
-->

# Running on Windows (and cross-platform)

This skill runs on **Windows with Docker Desktop** with **no host packages**: every compute step runs
**inside the DeepStream Linux container** via `docker run`. The host only needs `docker` + the NVIDIA
driver. The single exception is the host-side install (copying the skill into `<project>\.claude\skills\`),
which cannot run in a container — so the skill ships exactly one PowerShell script, **`install.ps1`**,
the twin of `install.sh`. There are no `.ps1` duplicates of the compute scripts.

## Why this works
The ONNX export, TensorRT engine build, custom nvinfer parser compile, DeepStream run, and PDF
report all execute **inside** `nvcr.io/nvidia/deepstream:9.1-triton-multiarch` (the venv
`build/.venv_optimum` + `wkhtmltopdf` are installed into that container by `setup.sh`). The container
is the portability layer. The only per-shell difference is the `docker run` **bind-mount token**.

## Prerequisites (Windows)
1. **Docker Desktop** with the **WSL2 backend enabled** (Settings → General → *Use the WSL 2 based
   engine*) — required for GPU (`--gpus all` works only through the WSL2 backend).
2. A recent **NVIDIA driver** with WSL/CUDA support. No CUDA toolkit / TensorRT / DeepStream needed
   on the host — the container ships them.
3. Docker Desktop → Settings → Resources → **File Sharing**: share the drive holding your working dir.
4. `docker pull nvcr.io/nvidia/deepstream:9.1-triton-multiarch`.

## The one thing that differs per shell: the mount token
Claude Code fills this in based on the host OS:

| Shell | working-dir mount |
|-------|-------------------|
| **PowerShell** | `-v "${PWD}:/work"` |
| **cmd** | `-v "%cd%:/work"` |
| **WSL2 / Linux bash** | `-v "$PWD":/work` |

## Bootstrap + preflight (PowerShell example)
```powershell
# one-time bootstrap: venv + torch/onnx/onnxruntime + wkhtmltopdf, all in-container
docker run --rm -it --gpus all --shm-size=16g -v "${PWD}:/work" -w /work `
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch `
  .claude/skills/deepstream-import-vision-model/setup.sh

# preflight — GPU + venv + trtexec (container-mode auto-detects /.dockerenv)
docker run --rm --gpus all -v "${PWD}:/work" -w /work `
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch `
  .claude/skills/deepstream-import-vision-model/scripts/preflight.sh
```
Every subsequent phase runs the same way — via `docker run … -lc '<commands>'` or the
`.claude/skills/deepstream-import-vision-model/scripts/dsrun.sh` wrapper.

## Notes
- The skill ships a `.gitattributes` forcing **LF** on all scripts, so a Windows checkout won't
  CRLF-corrupt them (CRLF breaks bash-in-container).
- `--shm-size=16g` works on the WSL2 backend.
- **Install:** on native Windows run the bundled **`install.ps1`** — the twin of `install.sh`, same
  sequence and flags (`-Target`=`--target`, `-NoCursor`=`--no-cursor`, `-DryRun`=`--dry-run`):
  `.\install.ps1 -Target C:\path\to\project` (copies the skill into `<project>\.claude\skills\`). On
  Linux/WSL2/Git Bash use `bash install.sh --target <project>`.
- Prefer a **WSL2 Ubuntu terminal** for the exact Linux experience — inside WSL2 everything runs
  unchanged.
