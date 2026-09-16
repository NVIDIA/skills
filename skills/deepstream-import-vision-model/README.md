# DeepStream Import Vision Model

Automated end-to-end pipeline: HuggingFace model → TensorRT engine → DeepStream
multi-stream benchmark → PDF report.

## Overview

This self-contained skill uses four phase-specific reference documents. Together they automate the full model bringup workflow for NVIDIA DeepStream, from
downloading a model on HuggingFace or NVIDIA NGC to a publication-ready benchmark report.

Supported input formats: ONNX (direct), SafeTensors (auto-exported via `torch.onnx.export`).

**Current scope:** object detection models only. Classification, segmentation, pose estimation, and other vision tasks are not yet supported — the pipeline fails fast if a non-detection architecture is detected in `config.json`.

## Prerequisites

**Host: only Docker + the NVIDIA driver.** Everything else runs **inside the DeepStream container** —
DeepStream, TensorRT/`trtexec`, the Python export venv (`torch`/`onnx`/`onnxruntime`), `wkhtmltopdf`,
`deepstream-app`/`gst-launch-1.0` — and is bootstrapped by `setup.sh`. Nothing is installed on the host.
Runs identically on Linux and **Windows** (Docker Desktop + WSL2 backend, required for GPU).

- **Docker** — Docker Desktop with the WSL2 backend on Windows; Docker Engine + NVIDIA Container
  Toolkit on Linux.
- **NVIDIA GPU + driver** (on Windows, the WSL2 GPU driver — no host CUDA/TensorRT/DeepStream needed).
- `docker pull nvcr.io/nvidia/deepstream:9.1-triton-multiarch`, then run `setup.sh` through the container.

See **[references/windows.md](references/windows.md)** for the cross-platform run model and the
per-shell `docker run` mount token.

## Installation

```bash
bash <path-to-deepstream-import-vision-model>/install.sh --target <your-project-path>
```

Preview what will be installed first with `--dry-run`:

```bash
bash <path-to-deepstream-import-vision-model>/install.sh --target <your-project-path> --dry-run
```

Where `<path-to-deepstream-import-vision-model>` is the location of this skill in your repo, e.g.:
- In **team-mind-hub**: `team-skills/deepstream-sdk/deepstream-import-vision-model`
- In **ds-copilot**: `team-skills/deepstream-sdk/deepstream-import-vision-model` (same path)

The script copies the complete skill into the target project for Claude Code, Codex, and Cursor. Re-running it safely refreshes an existing installation.

## Usage

**Claude Code:**
```text
Use deepstream-import-vision-model to run this model: https://huggingface.co/onnx-community/yolov8n
```

**Codex:**
```text
Use $deepstream-import-vision-model to deploy and benchmark https://huggingface.co/onnx-community/yolov8n
```

**Cursor:**
```text
@deepstream-import-vision-model run this model: https://huggingface.co/onnx-community/yolov8n
```

The skill runs the full pipeline autonomously — no manual steps required.

## Pipeline Steps

| Step | Phase reference | Action |
|------|-----------|--------|
| 1–3 | `references/model-acquire.md` | Browse HF repo, download ONNX or export SafeTensors |
| 4–5 | `references/engine-build.md` | Build dynamic TRT engine, run trtexec benchmarks |
| 6–7 | `references/pipeline-run.md` | Custom bbox parser, DeepStream single + multi-stream |
| 8 | `references/report-generation.md` | 5 charts, HTML report, PDF |

## Output Structure

Per-model outputs are written to `models/<model_name>/` in your project:

```text
models/<model_name>/
  model/          ONNX file(s)
  parser/         Custom nvinfer bbox parser (.cpp, .so)
  config/         nvinfer config, DS app config, labels.txt
  scripts/        Model-specific run helpers
  benchmarks/     TRT engines, trtexec logs
  reports/        benchmark_report.md / .html / .pdf + charts/
  samples/        Output videos, test frames, KITTI detections
```

## Files in this package

```text
deepstream-import-vision-model/
├── SKILL.md                    Top-level skill definition
├── README.md                   This file
├── references/                 Phase-specific runbooks
├── scripts/                    Utility scripts by pipeline phase
└── tests/                      Installer and script regression tests
```
