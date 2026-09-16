---
name: deepstream-import-vision-model
description: >
  Use this skill to bring a supported object-detection vision model from HuggingFace or NVIDIA NGC into
  an NVIDIA DeepStream pipeline with end-to-end automation: ONNX download,
  SafeTensors export, TRT engine build, custom nvinfer bbox parser, multi-stream
  benchmark, and PDF report. Object detection models only.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: "Tushar Khinvasara <tkhinvasara@nvidia.com>"
  owner: "Tushar Khinvasara <tkhinvasara@nvidia.com>"
  service: "deepstream"
  version: "1.5.2"
  reviewed: "2026-08-04"
  team: deepstream-sdk
  tags:
    - deepstream
    - tensorrt
    - object-detection
    - import-vision-model
  languages:
    - bash
    - python
    - cpp
  domain: computer-vision
---

# DeepStream Import Vision Model

When this skill is active, **read the relevant reference document before starting each phase**. Do not rely on memory — reference documents contain exact script paths, bash variable conventions, log filename contracts, and critical parsing rules.

**Current scope:** Object detection models only. Fail fast on classification, segmentation, or other architectures detected in `config.json`.

## Model choice — always offer two options

Before preflight, browsing, downloads, or file creation, present exactly these two choices. Do not
start with only an open-ended model-source prompt. If the user's request already clearly selects a
model, confirm the matching choice instead of asking redundantly.

### 1. Default model (recommended)

Use the validated Hugging Face RT-DETR model:

```yaml
model_id: PekingU/rtdetr_r50vd
source: huggingface
task: object-detection
precision_preference: fp16
```

### 2. Custom object-detection model

Ask for one supported source:

- Hugging Face model ID (`organization/model`) or full model URL.
- NVIDIA NGC catalog model URL including its version.

Explain that the skill currently rejects classification, segmentation, and other non-detection
architectures after inspecting `config.json`. Do not invent or silently substitute a model when the
custom source is missing or unsupported.

For a dry run, present the same two choices and simulate discovery, build, benchmark, and report
stages without browsing, downloading, launching Docker, writing files, or starting processes.

## Pipeline Overview

| Step | Phase | Reference | What it does |
|------|-------|-----------|--------------|
| 1–3 | Model Acquire | [references/model-acquire.md](references/model-acquire.md) | Browse HF/NGC, detect format, download ONNX or export SafeTensors |
| 4–5 | Engine Build  | [references/engine-build.md](references/engine-build.md) | Build dynamic TRT engine, run trtexec BS=1 and BS=MAX_BS |
| 6–7 | DS Pipeline   | [references/pipeline-run.md](references/pipeline-run.md) | Custom bbox parser, nvinfer config, single-stream + multi-stream benchmarks |
| 8   | Report        | [references/report-generation.md](references/report-generation.md) | 5 charts, HTML, PDF benchmark report |

Run the full pipeline autonomously without pausing for confirmation at each step.

## Runs entirely through Docker (no host packages)

**Every step runs INSIDE the DeepStream container.** The host needs only **Docker + the NVIDIA
driver** — no host python/venv/torch/trtexec/make/wkhtmltopdf. This works identically on Linux and
**Windows** (Docker Desktop + WSL2 backend, required for `--gpus`). The per-shell bind-mount
token is the only OS difference — `-v "$PWD":/work` (bash), `-v "${PWD}:/work"` (PowerShell),
`-v "%cd%:/work"` (cmd); full guide in [references/windows.md](references/windows.md). All venv/ONNX/
engine/parser/config/report artifacts live under the mounted working root and persist between the
ephemeral `--rm` containers.

## Pre-flight — bootstrap + verify (through the container)

**1. One-time bootstrap** — builds `build/.venv_optimum` (torch/onnx/onnxruntime/report deps; the
venv name is historical, optimum is no longer used) +
installs `wkhtmltopdf`, all in-container. From the working root:
```bash
docker run --rm -it --gpus all --shm-size=16g -v "$PWD":/work -w /work \
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  .claude/skills/deepstream-import-vision-model/setup.sh
```

**2. Preflight** — GPU + venv + trtexec, run THROUGH the container (container-mode auto-detects):
```bash
docker run --rm --gpus all -v "$PWD":/work -w /work \
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  .claude/skills/deepstream-import-vision-model/scripts/preflight.sh   # proceed only on PASS
```

**Every subsequent phase runs the same way** — issue the model's commands via
`docker run … --entrypoint bash … -lc '<commands>'` (or the
`.claude/skills/deepstream-import-vision-model/scripts/dsrun.sh` wrapper:
`bash .claude/skills/deepstream-import-vision-model/scripts/dsrun.sh '<in-container command>'`),
using `PY=build/.venv_optimum/bin/python` and
`trtexec` at `/usr/src/tensorrt/bin/trtexec` inside the container. `deepstream-app`,
`gst-launch-1.0`, and `/opt/nvidia/deepstream/…` sample paths all exist **in** the image.
TensorRT build+runtime share one image, so there is **no version skew** (the concern the old
"build on the host" rule tried to avoid — see [references/engine-build.md](references/engine-build.md)).
`sample_720p.mp4` ships in the image; set `DS_VIDEO` only to override.

## Mandatory Output Structure

Create once `MODEL_NAME` is known (Step 1). Never dump files flat.

```
models/{model_name}/
  model/           <- ONNX file(s)
  parser/          <- .cpp, Makefile, .so
  config/          <- nvinfer config, ds-app config, labels.txt
  scripts/         <- run helper scripts
  benchmarks/
    engines/       <- _dynamic_b{MAX_BS}.engine, timing.cache, build logs
    b1/            <- trtexec BS=1 log
    b{MAX_BS}/     <- trtexec BS=MAX_BS log
    ds/            <- DS benchmark logs
  reports/         <- benchmark_report.md, .html, .pdf, benchmark_data.json
    charts/        <- chart_*.png (5 charts)
  samples/         <- output .mp4 or .ogv (theoraenc fallback), test frames
    kitti_output/  <- KITTI detection .txt files
```

```bash
mkdir -p models/$MODEL_NAME/{model,parser,config,scripts,benchmarks/engines,benchmarks/ds,reports/charts,samples/kitti_output}
```

## Critical Rules

1. **Engine naming** — always `{model}_dynamic_b{MAX_BS}.engine`. Never bare `model_dynamic.engine`.
2. **batch_size == num_streams** — in DS runs, `batch-size` and stream count are always equal.
3. **Log filenames are fixed** — `trtexec_b1.log`, `trtexec_b${MAX_BS}.log`, `ds_s${N}_run1.log`, `ds_s${N}_run2.log`. No timestamps. Report generation reads exact paths.
4. **Parser zero-init** — always `NvDsInferObjectDetectionInfo obj = {};`. Required for DS 9.1 OBB support; bare `obj;` leaves `rotation_angle` uninitialized, causing tilted bounding boxes.
5. **KITTI validation gate** — do NOT proceed to Step 7 if KITTI frame count is zero or detection rate < 90%.
6. **Shared venv** — `build/.venv_optimum` reused across all models. Never create per-model venvs.
7. **trtexec `--noDataTransfers`** — GPU-only compute matches DeepStream's GPU-to-GPU data flow.
8. **Report HTML+PDF** — always use `.claude/skills/deepstream-import-vision-model/scripts/report/md-to-html-pdf.py`. Never write a custom HTML generator or call `wkhtmltopdf` directly.
9. **Object detection only** — reject non-detection architectures from `config.json` before building anything.
10. **Encoder fallback (MANDATORY)** — `x264enc` and `openh264enc` are **prohibited**. On NVENC-unavailable systems, use `theoraenc + oggmux` (LGPL; ships in gst-plugins-base; output is `.ogv`). If `theoraenc`/`oggmux` are absent, skip video creation (`DS_SINGLE_STREAM_MODE=skipped`). Report which mode was used: `nvv4l2h264enc` / `theoraenc-fallback` / `skipped`.
11. **Video source (MANDATORY)** — default is always `sample_720p.mp4` (1280×720). Never autonomously substitute `sample_1080p_h264.mp4` or any other file. Only use a different video when the user explicitly provides a path (via `DS_VIDEO` env var or script argument).

## Examples

**Default model, end to end.** Bootstrap once, then run the full pipeline:

```bash
docker run --rm -it --gpus all --shm-size=16g -v "$PWD":/work -w /work \
  --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch \
  .claude/skills/deepstream-import-vision-model/setup.sh
# then: "Use deepstream-import-vision-model to run PekingU/rtdetr_r50vd"
```

**SafeTensors model with no published ONNX.** Step 2b exports it first; the wrapper reports which
backend produced the graph and fails loudly if the batch dimension was baked in:

```bash
bash .claude/skills/deepstream-import-vision-model/scripts/model/safetensors-to-onnx.sh \
  models/$MODEL_NAME/hf_model models/$MODEL_NAME/onnx_export/
#   [export] backend=dynamo
#   [export] dynamo produced a static batch dimension; trying the next backend
#   [export] backend=legacy-torchscript
#   [export] pixel_values shape=['batch', 3, 640, 640]
```

**Pin a Hub revision** for a reproducible build — any exporter flag passes straight through:

```bash
bash .claude/skills/deepstream-import-vision-model/scripts/model/safetensors-to-onnx.sh \
  PekingU/rtdetr_r50vd models/rtdetr/onnx_export --revision <commit-sha> --opset 18
```

## Pipeline Timing

Wrap every step:

```bash
STEP_START=$(date +%s.%N)
# ... step commands ...
STEP_END=$(date +%s.%N)
STEP_DURATION=$(python3 -c "print(round($STEP_END - $STEP_START, 2))")   # bc is not in the container; python3 always is
echo "[Step N] completed in ${STEP_DURATION}s"
```

Track `PIPELINE_START` (before Step 1) and `PIPELINE_END` (after Step 8). Report all durations in the benchmark report.

## Report Output (MANDATORY — all 3 formats)

1. `benchmark_report.md` — markdown source (12 mandatory sections)
2. `benchmark_report.html` — styled HTML (charts base64-inlined, no local file access)
3. `benchmark_report_{model_name}.pdf` — via `md-to-html-pdf.py`; verify charts are embedded by counting `data:image/png` occurrences in the HTML output: `grep -o 'data:image/png' benchmark_report.html | wc -l` should equal 5

Run charts and report scripts with the shared venv active: `source build/.venv_optimum/bin/activate`.

## Reference Documents

**IMPORTANT**: Read the relevant reference before starting each phase. Do NOT generate code from memory.

| Document | Use When |
|----------|----------|
| [references/model-acquire.md](references/model-acquire.md) | Steps 1–3: HF/NGC URL parsing, format detection, ONNX download, SafeTensors export, label extraction |
| [references/engine-build.md](references/engine-build.md) | Steps 4–5: trtexec engine build, benchmarks, PEAK_GPU_STREAMS derivation, iterative scaling |
| [references/pipeline-run.md](references/pipeline-run.md) | Steps 6–7: custom bbox parser, nvinfer config, single-stream validation, KITTI dump, multi-stream benchmark |
| [references/report-generation.md](references/report-generation.md) | Step 8: benchmark_data.json, 5 charts, 12-section markdown report, HTML + PDF |

## Scripts

Installed into `.claude/skills/deepstream-import-vision-model/scripts/` by `install.sh`.

| Script | Phase | Purpose |
|--------|-------|---------|
| `model/hf-list-files.sh` | 1–3 | List HuggingFace repo files |
| `model/hf-download-config.sh` | 1–3 | Download config.json from HF |
| `model/ngc-list-files.sh` | 1–3 | List NGC model files |
| `model/ngc-download.sh` | 1–3 | Download NGC model archive |
| `model/safetensors-to-onnx.sh` | 1–3 | Export SafeTensors → ONNX via `torch.onnx.export` (wrapper) |
| `model/safetensors_to_onnx.py` | 1–3 | The exporter — dynamo backend, TorchScript fallback, verifies dynamic batch |
| `model/inspect-onnx.py` | 1–5 | Inspect ONNX input/output shapes |
| `model/make-static-batch-onnx.py` | 4–5 | Bake batch dim into ONNX |
| `model/cleanup.sh` | Any | Remove staging dirs, preserve shared venv |
| `engine/benchmark-trtexec.sh` | 4–5 | Run trtexec with standard flags |
| `deepstream/ds-single-stream.sh` | 6–7 | Single-stream visual validation (NVENC primary; theoraenc+oggmux fallback; skip if neither) |
| `deepstream/ds-sweep.sh` | 6–7 | 2-phase batch size sweep |
| `deepstream/benchmark-ds.sh` | 6–7 | Fixed-stream DS benchmark |
| `deepstream/ds-kitti-dump.sh` | 6–7 | KITTI detection dump via deepstream-app |
| `deepstream/ds-perf-run.sh` | 7 | Step 7c two-run benchmark — wraps `deepstream-app` with `enable-perf-measurement=1`, writes fixed-name log for the report parser |
| `deepstream/extract-frame.sh` | 6–7 | Extract sample frames from output video (`.mp4` NVENC path or `.ogv` theoraenc fallback) |
| `report/generate-benchmark-charts.py` | 8 | Generate 5 benchmark PNG charts |
| `report/md-to-html-pdf.py` | 8 | Markdown → styled HTML → PDF (canonical benchmark report path) |
| `report/md-to-pdf.sh` | Any | Markdown → PDF via pandoc/pdflatex — for design docs and references only, NOT for benchmark reports (use md-to-html-pdf.py for those) |
| `report/report-style.css` | 8 | CSS for HTML report |
| `report/render-mermaid-for-pdf.py` | 8 | Mermaid diagram → PNG |
| `report/mermaid-puppeteer.json` | 8 | Vetted Puppeteer config for Mermaid (sandboxed; non-root) |
| `report/mermaid-puppeteer-root.json` | 8 | Vetted Puppeteer config for Mermaid (used when running as root) |

## Quick Error Reference

| Error | Fix |
|-------|-----|
| Tilted/diagonal bounding boxes | Parser struct not zero-initialized — use `NvDsInferObjectDetectionInfo obj = {};` |
| Zero KITTI files | `gie-kitti-output-dir` not read by nvinfer — use `ds-kitti-dump.sh` (wraps `deepstream-app`) |
| Engine rebuilds every DS run | `model-engine-file` path wrong — check relative path from `config/` dir |
| `setDimensions` negative dims | Add `infer-dims=3;H;W` to nvinfer config for dynamic ONNX models |
| `--memPoolSize` workspace 0.03 MiB | Use `M` suffix not `MiB` — e.g. `--memPoolSize=workspace:32768M` |
| ForeignNode build failure (DETR) | Run `onnxsim` — see references/engine-build.md. Not reproduced on TRT 10.16 with either export backend |
| ONNX has a static batch dim | Both export backends specialized it — see the gotchas in references/model-acquire.md |
| Zero detections | Wrong `net-scale-factor` — check model family table in references/pipeline-run.md |
| `No module named 'pyservicemaker'` | Install into venv: `pip install /opt/nvidia/deepstream/.../pyservicemaker*.whl` |
