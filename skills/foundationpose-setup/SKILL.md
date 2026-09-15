---
name: foundationpose-setup
description: Install or repair the FoundationPose perception pipeline and build its FoundationStereo TensorRT engines. Use for SAM3/TAO dependency conflicts, CUDA library failures, and depth-engine shape or precision decisions.
license: Apache-2.0
metadata:
  author: "zwdoescode <zhengwang@nvidia.com>"
  version: "0.1.0"
---

# FoundationPose perception pipeline setup

## Purpose

Prepare the [FoundationPose perception pipeline](https://github.com/nvidia-isaac/foundationpose_perception_pipeline)
for depth, segmentation, and pose inference. Environment installation and engine construction
belong here; dataset adaptation, inference, and pose evaluation belong to
`foundationpose-pipeline` when that skill is installed.

## Requirements

Work from the product checkout, not the installed skill directory. Find the user's checkout by
checking for `pyproject.toml` (project `foundationpose-perception-pipeline`),
`tools/build_tao_engine.py`, and `config/defaults.yaml`. If absent and setup was requested, clone
the product URL above into the user's workspace and enter it. For advice-only requests, use the
supplied diagnostics without cloning or installing anything. Commands below use paths relative
to the product root; `references/` links are relative to this skill.

Read the checkout's `README.md` Requirements and Install sections for the matching revision.
The supported stack requires Linux x86_64, glibc >= 2.38, GLIBCXX_3.4.31, NVIDIA driver >= 580,
a CUDA toolkit >= 12.8 with `nvcc`, Python 3.12, uv, Git, Docker with GPU access, wget, and unzip.
Start GPU sizing at 24 GB and measure the densest scene; 32 GB was tested. Budget depth-cache
disk as roughly `width * height * 4 * 3` bytes per scene, plus predictions and models.

Keep sibling directories for `sam3/`, `foundation-pose-inference-library/`, and `models/` beside
the product checkout. `models/` contains the deployable ONNX and engine, not FoundationStereo source.

## Instructions

1. **Preflight before installing.** Check glibc, GLIBCXX, driver, `nvcc`, tools, and Docker GPU
   access. An Ubuntu 22.04 host with glibc 2.35 cannot load the shipped FoundationPose library;
   report the unsupported runtime and stop setup there. Do not replace system libc or try to
   solve this with `LD_LIBRARY_PATH`. See [installation](references/installation.md#preflight).
2. **Install into the product's Python 3.12 venv.** Follow
   [installation](references/installation.md#pipeline-and-sam3) for uv, SAM3, the FoundationPose
   build, and TAO Deploy. On a fresh venv use `uv sync --extra foundationpose`; on an existing
   venv use `uv sync --inexact --extra foundationpose` to preserve out-of-band packages.
3. **Verify checkpoint access.** SAM3 is gated at Hugging Face; an existing authorized token
   or usable cached checkpoint is sufficient. Request user action only if access is missing.
   FoundationPose and the documented FoundationStereo export are public NGC downloads;
   credential hunting is not the first response to a network failure.
4. **Prepare the depth engine.** Read [engine construction](references/engine.md). Use the
   user's ONNX location or the sibling `models/` directory. Adapt the dataset before measuring
   the engine shape; use `tools/bop_adapt/adapt.py --config <profile> --src <source>` as described
   in the checkout's README Dataset adaptation section. Only registered adapters are supported.
   Build with `--shape-from-scene` on an adapted scene and FP32 unless the user requests a
   precision experiment. Set `overrides.depth.engine` in `config/<profile>.yaml`.
5. **Set runtime paths and verify.** From the product root:

   ```bash
   source .venv/bin/activate
   export FOUNDATIONPOSE_ROOT="$(realpath ../foundation-pose-inference-library)"
   PIPELINE_SITE="$(realpath .venv/lib/python3.12/site-packages)"
   export LD_LIBRARY_PATH="${PIPELINE_SITE}/tensorrt_libs:${PIPELINE_SITE}/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
   python tools/verify_sam3.py
   python tools/verify_foundationpose.py
   python tools/verify_foundationstereo.py --config <profile> --engine <engine-path>
   python test/check_engine_depth_smoke.py --config <profile> --engine <engine-path>
   ```

   The first three verify components; the last also needs an adapted dataset. Expect
   `backend=tao`, `normalization=imagenet`, the intended fixed shape, and no `cropping N rows`
   warning. An unloaded or unavailable engine is an incomplete verification, not a pass.

## Troubleshooting

| Symptom | Action |
|---|---|
| `GLIBC_2.38 not found` | Use a supported OS/runtime; a venv or library search path cannot upgrade host libc. |
| `libcudart.so.13` missing | Check the product venv runtime wheels and absolute library paths before retrying pose. |
| SAM3 breaks after sync | Use `--inexact`; confirm numpy 1.26.x and reinstall the sibling SAM3 package if pruned. |
| pycuda build cannot find `cuda.h` | Check the CUDA toolkit, `nvcc` on PATH, or `CUDA_ROOT`. |
| TAO import or dependency conflict | Use TAO Deploy 7.1.0 with `--no-deps`; sync declared dependencies with `--inexact`. |
| Engine sidecar mismatch or cropping | Rebuild for this GPU, TensorRT version, precision, and adapted scene shape. |

## Examples

- "Install the FoundationPose perception pipeline on this Ubuntu 24.04 GPU machine."
- "The pipeline cannot load libcudart.so.13 after I moved the checkout."
- "Build the TAO depth engine for my adapted T-LESS scenes."

## Limitations and completion

Engines are machine-specific and must not be committed. With no dataset, download the ONNX
and report shape-dependent engine construction and scene validation as pending; do not invent
a rig shape. The pipeline's Apache license does not cover separately downloaded model weights;
retain their upstream terms and SAM3's access requirements.

Report which preflight, install, checkpoint, engine, and verification steps actually passed,
the checkout and engine paths, versions used, and remaining blockers. Do not equate installation
or a smoke check with measured pose accuracy.
