---
name: tao-port-huggingface-model
description: >
  Integrate a HuggingFace Computer Vision model into the NVIDIA TAO Toolkit
  ecosystem (tao-core config, tao-pytorch trainer, tao-deploy TensorRT
  pipeline). Use when the user asks to "integrate a HuggingFace model into
  TAO", "add an HF model to TAO Toolkit", "wire a HuggingFace ViT/DETR/
  SegFormer into tao-pytorch", "build a TAO trainer + deploy pipeline for an
  HF CV model", or pastes a HuggingFace model URL/ID and wants it turned
  into a TAO model. Covers the full 7-phase loop: prerequisites check,
  HuggingFace inspection and validation, codebase exploration, tao-core
  configuration and native trainer implementation, ONNX export plus TensorRT
  deploy integration, packaging and L0 testing, container-based end-to-end
  validation, and (conditional) accuracy/latency tuning. Supports
  classification, object detection, semantic / instance / panoptic
  segmentation, zero-shot detection, and depth estimation.
license: Apache-2.0
compatibility: Requires Python 3.10+, NVIDIA driver, CUDA 13.0+, docker + nvidia-container-toolkit, an NGC API key (`docker login nvcr.io`), an HF_TOKEN, and access to the TAO Toolkit container images on `nvcr.io` for `tao-pytorch`, `tao-deploy`, and (optionally) `tao-dataservices` — Phase 0 asks the user for the exact image references and prepares them locally as `tao-pytorch-base:latest`, `tao-deploy-base:latest`, `tao-dataservices-base:latest`. Local clones of `tao-core`, `tao-pytorch`, `tao-deploy`,...
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write Edit Grep Glob
tags:
- tao
- huggingface
- integration
- computer-vision
- deploy
---
<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# TAO-HF Integration Skill

Integrate a HuggingFace (HF) Computer Vision model into the NVIDIA TAO Toolkit ecosystem. Work the phases iteratively — not purely linearly — via a **build → test → debug → fix → retest** loop at every step: when something fails, diagnose and fix before moving on; when it passes, move to the next step.

This SKILL.md is the workflow coordinator. Each phase has a dedicated `references/phase-N-*.md` with the full step-by-step content, code, docker invocations, and gates. Read the matching reference at the start of each phase — the summaries below are not sufficient.

---

## Local-Only Rule

All work is strictly local. Do NOT push/commit/branch on any remote (GitLab, GitHub, HuggingFace), create merge/pull requests or issues, or upload/publish Docker images to any registry or artifact store. You may only read/clone from remotes — all edits, Docker builds, and test runs stay on the local machine.

---

## Submodule Override Strategy

The user clones the four TAO repos (`tao-core`, `tao-pytorch`, `tao-deploy`, `tao-dataservices`) independently into one working directory. The `tao-core/` submodule nested inside each repo points to the **original unmodified commit**; modifications only exist in the top-level `tao-core/`. **Always install from the top-level `tao-core/`, never `<repo>/tao-core/`** — the nested submodule silently ignores all modifications. Override rules: (1) mount the working directory `-v $(pwd):/workspace`; (2) `pip install /workspace/tao-core` FIRST, before tao-pytorch/tao-deploy; (3) PYTHONPATH top-level tao-core first, e.g. `-e PYTHONPATH=/workspace/tao-core:/workspace/tao-pytorch`. See `references/cross-cutting.md` for the directory tree.

---

## Execution platform

Every test, smoke run, and end-to-end validation executes inside a locally prepared TAO Toolkit container (`tao-pytorch-base:latest`, `tao-deploy-base:latest`, optionally `tao-dataservices-base:latest` — all from Phase 0). The platform skills own *how* to run them; this skill specifies *what*. **Default platform:** `local-docker`. Phase 0 delegates the driver / CUDA / NCT preflight to `tao-setup-nvidia-gpu-host`. See `references/cross-cutting.md` for the authoritative-skill table, bind-mount rationale, and canonical docker-run flag set.

---

## Phase Map

| Phase | Goal | Reference |
|---|---|---|
| 0 | Prerequisites + TAO Toolkit images + local image tags | [phase-0-prereqs.md](references/phase-0-prereqs.md) |
| 1 | Inputs, HF-inspection container, validate model + dataset | [phase-1-inspection.md](references/phase-1-inspection.md), [hf-inspection.md](references/hf-inspection.md) |
| 2 | Closest existing TAO reference model | [phase-2-codebase.md](references/phase-2-codebase.md), [task-type-guide.md](references/task-type-guide.md) |
| 3 | tao-core config + tao-pytorch trainer / eval / inference | [phase-3-implementation.md](references/phase-3-implementation.md), [tao-patterns.md](references/tao-patterns.md), [repo-structure.md](references/repo-structure.md) |
| 4 | ONNX export + tao-deploy TRT engine / inference / eval | [phase-4-deploy.md](references/phase-4-deploy.md) |
| 5 | Packaging (`console_scripts`) + L0 tests | [phase-5-packaging.md](references/phase-5-packaging.md) |
| 6 | Container testing + end-to-end validation | [phase-6-container-tests.md](references/phase-6-container-tests.md), [docker-patterns.md](references/docker-patterns.md) |
| 7 | (conditional) Accuracy / latency / size tuning | [phase-7-optimization.md](references/phase-7-optimization.md) |

Cross-cutting refs: [workflow-consistency.md](references/workflow-consistency.md) (CLI flow, config field paths, cross-phase dependencies); [cross-cutting.md](references/cross-cutting.md) (platform, isolation, module pitfalls, debugging).

**IMPORTANT — Continuous Execution Through Phase 6:** do NOT stop after Phases 3–5 to wait for the user to run tests. Phase 6 is mandatory — not complete until tests pass inside the containers and the end-to-end pipeline is validated.

---

## Development Loop

At every step: write code → test immediately (import check, unit test, or dry-run) → if it fails, read traceback → diagnose → fix → retest; if it passes, move on. Do NOT accumulate untested code — testing only at the end compounds bugs.

---

## Debugging Playbook

When something fails, consult the symptom → likely-cause → fix table in `references/cross-cutting.md` before trying random fixes — it covers `ModuleNotFoundError`, `BACKBONE_REGISTRY` `KeyError`, shape mismatch, NaN loss, ONNX/TRT build failures, TRT-vs-PyTorch accuracy gaps, OOM, DDP hangs, checkpoint load failures, and stale-submodule config issues.

---

## Environment Isolation Strategy

All Python work runs **inside Docker containers** — no host venvs, no `pip install`s into host Python (the host needs only Docker, from `tao-setup-nvidia-gpu-host`). Three contexts: A (Phase 1 HF inspection in `tao-hf-inspect`, `python:3.12-slim` fallback), B (Phase 3/4/6 smoke/L0/e2e in the prepared container, source via `pip install /workspace/tao-core && python setup.py develop`), C (host-bind-mount scratch). See `references/cross-cutting.md` for the contexts in full and the four numbered rules verbatim (`--check-only` host packages; Phase 1 `--user $(id -u):$(id -g)` vs. root; `HOME=/workspace`/`PIP_USER=1` fallback; distro package-manager list; `root:root` trade-off; `tao-hf-inspect` cleanup).

---

## Phase 0 — Prerequisites Check

**Goal:** verify Python 3.10+ and `git`; delegate the driver / CUDA / Docker / NVIDIA Container Toolkit host check to `tao-setup-nvidia-gpu-host`; verify NGC `docker login` for `nvcr.io`. Then **ask the user** for the TAO Toolkit image references (tao-pytorch, tao-deploy, optionally tao-dataservices), pull, and prepare local tags `tao-pytorch-base:latest`, `tao-deploy-base:latest`, `tao-dataservices-base:latest` for later phases — preparation removes the pre-installed released TAO packages so the user's `/workspace/...` clones install/load via `pip install /workspace/tao-core && python setup.py develop`. **Hard stop** on any failed check. Required user inputs: the image references + credentials (NGC login, `HF_TOKEN`). Full commands, prompt wording, and per-image `Dockerfile` snippets: the Phase 0 reference.

**Gate:** all prerequisite checks pass; the user supplied the required image references; `tao-pytorch-base:latest` and `tao-deploy-base:latest` exist locally; `tao-dataservices-base:latest` exists if dataservices work is anticipated.

---

## Phase 1 — Information Gathering & Validation

**Goal:** decide whether to proceed at all. Gather credentials, locate/clone the four TAO repos, create a consistent working branch, launch the `tao-hf-inspect` container (Context A), validate the HF model is CV with a supported `pipeline_tag`, extract config + state-dict schema, sanity-check ONNX export, clean up. Full steps: Phase 1 references.

**Reject if:** `pipeline_tag` is NLP / audio / LLM (non-CV); `AutoConfig` raises; or ONNX export fundamentally cannot work (no rewrite path).

**Gate:** all 4 TAO repos located/cloned with a consistent branch; `pipeline_tag` confirmed CV; `model_type`, `image_size`, `hidden_size`, `num_labels` extracted; state-dict keys documented + HF→TAO remapping plan drafted; ONNX export sanity check passed (or failure understood); user confirmed `model_short_name` + task type. (Full checklist: [phase-1-inspection.md](references/phase-1-inspection.md).) Present findings and get user confirmation first.

---

## Phase 2 — Codebase Exploration

**Goal:** find the closest existing TAO reference model for the detected `pipeline_tag`, read its implementation across `tao-core` / `tao-pytorch` / `tao-deploy`, and decide whether the backbone exists in `backbone_v2/` or is new.

The HF `pipeline_tag` → TAO reference model mapping (classification → `classification_pyt`, detection → `dino`/`rtdetr`, segmentation → `segformer`, instance → `mask2former`, panoptic → `oneformer`, zero-shot → `grounding_dino`, depth → `mono_depth`) drives **everything downstream** (config, architecture, loss, ONNX shape, TRT builder, deploy classes, metrics, dataset format). See the Phase 2 references for the full reference list (12 files per model), the `backbone_v2/` and `tao-dataservices` coverage checks, and per-task architecture.

If a new backbone is needed, decide the strategy (timm wrap > re-implement > HF black-box wrap) before Phase 3 — it changes weight loading, ONNX export, deploy. **Never dual-inherit from `transformers.PreTrainedModel` and `BackboneBase`** (metaclass conflict — compose instead).

**Gate:** reference TAO model identified + all 12 reference locations read; task-type implications understood (architecture, loss, ONNX outputs, deploy classes, metrics, dataset); backbone coverage decided (reuse / wrap timm / new); dataservices coverage checked. (Full checklist: [phase-2-codebase.md](references/phase-2-codebase.md).)

---

## Phase 3 — TAO Core Configuration & Native Implementation

**Goal:** write the tao-core config schema + the tao-pytorch trainer / native inference / evaluation, smoke-testing between steps. (`<model_name>` = `snake_case` short-name; `<ModelName>` = `PascalCase`.)

Steps 1–7 (each builds on the previous, smoke-test between): tao-core config (1), tao-pytorch trainer (2), multi-GPU/multi-node (3), native inference → `result.csv` (4), native evaluation → `results.json` (5), MLOps for training and eval/infer → `status.json` (6–7). The `ExperimentConfig(CommonExperimentConfig)` must contain `model`, `dataset`, `train`, `evaluate`, `inference`, `export`, `gen_trt_engine`, `quantize`. All `???` fields are `MISSING` (user supplies via YAML/CLI); the `augmentation.mean`/`std`, `model.head.in_channels`, checkpoint-name, and `onnx_file` matches are in the checklist below.

Full per-step bodies, code, the canonical `experiment_spec.yaml`, and smoke-test commands: the Phase 3 references.

**Gates:** Step 1 — `ExperimentConfig` imports cleanly in-container; Step 2 — `build_model(cfg)` runs + PLModel instantiates in-container; Phase 3 — all 7 steps complete, smoke tests pass, no missing `__init__.py`.

---

## Phase 4 — Export, Deployment & TensorRT Integration

**Goal:** ONNX export from tao-pytorch, then TRT engine builder + inference + evaluation in tao-deploy reusing the tao-core `ExperimentConfig`.

Steps 8–11: ONNX exporter (8 — task-specific input/output names, `batch_size=-1` ⇒ dynamic batch); TRT engine builder (9 — subclass `EngineBuilder` or reuse `ClassificationEngineBuilder`; write `specs/{gen_trt_engine,inference,evaluate}.yaml`, same `ExperimentConfig` schema, `augmentation.mean`/`std` MUST match training); TRT inference → `result.csv` (10); TRT eval → `results.json` (11). See the Phase 4 reference for full code and the Phase 3+4 gate (3 in-container checks: imports, model build + forward, ONNX round-trip).

**Module pitfalls:** tao-pytorch and tao-deploy have **separate** `hydra_runner` and `monitor_status` — use the deploy versions in deploy scripts. `ExperimentConfig` comes from `nvidia_tao_core` in both (same schema/field paths).

**Phase 3+4 gate:** all three in-container checks pass (`tao-pytorch` imports + model + ONNX export; `tao-deploy` imports).

---

## Phase 5 — Packaging & L0 Testing

**Goal:** register the model as a console_script in both repos and add unit tests.

Steps 12–15: register `'<model_name>=...:main'` in `console_scripts` of `tao-pytorch/setup.py` (12) and `tao-deploy/setup.py` (13, creating the deploy `entrypoint/<model_name>.py` via `entrypoint_hydra`); deploy L0 tests (14); trainer L0 tests — `Trainer(..., fast_dev_run=True)` + `@pytest.mark.cv_unit @pytest.mark.<model_name>` (15). See the Phase 5 reference for exact entry-point strings, code, and L0 test file lists.

**Gate:** entrypoints registered; pytest files exist and follow the marker convention. **Do NOT stop — go directly to Phase 6.**

---

## Cross-Phase Data Flow & Consistency Verification

Before Docker testing, verify the chain `train → export → gen_trt_engine → inference / evaluate` (the `*_model_latest.pth` → `.onnx` → `.engine` artifact flow + the config fields each stage reads/writes — full diagram in `references/cross-cutting.md`).

Consistency checklist (verify before proceeding): `self.checkpoint_filename` → the `*_latest.pth` name `evaluate.checkpoint` / `export.checkpoint` reference; `augmentation.mean`/`std` identical across training spec, `inference.yaml`, `evaluate.yaml`, engine-builder `preprocess_mode`; ONNX `input_names=['input']` / `output_names=['output']` (detection/instance-seg use task-specific names); `export.input_width`/`input_height` match `dataset.img_size`; `model.head.in_channels` matches `model_params_mapping.py`; `classes.txt` at `dataset.root_dir` readable by both repos; all `__init__.py` exist (incl. `scripts/__init__.py` for `get_subtasks()` via `pkgutil`). Full paths: [workflow-consistency.md](references/workflow-consistency.md).

---

## Phase 6 — Container Testing & End-to-End Validation

**Mandatory — start immediately after Phase 5.** All TAO models ship as Docker images; code that only works outside a container is incomplete. Testing runs **directly inside the TAO Toolkit container** — no image build in the loop: mount → install source (`setup.py develop`) → run `pytest` / `pylint` / `pydocstyle` / `flake8` directly. Use vanilla commands, NOT the `ci/run_functional_tests.py` / `ci/run_static_tests.py` wrappers (internal-mirror-only; public `github.com/NVIDIA-TAO/` mirrors have no `ci/` dir).

Steps 16–25: verify local image tags exist (16); unit tests for tao-core / tao-pytorch (`-m cv_unit`, `--shm-size=16G`) / tao-deploy (17–19); lint (20); wheels (21); end-to-end — train dry-run + export in **one** tao-pytorch session, then gen_trt_engine + inference + evaluate in **one** tao-deploy session (same session critical — `--rm` discards installs) (22); cross-check native vs TRT (23); debug shells (24); optional release images (25).

Full commands (every `docker run`, per-container env-vars, exact pytest / lint invocations + full-suite variants, the train/export/gen_trt_engine/inference/evaluate one-liner with all CLI overrides, the `ci/` note, the fix-and-retest loop) and build scripts / runner patterns: see the Phase 6 references.

**Phase 6 gate (Done criteria):** tao-core / tao-pytorch / tao-deploy unit tests pass in their containers; static tests pass (or only legacy lint warnings); wheels build; end-to-end `<model_name>_model_latest.pth` → `model.onnx` → `model.engine` → non-empty `result.csv` + `results.json`; native vs TRT agree within tolerance.

---

## Phase 7 — Optimization & Tuning (conditional)

Enter only if Phase 6 passes but accuracy / latency / size needs improvement. **Ask the user for target metrics first.**

Diagnostic categories: accuracy too low; TRT-vs-native gap; training too slow; inference too slow. Techniques: Step 27 — hyperparameter tuning; Step 28 — INT8 quantization (PTQ via torchao / modelopt, TRT INT8 + calibration); Step 29 — channel pruning + retrain; Step 30 — knowledge distillation; Step 31 — resolution tuning (TAO interpolates ViT positional embeddings automatically). See the Phase 7 reference for each category's checks, config blocks, YAML overrides, decision tree, and rationale.

---

## Argument

`$ARGUMENTS`

If provided, interpret `$ARGUMENTS` as the HuggingFace model ID or URL to start Phase 1. If credentials or model short-name are not included, ask the user for them before proceeding.
