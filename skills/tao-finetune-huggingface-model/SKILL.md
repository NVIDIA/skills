---
name: tao-finetune-huggingface-model
description: >
  Fine-tune any HuggingFace CV / VLM / LLM model on local NVIDIA GPUs inside an
  NGC PyTorch container. Use when the user wants to fine-tune a HuggingFace
  model (full or LoRA), train a vision / VLM / LLM model end-to-end, generate a
  reproducible HF training pipeline, smoke-test a HuggingFace model locally
  before scale-up, push a fine-tuned model to the HF Hub with a model card, or
  emit a self-contained rerun skill for an existing HuggingFace finetune.
  Supports image classification, object detection, semantic / instance /
  panoptic segmentation, depth estimation, image-text-to-text VLM (SFT / LoRA),
  and LLM SFT / DPO / GRPO. Six-step workflow: inspect and qualify, hardware
  and NGC image, research, generate and smoke, train + eval + infer, push and
  emit rerun skill.
license: Apache-2.0
tags:
  - finetuning
  - huggingface
  - nvidia-tao
  - computer-vision
  - training
compatibility: Requires docker + nvidia-container-toolkit, NVIDIA GPU (driver ≥ 545, ≥ 24 GB VRAM for ≤3B models), ~40 GB free disk. Optional credentials (read from the session environment, exported before launching) — HF_TOKEN is read only when the model/dataset is gated or `push_to_hub` is on; WANDB_API_KEY and WANDB_PROJECT only when WandB logging is enabled.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
---
<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. Licensed under the Apache License, Version 2.0; see http://www.apache.org/licenses/LICENSE-2.0 -->

# tao-finetune-huggingface-model

Local NVIDIA GPU fine-tuning for HuggingFace models, grounded in live-fetched
documentation with curated references as a fallback safety net. One NGC container,
a few focused scripts, one push to HF Hub. Follow the rules in this file; don't
improvise.

**Order of authority (highest first):**

1. **User input** — explicit `model_id`, `dataset_id`, `training_method`, `config.yaml` overrides.
2. **Live research** — model card, HF repo example, author finetune script, HF task docs, paper; always fetched (Step 3 + `references/research-priorities.md`).
3. **Curated references** (`references/*.md`) — fallback when live research is silent/ambiguous.
4. **Your training-data memory** — last resort; suspect, cross-check against (2)/(3).

Conflict resolution between (2) and (3) and the source-line discrepancy note are
in `references/research-priorities.md`.

---

## Inputs

**Required:**
- `model_id` — HuggingFace model ID, e.g. `google/vit-base-patch16-224`

**Conditional credentials (read from the session environment, exported before launching when present):**
- `HF_TOKEN` — only when the model/dataset is **gated** (read) or `push_to_hub` is on (write); public + public + `push_to_hub: false` needs none. Value never read — presence-only via `[ -n "$HF_TOKEN" ]`.
- `WANDB_API_KEY`, `WANDB_PROJECT` — only when WandB is enabled; `WANDB_MODE=disabled` opts out.

**Dataset — exactly one:**
- `dataset_id` — HuggingFace dataset ID *(source: `hf`)*
- `local_dataset_path` — local folder or file *(source: `local`)*; optional
  `local_dataset_format` ∈ {auto, imagefolder, coco, voc, jsonl, arrow, parquet,
  csv} (default: auto-detect).
- *(omit)* — agent recommends popular datasets *(source: `recommend`)*

**Optional (have defaults):**
- `task_type` — auto-detected from config + model card
- `n_train=10000`, `n_eval=1000`, `n_epochs=3`, `lora_r=16`
- `output_dir=./output/<model_short_name>`
- `hf_model_repo` — push target; if unset and HF_TOKEN has write access,
  auto-derived as `<whoami>/<model_short_name>-finetuned`.
- `push_to_hub=True` — set to `False` to skip
- `skip_baseline=False` — skip zero-shot baseline eval

**Optional deliverables (off by default):**
```yaml
emit_progress_log: false   # output_dir/PROGRESS.md (per-step journal)
emit_report:       false   # reports/report.{pdf,html} with curves & samples
emit_unit_tests:   false   # tests/ with fake-data heterogeneous-batch tests
```

All values live in `output_dir/config.yaml`. Never hardcode in Python.

---

## Execution platform

This skill orchestrates *what* to run; the platform skills own *how* to run it on
a GPU host — read them first.

| Concern | Authoritative skill |
|---|---|
| GPU host runtime (driver 580, CUDA Toolkit 13.0, NVIDIA Container Toolkit 1.19.0) | [`tao-skill-bank:tao-setup-nvidia-gpu-host`](../../platform/tao-setup-nvidia-gpu-host/SKILL.md) |
| `docker run` flags, NGC auth, mounts, env passthrough | [`tao-skill-bank:tao-run-on-docker`](../../platform/tao-run-on-docker/SKILL.md) |
| Local Docker job preflight (daemon, GPU smoke) | [`tao-skill-bank:tao-run-on-local-docker`](../../platform/tao-run-on-local-docker/SKILL.md) |

**Default platform:** `local-docker` — build a one-off image (`run-<short>:latest`)
and run it on the local Docker daemon. Ask only when the user explicitly needs a
different backend (Brev remote GPU, SLURM/Kubernetes); then run that platform's
Preflight first and route the Steps 4–5 `docker run` commands through it. The
GPU-runtime and presence-only credential preflights (values never read), the
canonical `docker run` flag set, the `list_tao_platforms.py` selection command, and
the workflow-specific flags (`--entrypoint /bin/bash -lc`, `PYTORCH_CUDA_ALLOC_CONF`,
`--name hft_train`) are in `references/workflow-intake-preflight.md`.

---

## References — fallback safety net

Consulted **only** when live research is silent, ambiguous, or unavailable; live
docs always win for the specific model and current API. Each step links the
references it needs; full catalog in `references/detailed-workflow.md`.

Always-on: `core-rules.md`, `error-playbook.md`, `compat-workarounds.md`,
`model-discovery.md`, `dataset-recommendations.md`, `dataset-sources.md`,
`dataset-patterns.md`, `hardware-container.md`, `research-priorities.md`,
`cv-scripts.md`, `vlm-scripts.md`, `docker-runs.md`, `hub-push.md`,
`pipeline-skill-template.md`, `deliverables.md`. Opt-in (when their flag/need
applies): `progress-tracking.md`, `testing.md`, `reporting.md`,
`workflow-intake-preflight.md`, `workflow-generate-train.md`, `workflow-push-rerun.md`.

**Rule:** before falling back, log the live source you tried and why it was
insufficient (`config.yaml` `notes:`, and PROGRESS.md if enabled). `[FETCH LIVE]`
markers in `cv-scripts.md` / `vlm-scripts.md` are a research checklist, not code to
inline — refetch the listed URL if a block has no Step 3 finding.

---

## Core rules

Non-negotiable behaviors. **Short version** (full enumeration —
hallucinated-imports list, never-without-approval list, full error-recovery and
hardware-sizing tables — in `references/core-rules.md`, consult before any
training-time decision):

- **Your HF-library knowledge is outdated.** Fetch live docs (model card, HF
  repo example, task doc) before writing any ML code — don't generate trainer
  args / collator / transforms from memory (Step 3).
- **Smoke-test on real data with `--max_steps 1`** before any full run; no batch
  launches without a verified smoke.
- **Never silently substitute** model_id, dataset_id, or training_method — if
  what the user asked for doesn't load, stop and ask.
- **Error recovery is minimal-change.** OOM → halve batch, double grad_accum,
  enable gradient checkpointing (no LoRA switch without approval); NaN → reduce
  LR 10×; flat loss → inspect collator; same error 3× → stop and ask. Don't loop.
- **Dataset columns verified BEFORE the collator** — rename in `prepare_data.py`;
  restructuring needed → stop and ask.
- **Hardware-sizing thumb (bf16):** ≤3B → 24 GB, 7–13B → 80 GB, 30B+ → multi-GPU
  or LoRA on 1× 80 GB, 70B+ → 8× 80 GB or LoRA. Full finetune won't fit and no
  LoRA requested → ask before switching.

---

## Workflow — 6 steps

Single pass, sequential; each step has a clear gate before the next begins.

### Step 1 — Inspect & qualify

**Goal:** decide whether to proceed. Probe model + dataset, apply accept/reject,
register applicable compat fixes, write the initial `config.yaml`.

Prerequisites: `MODEL_ID`, optional `DATASET_ID` / `local_dataset_path`,
optional `HF_TOKEN`, `OUTPUT_DIR` (default `./output/<model_short_name>`). Probes
run in a CPU-only `python:3.12-slim` Docker container (bind-mounted `.probe/`
scratch) so the host needs no virtualenv — Docker must exist first. Docker-presence
guard, container env, full probe invocation, and the model/dataset probe scripts
are in `references/workflow-intake-preflight.md`, `references/model-discovery.md`,
and `references/dataset-sources.md`.

Probe requirements:

- Model: load `AutoConfig`, read model-card tags, detect task from
  `architectures` + tags + card examples (fallback logging in `model-discovery.md`).
- Dataset: for recommended datasets, first present 3-5 choices from
  `dataset-recommendations.md`; for local data, bind-mount read-only and use
  `dataset-sources.md` format detection.
- Reject early if the model config fails, the task is out of scope, no recipe
  source exists, or the dataset cannot load / match the task schema.
- Evaluate `compat-workarounds.md` against the model/task; defer hardware-dependent
  rules to Step 2.

Write the initial `config.yaml` (`model_id`, `task`, `dataset_id` or
`local_dataset_path`, `research_sources: []` filled in Step 3,
`applicable_workarounds:` from Step 1, `notes: []` for reference fallbacks,
`push_to_hub: true` default — annotated template in
`references/workflow-intake-preflight.md`). Optionally `rm -rf "$OUTPUT_DIR/.probe"`
once the gate is met.

**Gate:** `config.yaml` exists with model, dataset, task, applicable_workarounds;
do not proceed if any field is missing.

---

### Step 2 — Hardware audit & NGC image

**Goal:** verify Docker + GPU + disk, pick the NGC PyTorch image live, finalize
hardware-dependent compat rules.

**2a. Audit (hard gate)** — three checks (commands in
`references/workflow-intake-preflight.md`):
1. GPU host runtime — `tao-setup-nvidia-gpu-host`'s
   `setup-nvidia-gpu-host.sh --backend docker --check-only`; on fail, ask approval
   then re-run with `--install --yes`.
2. Free-disk soft-warn — override via `MIN_DISK_GB` (default 100 GB); recommend
   ≥ 100 GB for NGC base (~20 GB) + HF cache + checkpoints + data.
3. Conditional credential presence (from the session environment, values never
   read) — `HF_TOKEN` only when gated or `push_to_hub` is on; `WANDB_*` only when
   WandB is on.

**Do not proceed to Step 4 on a hard-fail** — Step 4's `docker build` pulls a
20+ GB NGC base, and a missing `nvidia-container-toolkit` only surfaces later as
`could not select device driver "" with capabilities: [[gpu]]`. Record `gpu_count`,
`gpu_name`, `driver_major`, `vram_gb_per_gpu` in `config.yaml`.

**2b. Pick NGC image (live):** from the NVIDIA Deep Learning Frameworks support
matrix (<https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>),
PyTorch NGC container section, pick the highest-versioned image where
`Min driver ≤ detected driver_major` and container CUDA `≤` host CUDA Toolkit
(match closely so cuDNN / TensorRT line up). Do **not** reject an image for an
`aN`/`bN`/`rcN` PyTorch tag — NGC validates the full image; pick the newest
CUDA-aligned one and let `compat-workarounds.md` handle per-version issues. If the
matrix is unreachable, use the fallbacks in `references/hardware-container.md`;
default `nvcr.io/nvidia/pytorch:24.09-py3` (driver ≥ 545; SDPA+GQA bug — if
`num_key_value_heads < num_attention_heads`, set `attn_implementation: "eager"`).
Record `ngc_image` in `config.yaml`.

**2c. Re-evaluate hardware-dependent compat rules:** re-run the
`compat-workarounds.md` walk for entries whose `detect` needs `hw`; update
`applicable_workarounds:` in place.

**2d. Model-fit check:** estimate `param_bytes ≈ 2×param_count` (bf16); if
> 60% of `vram_gb_per_gpu × 1e9`, recommend LoRA in the user-facing summary.

**Gate:** `config.yaml` has `ngc_image`, `gpu_count`, `gpu_name`, `driver_major`,
`vram_gb_per_gpu`; hardware-dependent compat fixes recorded.

---

### Step 3 — Research the recipe

**Goal:** fetch the live recipe — training-data knowledge of
`transformers`/`trl`/`peft` is suspect, so Step 3 is non-negotiable. Walk
`references/research-priorities.md` in priority order (Priority 1 → 6); stop once
you have, for the detected task:

- `AutoModel` / processor class
- Train + eval transforms
- Collator
- `compute_metrics`
- Hyperparameter hints (LR, batch size, epochs, scheduler)

Record findings in `meta/recipe.md`, append source URLs to
`config.yaml: research_sources:`. A slot with no live finding falls back to the
matching scaffold (`cv-scripts.md` / `vlm-scripts.md`), logged as "fallback to
scaffold — no live source for <slot>" under `notes:`. Conflict-resolution rules
are in `references/research-priorities.md`.

**Gate:** every required slot filled, with a source URL or scaffold-fallback note.

---

### Step 4 — Generate project & smoke-test

**Goal:** write all scripts, build the image, prepare data, run a 1-step smoke on
real data (one `docker build`, two `docker run`s).

**4a. Generate project files** in `output_dir/`: `config.yaml`, `Dockerfile`,
`requirements.txt`, `prepare_data.py`, `train.py`, `run_eval.py`, `infer.py`,
optional `merge_lora.py`, optional `tests/`, `.gitignore`. Live Step 3 research is
authority; `cv-scripts.md` / `vlm-scripts.md` give scaffold shape only. Apply every
`applicable_workarounds` entry as a Dockerfile block, requirement pin, config
override, or runtime env var. Hard rules: `run_eval.py` keeps that exact filename
(avoids colliding with the HF `evaluate` package); every generated `.py` starts
with the NVIDIA Apache-2.0 copyright header and any emitter fails when it is
missing; `emit_unit_tests: true` generates and runs tests per
`references/testing.md`. Script bodies, Dockerfile shape, and the emitter contract
are in `references/workflow-generate-train.md`.

**4b. Build, prepare, smoke** — `docker build -t run-<short>:latest .`, then
`prepare_data` and the `--smoke --max_steps 1` run (`references/docker-runs.md`
§1-3). Smoke pass criteria (in `logs/smoke.log`):
- No exception
- Loss is finite (not `0.0`, not `NaN`)
- `grad_norm > 0` at step 1

If `emit_unit_tests: true`, also run `pytest tests/` in the container. Any failure → STOP.

**4c. Preflight summary** — before full training, print and verify: reference URL,
dataset columns, Hub target, monitoring target, NGC image, hardware, smoke loss/grad norm.

**Gate:** project files written, image built, smoke PASSED, preflight has no
blank fields.

---

### Step 5 — Train, evaluate, infer

**Goal:** baseline eval, full training, post-train eval, optional LoRA merge, 5
inference samples (all commands: `references/docker-runs.md` §4-8).

| Sub-step | docker-runs.md | Skip if |
|---|---|---|
| 5a. Baseline eval (zero-shot) | §4 | `skip_baseline: true` |
| 5b. Full training (detached) | §5 | — |
| 5c. LoRA merge | §6 | not VLM+LoRA |
| 5d. Post-train eval | §7 | — |
| 5e. Inference (5 samples) | §8 | — |

Multi-GPU: prepend `torchrun --nproc_per_node=$gpu_count` to `python train.py`.

While training streams, watch `docker logs -f hft_train`: loss should drop within
10-20 steps; flat loss (collator/label-masking bug), NaN (LR too high), and OOM
all stop the run — recovery in `references/core-rules.md`. If `emit_report: true`,
run `report.py` after Step 5e per `references/reporting.md`.

**Gate:** all of:
- `checkpoints/final/` (or `checkpoints/merged/` for LoRA) exists
- `reports/eval_results.json` has a numeric primary metric
- `reports/baseline_results.json` exists (unless skipped)
- `reports/inference_samples/` has 5 samples
- wandb URL shows descending loss

---

### Step 6 — Push & emit rerun skill

**Goal:** publish the run and make it reproducible without re-research.

Push per `references/hub-push.md` (weights, model card, eval/baseline JSONs,
`config.yaml`, `Dockerfile`, `requirements.txt`, inference samples, reports when
emitted) unless `push_to_hub: false` is explicit. Emit
`<output_dir>/skills/run-<short>/SKILL.md` from
`references/pipeline-skill-template.md` — substitute every placeholder, include
full YAML metadata + the NVIDIA copyright HTML comment, and make any emitter fail
if those are missing.

**Gate (Done criteria):** all of:
- Step 5 gate met
- HF Hub repo exists at the resolved URL with weights + card + `results/`
  (unless `push_to_hub: false`)
- `<output_dir>/skills/run-<short>/SKILL.md` exists, no `<placeholder>` left,
  with metadata + copyright HTML comment per `pipeline-skill-template.md`

Final message: wandb URL, HF Hub URL, baseline -> fine-tuned primary metric,
`reports/inference_samples/`, and the rerun skill path.

---

## Error playbook

On a known runtime error, consult the symptom → minimal-fix table in
`references/error-playbook.md` (NGC entrypoint, PyTorch/Transformers regressions,
numpy ABI, Albumentations bbox, PEFT/checkpointing, LoRA target breadth, CV
augmentation gaps, OOM at step 0) before redesigning anything. When a row there
fires twice across runs, lift it into `compat-workarounds.md` with a `detect` rule
— auto-applied in Step 1 before the error can fire.

---

## Communication style

- Terse. No filler, no restating the request; one-word answers when appropriate.
- Always include direct Hub and wandb URLs when referencing artifacts.
- On error: state what went wrong, why, what you changed — no menus.
- Never present "Option A/B/C" for a request with a clear answer. Act.

## Example pipelines

- [tao-rerun-convnext-cifar10](references/tao-rerun-convnext-cifar10.md)
- [tao-rerun-detr-cppe5](references/tao-rerun-detr-cppe5.md)
- [tao-rerun-segformer-foodseg103](references/tao-rerun-segformer-foodseg103.md)
- [tao-rerun-smolvlm-vqav2](references/tao-rerun-smolvlm-vqav2.md)
