# HuggingFace Fine-Tune Generate And Train

Detailed research, project generation, smoke testing, training, evaluation, and inference steps from the pre-refactor guide.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- Step 4 — Generate project & smoke-test
- Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
- Licensed under the Apache License, Version 2.0 (the "License");
- you may not use this file except in compliance with the License.
- You may obtain a copy of the License at
- http://www.apache.org/licenses/LICENSE-2.0
- Unless required by applicable law or agreed to in writing, software
- distributed under the License is distributed on an "AS IS" BASIS,
- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
- See the License for the specific language governing permissions and
- limitations under the License.
- {{COMPAT_DOCKERFILE_BLOCKS}}     ← injected from applicable_workarounds
- {{COMPAT_ENV_VARS}}                ← injected from applicable_workarounds
- → docker-runs.md §2: prepare_data
- → docker-runs.md §3: smoke (--smoke --max_steps 1)
- Step 5 — Train, evaluate, infer

### Step 3 — Research the recipe

**Goal:** fetch the live recipe. The agent's training-data knowledge of
`transformers`/`trl`/`peft` is treated as suspect — Step 3 is non-negotiable.

Walk `references/research-priorities.md` in priority order (Priority 1 → 6).
Stop once you have, for the detected task:

- `AutoModel` / processor class
- Train + eval transforms
- Collator
- `compute_metrics`
- Hyperparameter hints (LR, batch size, epochs, scheduler)

Record findings in `meta/recipe.md` and append source URLs to
`config.yaml: research_sources:`. If a slot has no live finding, fall back to
the matching scaffold reference (`cv-scripts.md` / `vlm-scripts.md`) and log
"fallback to scaffold — no live source for <slot>" under `notes:`.

**Conflict resolution rules** are in `references/research-priorities.md`.

**Gate:** every required slot above is filled, with a source URL or an explicit
scaffold-fallback note.

---

### Step 4 — Generate project & smoke-test

**Goal:** write all scripts, build the image, prepare data, run a 1-step smoke
on real data. One `docker build`, two `docker run`s.

**4a. Generate project files** in `output_dir/`:

| File | From | Notes |
|---|---|---|
| `config.yaml` | Steps 1-3 + user input | already started |
| `Dockerfile` | template below + compat injections | layer order: deps → compat → code |
| `requirements.txt` | task baseline + compat pins | don't pin without cause |
| `prepare_data.py` | scaffold + Step 3 | save Arrow to `data/{train,eval}` |
| `train.py` | scaffold + Step 3 recipe | reads `config.yaml`, supports `--smoke --max_steps N` |
| `run_eval.py` | scaffold + Step 3 | **MUST** be `run_eval.py` (collides with HF `evaluate` lib if named `evaluate.py`) |
| `infer.py` | scaffold + Step 3 | writes `reports/inference_samples/<i>_input.jpg`, `_pred.jpg`, `_meta.json` |
| `merge_lora.py` | scaffold | only for VLM with LoRA |
| `.gitignore` | `data/`, `checkpoints/`, `logs/`, `wandb/`, `reports/inference_samples/`, `.env`, `__pycache__/`, `*.pyc`, `.cache/`, `.probe/` | |

Authority order while writing: live research from Step 3 → scaffold reference
(`cv-scripts.md` / `vlm-scripts.md`) for **structure only**, never their
`[FETCH LIVE]` blocks. Apply each `applicable_workarounds` entry: Dockerfile
blocks, requirements pins, config overrides, runtime env vars.

Every generated `.py` file (`prepare_data.py`, `train.py`, `run_eval.py`,
`infer.py`, `merge_lora.py`, and any `tests/*.py`) must start with the NVIDIA
Apache-2.0 copyright header as a `#`-prefixed comment block — same text as the
HTML copyright comment used in the rerun skill, just commented for Python:

```python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

If you generate an emitter script, make it fail unless every emitted `.py`
begins with that header.

If `emit_unit_tests: true`, also generate `tests/` per `references/testing.md`.

**Dockerfile template:**

```dockerfile
ARG NGC_IMAGE=nvcr.io/nvidia/pytorch:24.09-py3
FROM ${NGC_IMAGE}

ENTRYPOINT ["/bin/bash", "-c"]
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# {{COMPAT_DOCKERFILE_BLOCKS}}     ← injected from applicable_workarounds
# {{COMPAT_ENV_VARS}}                ← injected from applicable_workarounds

COPY *.py ./
COPY config.yaml ./
```

**4b. Build, prepare, smoke** (commands: `references/docker-runs.md` §1-3):

```bash
docker build -t run-<short>:latest .
# → docker-runs.md §2: prepare_data
# → docker-runs.md §3: smoke (--smoke --max_steps 1)
```

Smoke pass criteria (in `logs/smoke.log`):
- No exception
- Loss is finite (not `0.0`, not `NaN`)
- `grad_norm > 0` at step 1

If `emit_unit_tests: true`, also run `pytest tests/` inside the container.
Failure → STOP. Do not proceed.

**4c. Preflight summary** — print and verify every field is filled before
launching full training:

```
─ PREFLIGHT ────────────────────────────────────────
reference implementation:  <URL from Step 3>
dataset columns verified:  <col1, col2, …>
push_to_hub:               <repo_id>
monitoring:                wandb <project>/<run_name>
ngc_image:                 <image tag>
hardware:                  <gpu_count>× <gpu_name>
smoke test:                PASSED (loss=X.XX, grad_norm=Y.YY)
────────────────────────────────────────────────────
```

**Gate:** project files written, image built, smoke PASSED, preflight has no
blank fields.

---

### Step 5 — Train, evaluate, infer

**Goal:** baseline eval, full training, post-train eval, optional LoRA merge,
5 inference samples. All commands: `references/docker-runs.md` §4-8.

| Sub-step | docker-runs.md | Skip if |
|---|---|---|
| 5a. Baseline eval (zero-shot) | §4 | `skip_baseline: true` |
| 5b. Full training (detached) | §5 | — |
| 5c. LoRA merge | §6 | not VLM-with-LoRA |
| 5d. Post-train eval | §7 | — |
| 5e. Inference (5 samples) | §8 | — |

Multi-GPU: prepend `torchrun --nproc_per_node=$gpu_count` to `python train.py`.

While training streams, watch `docker logs -f hft_train` for:
- Loss drops within 10-20 steps → working
- Flat loss → collator / label-masking bug; stop
- NaN loss → LR too high; stop, reduce LR, retry
- OOM → halve batch, double grad_accum, enable gradient checkpointing

If `emit_report: true`, run `report.py` after Step 5e per `references/reporting.md`.

**Gate:** all of:
- `checkpoints/final/` (or `checkpoints/merged/` for LoRA) exists
- `reports/eval_results.json` has a numeric primary metric
- `reports/baseline_results.json` exists (unless skipped)
- `reports/inference_samples/` has 5 samples
- wandb URL shows descending loss

---
