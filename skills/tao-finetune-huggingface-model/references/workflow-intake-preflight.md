# HuggingFace Fine-Tune Intake And Preflight

Detailed intake, platform delegation, reference routing, core rules, model/dataset probing, and hardware preflight from the pre-refactor guide.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- Inputs
- Execution platform
- References — fallback safety net
- Always-on (consulted in the workflow)
- Opt-in (only when their flag is set)
- Core rules
- Workflow — 6 steps
- Step 1 — Inspect & qualify
- HF source loadability + schema probe (catches gated / script-based / missing)
- Step 2 — Hardware audit & NGC image
- 1) GPU host runtime — delegated to tao-setup-nvidia-gpu-host
- 2) Free-disk soft-warn (override via MIN_DISK_GB; default 100 GB)
- 3) Conditional credential presence checks (no values are read)
- HF_TOKEN: only when the model/dataset is gated, or push_to_hub is on.
- WANDB_*:  only when WandB logging is enabled in config.yaml.

# tao-finetune-huggingface-model

Local NVIDIA GPU fine-tuning for HuggingFace models, grounded in live-fetched
documentation, with curated references as a fallback safety net. One NGC
container, a small set of focused scripts, one push to HF Hub. Behavior is
governed by the rules in this file — follow them, do not improvise around them.

**Order of authority (highest first):**

1. **User input** — explicit `model_id`, `dataset_id`, `training_method`, `config.yaml` overrides.
2. **Live research** — model card, HF repo example, author finetune script, HF task docs, paper. Always fetched. See Step 3 + `references/research-priorities.md`.
3. **Curated references** (`references/*.md`) — fallback when live research is silent or ambiguous.
4. **Your training-data memory** — last resort. Treat as suspect; cross-check against (2) or (3).

If (2) and (3) conflict on an API call, (2) wins (newer). If they conflict on a
method detail (collator, LoRA targets, augmentation), (2) wins for the *specific*
model; (3) for the generic shape. Note the discrepancy in a comment at the source
line.

---

## Inputs

**Required:**
- `model_id` — HuggingFace model ID, e.g. `google/vit-base-patch16-224`

**Conditional credentials (read from the session environment, exported before launching when present):**
- `HF_TOKEN` — required only when the model or dataset is **gated** (read access) or `push_to_hub` is on (write access). Public model + public dataset + `push_to_hub: false` runs do not need it. The agent never reads the value — only checks presence with `[ -n "$HF_TOKEN" ]`.
- `WANDB_API_KEY`, `WANDB_PROJECT` — required only when WandB monitoring is enabled. Set `WANDB_MODE=disabled` to opt out.

**Dataset — exactly one:**
- `dataset_id` — HuggingFace dataset ID *(source: `hf`)*
- `local_dataset_path` — local folder or file *(source: `local`)*. Optionally
  `local_dataset_format` ∈ {auto, imagefolder, coco, voc, jsonl, arrow,
  parquet, csv}. Default: auto-detect.
- *(omit)* — agent recommends popular datasets *(source: `recommend`)*

**Optional (have defaults):**
- `task_type` — auto-detected from config + model card
- `n_train=10000`, `n_eval=1000`, `n_epochs=3`, `lora_r=16`
- `output_dir=./output/<model_short_name>`
- `hf_model_repo` — push target. If unset and HF_TOKEN has write access,
  auto-derived as `<whoami>/<model_short_name>-finetuned`.
- `push_to_hub=True` — set explicitly to `False` to skip
- `skip_baseline=False` — skip zero-shot baseline eval

**Optional deliverables (off by default):**
```yaml
emit_progress_log: false   # output_dir/PROGRESS.md (per-step ✅/⚠️/❌ journal)
emit_report:       false   # reports/report.{pdf,html} with curves & samples
emit_unit_tests:   false   # tests/ with fake-data heterogeneous-batch tests
```

All values live in `output_dir/config.yaml`. Never hardcode in Python.

---

## Execution platform

This skill orchestrates *what* to run; the platform skills own *how* to run it
on a GPU host. Read those skills first and do not redraft their conventions
here.

| Concern | Authoritative skill |
|---|---|
| GPU host runtime — NVIDIA driver 580, CUDA Toolkit 13.0, NVIDIA Container Toolkit 1.19.0 | [`tao-skill-bank:tao-setup-nvidia-gpu-host`](../../platform/tao-setup-nvidia-gpu-host/SKILL.md) |
| `docker run` flags, NGC auth, `--gpus`, mounts, env passthrough, `--ipc=host`/`--shm-size`, common error modes | [`tao-skill-bank:tao-run-on-docker`](../../platform/tao-run-on-docker/SKILL.md) |
| Local Docker job preflight (daemon reachable, GPU smoke) | [`tao-skill-bank:tao-run-on-local-docker`](../../platform/tao-run-on-local-docker/SKILL.md) |

**Default platform:** `local-docker`. This workflow builds a one-off image
(`run-<short>:latest`) and runs it on the local Docker daemon — the same
pattern documented in `skills/platform/tao-run-on-local-docker/SKILL.md`. Ask the user only when
they explicitly need a different backend (Brev for a remote GPU instance,
SLURM/Kubernetes for managed scheduling); in that case run the chosen
platform's Preflight section first, generate the choices via
`${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py
--format text`, then route the `docker run` commands in Steps 4–5 through that
platform's execution pattern.

**GPU runtime preflight:** Step 2a runs the `tao-setup-nvidia-gpu-host` skill's
`--check-only` mode. Do not duplicate the NCT / driver / `--gpus all` smoke
logic here — if it needs to change, change it in `tao-setup-nvidia-gpu-host`.

**Credentials preflight:** credentials are read from the session
environment (exported in your shell before launching); the SessionStart hook
(`hooks/session_start.sh`) lists the variable names (never values) in the session banner.
Step 2a only confirms presence of credentials that the current run
*actually* needs — `HF_TOKEN` for gated downloads or `push_to_hub`,
`WANDB_API_KEY`/`WANDB_PROJECT` if WandB is enabled — instead of hard-
requiring them up front.

**Docker run conventions:** every `docker run` invocation in
`references/docker-runs.md` follows the canonical flag set from
`skills/platform/tao-run-on-docker/SKILL.md` (`--gpus all`, `--ipc=host` or `--shm-size=…`,
`-e VAR` passthrough, bind mounts, `--rm` for one-shots). Treat that skill as
the spec; this one only adds workflow-specific flags
(`--entrypoint /bin/bash -lc`, `PYTORCH_CUDA_ALLOC_CONF`, `--name hft_train`).

---

## References — fallback safety net

Consulted **only** when live research is silent, ambiguous, or unavailable. Live
docs always win for the specific model and current API.

### Always-on (consulted in the workflow)

| File | Step | Role |
|---|---|---|
| `core-rules.md` | all | Non-negotiable agent behaviours — full enumeration of the rules summarised in SKILL.md |
| `error-playbook.md` | 4, 5 | Runtime-error symptom → minimal-fix table (consulted on every failure) |
| `compat-workarounds.md` | 1 | Known-issue registry; auto-applied via `detect` rules |
| `model-discovery.md` | 1 | `model_type` → AutoModel/processor mapping (when card silent) |
| `dataset-recommendations.md` | 1 | Vetted datasets for `source = recommend` |
| `dataset-sources.md` | 1 | Local format detectors + COCO/VOC/imagefolder/jsonl loaders |
| `dataset-patterns.md` | 4 | Universal `prepare_data.py` skeleton |
| `hardware-container.md` | 2 | NGC selection (offline fallback), GPU/disk audit, multi-GPU |
| `research-priorities.md` | 3 | 6-priority live-fetch ladder + extract/record + conflict rules |
| `cv-scripts.md` | 4 | CV scaffold (file names, CLI, config schema). **Don't copy `[FETCH LIVE]` blocks** |
| `vlm-scripts.md` | 4 | VLM/LLM scaffold (TRL/PEFT). **Don't copy `[FETCH LIVE]` blocks** |
| `docker-runs.md` | 4, 5 | Canonical `docker run` invocations for every command |
| `hub-push.md` | 6 | HF Hub push Python block + model card template |
| `pipeline-skill-template.md` | 6 | `run-<short>/SKILL.md` rerun template |
| `deliverables.md` | 4, 6 | Final directory layout + README results section |

### Opt-in (only when their flag is set)

| File | Flag | Adds |
|---|---|---|
| `progress-tracking.md` | `emit_progress_log: true` | PROGRESS.md template |
| `testing.md` | `emit_unit_tests: true` | Fake-data heterogeneous-batch tests |
| `reporting.md` | `emit_report: true` | `report.py` (PDF + HTML, reads `trainer_state.json`) |

**Rule:** before falling back to a reference, log the live source you tried and
why it was insufficient (in `config.yaml` `notes:`, and PROGRESS.md if enabled).

**`[FETCH LIVE]` markers in `cv-scripts.md` / `vlm-scripts.md`** are a
research checklist, not code to inline. If a `[FETCH LIVE]` block has no Step 3
finding, refetch the listed URL.

---

## Core rules

The non-negotiable behaviors the agent must follow across the workflow. Full
text in `references/core-rules.md`. **Short version:**

- **Your HF-library knowledge is outdated.** Fetch live docs (model card, HF
  repo example, task doc) before writing any ML code. Don't generate trainer
  args / collator / transforms from memory — see Step 3.
- **Smoke-test on real data with `--max_steps 1`** before any full run. No
  batch launches without a verified smoke.
- **Never silently substitute** model_id, dataset_id, or training_method. If
  what the user asked for doesn't load, stop and ask.
- **Error recovery is minimal-change.** OOM → halve batch, double grad_accum,
  enable gradient checkpointing — don't switch to LoRA without approval. NaN
  → reduce LR 10×. Flat loss → inspect collator. Same error 3× → stop and
  ask. Don't loop.
- **Dataset columns verified BEFORE writing the collator.** Mismatch +
  rename → fix in `prepare_data.py`; restructuring needed → stop and ask.
- **Hardware-sizing rule of thumb (bf16):** ≤3B → 24 GB, 7–13B → 80 GB, 30B+ →
  multi-GPU or LoRA on 1× 80 GB, 70B+ → 8× 80 GB or LoRA. If a full finetune
  won't fit and the user didn't ask for LoRA, ask before switching.

Consult `references/core-rules.md` for the full enumeration (hallucinated
imports list, never-without-approval list, full error-recovery table, full
hardware sizing table) before training-time decisions.

---

## Workflow — 6 steps

Single pass, sequential. Each step has a clear gate before the next begins.

### Step 1 — Inspect & qualify

**Goal:** decide whether to proceed at all. Probe model, probe dataset, apply
accept/reject, register applicable compat fixes, write the initial `config.yaml`.

**Prerequisites (assumed set by the calling agent):**
- `MODEL_ID`, optional `DATASET_ID`, optional `HF_TOKEN` (loaded from the
  SessionStart hook when present).
- `OUTPUT_DIR` — defaults to `./output/<model_short_name>`. Same variable
  Steps 4–5 bind-mount into the training container, so any HF/pip cache the
  probe leaves behind under `$OUTPUT_DIR/.probe/.cache` survives for later
  inspection but is gitignored.

**Preflight — Docker must be installed.** Step 1's probes run inside Docker
(no host venv / pip needed), so Docker has to exist on the host before
Step 1a. The full GPU-runtime preflight still happens in Step 2a — this
just covers the Docker-daemon prereq earlier so the probe's `docker run`
doesn't fail with a bare `docker: command not found`:

```bash
TAO_SKILL_BANK_ROOT="${TAO_SKILL_BANK_PATH:-${TAO_SKILL_BANK_ROOT:-$PWD}}"
SETUP_SCRIPT="${TAO_SKILL_BANK_ROOT}/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh"

if ! command -v docker >/dev/null 2>&1; then
  echo "MISSING: docker is required for Step 1's containerized probe."
  echo "After user approval, run the platform installer (same one Step 2a uses):"
  echo "  bash \"$SETUP_SCRIPT\" --backend docker --install --yes"
  echo "Then re-source your shell or 'newgrp docker' so the new group membership applies."
  exit 1
fi
```

If you'd rather front-load the full driver/CUDA/NCT preflight (recommended
on a fresh host), just call `bash "$SETUP_SCRIPT" --backend docker --check-only`
here — same invocation Step 2a uses, repeated calls are cheap.

**1a. Probe model:**

The probe runs inside a small CPU-only `python:3.12-slim` container so the
host needs no Python prereqs (`python3-pip`, `python3-venv`, distro-managed
Python). Save the script to `$OUTPUT_DIR/.probe/model_probe.py` first so
it's diff-able, then run it with a bind-mounted scratch dir for cache reuse.

Docker rejects relative paths in `-v` (anything not starting with `/` is
parsed as a named-volume name and fails for `./output/...`). The snippet
normalizes `$OUTPUT_DIR` to an absolute path with a single bash case
before any `mkdir` / `cat` / `docker run`, so both the default
relative `./output/<short>` and an explicit absolute override resolve
correctly:

```bash
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$(pwd)/$OUTPUT_DIR" ;;
esac
mkdir -p "$OUTPUT_DIR/.probe/.cache"
cat > "$OUTPUT_DIR/.probe/model_probe.py" <<'PY'
import os, sys
from transformers import AutoConfig
from huggingface_hub import model_info
mid = os.environ["MODEL_ID"]; tok = os.environ.get("HF_TOKEN") or None  # optional — public models work without it
try:
    cfg = AutoConfig.from_pretrained(mid, token=tok, trust_remote_code=True)
except Exception as e:
    # If this is a gated model, the error message will name 401/access-denied;
    # tell the user to export HF_TOKEN and retry.
    print(f"REJECT: AutoConfig failed — {e}"); sys.exit(1)
info = model_info(mid, token=tok)
print("model_type:", cfg.model_type)
print("architectures:", getattr(cfg, "architectures", []))
print("tags:", info.tags)
print("hidden_size:", getattr(cfg, "hidden_size", None))
print("num_kv_heads:", getattr(cfg, "num_key_value_heads", None))
print("num_attn_heads:", getattr(cfg, "num_attention_heads", None))
PY

docker run --rm \
  --user $(id -u):$(id -g) \
  -e HOME=/probe -e PIP_USER=1 \
  -e MODEL_ID="$MODEL_ID" -e HF_TOKEN \
  -e HF_HOME=/probe/.cache -e PIP_CACHE_DIR=/probe/.cache/pip \
  -v "$OUTPUT_DIR/.probe":/probe -w /probe \
  python:3.12-slim \
  bash -c "pip install -q transformers huggingface_hub datasets Pillow && python model_probe.py"
```

Notes:
- `--user $(id -u):$(id -g)` keeps any cached files in `.probe/.cache`
  owned by the host user. Without it the cache ends up `root:root` and
  cleanup needs sudo.
- `HOME=/probe` + `PIP_USER=1` makes `pip install` resolve to
  `--user` mode (installing into `/probe/.local/lib/python3.12/site-packages`
  inside the bind mount). System `/usr/local/lib/python3.12/site-packages`
  in `python:3.12-slim` is root-owned, so without these env vars the pip
  install would fail with `PermissionError` once `--user $(id -u):$(id -g)`
  drops root. Python picks up the user-site automatically via `site.py`.
- The first invocation downloads `python:3.12-slim` (~50 MB) and a fresh set
  of HF wheels (~150 MB) into `.probe/.cache/pip` plus
  `.probe/.local/lib/python3.12/site-packages/`; subsequent probes reuse
  both.
- The probe never installs anything on the host — Docker is the only
  host-side prereq, and the Step 1 preflight above verifies it.

Detect `task` from `architectures` + `tags` + model-card body. If the card
doesn't show `from transformers import AutoModelFor...`, fall back to
`references/model-discovery.md` and log the fallback under `notes:`.

**1b. Probe dataset:**

For `source = recommend`, present 3–5 picks from
`references/dataset-recommendations.md` to the user, then re-run with the chosen
`dataset_id` / `local_dataset_path`.

Same in-container pattern as 1a — write the script to `.probe/dataset_probe.py`
first, then run it under `python:3.12-slim` with the bind-mounted cache.
Step 1b is a separate bash invocation, so it repeats the `$OUTPUT_DIR`
normalization (the variable doesn't survive across `bash -c` calls):

```bash
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$(pwd)/$OUTPUT_DIR" ;;
esac
cat > "$OUTPUT_DIR/.probe/dataset_probe.py" <<'PY'
# HF source loadability + schema probe (catches gated / script-based / missing)
import os
from datasets import load_dataset, load_dataset_builder
DID = os.environ["DATASET_ID"]; TOK = os.environ.get("HF_TOKEN") or None  # optional — public datasets work without it
try:
    load_dataset_builder(DID, token=TOK)
    ds = load_dataset(DID, split="train[:20]", token=TOK)
except Exception as e:
    print(f"REJECT dataset: {type(e).__name__}: {e}"); raise
rows = list(ds)
print("columns:", list(rows[0].keys()))
for col, val in rows[0].items():
    print(f"  {col}: {type(val).__name__}")
PY

docker run --rm \
  --user $(id -u):$(id -g) \
  -e HOME=/probe -e PIP_USER=1 \
  -e DATASET_ID="$DATASET_ID" -e HF_TOKEN \
  -e HF_HOME=/probe/.cache -e PIP_CACHE_DIR=/probe/.cache/pip \
  -v "$OUTPUT_DIR/.probe":/probe -w /probe \
  python:3.12-slim \
  bash -c "pip install -q transformers huggingface_hub datasets Pillow && python dataset_probe.py"
```

Same `HOME=/probe` + `PIP_USER=1` rationale as 1a — the install lands in
`.probe/.local/lib/python3.12/site-packages` and survives between probes
under the bind mount.

For `source = local`, see `references/dataset-sources.md` for format detection
and loaders. Bind-mount the local dataset path with an additional
`-v "<local_dataset_path>":"<local_dataset_path>":ro` so the container can
read it, and adapt `dataset_probe.py` to use the local loader instead of
`load_dataset(DID, …)`.

Verify columns match the task schema (Core rules → Dataset format). Mismatch +
rename fixes it → write the rename into `prepare_data.py`. Otherwise stop.

**1c. Apply accept/reject:**

REJECT if:
- `AutoConfig` raised
- task can't be determined
- task is not CV / VLM / SFT-LLM (out of scope)
- no recipe source exists at all (no card example, no HF repo script, no author
  finetune, no task doc, no paper)
- dataset is gated / script-based / missing (loadability probe failed)

Stop and report the specific reason. Do not proceed.

**1d. Walk compat-workarounds:**

For every entry in `references/compat-workarounds.md`, evaluate its `detect`
expression against `cfg` and the detected `task`. Hardware-dependent rules
(those needing `hw`) are deferred to Step 2.

Record matches in `config.yaml` under `applicable_workarounds:` (id + fix type +
one-line reason). Each becomes a Dockerfile block, requirements pin, config
override, or runtime env in Step 4.

**1e. Write `config.yaml` skeleton:**

```yaml
model_id: <…>
task: <…>
dataset_id: <…>             # or local_dataset_path
research_sources: []         # filled in Step 3
applicable_workarounds: [<…>]
notes: []                    # log any reference fallback
push_to_hub: true            # default
```

Optionally clean up the probe scratch dir once the gate is met:

```bash
rm -rf "$OUTPUT_DIR/.probe"
```

Keeping it around between reruns is fine — it caches `python:3.12-slim`
layers, pip wheels, and any HF model/dataset files already pulled, so a
re-probe is fast. Add `.probe/` to `.gitignore` (covered in Step 4a).

**Gate:** `config.yaml` exists with model, dataset, task, applicable_workarounds.
Do not proceed if any field is missing.

---

### Step 2 — Hardware audit & NGC image

**Goal:** verify Docker + GPU + disk, pick the NGC PyTorch image live, finalize
hardware-dependent compat rules.

**2a. Audit (hard gate):** the GPU host runtime check is owned by the
`tao-setup-nvidia-gpu-host` skill (driver branch 580, CUDA Toolkit 13.0, NVIDIA
Container Toolkit 1.19.0). Invoke it in `--check-only` mode; on failure, ask
the user to authorize the install, then re-run. Credentials are read from the
session environment (exported before launching) — only check the ones the current
run actually needs.

```bash
# 1) GPU host runtime — delegated to tao-setup-nvidia-gpu-host
TAO_SKILL_BANK_ROOT="${TAO_SKILL_BANK_PATH:-${TAO_SKILL_BANK_ROOT:-$PWD}}"
SETUP_SCRIPT="${TAO_SKILL_BANK_ROOT}/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh"

bash "$SETUP_SCRIPT" --backend docker --check-only || {
  echo "MISSING: TAO GPU host runtime not ready."
  echo "After user approval, run: bash \"$SETUP_SCRIPT\" --backend docker --install --yes"
  exit 1
}

# 2) Free-disk soft-warn (override via MIN_DISK_GB; default 100 GB)
min_disk_gb="${MIN_DISK_GB:-100}"
disk_free_gb=$(df -BG / | awk 'NR==2 {print $4}' | tr -d G)
if [ "${disk_free_gb:-0}" -lt "$min_disk_gb" ]; then
  echo "WARN: only ${disk_free_gb}G free on /; recommend ≥ ${min_disk_gb}G for NGC base (~20G) + HF cache + checkpoints + dataset." >&2
fi

# 3) Conditional credential presence checks (no values are read)
#    HF_TOKEN: only when the model/dataset is gated, or push_to_hub is on.
#    WANDB_*:  only when WandB logging is enabled in config.yaml.
```

**Do not proceed to Step 4 on a hard-fail** — Step 4's `docker build` pulls a
20+ GB NGC base image, and a missing `nvidia-container-toolkit` only surfaces
at `prepare_data.py` time as the cryptic `could not select device driver ""
with capabilities: [[gpu]]`.

Record `gpu_count`, `gpu_name`, `driver_major`, `vram_gb_per_gpu` in
`config.yaml`.

**2b. Pick NGC image (live):**

```
Fetch https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html with the available web or docs tool
```

Find the **PyTorch NGC container** section. Pick the highest-versioned image
where:
- `Min driver ≤ detected driver_major`
- Container CUDA is `≤` host CUDA Toolkit version (drivers are forward-
  compatible, but match closely so cuDNN / TensorRT versions line up with
  the host toolchain).

Do **not** reject an image because its PyTorch version carries an `aN` /
`bN` / `rcN` suffix. Every recent NGC PyTorch image ships a near-head
PyTorch build (`2.10.0a0`, `2.11.0a0`, …) — NVIDIA validates the full image
end-to-end (CUDA / cuDNN / TensorRT / NCCL / drivers / Python stack), so
the `aN` reflects upstream PyTorch's tag, not NGC instability. Treating
`aN` as disqualifying would force every run onto a ~year-old image. Pick
the newest CUDA-aligned image and let real compat workarounds
(`compat-workarounds.md`) handle any per-version issue.

If the live fetch fails: fallback rules in `references/hardware-container.md`. Default
fallback: `nvcr.io/nvidia/pytorch:24.09-py3` (driver ≥ 545; SDPA+GQA bug — if
the model has `num_key_value_heads < num_attention_heads`, set
`attn_implementation: "eager"` in config).

Record `ngc_image` in `config.yaml`.

**2c. Re-evaluate hardware-dependent compat rules:**

Re-run the `compat-workarounds.md` walk for entries whose `detect` expression
needs `hw`. Update `applicable_workarounds:` in place.

**2d. Model-fit check:** estimate `param_bytes ≈ 2×param_count` (bf16). If
> 60% of `vram_gb_per_gpu × 1e9`, recommend LoRA in the user-facing summary.

**Gate:** `config.yaml` has `ngc_image`, `gpu_count`, `gpu_name`, `driver_major`,
`vram_gb_per_gpu`. Hardware-dependent compat fixes are recorded.

---
