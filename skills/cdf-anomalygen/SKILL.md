---
name: cdf-anomalygen
description: >-
  Use when running the CDF AnomalyGen pipeline over a defect dataset — mask placement,
  fine-tuning, synthetic defect-image generation (SDG), evaluation, quality refinement, and
  pseudo-labeling — even when the user names only one stage.
license: Apache-2.0
compatibility: >-
  Requires a CUDA GPU and the built env (native .venv or the Docker image); Hugging Face auth +
  network access to download the model checkpoints.
metadata:
  owner: NVIDIA
  service: anomaly generation
  version: 1.2.0
  author: NVIDIA <nvidia@nvidia.com>
  tags:
      - physical-ai
      - anomaly-generation
      - cosmos
      - fine-tuning
  languages: [python]
  frameworks: [pytorch, cosmos, diffusion]
---

# Skill: cdf-anomalygen

## Purpose

AnomalyGen fine-tunes the frozen **Cosmos** base into an **anomaly-generation** model: given a clean
image + defect mask, it paints a realistic defect into that region (synthetic data for detectors).
Pick a mode, collect its params, run the steps — **all commands from the repo root.**

## When to Use This Skill

Use for the CDF AnomalyGen pipeline over a defect dataset — any single stage (mask placement,
fine-tuning, generation, evaluation, refinement, or pseudo-labeling) or the full run.

- "Run the AnomalyGen pipeline on phone_screen"
- "Fine-tune an anomaly-generation model for PCB defects"
- "Generate 25 synthetic PCB anomaly images"
- "Place defect masks", "evaluate quality", or "pseudo-label the generated defects"

Do not use this skill for generic image generation, or for synthetic data that isn't defect images
built from the `{texture}/{defect}` clean-image + defect-mask contract.

## Prerequisites

- A **CUDA GPU** and the project's built env (native `.venv` or the Docker image).
- **Hugging Face auth** — run `hf auth login` — plus network access; the downloader fetches
  the model checkpoints.
- A **`dataset_dir`** in the Step 1 contract layout (clean images + `{defect}` masks; plus
  `defect_spec.jsonl` **only when `mask_backend=amp`** — the default `amg` never reads it, and
  nothing after Step 2 does either).

Preflight the env + checkpoints (GPU, Python, CUDA/torch, deps, HF auth, model checkpoints):

```shell
scripts/preflight_env_ckpt.sh
```

**Read which group failed before reacting** — the summary separates them, and only one is fixed by
downloading:

- *missing checkpoints* → `scripts/download_checkpoints.sh` (run `hf auth login` first).
- *env / Python / deps* → rebuild the environment (`bash scripts/env_setup.sh`, then activate it).

**Third group — model access for mask placement (checked HERE, never mid-flow).** The default
`mask_backend=amg` needs a vision model, and a run that discovers this at Step 2 has already spent
the user's time. A **VLM API is a base, a key and a model name** — and all three live in ONE place:
**`~/.anomalygen/endpoints.yaml`**, the model registry (nothing is built into the code; the template
is `scripts/agentic_mask_generation_serving/endpoints.example.yaml`). It sits under `$HOME`, not in
the repo, so it works from any directory, serves every checkout, and is writable inside the
read-only product containers. The moment inputs are locked with `mask_backend=amg` (the default), probe:

```shell
[ -f ~/.anomalygen/endpoints.yaml ] && echo "amg model: ~/.anomalygen/endpoints.yaml present" \
    || echo "amg model: no ~/.anomalygen/endpoints.yaml — NO model can resolve; copy scripts/agentic_mask_generation_serving/endpoints.example.yaml there"
```

**The model resolves in the registry → use it.** That is the whole check: `--gen-model` and
`--judge-model` are REQUIRED flags (there is no default model), and every name they carry must have
a registry entry (`model` + `base_url` + `key`) — the entry point refuses to start when one does
not; there is no environment fallback. A user's own OpenAI-compatible endpoint is the same case:
one `endpoints.yaml` entry with their documented base URL, their key, and their model name in
`--gen-model`/`--judge-model`.

**Anything less → stop and ask the user now**, before Step 1 — unless the user's request already
names the model (e.g. "use local Gemma"): that IS the pick, act on it without re-asking. Three
options — whichever it is, set `AMG_MODEL` to the chosen name; the Step 2 command passes it to
both required flags:

- **(a) supply an API** — one `endpoints.yaml` entry: a model name, a base and a key
  (e.g. `AMG_MODEL=azure/openai/gpt-6-astra`).
- **(b) let the skill start the model service** — in a container, as a background process inside it;
  on a dev host, from `scripts/agentic_mask_generation_serving/`. Needs a GPU; the weights are a default checkpoint —
  `scripts/download_checkpoints.sh gemma` fetches them and preflight verifies them (the group
  argument fetches only this model; bare, the script fetches all eleven).
  `mkdir -p ~/.anomalygen && cp scripts/agentic_mask_generation_serving/endpoints.example.yaml ~/.anomalygen/endpoints.yaml` wires it: the template's gemma entry already routes
  `AMG_MODEL=nvidia/google/gemma-4-31b-it` to `http://localhost:8902/v1`, and the run
  carries `--beam-multi 4 --beam-pairwise --exemplar-captions --exemplar-visual-prompt --coverage-note` — the local model always pairs with the in-loop beam and the three information channels.
  Bootstrap, lifecycle and the why: [`agentic-mask-generation.md`](references/agentic-mask-generation.md).
- **(c) set `mask_backend=amp`** for this run.

Never fall back silently. Whichever they pick, the run's own capability probe confirms the model can
take images and answer in JSON before Step 2 spends anything.

## Inputs

Collect the run's inputs from the user before starting. **If the `mode` or any input it requires is
missing, ask the user first — never guess a required input or silently fall back to its default.**

**Lock the `mode` first** — it fixes which steps run and which inputs are required (ask for any not
provided):

- **`full`** — train + generate, Steps **1 → 7**. Required: `name`, `dataset_dir`, `num_sdg`.
- **`finetune_only`** — train only, Steps **1 → 3**. Required: `name`, `dataset_dir`, `num_sdg`. Here
  `num_sdg` sizes only the validation set.
- **`generation_only`** (a.k.a. `inference_only`) — generate from an existing checkpoint, Steps
  **2, 4 → 7**. Required: `name`, `checkpoint`, `recipe`, `dataset_dir`, `num_sdg`. The checkpoint's
  geometry stamp must equal `recipe`'s. A legacy checkpoint (trained before the stamp existed) has
  none: pass its run-dir recipe copy and it is stamped from that recipe at load (generation only; it
  cannot be resumed).

`name` is required in every mode — an experiment label that names the recipe, run dir, logs and
the generated mask-set dirs. It is never a dataset location: where the data lives is `dataset_dir`,
always passed separately and never derived from `name`.

**Optional in every mode** — all have working defaults, so ask only if the user raises them:
`num_gpus` (`1`), `num_search_run` (`3`), `score` (`nn`), `task` (`texture_ft`),
`base_seed` (`1`), `mask_backend` (`amg`), `legacy_fixed_size` (off), `per_clean_pairing` (`off`).
Anything not listed as **Required** above is optional.

`legacy_fixed_size` switches Step 3 to the legacy fixed-input-size training mechanism. It is off by
default: every fine-tune uses the current variable-input-size mechanism unless the user **explicitly**
asks for the legacy one ("FT using the legacy fixed-size mechanism", "old fixed input size setting",
"previous release FT mode", etc.). Never infer it from the dataset, the image sizes, or KPI numbers,
and never ask about it when the user has not raised it. When set, the run takes
`job_name: anomalygen_texture_ft_legacy` — in the recipe and as `--job_name` to `set_pipeline_vars.sh`
— so it never shares a run dir with a default run of the same `name`, and Step 3 appends the legacy
block to the recipe.

`mask_backend` picks how Step 2 places masks: **`amg`** (default) — the agentic engine proposes,
refines and judge-accepts every mask, all defect types included; **`amp`** — the
allocate → pair → place pipeline; no VLM beyond `text` ROIs. AMG
needs model access (preflight checks before spending); if the user cannot supply it and does not choose,
stop and ask — never fall back silently.

Parameter tables (each mapped to its CLI arg) and the step map: [`inputs.md`](references/inputs.md).
Confirm the inputs, set the shared variables below, then run straight through — in `full`,
`checkpoint` is derived after Step 3, so don't pass it.

## Instructions

Run the steps your `mode` selects (see **Inputs**), in order — each step's header notes when to skip it.

> **A step is done only when its output exists on disk — not when its command is *launched*.** Every
> step ends with a named gate (**dataset**, **amp**, **train**, …); run it before moving on. Never
> treat starting a long job as a stopping point, nor proceed on a partial artifact. Gate commands and
> the fix for each failure: [`references/verification.md`](references/verification.md).

Set the shared variables every step below refers to. **Re-source this in each new shell** — variables
do not survive one — and read the values it echoes back (`${DATASET_DIR}`, `${VAL_DIR}`, `${GEN_DIR}`,
`${OUT}`, `${RUN_DIR}`, `${LOG_DIR}`, …):

```shell
source scripts/skill_utility/set_pipeline_vars.sh --name <name> --dataset_dir <dataset_dir> \
    --num_sdg <num_sdg> [--num_gpus 1] [--num_search_run 3] [--task texture_ft] \
    [--job_name anomalygen_texture_ft_legacy]   # legacy_fixed_size only
export SCORE="<score>"  # repeat the confirmed choice in every shell; use nn when none was requested
```

**`--dataset_dir` is required and is never derived from `--name`.** The name is an experiment label; the
dataset is a separate input that comes from the user. They differ in the normal case — a label of
`uc1pega-opus5-e2e` over a dataset at `datasets/uc1pega` — so a name-derived path would be wrong far
more often than right, and wrong *silently*: the recipe's `dataset_path` is set separately, so the
recipe can point at the right dataset while `${DATASET_DIR}` hands Steps 5 and 6 a `--real_root` that
scores against nothing. The script does **not** check that the path exists; gate **dataset** does, one
step later.

**Every step's log goes under `${LOG_DIR}` (`results/`), never the repo root.** Use
`${LOG_DIR}/<step>_${NAME}.log`. The same goes for PID files — the launcher writes its own under `results/`;
do not add one.

The steps below pipe through `tee` to keep the log and the console. **Run `set -o pipefail` first** —
without it the pipeline's exit status is `tee`'s, so a step that died reports success:

```shell
set -o pipefail
```

`set_pipeline_vars.sh` **rejects** a `name`/`task` outside `[A-Za-z0-9][A-Za-z0-9._-]*` and any non-integer count rather
than sanitizing them — both reach command lines below. Apply the same care to user-supplied paths
(`dataset_dir`, `recipe`, `checkpoint`): quote every expansion. `texture_ft` is the only `task_type`
today; a new task family adds one `${TASK}`→`${SCRIPTS}` line to that script plus a matching
`task_type:` in the recipe.

Step 3 sets `CKPT` and repoints `RECIPE` at the run-dir copy for Steps 4–7. **`generation_only` skips
Step 3**, so set both yourself before Step 4 (use the user's `checkpoint` verbatim if given; `CKPT` and
`RECIPE` must come from one run or class ids drift):

```shell
RUN_DIR=<existing run dir>                      # only if it is not the ${RUN_DIR} already set above
CKPT="${RUN_DIR}/checkpoints/model/$(cat "${RUN_DIR}/checkpoints/best_checkpoint.txt")"
RECIPE="${RUN_DIR}/exp_${TASK}_${NAME}.yaml"    # run-dir copy — MUST match the checkpoint
```

### Step 1 — dataset (contract layout)

All modes need `dataset_dir` on disk in this layout — training and eval read the
`anomaly_image/`+`mask/` pairs; both Step 2 backends read `clean_image/` (AMG also samples those
same pairs as exemplars); `defect_spec.jsonl` has exactly one consumer, the AMP backend:

```text
{dataset_dir}/{texture}/anomaly_image/{defect}/<name>.png   # defect image
{dataset_dir}/{texture}/mask/{defect}/<name>_mask.png       # binary mask (>=128 = defect); same stem + _mask
{dataset_dir}/{texture}/clean_image/<name>.png              # clean images (Step 2 inputs — both backends)
{dataset_dir}/{texture}/cad_mask/<clean_stem>.png           # cad/cad_tht only: one per clean image, no {defect} level
{dataset_dir}/semantic_segmentation_labels.json             # cad/cad_tht only: label map for those CAD masks
{dataset_dir}/defect_spec.jsonl                             # amp only: one row per {texture}+{defect}; sets spatial_dependency (free/text/cad/cad_tht). amg never reads it — absent is normal
```

Ready-made subjects:

```shell
python -m anomalygen.scripts.datasets.prepare_phone_screen_defect --output_dir "${DATASET_DIR}" --zip <roboflow.zip>
# --zip is required (the images are a manual Roboflow download); --masks_only asks for a dataset with no images.
# also: prepare_pcb_defect, prepare_magnetic_tile_defect  (see anomalygen/scripts/datasets/README.md)
```

For custom data, emit the same layout and list every `[texture, defect]` pair in the recipe's
`anomaly_types`. Confirm the layout before proceeding — gate **dataset**.

### Step 2 — mask placement

Two backends produce the same artifact (placed masks + `testcase.jsonl`); everything downstream is
backend-blind. Default is **AMG**; set `mask_backend=amp` for the classic pipeline.

Model access was verified at **preflight** (third group). If preflight was skipped, run that
probe now — it must pass before anything below spends money.

One notice: if `defect_spec.jsonl` is present and declares `spatial_dependency`, tell the user —
**AMG is the default and does not read that field** (it decides placement by looking at the board);
to honour `free`/`text`/`cad` routing, run with `mask_backend=amp`. Do not change the default over
it. A dataset with no `defect_spec.jsonl` at all is normal under `amg` — nothing in the pipeline
reads it then, and the grep below is simply silent; under `amp` its absence is a **dataset** gate
failure.

```shell
grep -l spatial_dependency "${DEFECT_SPEC}" >/dev/null 2>&1 && echo   "NOTE: spatial_dependency present but mask_backend=amg ignores it; use mask_backend=amp to honour it"
```

#### AMG (default)

**Reference** (model resolution, the model-service bootstrap, the capability probe, cost, resume, the
run's own deviation ledger): [`agentic-mask-generation.md`](references/agentic-mask-generation.md).

One run per output set, each sized by `--num-sdg` — the same request AMP answers, split across the
types by the same rule (base each, remainder to the first types). **Both sets use `${NUM_SDG}`**, the
same convention as AMP, where `--num_sdg` sizes the validation and the generation set alike. The
deliverable is exactly `num_sdg` rows, so gate **amg** checks the number that was asked for:

```shell
AMG_MODE=validation; AMG_DIR="${VAL_DIR}"; ALLOC=validation   # names the log, the out-dir, the split
# then again:  AMG_MODE=generation AMG_DIR="${GEN_DIR}" ALLOC=inference

python -m anomalygen.scripts.agentic_mask_generation.generate \
    --dataset-path "${DATASET_DIR}" --anomaly-types <texture>+<defect> [...] \
    --gen-model "${AMG_MODEL}" --judge-model "${AMG_MODEL}" \
    --model-registry "${AMG_REGISTRY:-$HOME/.anomalygen/endpoints.yaml}" \
    --num-sdg "${NUM_SDG}" --mode "${ALLOC}" \
    --visual-prompt auto --defect-library auto \
    --out-dir "${AMG_DIR}" 2>&1 | tee "${LOG_DIR}/amg_${AMG_MODE}_${NAME}.log"
```

`--model-registry` is explicit, not left to the default, because Step 2 may run inside a container
whose `$HOME` is not the host's: the flag makes "which registry answered" a fact of the recorded
command. Set `AMG_REGISTRY` to wherever the registry actually is (e.g. the path it was mounted at).

**`--mode` is why the two sets are not interchangeable.** The generation set is `inference`: an even
split, and a type may legitimately get 0 when `num_sdg` is below the type count. The validation set is
`validation`: split in proportion to each type's real mask count, so it mirrors what the model trains
on — and a share that would round to 0 is **refused**, because validation scores per type and a type
with no samples drops silently out of the KPI that picks `best_checkpoint.txt`. Both are AMP's rules,
shared through one allocation module rather than reimplemented.

**`--num-sdg` is the only way to size a run**, as it is for AMP. It splits the total across the
types rather than sizing each alike, so 25 over 3 types is 9/8/8 and lands on 25 — where a per-type
count could only reach 27, paying for two masks before anything discards them. For exact per-type
targets instead, `--per-defect-counts '{"<texture>+<defect>": N}'`; omitted types get 0.

**`AMG_MODE` is in the log name because there are two runs.** Both would otherwise `tee` to one file
and the second would overwrite the first, silently discarding the validation run's deviation ledger —
the same reason AMP's log carries `${AMP_MODE}`.

**The validation set's size is a training-time cost, not just a one-off API charge.** Every validation
pass re-renders the whole `testcase_jsonl` set and runs one SAM2 forward per sample (~0.4 s each — see
[`fine-tuning.md`](references/fine-tuning.md)), repeated `max_iter / validation_iter` times. To spend
less, lower `num_sdg`; do **not** raise `validation_iter`, which also coarsens the resolution
`best_checkpoint.txt` is chosen at.

AMG places every type it is given. A stopped or interrupted run
**continues from where it was** when re-run with the same command. The run records every deviation as
`degraded[KIND]` in its log — read the summary before trusting the output. Verify the deliverable
directly against gate **amg** ([`verification.md`](references/verification.md)): `testcase.jsonl` rows
= `${NUM_SDG}`. The fill loop delivers in beam-sized batches and then trims to the target, so
`trimmed from N` with `N` well above a type's share is normal, not a defect — the surplus is kept under
`{texture}/_overflow/{defect}/` + `testcase_overflow.jsonl` and is deliberately **not** counted in
`testcase.jsonl`, so counting mask files on disk will exceed the row count with nothing wrong. When
the user asks for different backends on different types, each backend places its own subset and this
count is over the union of the runs' rows (the AMP gate's `roi_*` checks and remedies are for AMP
out-dirs only).

**When the model is served here, the service spans Step 2 and nothing else.** Both mask sets are Step
2 products, so serve once and run both against it — then stop it. Serving holds a large share of the
GPU, and Step 3 (`full`) and Step 4 (`generation_only`) need that GPU back:

```shell
scripts/agentic_mask_generation_serving/serverctl.sh health                   # idempotent: probe first, start only if down
case $? in                                                # 0 up · 1 not running · 2 alive but loading
  0) : ;;                                                 # ready — run the two generate runs below
  1) echo "DOWN — start it (references/agentic-mask-generation.md), then wait for it" ;;
  2) echo "LOADING — call: scripts/agentic_mask_generation_serving/serverctl.sh wait   (repeat while it exits 2)" ;;
esac
# ... the two generate runs above: AMG_DIR=${VAL_DIR}, then AMG_DIR=${GEN_DIR} ...
scripts/agentic_mask_generation_serving/serverctl.sh stop                     # part of this step's gate
scripts/agentic_mask_generation_serving/serverctl.sh health                   # ONLY exit 1 means the GPU is free
[ $? = 1 ] || echo "STILL HOLDING THE GPU — Step 3 will contend for it"
```

**Test for exit 1, not for failure.** A server that is up (0) and one still loading (2) both hold their
`--gpu-memory-utilization 0.9` reservation, so `health && echo ...` would clear the gate on a process
that never finished starting and is still sitting on the card.

**Waiting for the service is an action, exactly as waiting for a fine-tune is.** `serverctl.sh wait`
reports state as its exit code — **0** up, **1** no such process, **2** still loading, call again — and
each call is a ~9.5-minute slice, not the whole wait. A single-GPU cold start measured **~14 minutes**
to first answer `/v1/models`, so expect two or three calls. Do not read a `DOWN`/`LOADING` line during
that window as a failed start: exit **2** means the process is alive and has not finished loading, and
restarting on that reading puts a second vLLM on the same GPU at `--gpu-memory-utilization 0.9`.

**Step 2 is not done until that last line prints nothing.** The bootstrap that starts it — in a
container or on a dev host — is in [`agentic-mask-generation.md`](references/agentic-mask-generation.md).

#### Classic AMP (`mask_backend=amp`)

**Reference** (allocation modes, KPI floor, `--per_defect_counts`, ROI caching, `free/text/cad/cad_tht`
routing, pairing coverage): [`mask-placement.md`](references/mask-placement.md).

AMP builds a `testcase.jsonl` (clean image + placed mask + gen params) by running allocate → pair →
place. Build the **`validation`** set (fine-tune) and/or the **`generation`** set (SDG), each with a
distinct `--seed` — **full** → both, **finetune_only** → validation, **generation_only** → generation:

```shell
AMP_MODE=validation; AMP_DIR="${VAL_DIR}"; SEED=42    # or: AMP_MODE=inference AMP_DIR="${GEN_DIR}" SEED=43

python -m anomalygen.scripts.auto_mask_placement.roi_place_pipeline \
    --num_sdg "${NUM_SDG}" --mode "${AMP_MODE}" --defect_desc "${DEFECT_SPEC}" \
    --dataset_dir "${DATASET_DIR}" --output_dir "${AMP_DIR}" --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/amp_${AMP_MODE}_${NAME}.log"
# flags for roi_place go after a bare --, e.g.:  -- --refresh_roi
```

It derives every intermediate path from `--output_dir`, threads `roi_pair`'s `n_seeds` into
`roi_place`, and stops with a named stage if one produces nothing. Output is placed masks +
**`${AMP_DIR}/testcase.jsonl`**. The **generation** set must have exactly as many rows as the
allocation asked for — `num_sdg`, or the `--per_defect_counts` sum when you passed one — verify before
proceeding: gate **amp**.

**Only when the user set `per_clean_pairing=on`:** pass `--per_clean_pairing on`, and with it a
single-type `--per_defect_counts '{"<texture>+<defect>": '"${NUM_SDG}"'}'` with `num_sdg` at the
clean-image count — `on` is one record per clean, so it takes exactly one defect type. Absent, the
default `off` pairing applies: it maximises (submask, clean) diversity and so visits only *some*
clean images ([`mask-placement.md`](references/mask-placement.md)). Whether every clean must receive
an output is a property of the batch that only the user knows; do not infer it.

### Step 3 — fine-tuning (skip when generation_only)

**Reference** (recipe knobs, launch mechanics, early stopping, checkpoint selection):
[`fine-tuning.md`](references/fine-tuning.md).

**Before training:** copy the template recipe and edit its dataset block so `testcase_jsonl` points at
the validation set from Step 2 (AMG or AMP) (other keys ship at tested defaults):

```shell
TEMPLATE=ag_config/exp_texture_ft_phone_screen.yaml   # the worked example doubles as the template
[ -f "${USER_RECIPE:-}" ] && TEMPLATE="${USER_RECIPE}"   # a recipe the user passed replaces it
[ "${RECIPE}" = "${TEMPLATE}" ] || cp "${TEMPLATE}" "${RECIPE}"   # same file (name=phone_screen or a passed recipe): edit it in place
```

If the user passed a `recipe`, set `USER_RECIPE` to it first: an existing file replaces the template,
so a recipe already at `${RECIPE}` is edited in place, never overwritten.

```yaml
# edit the dataset block in ${RECIPE}
dataset_name: ${NAME}
anomaly_types: [[<texture>, <defect>], ...]   # every [texture, defect] pair on disk
dataset_path: ${DATASET_DIR}
testcase_jsonl: ${VAL_DIR}/testcase.jsonl        # ← Step 2 validation set
```

If you change `max_iter`, set `cycle_lengths` to the same value. `cycle_lengths` sets how long the
learning-rate schedule lasts, and training crashes if `max_iter` is larger.

Only when `legacy_fixed_size` is set: change the recipe's `job_name` line to
`anomalygen_texture_ft_legacy` (the value you passed as `--job_name`), and append this block to
`${RECIPE}`, keys and values exactly as shown — it is the legacy mechanism as one unit, never a subset:

```yaml
# legacy_fixed_size only — append to ${RECIPE}
input_size_ladder: [512]
crop_mode: ratio
crop_ratio_range: [1.5, 8.0]
background_dropout_prob: 0.5
ring_jitter_prob: 0.5
inst_aug_prob: 0.5
```

A fine-tune runs hours at the `15000` default, so it is launched detached (own session, PID file, log).
**Check `${RECIPE}` is the one you meant before launching** — it then runs unattended with your
credentials:

```shell
scripts/skill_utility/train_control.sh start --name "${NAME}" --recipe "${RECIPE}" \
    --num_gpus "${NUM_GPUS}" --scripts "${SCRIPTS}" --task "${TASK}"
```

**This launcher is the only sanctioned way to start a fine-tune. Never hand-roll a `torchrun` for
training.** It is what writes `results/train_${NAME}.log` and `results/train_${NAME}.pid`, records the
run dir, and makes `wait`/`stop` work; a direct `torchrun` bypasses all of it, and a literal
`--nproc_per_node=<n>` typed at the prompt silently discards `${NUM_GPUS}` and the user's GPU budget
with it. Pass `--num_gpus ${NUM_GPUS}` — never a literal.

Writes `${RUN_DIR}` (recipe copied in). `train_control.sh stop --name ${NAME}` ends a run;
`start` again resumes from `latest_checkpoint.txt`.

**Waiting is an action.** Call `wait` in the foreground until it stops returning **2** — arming a
background watcher and ending your turn leaves nobody to run Steps 4–7:

```shell
scripts/skill_utility/train_control.sh wait --name "${NAME}"   # 0 = finished · 1 = failed · 2 = still running, call again
```

**Each call is a ~9.5-minute slice, not the whole wait.** The `15000` default runs for hours
(dataset-dependent; the measured times are in `docs/tutorial/texture/7-agentic-end-to-end-flow.md`) —
expect **dozens of calls**.
Only exit **0** clears gate **train** — not a slice expiring, and not a checkpoint appearing
(the trainer writes an intermediate `iter_<N>.pt` every `save_iter`). On **1**, read the log
and fix the run; do not proceed.

**`finetune_only` stops here.** For `full`, take the **best** iteration from the pointer the trainer
writes at train end (peak `nn_score` on the `Average` column) — runs routinely peak mid-training:

```shell
CKPT="${RUN_DIR}/checkpoints/model/$(cat "${RUN_DIR}/checkpoints/best_checkpoint.txt")"  # iter_<best>.pt
RECIPE="${RUN_DIR}/exp_${TASK}_${NAME}.yaml"  # run-dir copy — MUST match the checkpoint
```

Never substitute `latest_checkpoint.txt` — it names the *last* iteration. `${RUN_DIR}` also resolves
correctly (best pointer first, then latest, then highest `iter_*.pt`), but pass the explicit path so
the record names the exact checkpoint. A missing pointer is a **train** gate failure — fix the run,
don't let it fall back to the latest.

### Step 4 — generation / SDG (skip when finetune_only)

Reads the **generation** AMP `testcase.jsonl` (Step 2, `GEN_DIR`). `--recipe` **must be the recipe the
checkpoint was trained with** (use the run-dir copy) so `anomaly_type`→class-id lines up.

```shell
torchrun --nproc_per_node="${NUM_GPUS}" "${SCRIPTS}/generate.py" \
    --checkpoint "${CKPT}" \
    --recipe "${RECIPE}" \
    --input_data_path "${GEN_DIR}/testcase.jsonl" \
    --output_dir "${OUT}" \
    2>&1 | tee "${LOG_DIR}/generation_${NAME}.log"
```

`--nproc_per_node=${NUM_GPUS}` — expand the variable, never a literal count.

`--base_seed` (default `1`) seeds the per-testcase noise. **Don't change it mid-pipeline** — Step 6
compares KPIs across re-generated rounds, which a different seed makes incomparable.

Output tree: `reconstructed_image/` (the synthetic defects), `original_image/`, `original_mask/`, and
`${TASK}_generation_result.csv`. `generate.py` emits **one image per testcase row** (no quality drop),
except for samples the content-safety guardrail blocks — those write no image and land in
`guardrail_blocked.csv`, which is expected, not a failure. Verify images + blocked = rows, and the
canonical CSV header, before Step 5 (gate **generate**); **never hand-edit the CSV** — a non-canonical
CSV crashes Step 6.

### Step 5 — evaluation (skip when finetune_only)

Scores the batch against the real dataset with the same correspondence KPI as training validation.

```shell
python "${SCRIPTS}/evaluate.py" \
    --gen_root "${OUT}" --real_root "${DATASET_DIR}" \
    --recipe "${RECIPE}" \
    --output_file "${OUT}/${NAME}_kpi.json" \
    2>&1 | tee "${LOG_DIR}/evaluation_${NAME}.log"
```

Writes per-type + `Average` blocks (`nn_score`, `mnn_score`, `fid`, `per_sample[]`) — verify it exists
(gate **eval**). **`nn_score` is the primary KPI** (higher is better); `mnn_score`/`fid` are secondary.
It also scores the anomaly-quality axes and `aq_nn`, the same metrics validation records
([`fine-tuning.md`](references/fine-tuning.md)). `--anomaly_types Phone+oil ...` overrides
recipe-derived types. The per-sample rows carry NN, completeness, precision, and boundary IoU, which
Step 6 needs when `--score aq_rank` is selected.

### Step 6 — quality refinement (skip when finetune_only)

**Reference** (draw strategy, ranges/seeding, `select`):
[`quality-refinement.md`](references/quality-refinement.md).

`quality_refine.py run` does the whole loop: `(draw → generate → evaluate) ×${NUM_SEARCH_RUN}`, then `select` and a final
score. Each round's `(guidance, crop_ratio)` is drawn per sample — **round 1 uniform**, **rounds 2+
per-sample Bayesian optimization** (see the reference):

```shell
python "${SCRIPTS}/quality_refine.py" run \
    --base_testcase "${GEN_DIR}/testcase.jsonl" \
    --original "${OUT}" --original_kpi "${OUT}/${NAME}_kpi.json" \
    --rounds_dir "${OUT}/rounds" --output "${OUT}/searched" \
    --final_kpi "${OUT}/searched/${NAME}_kpi.json" \
    --checkpoint "${CKPT}" --recipe "${RECIPE}" --real_root "${DATASET_DIR}" \
    --score "${SCORE}" \
    --num_search_run "${NUM_SEARCH_RUN}" --num_gpus "${NUM_GPUS}" \
    2>&1 | tee "${LOG_DIR}/refine_${NAME}.log"
```

It refuses to continue past a round that did not finish — completed/blocked rows, a generation CSV
that names exactly the reconstructed images, and its `kpi.json`. Both BO and `select` key off those
artifacts, so an unfinished round would otherwise be dropped silently and its GPU time wasted.
**`select` always runs**: `num_search_run=0` reduces it to cloning `${OUT}` into `searched/`, so
`searched/` always exists for Step 7. Verify the round dirs before Step 7: gate **refine**.

`nn` remains the default score. `--score aq_rank` is an optional cumulative composite: before each
draw it pools Original with every earlier round, groups candidates by anomaly type, and recomputes
AQ-Rank from NN, completeness, precision, and boundary IoU. Final selection recomputes it once more
over Original plus only the rounds requested by the current run. It therefore does not reuse the
bucket-local `aq_rank_score` written by `evaluate.py` whenever candidates compete; the zero-round
clone may copy it into audit output but makes no ranking decision. A candidate missing any of the
four finite raw inputs is logged and excluded from the rank pool, so it cannot earn a favorable rank
on a negatively oriented axis; the rest of the bucket remains usable. If one sample has no eligible
render across any bucket, selection logs the fallback and keeps its Original render without calling
it an AQ-Rank win. A bucket with no eligible candidate still fails explicitly.

AQ-Rank fits each geometry-axis direction across the pooled samples but applies it when comparing
round variants within one sample; those relationships can disagree. Every re-ranking logs each
axis's AUROC and sign for audit. Treat this as an opt-in heuristic, not a within-sample improvement
guarantee; changing to a within-sample orientation requires a separate experiment and human validation.

With `--score nn`, the `Average.nn_score` delta between `${OUT}/searched/${NAME}_kpi.json` and the
Step 5 `${OUT}/${NAME}_kpi.json` measures the refinement gain. AQ-Rank is pool-relative, so do not
compare its mean across passes. For `--score aq_rank`, inspect winner-source distribution and paired
raw-metric changes, then use human or downstream evaluation for the final quality claim. A
seven-dataset, one-reviewer blind pilot found that NN and AQ-Rank both continued selecting later-round
candidates and detected no statistically significant perceptual difference between their final
outputs; this supports AQ-Rank as an option, not a claim that it is superior or equivalent to NN.

### Step 7 — pseudo-labeling (skip when finetune_only)

Turns the refined `${OUT}/searched` tree into a labeled dataset: COCO instance annotations, a
classification layout, visualizations, and (unless `--no_caption`) VLM captions.

```shell
python "${SCRIPTS}/pseudo_label.py" \
    --gen_root "${OUT}/searched" \
    --output_dir "${OUT}/searched/pseudo_labels" \
    2>&1 | tee "${LOG_DIR}/pseudo_label_${NAME}.log"
# add --no_caption to skip the VLM captioning (fast, no GPU model load)
```

`--gen_root` must contain `reconstructed_image/`, `original_image/`, `original_mask/`, and
`${TASK}_generation_result.csv` (Step 6 always produces this at `${OUT}/searched`). Verify the labels
landed — gate **label**.

## Verification

The artifact each gate requires. Runnable commands + the fix for each failure:
[`verification.md`](references/verification.md).

| #   | Gate       | Step | Requires                                                                                                                     |
| --- | ---------- | ---- | ---------------------------------------------------------------------------------------------------------------------------- |
| 1   | `dataset`  | 1    | per-pair `clean_image/`, `anomaly_image/{defect}/`, `mask/{defect}/`; + `defect_spec.jsonl` **only when `mask_backend=amp`** |
| 2a  | `amg`      | 2    | each set's `testcase.jsonl`; rows = `num_sdg`; no `type_failed` / `model_smoke_failed`                                       |
| 2b  | `amp`      | 2    | `${AMP_DIR}/testcase.jsonl`; generation row count = the allocation                                                           |
| 3   | `train`    | 3    | `wait` exit **0** **and** `checkpoints/best_checkpoint.txt`                                                                  |
| 4   | `generate` | 4    | canonical CSV header **and** images + `guardrail_blocked.csv` rows = testcase rows                                           |
| 5   | `eval`     | 5    | `${OUT}/${NAME}_kpi.json` with per-type + `Average`                                                                          |
| 6   | `refine`   | 6    | `${NUM_SEARCH_RUN}` round dirs, **each complete** (images *and* `kpi.json`)                                                  |
| 7   | `label`    | 7    | `pseudo_labels/coco_annotations.json` + `classification/classes.txt`                                                         |

Gate 6 counts *complete* rounds — existing dirs aren't enough. An all-`original` `searched/` `source`
column is a valid outcome (no round beat Step 4), not a failure; missing round dirs are.

## Examples

Natural invocations — the `key=value` params map to the **Inputs** table (recipe knobs like `max_iter`
edit the recipe; see [`references/inputs.md`](references/inputs.md)). `pcb` is `cad`-routed,
`magnetic_tile` is `free`-routed, `phone_screen` is `text`-routed, and a through-hole solder-joint set
is `cad_tht`-routed, so their Step 2 AMP differs
(see [`references/mask-placement.md`](references/mask-placement.md)).

Full pipeline (Steps 1→7) on the text-grounded phone-screen set:

```text
Use AnomalyGen skill with
name=phone_screen
mode=full
dataset_dir=datasets/phone_screen
num_sdg=25
max_iter=15000
validation_iter=1000
save_iter=1000
```

To generate more from an existing checkpoint instead, switch to `mode=generation_only` and add
`recipe=${RUN_DIR}/exp_${TASK}_<name>.yaml` + `checkpoint=${RUN_DIR}/checkpoints/model/iter_<best>.pt`
(from `best_checkpoint.txt`), dropping the training knobs (`max_iter`/`validation_iter`/`save_iter`).

Only a prompt that explicitly asks for it — "FT using the legacy fixed-size mechanism" — adds
`legacy_fixed_size=true`: pass `--job_name anomalygen_texture_ft_legacy`, set the same `job_name` in the
recipe and append the legacy block (Step 3). The run lands under
`results/anomalygen/${NAME}/anomalygen_texture_ft_legacy`; SDG outputs under `results/generation_${NAME}`
are still shared with a default run of the same `name`. Every other prompt trains with the default
variable-input-size mechanism.

## Troubleshooting

Each fails *quietly* — plausible output, wrong result. Detail in
[`fine-tuning.md`](references/fine-tuning.md) and [`mask-placement.md`](references/mask-placement.md).

| Symptom                                                                                                     | Cause / fix                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Class ids misaligned in the labels                                                                          | A different `anomaly_types` set — the checkpoint stamp refuses it; use the run-dir recipe copy (order is sorted at parse, so it never matters)                                |
| Validation scores nothing                                                                                   | `testcase_jsonl` not wired to the Step 2 validation set                                                                                                                       |
| A fresh `max_iter` run exits immediately                                                                    | Resumed an old `iter_*.pt` — clear it or set a new `job_name`                                                                                                                 |
| A legacy run dies after model load with a geometry-stamp refusal (`crop_mode` / `input_size_ladder` differ) | It resumed a default run's checkpoint from the same run dir — the recipe's `job_name` and `--job_name` were not both `anomalygen_texture_ft_legacy`; fix both and start again |
| Fewer rows than the allocation                                                                              | A defect yielded zero placements — check `roi_place` logs                                                                                                                     |
| A worse checkpoint ships, numbers look fine                                                                 | Read column 1 of `${RUN_DIR}/valid/<iter>/valid_kpi.csv` (one per validated iteration, not one file), not `Average` — use `best_checkpoint.txt`                               |
