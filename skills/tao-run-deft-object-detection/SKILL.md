---
name: tao-run-deft-object-detection
description: >
  Run the full DEFT smart-data-augmentation loop for NVIDIA TAO Grounding DINO object detection:
  zero-shot baseline inference, KPI analysis, per-class gap analysis, SigLIP embedding of weak images,
  unique-neighbor mining against a source pool, ODVG dataset staging, and retraining — repeated for a
  fixed number of iterations. Also prepares the source pool the loop mines from, as a separate run:
  Co-DETR pseudo-labeling, folding to the target classes, KITTI→COCO→ODVG conversion, and embedding.
  Use for prompts like "run the DEFT OD loop", "run smart data augmentation for grounding dino",
  "mine and retrain my detection model", "improve OD mAP with gap analysis and mining", "prep the
  source pool", or "pseudo-label my unlabeled images for mining"; do not use for standalone TAO
  training, one-off inference, or gap analysis alone.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit and one or more CUDA GPUs. Workflows declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Skill Task Bash Write
tags:
- application
- workflow
- deft
- object-detection
- grounding-dino
- mining
---

# Skill: tao-run-deft-object-detection

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

## Execution Contract

Treat this as a disk-backed state machine, not as a prose recipe.

1. Preserve every explicit user value. `epoch 1` means `train.num_epochs=1`; a spec value or documented default applies only when the user did not supply that parameter. Show the source of every run parameter (`user`, `spec`, or `default`) in the Pre-Flight Summary.
2. After the user approves the Summary, initialize `deft_state.json` once with `scripts/init_deft_state.py`. Never hand-author or reinitialize it on resume.
3. Run every bundled or inline host-Python command through `scripts/deft_python.sh`. On startup, after context compaction, before every stage, and before any completion claim, run:

   ```bash
   <skill_root>/scripts/deft_python.sh \
     <skill_root>/scripts/audit_deft_run.py --results-dir "${RESULTS_DIR}"
   ```

   If it prints `DEFT_RUN_STATUS=INVALID`, stop and repair the listed disk inconsistency; do not launch another stage. Read the path printed as `read_before_action` before continuing.
4. Invoke the mapped underlying skill after reading the DEFT overlay. Do not replace a missing or unread stage reference, or a failed skill call, with guessed shell commands, inline Python, a different output tree, or fabricated data.
5. Commit every stage with `scripts/commit_stage.py`; it verifies artifacts, updates `deft_state.json`, appends exactly one ordered `loop_log.jsonl` event, and rolls back if its audit fails.
6. Claim the loop complete only when this exits zero:

   ```bash
   <skill_root>/scripts/deft_python.sh \
     <skill_root>/scripts/audit_deft_run.py \
     --results-dir "${RESULTS_DIR}" --require-complete
   ```

## Context Discipline

- Load references just in time. Run the audit, read only its `read_before_action` file and the current stage's named section, then act. Never preload all references.
- Redirect verbose train, inference, and Docker output to files. Inspect at most the final 40 lines or a one-line artifact check; never print a full spec, state file, or loop log into the conversation.
- A Skill-tool call loads stage instructions; it does not start a background orchestrator. Continue the documented stage in the parent immediately after it returns.

## When to Use This Skill

Use this skill when the user wants an agent to run the full smart-data-augmentation loop for a TAO Grounding DINO detection model: zero-shot baseline, gap analysis, mining, dataset growth, and retraining across N iterations.

- "Run the DEFT OD loop"
- "Run smart data augmentation for grounding dino"
- "Mine more training data for my detection model and retrain"
- "Improve detection mAP with gap analysis and unique-neighbor mining"

Also use it to prepare the source pool, which is a **separate run** that completes before the
loop launches (see `## Two Invocations: Prep, Then Loop`):

- "Prep the source pool"
- "Pseudo-label my unlabeled images for mining"
- "Build the mining pool from these raw images"

Do not use this skill for a single standalone TAO training run, one-off inference, or gap analysis alone. Invoke the relevant leaf skill directly instead.

## Scope: Grounding DINO + ODVG

This loop targets **Grounding DINO** with **ODVG** training annotations (`tmm_odvg.jsonl` + `labelmap.json`), matching the reference pipeline. `dataset.train_data_sources` is a **list**; each iteration appends one new ODVG source rather than rewriting a combined CSV. DINO and RT-DETR (COCO) are not supported by this workflow — use the leaf skills directly for those.

The loop does **not** train at baseline. It evaluates the supplied zero-shot / pretrained checkpoint as iteration 0 and only trains from iteration 1 onward, once mining has produced data to add.

## Two Invocations: Prep, Then Loop

The source pool is prepared by its own run, before the loop launches. They are separate
because a pool is prepped once and then serves many loop runs — coupling them would re-label
and re-embed the same images on every launch.

| Invocation | Does | Produces |
|---|---|---|
| "Prep the source pool" | Co-DETR pseudo-labels raw pool images, folds to the target classes, converts KITTI→COCO→ODVG, verifies, embeds | `coco.json`, `odvg/`, `source_embeddings.parquet`, `pool_report.json` |
| "Run the DEFT loop" | baseline → iterations | checkpoints, KPI, the mAP trend |

Follow `references/prep-source-pool.md` for the first. The loop then takes those four paths as
inputs; Pre-Flight validates them and `init_deft_state.py` pins them, so a run cannot reach
`mine` with no corpus to search.

## Launch Intake

After the user confirms they want to run this workflow, ask which supported platform they intend to run on. Discover the execution platforms from the installed platform skills (tao-run-on-docker / -slurm / -kubernetes / -brev). After platform selection, read the chosen platform skill's `## Credentials` section.

Never ask for or read credential values. Check only whether the required environment variable is set; if unset, tell the user which variable to export.

## Agent Behavior

> **There is exactly one user gate: pre-flight confirmation.** Print the Pre-Flight Summary (see `references/preflight.md`), then STOP and wait for explicit approval ("go", "yes", "looks good"). Do not launch any side-effecting step before that approval.
>
> **After the gate, the skill is fully autonomous.** Run the entire loop without asking for confirmation. Only stop if a step fails with an unrecoverable error or a hard-stop gate fires. Print a one-line status update at each stage milestone.
>
> **Auto-mode required.** The post-gate loop fires constant side-effecting calls; without auto-accept mode it stalls on the first prompt. Remind the user at the Pre-Flight Summary to enable auto-mode (shift+tab) before approving.
>
> **Non-zero command rule.** Never repeat an unchanged failed command. Read the final error block, map it to the loaded stage reference, make one evidence-based correction, and rerun its documented verification. If the reference does not cover the failure, commit `status=error` and halt.
>
> **Revised plan.** If any run parameter changes after the Summary was shown, re-run Pre-Flight and show an updated Summary before proceeding.

## Workflow

Full detail in `references/pipeline-and-state.md`.

1. **Pre-Flight.** Run every check in `references/preflight.md`. Resolve workspace, specs, annotations, the zero-shot checkpoint, the source-pool embedding parquet, and container images. Hard stop only on missing input you cannot resolve yourself.
2. **Prep (once, before baseline).** If the source pool is not already labeled and embedded, pseudo-label it with Co-DETR, fold the predictions onto the user's target classes, convert KITTI→COCO→ODVG, and embed the pool. Idempotent — each artifact is skipped when it already exists. See `references/prep-source-pool.md`.
3. **Baseline (iter_0) — no training.** Run `inference` with the supplied zero-shot / pretrained checkpoint, then `kpi_analyze`. Seed `train_grounding_dino.yaml` from the user's template for later iterations to extend.
4. **Iterate.** For each iteration 1..`max_iterations`, run the seven stages in order:
   `gap_analysis` → `embed` → `mine` → `stage` → `train` → `inference` → `kpi_analyze`.
   Each iteration's `gap_analysis` consumes the **previous** phase's inference labels. Between stages run the audit and follow its one-line disk-backed next action.
5. **Stop** when `max_iterations` is reached or a hard-stop gate fires. mAP is reported, not gated — the loop does not early-exit on a metric target.
6. **Render** `results/DEFT_Loop_Report.md` after each completed iteration and once more at loop end by spawning the `reporter` subagent (`agents/reporter.md`). Never render inline.

All stages run inline in the parent context. Prefer invoking the underlying `tao-skill-bank:*` skills via the Skill tool, layering loop conventions on top via the matching `references/*.md` overlay.

### Using Bundled Scripts

Run bundled scripts through `<skill_root>/scripts/deft_python.sh`. Resolve every path argument to an absolute host path first. Use `commit_stage.py` for all state and log writes. See `references/scripts-and-agents.md`.

## Stage Reference Modules

Each stage maps to one underlying skill or to bundled glue. **Read only the current stage's overlay, then invoke.**

The overlays are written against two loop variables that Pre-Flight sets and every stage uses: `N`, the current iteration number, and `PHASE`, which is `baseline` or `iter${N}`. They are stated here because a stage overlay is read on its own — see `references/preflight.md` for the full set.

If an *overlay* is missing, stop and ask the user to reinstall the plugin — the loop cannot run a stage whose settings it does not have. If a mapped *skill* is unavailable, do not stop and do not improvise: fall back to the overlay's documented `docker run` as described in `references/scripts-and-agents.md`, and record `execution_path=direct-container`. The overlay carries everything the invocation needs, so the fallback produces the same artifacts.

| Stage | Overlay | Underlying skill |
|---|---|---|
| `prep` (once) | `references/prep-source-pool.md` | `tao-skill-bank:tao-train-codetr` + `tao-generate-image-embeddings` (+ bundled glue) |
| `gap_analysis` | `references/tao-analyze-gaps-od-map.md` | `tao-skill-bank:tao-analyze-gaps-od-map` |
| `embed` | `references/tao-generate-image-embeddings.md` | `tao-skill-bank:tao-generate-image-embeddings` |
| `mine` | `references/tao-mine-od-images.md` | `tao-skill-bank:tao-mine-od-images` |
| `stage` | `references/stage-mined-data.md` | *(bundled glue — no leaf skill)* |
| `train`, `inference` | `references/grounding-dino.md` | `tao-skill-bank:tao-train-grounding-dino` |
| `kpi_analyze` | `references/tao-analyze-detection-kpi.md` | `tao-skill-bank:tao-analyze-detection-kpi` |

**Path rule (invariant).** Record absolute host paths under `${RESULTS_DIR}`. Mount `"$WORKSPACE:$WORKSPACE"` with identical host and container paths. TAO's `update_results_dir` **appends the task name** to `results_dir`, so passing `results_dir=X` to train writes `X/train/` and to inference writes `X/inference/`. Never append the subdirectory yourself.

## Data, Pre-Flight, Pipeline, and State References

| Topic | Reference |
|---|---|
| Data contract, ODVG layout, source pool, output tree | `references/data-layout.md` |
| One-time source-pool prep (pseudo-label, remap, convert, embed) | `references/prep-source-pool.md` |
| Pre-Flight checks, defaults, Summary template | `references/preflight.md` |
| Pipeline stages, state schema, loop-end sequence | `references/pipeline-and-state.md` |
| Bundled scripts, glue, reporter agent, stage table | `references/scripts-and-agents.md` |

**`max_iterations` defaults to `1`** — one mine, train and score pass, the smallest run that yields a comparison against the baseline. Confirm it with the user when they have not said how many iterations they want; an unattended run takes the default rather than stopping to ask.

## Gating

Run the full Pre-Flight, print the Summary, then STOP at the one user gate. After approval, run the baseline and the seven-stage iteration pipeline.

Hard-stop and never auto-retry on: any stage `status=error`; a missing or zero-row source-pool embedding parquet; a zero-row mining result when weak images were present; a missing ODVG annotation source; an image/annotation mismatch after staging; or a train exit that emits no new iteration checkpoint. The loop stops when `max_iterations` is reached or an unrecoverable gate fires. Each terminal path commits `loop_stop` through `commit_stage.py`, then follows the loop-end sequence in `references/pipeline-and-state.md`.
