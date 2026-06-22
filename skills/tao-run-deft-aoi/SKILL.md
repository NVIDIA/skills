---
name: tao-run-deft-aoi
description: >
  Run the full DEFT AOI improvement loop for NVIDIA TAO VisualChangeNet / ChangeNet PCB inspection models:
  baseline evaluate, RCA, Cosmos AnomalyGen / AMP synthetic defects, k-NN mining, retraining, and deployment
  gating until FAR / recall KPI targets are met. Use for prompts like "run the DEFT loop", "fine-tune until
  FAR below 0.1% at recall=100%", or "improve my AOI ChangeNet model with RCA and synthetic defects"; do not use
  for standalone TAO training, one-off inference, generic anomaly generation, or RCA-only analysis.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Requires docker + nvidia-container-toolkit. Workflows declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Task Bash Write
tags:
- application
- workflow
- deft
- aoi
- loop
---

# Skill: tao-run-deft-aoi

## When to Use This Skill

Use this skill when the user wants an agent to run the full DEFT AOI improvement loop for an NVIDIA TAO VisualChangeNet / ChangeNet PCB inspection model: baseline evaluation, RCA, synthetic defect generation, data mining, retraining, and deployment gating until a KPI target is met.

- "Run the DEFT loop"
- "Fine-tune until FAR below 0.1% at recall=100%"
- "Improve my AOI ChangeNet model using RCA and synthetic defects"
- "Iterate training until false accept rate meets the target"

Do not use this skill for a single standalone TAO training run, one-off inference, generic anomaly generation, or RCA-only analysis. Use the relevant agent directly when the user asks for only that step.

## Base Model

The loop operates on **NVIDIA TAO Visual ChangeNet** classify with the **NVIDIA C-RADIOv2-B** backbone, fine-tuned end-to-end. The architecture is defined in `specs/baseline_spec.yaml` — that file is the source of truth. All pretrained weights come from HuggingFace (`HF_TOKEN` required); `NGC_KEY` only gates container pulls. ChangeNet backbone resolution + the staged-file/HF-URL fallback for `model.backbone.pretrained_backbone_path` are owned by `references/visual-changenet.md`. SigLIP for k-NN mining is owned by `references/tao-mine-aoi-images.md`. AnomalyGen-side checkpoints (Cosmos-Predict2, T5, NVDINOV2, C-RADIO-V3, DINOv2-large, SAM2, Qwen3-VL — ~22 GB for 2B-only, ~140 GB with 14B + T5-11b) live under `<workspace>/augmentation/anomalygen/base_checkpoints/`; the paidf-anomalygen container auto-downloads them on first use. The PCB reference dataset under `<workspace>/augmentation/anomalygen/datasets/<project>/` is also auto-fetchable. See `references/paidf-anomalygen.md`.

## Train AutoML Policy

DEFT AOI owns the iterative data-improvement loop, retraining cadence, and KPI
checkpoint selection. For this workflow only, bypass model-level AutoML even
when the underlying Visual ChangeNet model metadata has `automl_enabled: true`.

`automl_policy: off` is a **workflow argument** to the Visual ChangeNet skill
invocation (the value the parent passes when calling `tao-skill-bank:tao-train-visual-changenet`
via the Skill tool), **not** a TAO spec field. Two cases:

- **Direct `docker run visual_changenet train -e <spec>`** (the path this workflow
  actually uses inline): no action needed. The TAO entrypoint is plain training
  by default; AutoML lives behind a different code path that the SDK orchestrates.
  Effectively, every direct `docker run` is already `automl_policy: off`.
- **SDK-orchestrated dispatch** (Brev/SLURM/k8s with the SDK building the
  command): pass `automl_policy: off` to `VisualChangeNetSDK.train(...)` or the
  equivalent runner argument. The SDK uses it to pick the plain-train command
  instead of the AutoML wrapper.

**Never add `automl_policy` or a `workflow` key to the spec YAML.** TAO's Hydra
`ExperimentConfig` schema does not recognize these keys and the train job
fails at config-merge time with
`Error merging '<spec>.yaml' with schema: Key 'workflow' not in 'ExperimentConfig'`.
This is a workflow-level override only; do not change model metadata, and do
not apply this policy to other workflows.

## Launch Intake

After the user confirms they want to run this workflow, ask which supported
platform they intend to run on. Generate the platform choices with:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} --format text
```

After platform selection, run:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --platform <platform> --format text
```

Ask only for credentials relevant to that platform, plus model-specific
credentials required by the selected workflow.

## Agent Behavior

> **There is exactly one user gate: pre-flight confirmation.** Print the Pre-Flight Summary
> (see `references/preflight.md` → Pre-Flight Summary), then STOP and wait for the user to type "go", "yes",
> "looks good", or similar explicit approval. Do not launch any side-effecting step
> (`docker run`, training, SDG, mutations under `${RESULTS_DIR}/`) before that approval —
> reading specs, listing files, `docker image inspect`, and populating the summary table
> are fine. **"Autonomous" describes behavior *after* this gate, not before it.** Do not
> skip the gate even if the user's original prompt sounded urgent ("just run it", "go
> ahead") — the summary itself is the artifact they need to see before approving.
>
> **After the gate, the skill is fully autonomous.** Run the entire loop without asking
> for confirmation. Do not pause between steps. Do not ask "want me to continue?" — just
> continue. Only stop if a step fails with an unrecoverable error or a hard-stop gate
> fires. Print a one-line status update at each step milestone so the user can follow
> progress.
>
> **Auto-mode required.** The post-gate loop fires constant side-effecting calls
> (`docker run`, `${RESULTS_DIR}/` writes); without auto-accept / bypass-permissions mode it
> stalls on the first prompt. Remind the user at the Pre-Flight Summary to enable auto-mode
> (shift+tab) before approving.
>
> **Blocker recovery.** Fix recoverable blockers yourself — missing image (pull), unstaged
> C-RADIO backbone (stage `.pth` per `references/visual-changenet.md`), missing pydeps (venv),
> absent AnomalyGen assets (paidf auto-fetches) — then resume the Pre-Flight step you were on
> (`<blocker> cleared → resuming step N`) and continue to the Summary. Halt only for what you
> can't fix (missing workspace/specs/CSVs/credentials, empty pool, leakage). A fix is not the
> user gate.
>
> **Revised plan.** If any run parameter changes after the original summary was shown (user imposes a time limit, overrides epochs, changes max_iterations, etc.), always re-run Pre-Flight and show an updated summary before proceeding.

## Workflow

Execute the loop in this order (full detail in `references/pipeline-and-state.md` → Pipeline + Stage Execution):

1. **Pre-Flight.** Run every check in `references/preflight.md`. Resolve workspace, specs, CSVs, checkpoints, container images. Hard stop only on missing input you can't resolve yourself (see `## Agent Behavior` → Blocker recovery).
2. **Baseline.** If `deft_state.json` already has `iterations.baseline.stage_completed == "train"` and a `best_ckpt_path` pointing at an existing file (the upstream `automl-deft-pipeline` pre-seeds these from its Phase 1 AutoML winner — see its Phase 1 → Phase 2 handoff), **skip the train sub-step** and resume at `inference -> evaluate` against the pre-seeded checkpoint. Otherwise run `train -> inference -> evaluate` by invoking the `tao-skill-bank:tao-train-visual-changenet` skill. Either way, then `rca` by invoking `tao-skill-bank:tao-analyze-gaps-visual-changenet`. Read `references/visual-changenet.md` and `references/tao-analyze-gaps-visual-changenet.md` first for DEFT-loop-specific args (mounts, output dirs, `deft_state.json` updates).
3. **Iterate.** For each iteration up to `max_iterations`, execute Pipeline steps 1-7. Between every step, re-read `results/loop_log.jsonl` tail + `results/deft_state.json` from disk — disk is canonical.
4. **Stop** when the KPI target is met, `max_iterations` is reached, or a hard-stop gate fires (silent-drop, AMP allocation mismatch, train/val leakage). Never auto-retry hard stops.
5. **Render** `results/DEFT_Loop_Report.html` after each completed iteration (and once more at loop end) by spawning the `reporter` subagent (`agents/reporter.md`). Per-stage renders are not done — every stage already appends one line to `loop_log.jsonl`, which is enough for a tail-watching user; the HTML render carries an iteration's worth of state and one render per iteration keeps the per-loop token cost roughly linear in iteration count, not in stage count. Do not render inline.

All pipeline stages run inline in the parent context — the parent invokes the underlying `tao-skill-bank:*` skills directly via the Skill tool, layering DEFT-loop conventions on top via the matching `references/*.md` file. The **only** delegated work is HTML report rendering, handled by the `reporter` subagent in a fresh context so an end-of-loop render is never silently dropped when the parent's context is saturated. See `references/scripts-and-agents.md` → Agents for the `reporter` spawn contract.

### Using Bundled Scripts

Run bundled scripts from `scripts/` via `run_script()` when the harness provides it (a Claude Code plugin runtime helper, not a function defined in this repo); otherwise fall back to direct `python`. Resolve every path argument to an absolute host path first. Never write `loop_log.jsonl` via `echo` or inline `jq` — the `seq` invariant requires reading the live tail through `next_seq()`. See `references/scripts-and-agents.md` for the full **Available Scripts** table, the `agents/reporter.md` spawn contract, the **Stage Reference Modules** stage→skill mapping, the path-rule invariant, and the workflow-level AutoML-policy pitfall. For per-script invocation examples, see `references/SCRIPT_USAGE.md`.

## Stage Reference Modules

Each pipeline stage maps to one underlying skill in the bank; the matching `references/*.md` file layers DEFT-loop conventions (mounts, output dirs, `deft_state.json` updates, `log_stage.py` summary string) on top of the skill's generic instructions. **Read the reference file first, then invoke the skill via the Skill tool.** If a reference file is missing, stop and ask the user to reinstall the plugin. The full stage→reference→skill→ownership table lives in `references/scripts-and-agents.md` → **Stage Reference Modules**. The stages: `train`/`evaluate` (`references/visual-changenet.md`), `anomalygen` (`references/paidf-anomalygen.md`), `rca` (`references/tao-analyze-gaps-visual-changenet.md`), `routing` (`references/tao-route-visual-changenet-samples.md`), and `data_mining` (`references/tao-mine-aoi-images.md`).

**Path rule (invariant).** Use absolute host paths under `${RESULTS_DIR}/iter${ITER}/` for every stage's output, mount `<workspace>` into the container at the same path, pre-create dirs world-writable, and reject any config containing `output: /results/...` or any path outside `<workspace>`.

## Data, Pre-Flight, Pipeline, and State references

| Topic | Reference | Contents |
|---|---|---|
| Bring-your-own-data, data contract, output layout, augmentation pool | `references/data-layout.md` | No public AOI dataset; full `<workspace>` input tree, ChangeNet 14-column CSV schema pointer, `${RESULTS_DIR}/` output tree, and the two-source mining-pool table |
| Pre-Flight checks, defaults, Pre-Flight Summary template, runtime estimate | `references/preflight.md` | The 10 ordered Pre-Flight checks, required input `max_iterations`, all defaults, the full Pre-Flight Summary table + populate commands, and the per-iteration runtime estimate |
| Pipeline steps, state/logging, stage execution, reports, runtime behavior | `references/pipeline-and-state.md` | Baseline pre-seed/skip-train logic, the 7 iteration Pipeline steps, `deft_state.json` + `loop_log.jsonl` schema and `seq` cadence, post-stage check, per-iteration HTML render, and the loop-end sequence |
| Bundled scripts, reporter agent, stage modules, AutoML pitfall | `references/scripts-and-agents.md` | Available Scripts table, `agents/reporter.md` spawn contract, Stage Reference Modules table, path-rule invariant, AutoML-policy spec trap |

**Required input — `max_iterations`.** No default; ask the user if not supplied and do not proceed past Pre-Flight without it. If the user gives a time limit instead, convert it to an estimated `max_iterations` using the per-iteration runtime figure in `references/preflight.md` and surface the estimate for confirmation. All other run parameters have defaults — never ask about a parameter with a default. The full defaults list and the Pre-Flight Summary the user approves at the single gate are in `references/preflight.md`.

## Gating

Run the full Pre-Flight (`references/preflight.md`), print the Pre-Flight Summary, then STOP at the one user gate. After approval, run the baseline (with the pre-seed/skip-train logic) and the 7-step iteration Pipeline, all detailed in `references/pipeline-and-state.md`.

Hard-stop and never auto-retry on: any stage `status=error`; train/validation leakage (the mid-iteration check on `mining_filter/mining_pool.csv` right after mining, and the post-assembly check on the combined CSV); a missing or zero-row mining pool; a failed CSV existence check; silent-drop; and AMP allocation mismatch. The loop stops when the KPI target is met, `max_iterations` is reached, or an unrecoverable gate fires. Each terminal path runs the loop-end sequence: append the final `loop_stop` entry via `scripts/log_stage.py`, backfill token usage with `scripts/align_token_usage.py`, spawn the `reporter` agent one final time (`trigger="loop-end"`), then run `scripts/prepare_inference_spec.py` — skipped only when no valid checkpoint exists. Per-stage state cadence (one `loop_log.jsonl` entry per stage, `seq=last+1` from disk, disk is canonical, HTML render once per iteration and at loop end) is specified in `references/pipeline-and-state.md`.
