---
name: tao-run-deft-aoi
description: >
  Run the full DEFT AOI improvement loop for NVIDIA TAO VisualChangeNet / ChangeNet PCB inspection models:
  baseline evaluate, RCA, Cosmos AnomalyGen / AMP synthetic defects, k-NN mining, retraining, and deployment
  gating against a customer-defined primary metric and optional constraints. Use only when the request identifies
  an AOI / automated-optical-inspection, PCB-defect, VisualChangeNet, or ChangeNet workflow. Supports air-gapped/offline
  runs with pre-staged assets. Never infer AOI from generic iterative-improvement language. Do not use for CLIP / SigLIP
  image retrieval, attribute-labelled image data, standalone TAO training, one-off inference, generic anomaly generation,
  or RCA-only analysis.
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

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

## Execution Contract

Treat this as a disk-backed state machine, not as a prose recipe.

1. Preserve every explicit user value. `epoch 1` means `num_epochs=1` and
   `iteration 1` means `max_iterations=1`; a heuristic or spec default applies
   only when the user did not supply that parameter. Show the source of every
   run parameter (`user`, `spec`, or `default`) in the Pre-Flight Summary.
   Preserve the customer's metric name, operator, target, unit, evaluator, and
   constraints. The approved `metric_contract` is the source of truth for
   evaluation, checkpoint selection, completion, and reporting.
2. After the user approves the Summary, set
   `PYTHON=$(bash scripts/deft_python.sh)`, then initialize `deft_state.json`
   once with `"$PYTHON" scripts/init_deft_state.py`, passing Preflight's exact GPU model/memory,
   resolved `--network-mode`, activation source, and selected absolute Python.
   The resulting `execution_policy` is immutable run state. Never hand-author
   or reinitialize it on resume.
3. On startup, after context compaction, before every stage, and before any
   completion claim, run `"$PYTHON" scripts/deft_context.py --state ... --stage ...`.
   Use its durable `next_stage` plus the state file's
   `status`, `current_iteration`, `iterations.*.status`, `stage_completed`,
   and latest `events` entry to resume. Do not infer progress from assistant
   prose or from an artifact that is not recorded in state.
4. Invoke the mapped underlying skill after reading the DEFT overlay. Do not
   replace a missing/unread stage reference or a failed skill call with guessed
   shell commands, inline Python, a different output tree, or data fabricated
   from the KPI set.
5. After initialization, run install/fetch/login/container commands through
   `"$PYTHON" scripts/deft_exec.py --state ... -- <command>`. Air-gap mode rejects egress
   and installs, injects offline flags, and enforces no-pull. Selected
   platforms must enforce the equivalent policy.
6. Commit every stage with `"$PYTHON" scripts/commit_stage.py`; it verifies the stage's
   required inputs and atomically updates both the resume snapshot and ordered
   `events` array inside `deft_state.json`. Never edit the state file with
   inline Python, jq, heredocs, or an editor. Fix rejected evidence; never
   fabricate state. For evaluate, pass the metric result,
   checkpoint, inference CSV, and threshold directly to `commit_stage.py`.
   Pass positive measured `--duration-sec` from backend elapsed time or a host
   timer for executed stages. A documented `--skip` may record `0`; negative
   durations are always rejected.
7. Claim the loop complete only after `"$PYTHON" scripts/finalize_run.py` creates the
   handoff artifacts, successfully commits `loop_stop`, and a
   fresh read of `deft_state.json` shows `status == "complete"`,
   `iterations.baseline.status == "complete"`, and the final iteration's
   `status == "complete"`. A checkpoint, inference CSV, report, or assistant
   message is not completion evidence by itself.

## Context Discipline

- Load references just in time. Re-read state, then read only the current
  stage's named section and act.
  Never preload/cat every reference or underlying skill, recursively list the
  skill tree, or re-read a reference already present in the current context.
- Redirect verbose train, inference, Docker, and SDG output to files. Inspect
  at most the final 40 lines or a one-line status/artifact check; never print a
  full spec, state file, loop log, or generated script into the conversation.
- A Skill-tool call loads stage instructions; it does not start a background
  orchestrator. Continue the documented stage in the parent immediately after
  it returns. Never sleep or poll waiting for a Skill-tool process. For actual
  background Docker work, save the PID and poll at intervals no longer than 30s.
- At the start of Pre-Flight, resolve network mode before dependencies. Read
  exactly one branch: `references/air-gap.md` for air-gap mode or
  `references/network-bootstrap.md` for network-enabled mode. Never load the
  network bootstrap in an air-gapped run.

## When to Use This Skill

Use this skill when the user wants an agent to run the full DEFT AOI improvement loop for an NVIDIA TAO VisualChangeNet / ChangeNet PCB inspection model: baseline evaluation, RCA, synthetic defect generation, data mining, retraining, and deployment gating until a KPI target is met.

- "Run the DEFT loop"
- "Fine-tune until the configured quality metric meets its target"
- "Optimize a customer-defined metric while preserving its constraints"
- "Improve my AOI ChangeNet model using RCA and synthetic defects"
- "Iterate training until the deployment KPI meets the target"

Do not use this skill for a single standalone TAO training run, one-off inference, generic anomaly generation, or RCA-only analysis. Use the relevant agent directly when the user asks for only that step.

## Base Model

The loop uses **NVIDIA TAO Visual ChangeNet** classify with either end-to-end
C-RADIOv2-B or a frozen DINOv3 backbone. `specs/baseline_spec.yaml` defines the
architecture. Backbone variants, staging, `HF_TOKEN`, and mount rules are owned
by `references/visual-changenet.md`; the spec always points to a local mounted
file. `NGC_KEY` gates container pulls. SigLIP mining is owned by
`references/tao-mine-aoi-images.md`; AnomalyGen assets and network/air-gap rules
are owned by `references/tao-generate-anomalies.md` and
`references/air-gap.md`. The container owns its base-asset list; this workflow
keeps bootstrap on Text2Image 2B by passing `--model_sizes 2B` explicitly.

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
platform they intend to run on. Discover the execution platforms from the
installed platform skills (tao-run-on-docker / -slurm / -kubernetes / -brev,
plus any external one); on a runtime that surfaces only the core router skills,
read `skills/platform/tao-run-on-*/SKILL.md` frontmatter.

After platform selection, read the chosen platform skill's `## Credentials`
section and `references/skill_info.yaml` (required_credentials /
credential_groups).

Never ask for or print credential values. Check only whether the variable is
set (`[ -n "$VAR" ] && echo SET || echo UNSET`); if unset, name it so the user
can export it or put it in a user-approved env file (`~/.tao/secrets.env`,
`~/.config/tao/.env`, or one they point at), loaded with
`set -a; source /path/to/.env; set +a`. The run never creates or writes that
file.

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
> **Blocker recovery.** Before the user gate, select a complete installed host
> interpreter through `deft_python.sh`. If none exists, follow only the
> already-selected network-mode reference. Air-gap mode hard-stops without a
> package-manager command; network-enabled bootstrap is isolated in
> `references/network-bootstrap.md`.
> Apply the network-mode branches in `references/air-gap.md`; record permitted
> fetches and directory creation as
> post-approval work, or validate staged assets in air-gap mode. After
> approval, fix recoverable blockers yourself, then
> resume the Pre-Flight step you were on (`<blocker> cleared → resuming step N`).
> Halt only for what you cannot fix (missing workspace/specs/CSVs/credentials,
> empty pool, leakage). A fix is not another user gate.
>
> **Non-zero command rule.** Never repeat an unchanged failed command and never
> switch to an undocumented CLI/module path by trial and error. Read the final
> error block (not only the container banner), map it to the loaded stage
> reference/underlying skill, make one evidence-based correction, and rerun its
> documented verification. If the reference does not cover the failure, commit
> `status=error` and halt instead of improvising a reduced workflow.
>
> **Revised plan.** If any run parameter changes after the original summary was shown (user imposes a time limit, overrides epochs, changes max_iterations, etc.), always re-run Pre-Flight and show an updated summary before proceeding.

## Workflow

Execute the loop in this order (full detail in `references/pipeline-and-state.md` → Pipeline + Stage Execution):

1. **Pre-Flight.** Run every check in `references/preflight.md`. Resolve workspace, specs, CSVs, checkpoints, container images. Hard stop only on missing input you can't resolve yourself (see `## Agent Behavior` → Blocker recovery).
2. **Baseline.** If `deft_state.json` already has `iterations.baseline.stage_completed == "train"` and a `best_ckpt_path` pointing at an existing file (the upstream `automl-deft-pipeline` pre-seeds these from its Phase 1 AutoML winner — see its Phase 1 → Phase 2 handoff), **skip the train sub-step** and resume at `inference -> evaluate` against the pre-seeded checkpoint. Otherwise run `train -> inference -> evaluate` by invoking the `tao-skill-bank:tao-train-visual-changenet` skill. Evaluate with the approved contract and evaluator in `references/metric-contract.md`. Either way, then `rca` by invoking `tao-skill-bank:tao-analyze-gaps-visual-changenet`. Read `references/visual-changenet.md`, `references/metric-contract.md`, and `references/tao-analyze-gaps-visual-changenet.md` first for DEFT-loop-specific args.
3. **Iterate.** For each iteration up to `max_iterations`, execute Pipeline steps 1-7. Between steps re-read `deft_state.json` and continue from its `stage_completed` value; do not print the full state.
4. **Stop** when the KPI target is met or `max_iterations` is reached by running
   `"$PYTHON" scripts/finalize_run.py` with the matching reason. Hard-stop failures are
   committed as errors and are never relabeled as successful `loop_stop`.
5. **Render automatically.** `scripts/init_deft_state.py` writes the initial
   `results/DEFT_Loop_Report.html`; every successful `commit_stage.py` call
   then refreshes it with the deterministic report hook implemented in
   `scripts/render_report.py`. The `loop_stop` commit therefore produces the final
   report even when the parent context is saturated. If a hook reports an
   error, run `"$PYTHON" scripts/render_report.py --results-dir "${RESULTS_DIR}"`
   directly after repairing the named presentation input; never hand-author
   report HTML.

All pipeline stages run inline in the parent context. Prefer invoking the underlying `tao-skill-bank:*` skills directly via the Skill tool, layering DEFT-loop conventions on top via the matching `references/*.md` file. If the mapped Skill tool is unavailable but Docker, the skill source tree, and the stage reference modules are present, use the documented direct-container fallback in `references/scripts-and-agents.md`; before the first fallback stage, write `execution_path=direct-container` to the transcript, and for each fallback stage record the mapped underlying skill name plus the exact direct command used. Preserve the same `deft_state.json`, artifact, and script-backed report contracts. HTML rendering is not delegated.

### Using Bundled Scripts

For each tool call, set `PYTHON=$(bash <skill_root>/scripts/deft_python.sh)`;
then use `"$PYTHON" <skill_root>/scripts/<name>.py`. Resolve every path
argument to an absolute host path first. Use
`deft_context.py` before each stage, `deft_exec.py` for external execution, and
`commit_stage.py` for all state writes. See
`references/scripts-and-agents.md` for script invocations, the automatic
report hook, stage mapping, direct-container fallback, and path invariants.

## Stage Reference Modules

Each pipeline stage maps to one underlying skill in the bank; the matching `references/*.md` file layers DEFT-loop conventions (mounts, output dirs, and `commit_stage.py` arguments) on top of the skill's generic instructions. **Read only the current stage's relevant section, then invoke the skill via the Skill tool or the documented direct-container fallback; never preload all stage references.** If a reference file is missing, stop and ask the user to reinstall the plugin. The full stage→reference→skill→ownership table lives in `references/scripts-and-agents.md` → **Stage Reference Modules**. The stages: `train`/`evaluate` (`references/visual-changenet.md`), `anomalygen` (`references/tao-generate-anomalies.md`), `rca` (`references/tao-analyze-gaps-visual-changenet.md`), `routing` (`references/tao-route-visual-changenet-samples.md`), and `data_mining` (`references/tao-mine-aoi-images.md`).

**Path rule (invariant).** Record absolute host artifact paths under
`${RESULTS_DIR}`. For ChangeNet direct containers, mount
`"$WORKSPACE:/data/workspace"` and `"$RESULTS_DIR:/results"`; specs use
`/results/baseline/<stage>` or `/results/iterN/<stage>`. Other stages retain
their reference module's required workspace mount. Never remap the run
directory to `/results/iterN`.

## Data, Pre-Flight, Pipeline, and State references

| Topic | Reference | Contents |
|---|---|---|
| Air-gap activation and offline execution | `references/air-gap.md` | Global mode triggers, precedence, prohibited network actions, staged-asset requirements, and Pre-Flight evidence |
| Bring-your-own-data, data contract, output layout, augmentation pool | `references/data-layout.md` | No public AOI dataset; full `<workspace>` input tree, ChangeNet four-column required CSV schema, `${RESULTS_DIR}/` output tree, and the two-source mining-pool table |
| Customer metric contract and evaluator adapter | `references/metric-contract.md` | Primary metric schema, comparison direction, evaluator JSON, constraints, evaluate commit, and compatibility behavior |
| Pre-Flight checks, defaults, Pre-Flight Summary template, runtime estimate | `references/preflight.md` | The 10 ordered Pre-Flight checks, required input `max_iterations`, all defaults, the full Pre-Flight Summary table + populate commands, and the per-iteration runtime estimate |
| Pipeline steps, state, stage execution, reports, runtime behavior | `references/pipeline-and-state.md` | Baseline pre-seed/skip-train logic, the 7 iteration Pipeline steps, the `deft_state.json` snapshot + event schema, post-stage check, per-iteration HTML render, and the loop-end sequence |
| Bundled scripts, report hook, stage modules, AutoML pitfall | `references/scripts-and-agents.md` | Available Scripts table, deterministic report renderer and post-commit hook, Stage Reference Modules table, path-rule invariant, AutoML-policy spec trap |

**Required input — `max_iterations`.** No default; ask the user if not supplied and do not proceed past Pre-Flight without it. If the user gives a time limit instead, convert it to an estimated `max_iterations` using the per-iteration runtime figure in `references/preflight.md` and surface the estimate for confirmation. All other run parameters have defaults — never ask about a parameter with a default. The full defaults list and the Pre-Flight Summary the user approves at the single gate are in `references/preflight.md`.

## Gating

Run the full Pre-Flight (`references/preflight.md`), print the Pre-Flight Summary, then STOP at the one user gate. After approval, run the baseline (with the pre-seed/skip-train logic) and the 7-step iteration Pipeline, all detailed in `references/pipeline-and-state.md`.

Hard-stop and never auto-retry on: any stage `status=error`; train/validation leakage; a missing or zero-row mining pool; a failed CSV existence check; silent-drop; AMP allocation mismatch; a PAIDF-incompatible AnomalyGen fine-tuned checkpoint; a missing AnomalyGen Guardrail checkpoint; or an SDG log showing disabled screening. The loop stops when the KPI target is met, `max_iterations` is reached, or an unrecoverable gate fires. Each terminal path commits `loop_stop` through `commit_stage.py`, then follows the loop-end sequence in `references/pipeline-and-state.md`.
