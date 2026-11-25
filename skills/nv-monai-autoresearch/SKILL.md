---
name: nv-monai-autoresearch
description: Used for NV-MONAI-Autoresearch medical-imaging research, experiment planning, diagnosis, and resumable optimization. Not for clinical decisions or standalone model inference.
license: Apache-2.0
allowed-tools: Read Write Edit Bash WebSearch WebFetch
metadata:
  author: "NVIDIA MedTech <noreply@nvidia.com>"
  version: "0.1.0"
  tags: [medical-imaging, research, experiments]
---


# NV-MONAI-Autoresearch

## Purpose

Run the MIAgent v3.5 medical-imaging research workflow with a durable,
human-inspectable research record. The host agent performs research and execution;
this skill supplies the operating contract, original artifact templates, and
supporting tools. Use for a complete research task or resuming one. Engineering
and research use only.

## Instructions

Resolve `SKILL_ROOT` to the directory containing this `SKILL.md`.
Every `assets/`, `references/`, and `scripts/` path below is relative to that
installed directory. Workspace paths such as `challenge/<name>` belong to the
caller's working directory, independently of the installation. Resolve each
`miagent-template-source` against `SKILL_ROOT`; never write run artifacts
into the installed skill.
For v3.5 records, map the legacy `challenge/challenge_template/` prefix to
`assets/challenge_template/`; the checker accepts both pointer formats.

For a new task, run:

```bash
python3 "${SKILL_ROOT}/scripts/init_challenge.py" request.json --output-dir challenge/my-task
```

The `task_request` input is JSON with `name` and `task` strings. The
`initialization_report` output lists the created records and confirms that no
experiments ran. The initializer creates only
the initial task, dashboard, and trajectory records. It does not approve a
candidate, create a baseline, or execute an experiment. Create the remaining
artifacts from their templates at the workflow's prescribed phases. After
initialization, fill known task facts and open questions under step 1; record
authorization only to the extent the user actually supplied it.

If the host exposes it, the equivalent invocation is
`run_script("scripts/init_challenge.py", args=["request.json", "--output-dir", "challenge/my-task"])`.
For an existing task, read its records and verify live process state before
resuming; never reinitialize it. Read `skill_manifest.yaml` when using the machine
entrypoint: it describes initialization, not execution of the whole research loop.

Follow the ten steps below. Existing explicit user authorization remains valid;
record it at the relevant gate rather than asking for the same approval again.
A research subagent or critic may be used when the host permits delegation.
Otherwise use the workflow's local research and dedicated self-critique fallback.
Use host scheduling/heartbeat tools when available; otherwise monitor within the
active session and record an explicit resume command and remaining jobs before
handoff. Never claim monitoring will continue after the host stops running.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/init_challenge.py` | Initialize the first three research records; JSON on stdout. | `request.json --output-dir challenge/my-task` |
| `scripts/check_consistency.py` | Existing v3.5 artifact checker and discovery seal. | `challenge/my-task --strict --json`; discovery: `challenge/my-task --phase discovery --strict --seal --json` |
| `scripts/webui/app.py` | Optional read-only localhost artifact browser. | `--challenge challenge/my-task --port 8765` |
| `scripts/export_skill.py` | Export only the distributable skill files into a fresh directory. | `--output-dir /path/to/skills/nv-monai-autoresearch` |

Use these checker commands at the corresponding workflow gates, replacing
`challenge/my-task` with the actual workspace. Seal discovery only after the
candidate gate; run the general strict check before training and final evaluation.

```bash
python3 "${SKILL_ROOT}/scripts/check_consistency.py" --phase discovery --strict --seal challenge/my-task
python3 "${SKILL_ROOT}/scripts/check_consistency.py" --strict challenge/my-task
```

## Prerequisites

- Python 3.10+; initialization, checking, and the browser use the standard library.
- Set `SKILL_ROOT` to the installed skill directory for the commands above.
- Research uses available primary-source search tools. Models, datasets, GPUs,
  Docker, schedulers, and network access are task-specific, not prerequisites
  for the initialization entrypoint. Establish them in readiness before use.
- An external critic is a caller-configured host tool. Use only a destination
  authorized for the selected research records; otherwise use the local critique
  fallback. No provider client or credential handling is bundled. Read the
  [critic review instruction](references/critic-review.md) when preparing a review.

## Workflow

- One active session operates one durable `challenge/<task>` folder initialized
  from `assets/challenge_template/`.
- This file owns phase order, gates, continuous execution, and handoff. Raw
  templates own artifact fields and update rules; installed method/compute skills supply optional techniques.

```text
understand -> readiness -> independent thesis -> open research -> candidate gate
-> discovery seal -> implementation-plan gate -> implement/smoke
-> internal loop -> external loop when valuable -> freeze
```

## File Ownership

A filled artifact remains governed by the raw template named in its
`miagent-template-source`. Reload that template before creating or materially
updating the artifact, and after resume or context compaction. Write an event in
its narrowest owner and link it elsewhere instead of copying it.

| Artifact | Purpose | Update When |
|---|---|---|
| `challenge.md` | Stable task, data, metric, rule, and submission contract | Initial understanding, or a fact/assumption changes |
| `dashboard.md` | Live authorization, resources, jobs, heartbeat, blockers, and important-result index | Operational truth, queue, or next wake changes |
| `literature.md` | Preserved independent thesis, decision-changing research, and revised direction | Initial discovery or focused re-search changes evidence |
| `candidates.md` | Scientific selection and living external-direction map | A direction, judgment, blocker, or decisive next test changes |
| `baselines/<baseline>/baseline.md` | Current implementation contract and living Internal Exploration | Plan/adopted implementation changes, or valid evidence changes an internal possibility |
| `runs/<exp>/experiment.md` | One scientific intervention's immutable Intent, evidence, Outcome, and diagnosis | Before launch and after a valid decision-bearing result |
| `runs/<exp>/jobs/` | Commands, jobs, retries, resumes, and execution evidence | An execution stage changes |
| `trajectory.md` | Concise causal index of consequential agent decisions and human steering | Focus, direction, plan, authority, or stop outcome changes |
| `state.md` | Small current scientific-state reset and continue/switch/freeze judgment | Current focus, exploration pointer, constraint, or judgment changes |

For a valid result: write the experiment Outcome first; update the affected
baseline or candidate map; append the decision and decisive evidence to
`trajectory.md` only if the decision changed; update `state.md` last only if the
current scientific state changed. Keep `dashboard.md` operationally current.
Retain human steering in `state.md` only while it constrains current work.

## The 10 Steps

1. **Understand.**
   - Initialize or verify `challenge.md`. Resolve blocking questions; use a
     stated assumption only when work can safely continue.
   - Natural-language human steering can redirect, pause, or stop the run at any
     time. On pause/stop, verify process state and launch nothing; kill work only
     when requested or when a recorded kill rule applies.

2. **Establish readiness and authority.**
   - Record readiness, objective/target, compute budget, and authorization in
     `dashboard.md`. Before discovery is sealed, allow only task facts,
     split-safe data characterization, rule/evaluator clarification, downloads,
     and general resource checks—no method-specific artifact, probe, or GPU job.
   - Prior MIAgent or comparison-session results are benchmark history when the
     user excludes them. Public papers, official leaderboards, winner methods,
     and released implementations remain normal research evidence unless the
     user forbids them. Never inspect protected labels or case-level protected
     summaries.
   - `:/auto` supplies the initial candidate and plan approvals, plus later
     direction authority, only after each gate's reasoning is complete. It does
     not authorize ambiguous rules, invalid evaluation, human-only access, or
     unapproved expensive compute. Record it in `dashboard.md` and `trajectory.md`.

3. **Think, search, and synthesize.**
   - Before literature, specialized method skills, implementations, package
     probes, prior solutions, or candidate ranking, the main agent writes the
     Independent Scientific Thesis in `literature.md` from the task and its own
     knowledge. Preserve it rather than rewriting hindsight into it.
   - Use one fresh research subagent when available. Give it `challenge.md` and
     the thesis without inherited chat, prior candidates, or a prescribed search
     taxonomy. It searches freely for evidence that may challenge or extend the
     thesis, including current primary sources, implementations, weights,
     applicable installed skills, and permitted prior-task evidence. During an
     external loop, also give it the current bottleneck and affected directions.
   - The subagent fills only Open Research Findings and returns decisive Evidence
     IDs and blockers; it does not choose the solution. If delegation is
     unavailable, preserve the same think-first separation locally. While it
     runs, the main agent may do only the neutral readiness work from Step 2.
   - The main agent spot-checks decisive sources, writes the Revised Scientific
     Direction, and completes the post-search omission audit. Search only gaps
     that could change the decision. Published pipelines are evidence and
     reusable components, not boundaries on the agent's solution.
   - Draft `candidates.md`, then run the external critic on the thesis, findings,
     revised direction, and candidates. Integrate only objections that change the
     judgment. If the provider fails, use a fresh critic subagent, or a dedicated
     self-critique only when delegation is unavailable. Provider failure is not
     a passed critic gate. Clear all discovery-checker warnings.

4. **Select one baseline.**
   - Select the complete pipeline the agent judges most likely to maximize the
     objective. Record how research verified, refined, replaced, or rejected the
     thesis and compare the strongest alternative. Evidence informs judgment but
     does not force an existing published pipeline.
   - Do not demote the strongest direction for setup time, unfamiliarity, cached
     dependencies, or a desire to start. Only a verified rule/license/access,
     asset, interface, or minimum-compute blocker can justify selecting another
     direction. A simple control may be diagnostic but cannot replace the
     selected pipeline.
   - Present the candidate gate and wait unless `:/auto` is active. Record the
     result in `candidates.md`, `trajectory.md`, and `dashboard.md`, then run the
     discovery-seal command in Available Scripts.

     The initial seal must precede baseline, code, environment, experiment, or
     job artifacts. Do not initialize `state.md` yet.

5. **Plan implementation.**
   - After the seal, create the selected baseline and write its serious,
     multi-component Implementation Plan. Prefer and inspect mature reusable
     implementations; define component roles, interfaces, reuse/build
     boundaries, compute use, GPU efficiency, and human-inspectable evidence.
   - Leave Internal Exploration empty until a valid performance result reveals
     what is worth improving. Do not pre-commit a generic science checklist.
   - Draft the shared visualization contract. Start no experiment before the
     plan gate.

6. **Pass the implementation-plan gate.**
   - Present `baseline.md` and wait unless `:/auto` is active; revise in place.
   - After approval, append the decision to `trajectory.md`, initialize
     `state.md` with one focus plus baseline/external pointers, and activate the
     dashboard. Its internal pointer may say the first decision-bearing result
     will seed Internal Exploration. Later directions retain authorization unless
     paused or new authority is required.

7. **Implement and smoke.**
   - Implement the complete selected composition and shared visualizer. Before
     launch, create the `experiment.md` Intent in `exp001_smoke/`, add it to the dashboard,
     and establish one session heartbeat covering every asynchronous job.
   - Keep every stage under the experiment's `jobs/`; reconcile real process
     state, keep evaluation provenances distinct, profile representative GPU
     use, produce the viewer, and complete Implementation Acceptance. A
     control-only smoke does not prove the selected composition.
   - Smoke/probes are diagnostic, not performance Outcomes. Establish a
     meaningful reference result; after its Outcome, seed Internal Exploration
     with only task-specific possibilities supported by evidence or judgment.
   - Before non-smoke training, complete readiness/acceptance, write every
     planned Intent, and pass the general strict command in Available Scripts.
     Every warning blocks admission.

8. **Run the internal loop.**
   - From the state focus and living Internal Exploration, choose the
     highest-value question; add a new one when evidence or intuition reveals a
     better opportunity. Write Intents before scheduling.
   - Within the one active baseline, use adaptive racing when resources allow:
     run fair recipe, component, architecture, fold, or seed screens in parallel
     on the same evaluation contract; promote the best one or two and useful
     complements; deepen them; then confirm robustly. A screen must be
     decision-bearing, and one weak fold/seed or undertrained run cannot reject a
     method family. Mature evidence may justify going directly to robust training.
   - Monitor actual processes. Repair code, environment, OOM, preemption,
     timeout, and scheduler failures in the same experiment without a
     performance Outcome; they are not scientific evidence.
   - For each valid result, create the viewer and Outcome, update only affected
     exploration/implementation judgments, and follow the artifact propagation
     order above. Use [result analysis](references/result-analysis.md) for important,
     surprising, weak, or suspicious results.
   - At each focus boundary, ask whether any plausible within-direction change
     is still likely to produce a large gain. Use the state template's reminders
     to challenge the judgment, but test only relevant ideas. Checklist
     completion, implementation difficulty, default/smoke completion, a weak
     plateau, ordinary failure, or an empty queue does not justify exit.
   - Switch to the external loop when its expected scientific value exceeds
     further local tuning. A competent reference is useful, but need not be
     exhaustively optimized when evidence points to a method-class mismatch or
     stronger formulation.
   - Maintain the heartbeat until freeze, completion, or human pause/stop. At
     each wake, do one bounded scheduler/log/result reconciliation. Before
     yielding while work continues, schedule/execute the next action and record
     its dashboard link and next wake.

9. **Run the external loop.**
   - Record the switch in Internal Exploration, `trajectory.md`, and `state.md`.
     Return to Step 3 for focused research on the current bottleneck; preserve
     the original thesis, add only decision-changing findings, rerun the critic,
     and update the external map.
   - Select the unblocked direction now judged most likely to maximize the
     objective and reseal discovery. Reuse the baseline only for a compatible
     extension; otherwise create and approve a new baseline. Keep one active
     baseline—select and activate an independent direction before its first
     experiment, then use parallel racing within it. Do not require experimental
     proof that can only exist after the switch.
   - Continue while an external direction is plausibly likely to produce a large
     gain. Candidate rows need not all be mechanically closed; the agent may
     judge one not worthwhile with a causal comparison. Difficulty or lack of an
     implementation attempt is not evidence that it is scientifically weak.

10. **Freeze or stop.**
    - A one-shot held-out limit controls evaluation timing, not exploration.
      Freeze only when the result is trustworthy and `state.md` explains why
      neither internal work nor an external direction is likely to produce a
      large gain. An explicit terminal target or human stop may end sooner;
      crossing a reference score alone does not end budgeted maximization.
    - Compute limits execution, not scientific imagination, and need not be
      depleted. Useful remaining compute plus a plausible large-gain test means
      continue. A verified resource limit may narrow the fair test but must not
      become a scientific rejection.
    - If the frozen method retrains on all data or changes checkpoints, code,
      preprocessing, or inference from the validated path, run a train-only
      final-path canary. It proves technical equivalence only and cannot replace
      validation, folds, seeds, or ensembles; prefer the validated ensemble when
      all-data equivalence is uncertain.
    - Record a justified freeze in `trajectory.md`, then `state.md`, then
      dashboard readiness. Pass the strict checker before protected/final
      evaluation. Record completion through the same owner-to-state order and
      end the heartbeat only on freeze/completion or human pause/stop.

## Artifact Templates And Analysis References

Read only the templates and evidence guidance needed for the current phase.
Template field names, schemas, artifact ownership, and update rules follow v3.5.

| When | Source |
|---|---|
| Task understanding | [Challenge](assets/challenge_template/challenge.md) |
| Resources, authorization, and jobs | [Dashboard](assets/challenge_template/dashboard.md) |
| Scientific research | [Literature](assets/challenge_template/literature.md) |
| Candidate selection | [Candidates](assets/challenge_template/candidates.md) |
| Current focus and stop judgment | [State](assets/challenge_template/state.md) |
| Consequential decisions | [Trajectory](assets/challenge_template/trajectory.md) |
| Implementation plan and internal exploration | [Baseline](assets/challenge_template/baseline_template/baseline.md) |
| Experiment Intent and Outcome | [Experiment](assets/challenge_template/baseline_template/runs/exp_template/experiment.md) |
| Execution logs and repairs | [Jobs](assets/challenge_template/baseline_template/runs/exp_template/jobs/README.md) |
| Discovery critique | [Critic contract](assets/challenge_template/critics/README.md) |
| Shared result viewer | [Visualization contract](assets/challenge_template/shared/visualization/README.md), [renderer template](assets/challenge_template/shared/visualization/visualizer.py) |
| Valid-result diagnosis | [Result analysis](references/result-analysis.md) |
| Task-specific evidence | [Segmentation](references/segmentation-analysis.md), [registration](references/registration-analysis.md), [generation](references/generation-analysis.md), [representation](references/representation-analysis.md), [VLM](references/vlm-analysis.md) |

## Limitations

- Does not provide a clinical diagnosis, treatment recommendation, or clinical validation.
- Method implementations and compute adapters are supplied by the user's
  environment. No model weights, datasets, benchmark launchers, or private
  compute profiles are bundled.
- The checker validates recorded consistency; it cannot prove that prose is
  truthful, that a metric is scientifically valid, or that protected data was
  never accessed. Inspect primary evidence before accepting results.
- Initialization and static validation do not establish end-to-end research
  effectiveness or NVIDIA publication approval.

## Troubleshooting

| Problem | Action |
|---|---|
| Workspace already exists | Read its records and resume; choose a new directory only for a new task. |
| Missing template or helper | Verify `SKILL_ROOT` and reinstall the complete skill directory. |
| Strict checker rejects a record | Read the finding and owning raw template; repair the evidence or record before proceeding. |
| Critic provider unavailable | Use a permitted fresh critic subagent or dedicated self-critique with the same review instruction; retain the report. |
| Scheduler or heartbeat unavailable | Verify processes directly; keep live state and handoff instructions in the dashboard. |
