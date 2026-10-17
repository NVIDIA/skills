# Validate a physics selection

Use these checks when the recommendation depends on a capability, migration, gradient, or performance claim. Keep checks within the user's task and execution budget. An engine comparison request does not by itself require installing every candidate.

## Evidence to retain

Record the input scene and asset versions, required physics, engine/solver/integration versions, hardware, initial conditions, controls, timestep, contact settings, observation/sensor workload, and acceptance criteria. Separate source claims, observed behavior, and untested assumptions.

Choose a representative task check, for example:

| Task | Useful evidence |
| --- | --- |
| Learning throughput | Simulated experience per wall-clock second, batch size, reset/observation costs, and task fidelity checks |
| Grasp or insertion | Success definition, slip, penetration/constraint error, applied forces, and repeated trials |
| Deformable manipulation | Required material/attachment behavior, contact handling, stability, and representation/import coverage |
| Shared-stage application | Intended state/drive edits reach the solver; published motion reaches the consumer with correct ownership and ordering |
| Gradient-based optimization | Required derivative exists, is finite, and agrees with a suitable numerical or analytic reference |

**Human note:** a fast but physically different task is not a valid performance win. Equally, a scene that loads successfully may still omit an actuator, misinterpret a joint, or fail during contact.

## A fair comparison

Use equivalent geometry, masses, controls, duration, and task acceptance criteria. Engine-specific parameter names are not interchangeable; tune each configuration to the same physical target and disclose residual model differences. Measure setup/compilation separately from steady-state runtime. Synchronize GPU work before reading elapsed time. State whether rendering, sensors, publication, and policy inference are included.

Match whether the objective is single-rollout latency or many-environment throughput. Use repeated trials and report variation when ranking close candidates. A benchmark from another scene or vendor is a lead to investigate, not a measured result for this task. Do not claim better sim-to-real transfer without relevant physical validation.

For gradients, test the requested loss and parameter through the selected solver path. Compare automatic derivatives with finite differences or an analytic reference where appropriate. Include contact regimes relevant to the task, vary the perturbation scale, and account for nonsmooth contact transitions. A smooth no-contact example does not qualify a contact-heavy optimization workload. Inspect whether stage/import, controller, or sensor operations break the required gradient path.

If a check cannot run, provide its exact purpose and procedure and label it not run. A documentation-based recommendation can remain useful without inventing runtime evidence.

## Decision record

Report the selected engine, solver, and integration separately, with:

- Task requirements that the selection satisfies, and any conditions or unknowns.
- The closest viable alternative and the specific reason for the preference.
- Source and runtime versions, commands or experiments actually run, and measured outcomes.
- Migration/setup cost that affects the recommendation and the next bounded action.

If two options remain plausible, identify the smallest discriminating experiment instead of asserting a winner. If no option satisfies a hard requirement, state the unresolved requirement.

## Evaluate the skill itself

[Evaluation cases](../evals/evals.json) cover Newton recommendations, bridge selection, alternatives, and missing evidence. For a live-agent evaluation, give the host the prompt, installed skill, and permitted project artifacts. Assess the engine/solver/integration decision, requirements coverage, factual accuracy, and honesty about tests. For discovery checks, use the generic prompt without naming the skill.

These case definitions are not execution results. A paired with-skill/without-skill trial can measure behavior changes. Count justified alternatives as successful decisions; do not use the fraction of Newton recommendations as a quality metric.
