# Result Analysis

Use this for a valid decision-bearing result in workflow step 8. It supplies the
diagnostic method; the raw `experiment.md` template owns the recorded evidence and
analysis fields. The purpose is to decide what to do next, not to produce a long
report.

## 1. Verify The Result

- If the job failed or killed due to time limit, repair or resume it in the same
  experiment. Do not append an Outcome.
- Identify the exact config/checkpoint, training data and split, evaluated
  cases and provenance, evaluator or submission, and numerical result.
- Run and inspect the challenge's shared viewer for prediction-producing
  experiments. Read the relevant `references/<task>-analysis.md` for the
  task-specific evidence.
- Compare only against results with compatible splits and evaluators.
- If a known-good path collapses after an implementation change, freeze
  dependent work, compare the known-good and broken paths, run the smallest
  discriminating test, and repair it before judging the method.

## 2. Decide Whether Diagnosis Is Needed

If the result is expected and answers a routine ablation, record the result and
decision briefly. No component audit is needed.

If an important full run is below target, surprising, or visually suspicious,
work through every applicable component in the task-specific result-analysis
skill and any further explanation suggested by the evidence. Record the grouped
disposition using the raw `experiment.md` template:

```text
verified: evidence shows it is working as intended
suspect: evidence suggests it may explain the result
untested: still plausible and needs a diagnostic or experiment
```

Do not call a component healthy merely because the run completed or converged.
Stable training does not mean the recipe is optimized. Use saved outputs,
curves, viewers, cheap oracles, and controlled diagnostics before another full
training run when possible.

If the diagnosis points to a method-class mismatch, such as instance separation,
topology, rare-class recall, poor domain transfer, or a missing foundation
model, revisit `candidates.md` and the affected `literature.md` evidence before
spending more time on local hyperparameter tweaks. A valid low-ceiling fallback
should be frozen only when stronger alternatives are blocked, disproven, or
outside the remaining budget and finalization reserve.

## 3. Act

Append a concise `Outcome` to `experiment.md` containing the result, diagnosis,
evidence links, interpretation, decision, and one concise downstream update
covering Internal Exploration, `state.md`, and `trajectory.md` when applicable.
Then do exactly one of these:

1. add a small set of controlled, independent follow-up experiments to
   `dashboard.md` Planned Jobs and create their `experiment.md` Intent sections;
2. continue a planned comparison that this result enables;
3. record that the solution direction has reached an evidence-backed plateau.

A below-target baseline cannot stop after only one default full run and one
failed modification while a plausible large-gain change remains. Hyperparameter
experiments must test a stated hypothesis, not form a random sweep.

Update the active baseline's Internal Exploration whenever evidence reveals or
changes a material improvement possibility. Do not backfill generic rows. Update
`state.md` only when its current focus, exploration pointers, active constraint,
or continue/switch/freeze judgment changes. Put consequential decisions and
their decisive evidence in `trajectory.md`; keep live jobs and planned work in
`dashboard.md`.

When several fair independent tests answer the current focus, launch them in
parallel if resources permit. Use a representative screen to promote promising
variants, then deepen and confirm them; do not reject a method family from one
undertrained fold or seed.

**Research escape:** when repeated valid evidence no longer changes the same
bottleneck, or evidence contradicts the method's central assumption, use the
agent's judgment to broaden the internal loop with a discriminating diagnostic,
structural change, or targeted literature/code search. That search may replace
or recombine components within the same baseline; start another baseline only
when the resulting formulation is independent. When the direction is no longer
worth pursuing, record its best evidence, uncertainty, and switch reason in
`trajectory.md`, then establish the next focus in `state.md`.
