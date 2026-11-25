<!-- miagent-template-source: assets/challenge_template/state.md -->
<!-- miagent-template-contract: active; keep the miagent-template-source loaded for creation/material updates; reload after resume or compaction -->
<!-- miagent-template-schema: state-stop-check-v8 -->
# Challenge State

Small authoritative attention reset. It prevents drift after long experiment and
monitoring histories; it is not a result log, research queue, or task summary.

Update rule:

- Keep one current focus. Change its ID only when the stage objective,
  bottleneck, or decision changes.
- Update only when the current scientific situation or judgment changes. Never
  add an experiment, config, score, job, or planned action merely because it
  happened or may happen.
- Keep within-direction possibilities in the active baseline's Internal
  Exploration and independent directions in `candidates.md`; point to them here.
- Reevaluate the judgment after each valid result, focus resolution, plateau,
  empty queue, authority/resource change, and before freeze. Rewrite only if its
  answer changed.
- Preserve decisions and their decisive evidence in `trajectory.md`, not here.
  Retain human steering here only while it constrains the current work.

## Current Focus

An experiment must name this focus; change it before unrelated work becomes the
real priority.

| Field | Current Answer |
|---|---|
| focus ID / last changed | `F0001` / YYYY-MM-DD HH:MM TZ |
| stage / active direction | discovery / implementation / internal / external / monitoring / freeze; `baselineXXX_<direction>` when applicable |
| objective now | `<concrete outcome for this stage>` |
| bottleneck / current diagnosis | `<specific issue preventing the objective>` |
| internal exploration pointer | `<E001 and active baseline Internal Exploration link; or not yet created: first decision-bearing result will seed it; or not applicable: reason>` |
| external exploration pointer | `<strongest relevant alternative and candidates.md link; or not applicable: reason>` |
| active human constraint | `<current instruction and owning link; or none>` |
| focus decision / authority | `TXXXX` and owning human/agent decision link |

## Continue / Switch / Freeze Judgment

- This is a scientific judgment, not a checklist-completion gate. The agent
  decides which possibilities matter and records the causal reason.
- Before declaring no likely large internal gain, deliberately reconsider the
  applicable opportunities: data/domain coverage and external data;
  input/preprocessing/context/resolution/patch; sampling/augmentation;
  architecture/initialization/pretrained or foundation models; loss/objective;
  optimizer/LR/schedule/training length/effective batch; inference/
  postprocessing; folds/seeds/ensembles/robustness; diagnosed failure modes; and
  GPU efficiency. Add a promising task-specific question to Internal
  Exploration; do not log an answer for every reminder.
- Before declaring no likely large external gain, reconsider the strongest
  alternative compositions, whether strong current methods, implementations,
  and weights were adequately researched, new evidence, and whether the current
  failure is a method-class mismatch.
- Compute is an execution constraint, not a scientific rank. It is a ceiling,
  not a requirement to spend everything; however, useful remaining compute plus
  a plausible large-gain test means continue.
- Smoke/default completion, an ordinary technical failure, a weak plateau, an
  empty immediate queue, or final-evaluation timing cannot justify freeze.

| Field | Current Answer |
|---|---|
| internal improvement potential | `continue: <largest plausible gain and baseline Internal Exploration link>` / `no likely large gain: <concise current reason>` / `not applicable: explicit terminal target or human stop link` |
| external improvement potential | `continue: <strongest plausible alternative and candidates.md link>` / `no likely large gain: <concise current reason>` / `not applicable: explicit terminal target or human stop link` |
| resource effect | `<dashboard.md link>; not decision-limiting / limits execution to <fair test> / verified blocker: <evidence>` |
| agent judgment | `continue: internal` / `continue: external` / `continue: monitoring` / `blocked: <verified blocker>` / `ready to freeze: [TXXXX](trajectory.md)` / `paused: human [TXXXX](trajectory.md)` / `stopped: human [TXXXX](trajectory.md)` / `completed: final evaluation [TXXXX](trajectory.md)` |
