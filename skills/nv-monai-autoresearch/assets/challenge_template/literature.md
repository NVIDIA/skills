<!-- miagent-template-source: assets/challenge_template/literature.md -->
<!-- miagent-template-contract: active; keep the miagent-template-source loaded for creation/material updates; reload after resume or compaction -->
<!-- miagent-template-schema: literature-evidence-v7 -->
# Literature Notes

Decision record for scientific thinking and research. It is not a search
protocol, bibliography, or menu of allowed solutions.

## Independent Scientific Thesis

The main agent completes this from the task, data, metric, and its own model
knowledge before inspecting specialized method skills, implementations, package
availability, literature, prior solutions, or candidates. Choose the
decomposition and solution freely. Preserve this answer after search; record
corrections in the later sections instead of rewriting the starting view.

| Field | Pre-Search Answer |
|---|---|
| agent-chosen decomposition | <Scientific components, failure mechanisms, interactions, and metric implications chosen by the agent> |
| strongest complete pipeline before search | <The end-to-end method the agent would implement now, including the important training and inference recipe> |
| causal rationale | <Why this pipeline should perform strongly on this task> |
| serious alternatives | <Other genuinely competitive directions, not token controls> |
| memory / uncertainty boundary | <Which named methods, weights, task facts, or performance claims are recalled but not yet verified> |

## Open Research Findings

Search freely to challenge, verify, and extend the thesis. Follow promising
leads rather than prescribed lanes. A fresh research subagent may fill this
section, but it must not alter the Independent Scientific Thesis or choose the
final direction.

Before calling a method strongest, latest, or state of the art, challenge that
claim through the current date using the exact task/dataset and the immediate
predecessor-to-successor citation chain. Record only findings that affect the
decision, not the search transcript.

Keep only decision-changing findings. Use current primary papers and official
challenge, repository, checkpoint, model-card, or dataset sources. Name exact
weights, training data, license/access, and usable interface when they affect
the decision. Use stable `L001`-style IDs. Mark each effect `verify:`, `refine:`,
`replace:`, `reject:`, or `uncertain:`.

| Evidence ID | Finding / Method | Decision-Relevant Evidence | Primary Source | Implementation / Weights / Data | Effect On Thesis |
|---|---|---|---|---|---|
| `L001` |  |  |  |  | verify / refine / replace / reject / uncertain: |

## Revised Scientific Direction

The main agent compares the independent thesis with the research findings and
makes the scientific judgment. Literature is evidence, not authority. The
chosen method may preserve, combine, revise, reject, or exceed searched methods.

| Field | Post-Search Judgment |
|---|---|
| chosen complete pipeline after search | <Coherent end-to-end scientific direction with component interfaces, training, validation, inference, and ensemble/postprocessing choices that affect predictions> |
| what changed from the thesis | <What was verified, refined, replaced, or rejected, with decisive Evidence IDs> |
| why this is now strongest | <Causal comparison with the strongest exact/close-task evidence and strongest serious alternative> |
| strongest serious alternative | <The direction most likely to displace the chosen pipeline> |
| remaining scientific uncertainty | <Evidence that could still change the choice> |

## Post-Search Coverage Audit

Complete this only after the revised direction is written. It is a final
omission check, not the search plan. Each row needs concrete inspected evidence
and a decision; a checkmark is not evidence. If it exposes a meaningful gap,
search that gap and then reconsider the revised direction.

| Coverage Area | Evidence IDs / Concrete Findings | Decision / Remaining Gap |
|---|---|---|
| strongest exact/close-task methods, organizer baselines, and prior winners |  |  |
| recent task-specific and general CV/ML frontier through the current date, including predecessor, related-work, forward-citation, and official-release successors |  |  |
| current general and domain/task-family foundation or pretrained models |  |  |
| usable official implementations, checkpoints, and applicable MIAgent skills |  |  |
| external/pretraining data, training-distribution overlap, license, and leakage risk |  |  |

## External Critic Summary

Follow `assets/challenge_template/critics/README.md`. The critic challenges
the completed thesis/research synthesis; it does not supply or select the
direction. Integrate accepted objections before finalizing `candidates.md`.
A provider failure is not a completed critic pass; use a fresh critic subagent,
or write a dedicated self-critique report only when delegation is unavailable.

| Field | Value |
|---|---|
| search pass / trigger | initial discovery / external loop from `<current focus or bottleneck>` |
| critic report |  |
| main objections |  |
| accepted changes |  |
| rejected or deferred objections |  |

## Human Steering

Record only human guidance that changes scientific-search scope or direction
taste.

| Date | Human Decision | Reason | Applied Action |
|---|---|---|---|
| YYYY-MM-DD | Example: preserve the agent's independent direction before searching exact-task methods | Avoid letting literature anchors erase useful model intuition | Record the thesis first, then use research to challenge it |
