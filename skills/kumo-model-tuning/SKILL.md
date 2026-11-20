---
name: kumo-model-tuning
description: Configure a Kumo model plan for a user-defined task, including encoders, run mode, bounded experiments, sampling, optimization, architecture, metrics and refits. Preserve target and evaluation constraints and use existing diagnostics for comparisons.
---

# Configure a controlled experiment

Start from the existing task, graph, snapshot, populations and evaluation protocol. Reuse
known constraints and ask only for unresolved objectives or workload budget. Suggesting a plan
is not permission to execute its experiments. Keep a request for inspection separate from a
request for another fit.

Read [plan configuration](references/plans.md). To change an existing plan, start from it —
the plan the user supplies, or `trainer.model_plan` of the fitted or loaded trainer — since a fresh
suggestion resets every setting the user did not mention; if it is not available, ask for it.
For a new plan, begin with `pq.suggest_model_plan`. Apply only the requested changes and present
the resolved plan and workload. Preserve target, class
selection, threshold, ranking K, forecast horizon, population, splits and tuning metric unless
the user requests a change. A smoke test is an explicitly limited functional check, not a
production quality choice. Do not silently reduce epochs or turn off refits.

Use [training diagnostics](../kumo-training-diagnostics/SKILL.md) for metrics, selected trials
and curves. Compare validation evidence under the same protocol; hold test results for final
evaluation. Explain hypotheses and tradeoffs without promising improvement. This skill does
not supply custom executable metrics, autonomous target selection or deployment.
