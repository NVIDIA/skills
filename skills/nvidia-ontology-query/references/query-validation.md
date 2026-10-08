<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Grounded query and answer validation

Use this workflow for descriptive and diagnostic questions after the ordinary
MCP or REST readiness and answerability sequence in the parent skill. It checks
the generated SQL, returned rows, and answer separately; successful execution
alone does not prove that the intended question was answered.

## Resolve the material question

Before `ask_question`, restate the required:

- population and exclusions;
- measure, unit, and result grain;
- grouping, ordering, and distinctness;
- time range, as-of time, and comparison basis;
- run/scenario identity, parent lineage, status, validity, and supersession;
- required source or artifact role.

Clarify material ambiguity instead of selecting a convenient default. Resolve
the approved Term, SQL attribute, source expression, denominator, filters, and
join path. Do not substitute a similarly named measure or source field.

## Inspect the SQL

Check that the generated SQL preserves every material constraint. In
particular:

- required tables, columns, joins, literals, and time predicates are present;
- joins use approved keys and direction;
- detail rows are aggregated at detail grain before joining to run summaries;
- an already run-level metric is not summed after a one-to-many detail join;
- quantity is not replaced by a ratio, rate, average, or horizon total;
- requested run, scenario, parent, status, validity, and supersession filters
  are explicit;
- the selected hash or identifier has the requested source role.

If a material constraint is missing or substituted, do not present the result
as an answer. Preserve the SQL and report the mismatch or request clarification.
When a one-to-many fanout is detected, obtain and execute a corrected run-grain
query before presenting a validated result. A supplied run-level value may be
identified as source evidence while diagnosing the multiplication, but it must
not be relabeled as a validated query result.

## Check rows independently

Validate row keys, population, cardinality, values, units, time scope,
denominator, provenance, and literal states. Recompute simple totals and ratios
when the returned rows permit it. Empty or null is not automatically zero.

A truncated result cannot prove a complete list, ranking, or total. Narrow the
claim or request an aggregate over the complete population. Keep exact values
separate from presentation rounding.

## Check the answer

Lead with findings from the returned rows, not a description of what the SQL
would do. Require the prose to match the validated values, units, scope, and
status. Preserve identifiers and literal recommendation state when requested.

Observed, computed, modeled, predicted, recommended, and committed are distinct
evidence states. A retrieved recommendation does not become an approved or
executed action. Diagnostic association is not causality; counterfactual claims
need appropriate evidence.

## Follow-ups and failure behavior

Wait for the prior SSE stream's `[DONE]` before a conversational follow-up and
preserve its load-bearing constraints. Use a fresh conversation for independent
evaluation questions. Do not use expected or evaluator-gold answers as model
context.

Retain a concise receipt containing question identity, ontology/source identity,
SQL, rows, row count, truncation, answer, attempts, and observed gaps. Never
record credentials. On missing evidence, wrong population, invalid state, or
unverifiable truncation, fail closed instead of silently trimming or choosing
the latest available row.
