<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Source-grounded ontology modeling

Use this workflow when a request changes business meaning rather than only
correcting copy. Routine API mechanics remain in [write-api.md](write-api.md).
Publication, promotion, and rollback belong to
[publication.md](publication.md).

## Start with the question contract

Before proposing an edit, record representative business questions and the
required population, measure, grouping, time range, comparison, and exclusions.
Use the exported physical catalog and approved source evidence for schemas,
types, keys, nullability, time columns, and relationships. Names and small value
samples are supporting evidence, not proof of identity or cardinality.

Define each concept by business identity and population. Similar names or equal
current row sets do not make two concepts equivalent. Every attribute needs an
exact physical source or an explicit derivation. Preserve physical and semantic
IDs when editing an existing model; generated UUIDs are not natural-name
upserts.

## Relationship contract

For every relationship record:

- source and target concepts and columns;
- direction and business role;
- expected one-to-one, one-to-many, or many-to-many multiplicity;
- key/null behavior and observed join coverage;
- approved query direction and known fanout risk;
- validity, version, and source evidence.

A matching column name is not proof of a join. Test duplicate and null keys,
reverse joins, missing mappings, and cardinality violations. A semantic
relationship is not automatically a prediction edge, graph edge, or solver
coefficient.

## Measure contract

For every reusable measure record:

- expression and source evidence;
- row grain and permitted query/grouping grain;
- physical unit, separately from time grain;
- additive dimensions and forbidden reaggregation;
- numerator, denominator, weights, and population filters;
- null, empty-population, and zero-denominator behavior;
- effective time, availability time, revision, validity, and supersession.

A quantity, fraction, period average, rate, and horizon total are different
measures. Ratio-of-totals is not generally mean-of-ratios. A run-level total
must not be summed after a one-to-many detail join.

Policies need authority, applicability, constrained fields, permitted changes,
and required identities. Descriptive text is not executable procedure or
permission to act.

## Map the design to supported Auto Ontology operations

Use the smallest supported change:

- patch an existing Term or ColumnAttribute for metadata corrections;
- validate and create/update a SQL attribute for a reusable derivation;
- use scoped native model export/import for concept population or relationship
  changes.

Auto Ontology has no direct public create-empty-Term endpoint and no complete direct
structural relationship mutation workflow. Do not invent one. If the requested
design cannot be expressed through the installed revision's public API, return
a model gap instead of weakening the design.

## Validate before handoff

Test source-backed examples and counterexamples for duplicate/null keys, fanout,
reverse joins, missing mappings, stale versions, ambiguous periods, unit
confusion, and intended empty populations. Verify that each scoped question is
answerable without changing its material meaning.

Deliver:

- a versioned semantic specification;
- exact source and ID mappings;
- relationship and measure contracts;
- approved join paths;
- positive and negative tests;
- unresolved model or product gaps.

A successful SQL execution, import response, or non-empty answer does not
certify semantic correctness. Hand the reviewed specification to the publication
workflow for backup, staged application, exact readback, and rollback handling.
