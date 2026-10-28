<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Spec handoff requirements

Use this checklist when filling the three JSON templates. Preserve every
template field and its object or array type, including empty containers.
Preserve the templates' `schema_version` and `schema_authority` values.
The finalizer checks structure and
consistency; Spec remains responsible for complete, meaningful requirements.
These JSON files are fill-in templates, not JSON Schema documents. The Python
finalizer enforces the rules below and rejects changed inputs during `--check`.
It does not prove the algorithm, judge the authority of a source, or validate
every free-form behavior field or the text of the Doctor report.

## Operator specification

- Set `status` to `CONFIGURED` before finalization.
- Set a nonempty `operator.name` and `task_mode` to `new`, `change`, or `repair`.
- Keep `operator.public_api`, `platform`, `resources`, `behavior`, and
  `performance_contract` as objects. Fill their template fields from the
  approved contract. Do not invent unknown behavior.
- Set `platform.emulator.policy` to `optional` or `required`.
- Keep `requirements`, `parameters`, `design_constraints`,
  `correctness_contracts`, and `unresolved_requirements` as arrays.
- Every requirement is an object with a unique, nonempty string `id`.
- Every unresolved requirement is an object with a boolean `blocking` field.
  No entry may have `blocking: true` at finalization. Return `NEEDS_INPUT`
  while blocking requirements remain.

## Test matrix

- Set `operator` to the exact spec operator name and `specification` to the
  literal `operator-spec.json` (the associated file, not a path from the CWD).
- Set `execution_policy.hardware` to `required` and
  `execution_policy.emulator` to the spec's EMU policy. When EMU runs, use
  the same correctness cases, inputs, references, and tolerances as hardware.
- Keep `correctness_cases` and `performance_cases` as arrays. Each case is an
  object with a nonempty string `id`, unique across both arrays, and a nonempty
  `requirement_ids` array containing only IDs from the spec.
- Include at least one correctness case with an independent expected result.
  The finalizer rejects an empty correctness array, even when performance cases
  cover every requirement. Performance cases remain optional.
- Set `coverage.requirement_ids` to all spec requirement IDs. Every requirement
  must also be linked from at least one actual test case; listing it only in
  coverage is insufficient.
- Set `coverage.uncovered_requirement_ids` and `coverage.unresolved_cases` to
  empty arrays only after their issues are resolved. The finalizer rejects
  nonempty arrays. Return `NEEDS_INPUT` for unresolved test behavior.

## Evidence

This is the provenance record for the operator contract, not a record of later
hardware test results. It lets OpGen trace requirements to their sources and
detect changes to recorded project files after Spec.

| Field | Purpose |
| --- | --- |
| `public_sources` | Public documentation or requirements used to define behavior, with citations. |
| `user_sources` | User-supplied requirements and decisions. |
| `additional_evidence_used` | Project files used to define the contract, with their requirement links. |
| `assumptions` | Assumptions and the user's disposition of them. |
| `conflicts` | Conflicting sources and how the stated precedence resolves them. |

- Preserve the five arrays: `additional_evidence_used`,
  `public_sources`, `user_sources`, `assumptions`, and `conflicts`.
- Record project files in `additional_evidence_used`.
- Each project-file record needs nonempty `category`, `relative_path`, and
  `used_for` strings. `relative_path` starts at the active `PROJECT_ROOT`;
  the file must exist inside it. Absolute paths, parent traversal, and escaping
  symlinks are rejected. Use a `requirement_ids` array containing only spec IDs
  (empty when the file supports the workflow rather than a requirement).
- Do not repeat a path in `additional_evidence_used`.

## Finalizer paths

Pass existing absolute paths for the three JSON files, Doctor report, and active
workspace. Create the output's parent directory before finalizing. The finalizer
generates hashes, sizes, counts, and `opgen-inputs.json`; the model does not write
them. Run `--check` afterward and leave the recorded inputs unchanged.

The output must differ from all three JSON inputs, the Doctor report, and every
recorded project evidence file. The finalizer rejects collisions, including
symlink and hard-link aliases, before writing the handoff.
