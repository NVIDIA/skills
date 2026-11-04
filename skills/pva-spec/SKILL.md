---
name: pva-spec
description: >-
  Create and validate an NVIDIA PVA operator specification, test matrix,
  evidence manifest, and deterministic OpGen handoff. Use after Doctor for
  new operators, changes, or repairs, using requirements, a partial spec,
  reference code, or an existing operator. Accept text, files, and public
  URLs. Record unresolved requirements without inventing behavior.
metadata:
  author: "Venkat Balan <venkatramanb@nvidia.com>"
  tags: [pva, specification, test-design]
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Specify a PVA operator

Define behavior before implementation. Spec owns the behavior contract and
test matrix. It does not implement the operator or invoke OpGen.

## Inputs

Accept one or more of these sources:

- A plain-language description or local requirements file.
- A public requirements URL.
- A partial or complete specification.
- A local or public reference implementation.
- An existing operator name or path.

Resolve:

- Task mode: `new`, `change`, or `repair`.
- PVA generation: `GEN2`, `GEN3`, or both.
- PVA SDK version and PVA Solutions version, when used.
- Host API: C, C++, or both.
- EMU policy: `optional` by default or `required` by user request.
- Work directory (`WORK_DIR`) and project root (`PROJECT_ROOT`).
- A Doctor report that is ready for `spec`, or accepted limitations.

Default `WORK_DIR` to `./pva-operator/<operator-name>/`. Resolve it and
user-supplied relative paths against the invocation directory before changing
directories. `PROJECT_ROOT` is the absolute supplied project directory. Work
directly in that project. Do not overwrite an existing configuration.

`SKILL_ROOT` is the absolute directory containing this skill's `SKILL.md`,
after resolving symlinks. Bundled files are relative to that directory. All
command paths below use these absolute roots.

## Verify references

Require Doctor's `public-reference-corpus` check to be `PASS`. Set
`SDK_REFERENCES` to the report's absolute `sdk_reference_root`. Require a readable
`index.md` and `reference-lock.json`, a lock SHA-256 matching the report's
`sdk_reference_lock_sha256`, and the reported SDK version matching this task.
If these fields are absent or no longer match, return `BLOCKED` and request a
fresh Doctor report. Do not fall back to an installed package or CWD path. Read the
`pva-spec` route in `$SDK_REFERENCES/index.md`.
Before reading Markdown, check its SHA-256 against the lock
`outputs` entry; return `BLOCKED` on a mismatch. Resolve links relative to each
generated file.

Use `rg` to select only relevant API, requirement, architecture, and
data-movement Markdown. Do not fetch the same NVIDIA product documentation.
For each product fact, record the SDK documentation root, local Markdown path,
packaged source path, source SHA-256, version, and section or PDF page.

Do not inspect the VPU ISA instruction-by-instruction reference during Spec
(the Programmer's Guide chapter titled "Instruction Set Reference"). The
architecture overview remains available for defining platform constraints.

## Collect requirements

Inspect supplied implementations within the user's authorized scope and their
license terms.

Use this precedence:

1. Explicit user requirements.
2. An approved supplied specification.
3. The public API and independent CPU reference.
4. Correctness tests, workloads, fixtures, and expected results.
5. Public operator API documentation.
6. Public algorithm and format documentation.
7. Public SDK constraints.
8. An assumption that the user must approve.

For a URL, record its title, URL, access date, license when available, and
SHA-256 for downloaded source. Follow only links that resolve the contract.

For a change or repair, reconcile the existing API, CPU reference, tests, workloads,
KPI cases, benchmark cases, and build wiring. Record each conflict. Do not
silently select one conflicting source.

Do not copy source into the output. Do not use an example as proof for an
unspecified case.

## Write the three model outputs

Copy these templates into `$WORK_DIR/configure/` and fill them according to
[the handoff requirements](references/handoff-requirements.md):

- [operator-spec-template.json](assets/operator-spec-template.json) as
  `operator-spec.json`.
- [test-matrix-template.json](assets/test-matrix-template.json) as
  `test-matrix.json`.
- [evidence-template.json](assets/evidence-template.json) as `evidence.json`.

The templates own the field names. Write strict JSON. Do not write comments,
duplicate keys, trailing commas, `NaN`, or infinity tokens. Use `null` for an
unknown optional value.

Before finalizing, set the spec's `status` to `CONFIGURED` only when no blocking
requirement remains. Set the matrix's `operator` to the spec's `operator.name`.
Both `coverage.uncovered_requirement_ids` and `coverage.unresolved_cases` must
be empty; resolve outstanding cases before creating the handoff.

For each requirement:

1. Assign a stable ID.
2. State the source and authority: `public_nvidia`, `user_input`,
   `skill_workflow`, or `assumption`.
3. Define the observable API, resource, parameter, arithmetic, border,
   aliasing, execution, error, safety, and interoperability behavior that
   applies.
4. Add an unresolved item when a missing value can change behavior.
5. Mark the item as blocking only when implementation cannot proceed safely.

Record design constraints, not implementation choices. Mark a possible
dataflow, tile, VMEM layout, vector-tail method, address generator, lookup
method, or L2 use as `candidate` unless the contract requires it.

## Define the test matrix

Require at least one correctness case with an independent expected result.
Each requirement must have a test. Link every test to one or more requirement
IDs. Include the cases that apply:

- Each supported type, format, plane, and parameter family.
- Minimum, maximum, odd, partial-tile, and vector-tail dimensions.
- Tight and padded pitch.
- Each border mode and corner.
- Rounding, saturation, overflow, NaN, and infinity rules.
- Structured and seeded-random input.
- Valid and invalid aliasing.
- Repeated submission and asynchronous execution.
- Invalid resources, formats, dimensions, pitches, and parameters.
- Required hardware correctness.

Set `execution_policy.hardware` to `required`. Set
`execution_policy.emulator` to the selected EMU policy. Use the same approved
correctness cases, inputs, references, and tolerances on EMU and hardware when
EMU runs. EMU execution is an early aid. It is not hardware verification.

Add performance cases only when the user or source contract requires them.
For each performance case, define the workload, metric, warm-up count, at least
1000 hardware samples, target conditions, and acceptance rule when one exists.
Do not turn a public benchmark into a universal threshold. Do not derive a
Speed-of-Light target from the implementation under test.

A missing performance target does not block correctness configuration. Use
`not_set` for performance acceptance.

## Record evidence

`evidence.json` records where requirements came from and which files support
them. See the field purposes in
[the handoff requirements](references/handoff-requirements.md#evidence).

In `evidence.json`, record each used source once. For a project file, record
only its category, path relative to `PROJECT_ROOT`, requirement IDs, and use.
Do not calculate a hash, size, or absolute path. The finalizer calculates those
values.

Record assumptions and conflicts in their named lists.

## Create and check the handoff

| Script | Command | Inputs | Outputs | Exit codes |
| --- | --- | --- | --- | --- |
| [finalize_opgen_inputs.py](scripts/finalize_opgen_inputs.py) | `python3 "$SKILL_ROOT/scripts/finalize_opgen_inputs.py" --operator-spec "$WORK_DIR/configure/operator-spec.json" --test-matrix "$WORK_DIR/configure/test-matrix.json" --evidence "$WORK_DIR/configure/evidence.json" --doctor-report "$DOCTOR_REPORT" --workspace "$PROJECT_ROOT" --output "$WORK_DIR/configure/opgen-inputs.json"` | Spec, test matrix, evidence, Doctor report, and project root. | `opgen-inputs.json` and a JSON validation summary. | `0`: valid; `1`: failed; `2`: invalid command arguments. |
| [finalize_opgen_inputs.py](scripts/finalize_opgen_inputs.py) | `python3 "$SKILL_ROOT/scripts/finalize_opgen_inputs.py" --check "$WORK_DIR/configure/opgen-inputs.json"` | Finalized handoff and its recorded inputs. | JSON validation summary; read-only check. | `0`: valid; `1`: failed; `2`: invalid command arguments. |

After the three JSON files are final, resolve `DOCTOR_REPORT` to the absolute
path of the supplied Doctor report and run:

```bash
python3 "$SKILL_ROOT/scripts/finalize_opgen_inputs.py" \
  --operator-spec "$WORK_DIR/configure/operator-spec.json" \
  --test-matrix "$WORK_DIR/configure/test-matrix.json" \
  --evidence "$WORK_DIR/configure/evidence.json" \
  --doctor-report "$DOCTOR_REPORT" \
  --workspace "$PROJECT_ROOT" \
  --output "$WORK_DIR/configure/opgen-inputs.json"
```

Do not create `opgen-inputs.json` with the model.

If this command reports a missing test for a requirement, add the test or
link an existing test that checks the requirement. Run the command again.

Then run:

```bash
python3 "$SKILL_ROOT/scripts/finalize_opgen_inputs.py" \
  --check "$WORK_DIR/configure/opgen-inputs.json"
```

Do not modify a recorded input after finalization. Return `BLOCKED` if either
command still fails. Give the files to OpGen only after both commands pass.

## Output Format

Write `$WORK_DIR/reports/configure-report.md` with
[the report template](assets/configure-report-template.md).

Return one result:

- `CONFIGURED`: No blocking question remains, and both finalizer commands pass.
- `NEEDS_INPUT`: Missing behavior can change the API or required result.
- `INVALID_INPUT`: Sources conflict, and no precedence rule resolves them.
- `BLOCKED`: A required path, tool, or permission is unavailable.

Return:

```text
Result: CONFIGURED | NEEDS_INPUT | INVALID_INPUT | BLOCKED
Operator: <name>
Task mode: <new | change | repair>
EMU policy: <optional | required>
Requirements: <count>
Correctness cases: <count>
Performance cases: <count>
Blocking questions: <count>
Specification: <absolute path>
Test matrix: <absolute path>
Evidence: <absolute path>
OpGen inputs: <absolute path>
Active workspace: <absolute path>
```

Stop after these outputs. Do not invoke OpGen.
