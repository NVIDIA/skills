---
name: hsb-ip-def
author: "Holoscan Team <holoscan-team@nvidia.com>"
description: Generate, validate, compare, or explain HSB HOLOLINK_def.svh macros. Do not use for FPGA_top.sv wrappers or packetizer-only derivation. Generation runs bundled Python scripts locally through shell commands and writes validated .svh output files after user-confirmed paths.
version: "0.1.0"
tags:
  - holoscan
  - hsb
  - fpga
  - systemverilog
  - configuration
permissions: [file_read, file_write, shell]
license: Apache-2.0
compatibility: Targets HSB IP rev 16'h2604; backward-compatible with 16'h2603 (the public release rev). Prefer live HSB IP source when available; warn on unknown revisions. Requires Python 3.9+ with PyYAML for the generator.
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  team: holoscan
  domain: fpga
  vendor: nvidia
  tags:
    - holoscan
    - hsb
    - fpga
    - systemverilog
    - configuration
  languages:
    - systemverilog
    - python
  artifact: HOLOLINK_def.svh
  hsb_ip_version: "16'h2604"
  min_compat_rev: "16'h2603"
---

# HSB IP Def Skill

## Purpose

Use this skill through four workflows:

- **Generate** a `HOLOLINK_def.svh` from confirmed board requirements.
- **Validate** an existing def file with the bundled validator.
- **Explain / Reason** about HSB IP macros, legality, and macro-driven ports.
- **Compare** two def files semantically.

## Scope

This skill owns the contents of `HOLOLINK_def.svh`: `\`define` directives, `localparam` arrays, the `HOLOLINK_pkg` wrapper, and boot-time `init_reg[]` sequence. The surrounding top-level wrapper is owned by `hsb-ip-create-top`; packetizer-only profile derivation can be delegated to `hsb-ip-packetizer`.

The file must use the standard guard plus `package HOLOLINK_pkg` wrapper. Load `references/macro-reference.md` for the full wrapper requirement and per-macro semantics.

## Prerequisites

- Python 3.9+ is required for the bundled scripts; PyYAML is required when reading YAML profiles.
- The user must confirm file writes and shell commands unless they already asked for that exact operation.
- A concrete def file path, pasted content, or confirmed generation profile is required before script-backed validation or generation.
- Live HSB IP source is optional but preferred when validating against a specific checked-out IP revision.

## Instructions

- Run bundled script preflight from `references/script-usage.md` before Generate, Validate, Compare, or script-backed legality checks.
- Never silently default a macro during Generate. Show the proposed value or inferred requirement and get user confirmation.
- Ask one requirements question per turn during chat-driven Generate.
- Use sensor-agnostic language unless the user says the design is camera-specific.
- Avoid unsupported "typical", "common", "most designs", or corpus-frequency claims. Anchor choices to IP behavior, documented constraints, or user requirements.
- Always surface HD-W3xx footgun warnings, even when the user asks only about errors.
- Do not generate `FPGA_top.sv`; offer handoff to `hsb-ip-create-top` after a generated def file is validated.
- Treat unfamiliar macros as project-specific unless they are documented in this skill's references or live HSB IP source.

## Security Considerations

This skill can read and write local files and run shell commands through its bundled scripts. Before running a command or writing a file, state the command or path and get user confirmation unless the user already explicitly requested that exact operation. For pasted `HOLOLINK_def.svh` content, write only to a safely generated file in an isolated temporary directory and remove it after validation unless the user asks to keep it.

## Version, Compatibility, And Live Source

This skill targets HSB IP rev `16'h2604` and is backward-compatible with `16'h2603`. Live HSB IP source supersedes bundled references.

When source-sensitive behavior matters:

1. Locate `<hsb-ip-root>/top/HOLOLINK_top.sv`. Known roots include `hw/nvcpu_dgx_fpga/vrtl/hololink/` and public-release `fpga/nv_hsb_ip/`.
2. Read `HOLOLINK_REV` and `HOLOLINK_BACKWARD_COMPAT_REV`.
3. Trust live source for consumed macros, port gates, and RTL behavior when it differs from bundled references.
4. Pass `--ip-source <root>` to `scripts/validate_def.py` when validating against a known source root.

Public-doc baseline:

- `https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/release-2.6.0-EA/docs/user_guide/ip_integration.md`
- `https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/release-2.6.0-EA/docs/user_guide/port_description.md`

## Workflow Decision

1. **Generate** when the user asks to create, scaffold, draft, design, or produce a `HOLOLINK_def.svh`.
2. **Validate** when the user asks to lint, check, validate, or review a `HOLOLINK_def.svh`.
3. **Explain / Reason** when the user asks what a macro does, whether a combination is legal, why validation fails, or how a macro affects `HOLOLINK_top`.
4. **Compare** when the user asks to diff two def files or understand what changed between configurations.

## Generate

Load `references/generate-workflow.md` and `references/script-usage.md`.

Follow the detailed Generate workflow in the reference: run preflight, classify supplied requirements, ask one requirements question per turn, build a flat YAML profile, run `scripts/generate_def.py`, show provenance for generated fields, and offer handoff to `hsb-ip-create-top`.

When packetizer fields are needed, invoke `hsb-ip-packetizer` with the known RX count, RX widths, and the user's data-manipulation description. Consume only that skill's `packetizer_profile_overlay` YAML keys, merge them into the in-progress profile, and continue full-file generation and validation here.

## Validate

Load `references/script-usage.md`. Load `references/validation-rules.md` only when explaining specific rule IDs or validation behavior.

Steps:

1. Run bundled script preflight once per session.
2. Locate the file. If the user pastes content, tell them the generated temporary path before writing, ask for confirmation, use an isolated safe temp path, and clean it up after validation unless they ask to keep it. Otherwise use the provided path.
3. Run `<PY> scripts/validate_def.py <path> --json` and do not reimplement validation in context.
4. Group findings by severity: errors, warnings, then info. For each error, cite rule ID, line number, and macro when available.
5. Always surface footgun warnings, especially HD-W3xx silent-fallback warnings.
6. If clean, confirm the inferred archetype and IP version, then suggest the likely next check or handoff.

## Explain / Reason

Load only the reference needed for the question:

| Question type | Reference |
|---|---|
| Macro semantics or legal values | `references/macro-reference.md` |
| Validation rule behavior | `references/validation-rules.md` |
| Macro-driven port effects | `references/top-port-map.md` |
| `init_reg[]` and `N_INIT_REG` | `references/init-reg-cookbook.md` |
| Advanced macros | `references/advanced-macros.md` |
| Example legal configurations | `references/archetypes.md` |

For grounded legality questions, run preflight and prefer validating a concrete file or minimal synthetic file with `scripts/validate_def.py` over hand reasoning. Cite RTL line ranges from references when explaining why a rule exists.

## Compare

Load `references/script-usage.md`, run preflight, then use `<PY> scripts/compare_defs.py <a.svh> <b.svh> [--json|--text]`. Summarize semantic differences, not whitespace or comment-only changes.

## Limitations

- Do not generate `FPGA_top.sv`; use `hsb-ip-create-top` after the defs file validates.
- Do not derive packetizer-only field sets here when the packetizer behavior is underspecified; delegate that slice to `hsb-ip-packetizer`.
- Do not treat bundled archetypes or corpus metadata as norms. They are examples and maintenance metadata, not defaults.
- Do not silently accept unknown macros as validated HSB IP behavior unless live source or references document them.

## Troubleshooting

- Script preflight fails: report the missing Python or PyYAML requirement and stop before generation or validation.
- Validation reports errors: group by severity, cite rule ID and line, and fix the defs file before offering top-level handoff.
- Validation reports HD-W3xx warnings: surface them even when there are no errors because they describe silent RTL fallback risks.
- Unknown macro appears: treat it as project-specific unless live HSB IP source or bundled references document it.

## Available Scripts

Use `<PY>` selected during preflight from `references/script-usage.md` for every command.

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/generate_def.py` | Generate `HOLOLINK_def.svh` from an archetype and/or YAML/JSON profile; validates before writing | `--profile <path>`, `--archetype <slug>`, `-o <output>`, optional `--allow-random-uuid` compatibility flag |
| `scripts/validate_def.py` | Validate a `HOLOLINK_def.svh` and emit JSON or text findings | `<path/to/HOLOLINK_def.svh>`, optional `--json` or `--text`, optional `--ip-source <root>` |
| `scripts/compare_defs.py` | Compare two def files semantically, ignoring whitespace/comment-only changes | `<a.svh> <b.svh>`, optional `--json` or `--text` |
| `scripts/build_corpus_metadata.py` | Maintenance helper to rebuild anonymized corpus metadata; do not run during normal user workflows | `<path1> [<path2> ...]` |

## Bundled Resources

| Resource | Use |
|---|---|
| `references/generate-workflow.md` | Detailed Generate workflow, requirement order, question style, and per-topic prompt guidance |
| `references/script-usage.md` | Preflight, command forms, and `run_script()` examples for bundled scripts |
| `references/macro-reference.md` | Wrapper requirement, macro semantics, legal constraints, and RTL citations |
| `references/validation-rules.md` | Rule catalog for validator findings |
| `references/archetypes.md` | Illustrative legal configurations; never treat as templates or frequency guidance |
| `references/init-reg-cookbook.md` | Boot-time APB write sequence patterns and address conventions |
| `references/top-port-map.md` | Macro-to-`HOLOLINK_top` port effects |
| `references/advanced-macros.md` | `SYNC_CLK_HIF_APB`, `SYNC_CLK_HIF_PTP`, `PERI_RAM_DEPTH`, and `DISABLE_COE` |
| `assets/metadata/corpus.json`, `assets/metadata/corpus-stats.json` | Maintenance metadata only; do not cite corpus counts as user guidance |
| `scripts/generate_def.py` | Generate a def file from a profile |
| `scripts/validate_def.py` | Validate a def file and emit JSON/text findings |
| `scripts/compare_defs.py` | Compare two def files semantically |

## Examples

- `Use hsb-ip-def to generate a HOLOLINK_def.svh for a new HSB board.` Treat as Generate, run script preflight, classify supplied requirements, ask one requirement question per turn, and run `scripts/generate_def.py` only after the profile is confirmed.
- `Use hsb-ip-def to validate my existing HOLOLINK_def.svh and tell me whether any warnings are important.` Treat as Validate, ask for or locate the file, run `scripts/validate_def.py <path> --json`, group findings by errors, warnings, and info, and always surface HD-W3xx footgun warnings.
- `Use hsb-ip-def to explain whether HOST_WIDTH=512 and PTP_CLK_FREQ=90_000_000 is legal.` Treat as Explain / Reason, prefer a concrete validator-backed check over hand reasoning, and load only the macro or validation reference needed to explain the result.
