---
name: hsb-ip-create-top
author: "Holoscan Team <holoscan-team@nvidia.com>"
description: Create or explain fixed-format HSB FPGA_top.sv wrappers from validated HOLOLINK_def.svh files. Do not use for def generation or validation.
version: "0.1.0"
tags:
  - holoscan
  - hsb
  - fpga
  - systemverilog
  - top-level
license: Apache-2.0
compatibility: Targets HSB IP rev 16'h2604; backward-compatible with 16'h2603. Prefer live HSB IP source when available; warn on unknown revisions. Designed to work standalone or as a companion to hsb-ip-def.
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  team: holoscan
  domain: fpga
  vendor: nvidia
  tags:
    - holoscan
    - hsb
    - fpga
    - top-level
    - systemverilog
  languages:
    - systemverilog
  artifact: FPGA_top.sv
  source_artifact: HOLOLINK_def.svh
  hsb_ip_version: "16'h2604"
---

# HSB IP Create Top Skill

## Purpose

Create a fixed-format SystemVerilog top-level scaffold that instantiates `HOLOLINK_top` and declares the HSB-facing signals required by a validated `HOLOLINK_def.svh`.

This skill owns only the top-level scaffold around the HSB IP:

- SPDX/Apache license header
- `HOLOLINK_def.svh` include
- `FPGA_top` module shell
- HSB-facing signal declarations
- `HOLOLINK_top` instantiation
- comments/TODOs for top-level integration

Use `hsb-ip-def` for `HOLOLINK_def.svh` generation and validation. Do not create or validate the defs file here.

## Prerequisites

- A validated `HOLOLINK_def.svh` path or pasted file content is required before generating `FPGA_top.sv`.
- Live `HOLOLINK_top.sv` source is preferred when source-sensitive port extraction is needed; bundled references cover known rev `16'h2604` and compatible rev `16'h2603`.
- The user must provide or accept the output path when writing a scaffold file.

## Instructions

- Require a validated `HOLOLINK_def.svh` before creating `FPGA_top.sv`. If the user does not have one, invoke or direct them to `hsb-ip-def` first.
- Treat live `HOLOLINK_top.sv` as the source of truth for ports, directions, macro gates, and IP revision whenever it is available.
- First classify the request as Standalone Create Top, Def-Skill Handoff, or Explain. Load only the references required for that workflow and do not re-read the same reference just to reconfirm formatting.
- Use `references/fixed-format.md` as the baked formatting and naming style. Do not inspect example top-level files just to confirm formatting during normal generation.
- Emit a fixed-format scaffold, not a complete project top. Because the surrounding design is unknown, the generated `FPGA_top` module normally has no top-level ports.
- Derive active signal groups from the validated defs file. Do not connect ports that are absent from `HOLOLINK_top` under the active macro set.
- Preserve `HOLOLINK_top` port shapes exactly: keep packed vector widths before the signal name and unpacked interface-array dimensions after the signal name.
- When using bundled known-rev references, still preserve the signal shapes in `references/fixed-format.md`, including unpacked arrays for multi-interface `*_tdata`, `*_tkeep`, and `*_tuser` buses and the APB shape documented there.
- Use sensor-agnostic language. Say "sensor data", "sensor interface", and "stream"; use camera-specific language only after the user says the design is camera-specific.
- Ask one plain-language question per turn when information is missing. Keep source-selection questions as plain-language prompts rather than radio buttons, structured choices, or multi-question prompts.
- Do not invent project-specific integration details. Leave concise TODO comments where the surrounding design must connect to HSB.
- For packetizer comments in `FPGA_top.sv`, use generic wording such as "data manipulation behavior".
- In final summaries, include the defs validation/source status, the port-map source used, and the emitted HSB signal groups. Do not enumerate what is outside the scaffold's scope. Use only this generic note: "This file is a reference/template for hooking up top-level signals and companion IP to the HSB IP."

## Live HSB IP Source Policy

Prefer live HSB IP source over bundled references whenever the user's workspace provides it. The bundled references describe known rev `16'h2604` and backward-compatible rev `16'h2603`; live source is the authority for the checked-out IP.

When source is available:

1. Locate `<hsb-ip-root>/top/HOLOLINK_top.sv`. Known roots include `hw/nvcpu_dgx_fpga/vrtl/hololink/` and public-release `fpga/nv_hsb_ip/`.
2. Read `HOLOLINK_REV` and `HOLOLINK_BACKWARD_COMPAT_REV` from `HOLOLINK_top.sv`.
3. Read the `HOLOLINK_top` module declaration to confirm port names, directions, widths, ordering, and `ifdef`/`ifndef` gates.
4. If the rev is newer, unknown, or the live source disagrees with this skill's references, state the mismatch. Trust the live source for IO extraction and warn that bundled references may need an update.

When no source is available, use the bundled references as known-revision guidance and say the scaffold is based on the skill's supported HSB IP rev. If the user already authorized using the bundled known-rev port reference when no checkout is obvious, do a brief search of obvious roots and then proceed with bundled references instead of asking the live-source question.

If the source root is not obvious, ask conversationally in one turn:

```text
Do you have a current HSB IP `HOLOLINK_top.sv` source checkout you want me to use for port extraction, or should I use the skill's bundled known-rev port reference?
```

Accept either a path or a "use bundled reference" answer.

## Reference Loading

Load only the reference needed for the user's task:

| File | When to load |
|---|---|
| `references/source-policy.md` | Finding live HSB IP source, handling unknown revisions, or deciding source/reference precedence |
| `references/fixed-format.md` | Generating or explaining the fixed `FPGA_top.sv` scaffold format |
| `references/handoff-contract.md` | Standalone prerequisite handling or def-skill handoff behavior |

## Workflow Decision

1. If invoked by `hsb-ip-def` after a successful defs generation/validation, use **Def-Skill Handoff**.
2. If the user directly asks to create `FPGA_top.sv`, instantiate `HOLOLINK_top`, or make a top-level wrapper, use **Standalone Create Top**.
3. If the user asks about version handling, port groups, or why a signal appears, use **Explain**.

## Standalone Create Top

Use this when the user invokes this skill directly.

Steps:

1. Ask for the path to a validated `HOLOLINK_def.svh`, or accept pasted content. If the user does not have a defs file, stop and direct them to `hsb-ip-def` to create and validate it first.
2. If validation status is unknown, invoke or direct the user to `hsb-ip-def` validation. Do not proceed from an unvalidated defs file.
3. If the HSB IP source root is not obvious and source-sensitive output is needed, ask the live-source question from "Live HSB IP Source Policy" unless the user already authorized bundled known-rev fallback. Do not show path options as radio buttons. The known roots are search hints for the agent, not choices to force on the user.
4. Load `references/source-policy.md` and `references/fixed-format.md`.
5. Derive the active macro set from the validated defs file.
6. Derive or verify `HOLOLINK_top` port groups from live source when available.
7. Produce `FPGA_top.sv` in the fixed format. If writing a file, default filename is `FPGA_top.sv` unless the user provides a path.
8. Explain that the result is an HSB integration scaffold. Use the generic reference/template note in `## Instructions`; do not enumerate what is outside the scaffold's scope.

## Def-Skill Handoff

Use this when `hsb-ip-def` has just completed a validated `HOLOLINK_def.svh` and the user chooses to create a matching top-level scaffold.

Steps:

1. Accept the validated defs file path/content and any known profile/source-root context from `hsb-ip-def`.
2. Do not re-ask for defs-file decisions already completed by the def skill.
3. Ask only for missing top-generation facts, such as output path or HSB IP source root when needed. If source root is missing, ask the live-source question from "Live HSB IP Source Policy" unless the user already authorized bundled known-rev fallback; do not use radio buttons or structured choices.
4. Load `references/handoff-contract.md` and `references/fixed-format.md`.
5. Generate the fixed-format `FPGA_top.sv` scaffold from the active defs macros and live `HOLOLINK_top.sv` when available.
6. Return a concise summary of which HSB signal groups were emitted, then use the generic reference/template note in `## Instructions`. Do not enumerate what is outside the scaffold's scope.

## Explain

Use this when the user asks why a port or signal group appears, how macro settings affect the top scaffold, or how the skill handles IP revisions.

Steps:

1. For port questions, prefer live `HOLOLINK_top.sv` when available. Otherwise load `references/fixed-format.md`.
2. For revision/source questions, load `references/source-policy.md`.
3. For defs-file macro meaning, defer to `hsb-ip-def`.

## Troubleshooting

- Missing or unvalidated defs file: stop before generation and send the user to `hsb-ip-def` validation.
- Unknown or newer HSB IP revision: prefer live source if available, state the mismatch, and warn that bundled references may need updates.
- Port missing under the active macro set: omit that connection and explain which validated def macro disabled the signal group.

## Examples

- `Use hsb-ip-create-top to create an FPGA_top.sv scaffold from my validated HOLOLINK_def.svh. Use the bundled known-rev port reference if no HSB IP source checkout is obvious.` Treat as Standalone Create Top, require the validated defs file, load `references/source-policy.md` and `references/fixed-format.md`, use bundled known-rev ports after a brief source search, and report the defs status, port-map source, and emitted signal groups.
- `Create a top-level FPGA_top.sv wrapper for HSB, but I do not have a HOLOLINK_def.svh yet.` Stop before generation, direct the user to `hsb-ip-def`, and ask for a validated defs file path or content before proceeding.
- `Why does the generated FPGA_top.sv include different HSB signal groups depending on my HOLOLINK_def.svh macros?` Treat as Explain, say that validated defs macros gate the HSB-facing signal groups and that the live `HOLOLINK_top.sv` port map wins when available; defer detailed macro semantics to `hsb-ip-def`.

## Limitations

- Do not generate `HOLOLINK_def.svh`.
- Do not validate full `HOLOLINK_def.svh` files.
- Do not generate project-specific integration outside the HSB-facing scaffold.
- Do not infer signal meanings, polarity, sources, or behavior outside the HSB port map.
- Do not copy example project logic into the scaffold unless the user explicitly requests project-specific adaptation.
