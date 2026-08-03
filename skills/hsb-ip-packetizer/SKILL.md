---
name: hsb-ip-packetizer
author: "Holoscan Team <holoscan-team@nvidia.com>"
description: Choose or explain HSB Sensor RX packetizer fields for HOLOLINK_def.svh. Do not use for full defs, validation, or runtime APB programming.
version: "0.1.0"
tags:
  - holoscan
  - hsb
  - packetizer
  - systemverilog
  - fpga
license: Apache-2.0
compatibility: Targets HSB IP rev 16'h2604; backward-compatible with 16'h2603. Prefer live HSB IP packetizer source when available; warn on unknown revisions. Designed to work standalone or as a companion to hsb-ip-def.
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  team: holoscan
  domain: fpga
  vendor: nvidia
  tags:
    - holoscan
    - hsb
    - fpga
    - packetizer
    - systemverilog
  languages:
    - systemverilog
  artifact: HOLOLINK_def.svh
  hsb_ip_version: "16'h2604"
---

# HSB IP Packetizer Skill

## Purpose

Choose packetizer-related `HOLOLINK_def.svh` fields for the NVIDIA Holoscan Sensor Bridge Sensor RX path.

This skill owns only the packetizer slice of `HOLOLINK_def.svh`:

- `SIF_RX_PACKETIZER_EN[]`
- `SIF_RX_VP_COUNT[]`
- `SIF_RX_SORT_RESOLUTION[]`
- `SIF_RX_VP_SIZE[]`
- `SIF_RX_NUM_CYCLES[]`

Use `hsb-ip-def` for whole-file generation, validation, comparison, and non-packetizer macros.

## Prerequisites

- `SENSOR_RX_IF_INST` must be defined and nonzero before packetizer fields are relevant.
- Per-RX-interface `SIF_RX_WIDTH[]` values and the user's data manipulation intent are required before deriving enabled packetizer values.
- Live packetizer RTL is preferred for source-sensitive claims; bundled references cover known rev `16'h2604` and compatible rev `16'h2603`.
- Whole-file generation and validation remain the responsibility of `hsb-ip-def`.

## Instructions

- Only configure packetizer fields when `SENSOR_RX_IF_INST` is defined. If Sensor RX is disabled, say no packetizer fields are needed.
- First classify the request as Standalone SVH, Def-Skill Handoff, Explain, or runtime APB pattern-RAM. For Def-Skill Handoff, load `references/handoff-contract.md` and make `packetizer_profile_overlay` the first non-empty output after any required clarification.
- Ask one question per turn only while collecting the data-description facts needed to derive a grounded profile. Prefer deriving macro values from those facts instead of requesting per-macro confirmation.
- Use sensor-agnostic language. Say "sensor data", "data word", "stream", "bandwidth", and "packetization"; use camera-specific language only after the user says the sensor data is image/camera data.
- Never silently default enabled packetizer fields. Derive them from the user's stated data layout and manipulation intent, print the complete field set in one block, and then explain what the selected settings enable and what assumptions they encode.
- Avoid ungrounded "typical", "common", "most designs", or corpus-frequency claims. Anchor statements to HSB docs, RTL behavior, or the user's stated requirements.
- Keep runtime APB pattern-RAM programming out of v1 generation. `HOLOLINK_def.svh` enables/sizes hardware; runtime software still controls pattern RAM, virtual-port selection, sort controls, bypass, padding, duplication, and latency.
- In a handoff from `hsb-ip-def`, use facts already known by that def flow. If the prompt gives RX count, enabled/pass-through intent, lane or stream width, virtual stream count, and cycle window, emit the overlay without re-asking for macro names or whole-file details. Leave final merge and full validation to `hsb-ip-def`.

## Live HSB IP source policy

Prefer live HSB IP source over bundled packetizer references whenever the user's workspace provides it. The bundled references describe known rev `16'h2604` and backward-compatible rev `16'h2603`; live source is the authority for the checked-out IP.

When source is available:

1. Locate `HOLOLINK_top.sv` plus the packetizer RTL files:
   - `<hsb-ip-root>/top/HOLOLINK_top.sv`
   - `<hsb-ip-root>/packetizer/packetizer_top.sv`
   - `<hsb-ip-root>/packetizer/packetizer.sv`
   - `<hsb-ip-root>/packetizer/virtual_port.sv`
   - `<hsb-ip-root>/packetizer/odd_even_gen.sv`
2. Read `HOLOLINK_REV` and `HOLOLINK_BACKWARD_COMPAT_REV` from `HOLOLINK_top.sv` when present.
3. Verify that `HOLOLINK_top` still maps the five def fields into `packetizer_top` as documented before making source-sensitive claims.
4. If the rev is newer, unknown, or live RTL disagrees with this skill's references, state the mismatch. Trust the live RTL for explanation, avoid unsupported assumptions, and do not emit an enabled packetizer overlay unless the field mapping and RTL constraints can be verified from the live source.

When no source is available, use the bundled references as known-revision guidance and say that the answer is based on the skill's supported HSB IP rev.

## Reference Loading

Load only the reference needed for the user's task:

| File | When to load |
|---|---|
| `references/packetizer-def-fields.md` | Choosing or explaining `HOLOLINK_def.svh` packetizer fields, array length rules, bypass placeholders, RTL-derived constraints, or standalone SVH output formatting |
| `references/packetizer-architecture.md` | Explaining packetizer blocks, data flow, clocks, APB registers, runtime-vs-def split, or why a field exists |
| `references/handoff-contract.md` | Producing `packetizer_profile_overlay` YAML for `hsb-ip-def` or enforcing output-mode separation |

## Workflow Decision

1. If the user is working inside a `HOLOLINK_def.svh` generation flow, use **Def-Skill Handoff**.
2. If the user asks for lines to add to an existing/new `HOLOLINK_def.svh`, use **Standalone SVH**.
3. If the user asks what a field does, why a value matters, or whether a combination is safe, use **Explain**.
4. If the user asks for APB pattern RAM programming, explain that it is runtime configuration and outside v1 generated output; load `references/packetizer-architecture.md` for the register context.

## Requirement Discovery

Collect these facts in order, skipping facts the user already supplied. Stop asking as soon as the data manipulation description is specific enough to derive the profile.

1. `SENSOR_RX_IF_INST`: number of Sensor RX interfaces. If undefined or zero, stop.
2. `SIF_RX_WIDTH[]`: per-RX-interface input width. This is `DIN_WIDTH` for each packetizer instance.
3. Packetizer intent per RX interface: pass-through, data rearrangement/manipulation, split into virtual ports, replicate/duplicate, or other manipulation.
4. The data layout for enabled interfaces: lane/chunk width, number of output virtual streams, whether the manipulation stays within one input cycle or spans multiple cycles, and any exact ordering/replication requirement.

Skip per-macro confirmation prompts such as "Confirm `SIF_RX_VP_COUNT`?", "Confirm `SIF_RX_SORT_RESOLUTION`?", "Confirm `SIF_RX_VP_SIZE`?", or "Confirm `SIF_RX_NUM_CYCLES`?" Ask for missing data-shape facts, such as "How many virtual streams should RX0 split into?" or "What bit granularity is being rearranged?"

## Deriving Fields

After the required facts are known, derive all packetizer arrays together:

- `SIF_RX_PACKETIZER_EN[i] = 1` for RX interfaces that need packetizer hardware for rearrangement, split, replication, or other runtime-controlled data manipulation; `0` for pass-through interfaces.
- `SIF_RX_VP_COUNT[i]` equals the number of virtual output streams/ports required by the user's manipulation for that RX interface.
- `SIF_RX_VP_SIZE[i]` is the per-virtual-port data size/width in bits. For an equal contiguous split of one `SIF_RX_WIDTH[i]` input word into `N` virtual ports, use `SIF_RX_WIDTH[i] / N` if it is a positive power-of-two divisor and matches the user's intent.
- `SIF_RX_SORT_RESOLUTION[i]` is the bit granularity of the data rearrangement operation. If the user describes fixed-size lanes, use that lane size when it passes the RTL-derived constraints. If no ordering manipulation is needed and the packetizer is used only for contiguous split/replication, use `SIF_RX_WIDTH[i]` to disable the sort network internally when legal.
- `SIF_RX_NUM_CYCLES[i]` is the number of input sensor-data cycles that participate in the manipulation. Use `1` for single-cycle rearrange/split/replication; use the stated cycle window when the manipulation spans multiple input cycles.

If several profiles remain plausible after reading the data description, ask one clarifying question about the behavior or data shape that distinguishes them. Do not fall back to "typical" packetizer values.

For disabled entries in a mixed enabled/disabled design, fill the peer arrays with ignored placeholder values matching the RTL bypass constants:

- `SIF_RX_VP_COUNT[i] = 1`
- `SIF_RX_SORT_RESOLUTION[i] = 2`
- `SIF_RX_VP_SIZE[i] = 32`
- `SIF_RX_NUM_CYCLES[i] = 1`

## Emitted Field Format

- The enable array must always have exactly `SENSOR_RX_IF_INST` entries.
- If any enable entry is `1`, emit all four peer arrays at exactly `SENSOR_RX_IF_INST` length.
- If every enable entry is `0`, emit only `SIF_RX_PACKETIZER_EN[]`; the four peer arrays are not required for the packetizer decision.
- Use positional array syntax for mixed per-port values. Use `'{default:<value>}` only when every entry is identical and that form improves readability.
- Do not emit or modify `SENSOR_RX_IF_INST`, `SIF_RX_WIDTH[]`, `DATAPATH_WIDTH`, host fields, clocks, or any non-packetizer macros from this skill.
- After emitting the block or overlay, explain the selected settings in terms of the user's input data and manipulation needs. Keep the explanation after the fenced output so downstream tools can consume the block first.

## Standalone SVH

Use this when the user asks for packetizer lines to add to a `HOLOLINK_def.svh`.

Steps:

1. Gather the required facts.
2. Load `references/packetizer-def-fields.md` for choosing values, checking constraints, and using the canonical standalone output formatting.
3. Emit a fenced `systemverilog` block containing only the packetizer localparams.
4. Explain what the settings allow for the user's data manipulation: which RX interfaces use packetizer hardware, how many virtual streams they expose, the per-stream width, the data-rearrangement granularity, and whether the operation is single-cycle or multi-cycle.
5. Add a short note that whole-file validation should be done with `hsb-ip-def` or its `validate_def.py`.


## Def-Skill Handoff

Use this when invoked by `hsb-ip-def` while that skill is generating a full `HOLOLINK_def.svh`.

Steps:

1. Gather only packetizer-specific requirements not already known from the def flow.
2. Load `references/handoff-contract.md`.
3. Emit one labeled YAML overlay first. Do not write an introductory sentence before the `packetizer_profile_overlay` label once requirements are known. The def skill consumes the flat keys directly.
4. After the overlay, add a concise explanation of what the settings allow for the user's data description and manipulation needs.

Required label:

`packetizer_profile_overlay`

YAML keys:

- `sif_rx_packetizer_en`
- `sif_rx_vp_count` only when any enable is `1`
- `sif_rx_sort_resolution` only when any enable is `1`
- `sif_rx_vp_size` only when any enable is `1`
- `sif_rx_num_cycles` only when any enable is `1`

Example shape:

packetizer_profile_overlay
```yaml
sif_rx_packetizer_en: [1, 0]
sif_rx_vp_count: [4, 1]
sif_rx_sort_resolution: [16, 2]
sif_rx_vp_size: [128, 32]
sif_rx_num_cycles: [1, 1]
```

The def skill remains responsible for merging this overlay into its profile, generating the final SVH, and running full validation.

## Explain

Use this when the user asks what a packetizer field does, why a value matters, or whether a packetizer combination is suitable.

Steps:

1. Load `references/packetizer-def-fields.md` for field-level answers.
2. Load `references/packetizer-architecture.md` for architecture, data flow, clock domains, or runtime APB behavior.
3. Cite source locations in the form `top/HOLOLINK_top.sv:1356-1362` or `packetizer/packetizer.sv:69-85` when explaining RTL behavior.
4. For legality questions that affect the whole `HOLOLINK_def.svh`, state the packetizer-local concern and tell the user to validate the full file with `hsb-ip-def`.

## Troubleshooting

- Sensor RX disabled: emit no packetizer arrays and explain that packetizer hardware is only for Sensor RX interfaces.
- Missing data-shape facts: ask one clarifying question about stream count, lane width, cycle window, or ordering before deriving fields.
- Runtime APB pattern-RAM request: explain that runtime programming is outside v1 generated output and load `references/packetizer-architecture.md` only for context.
- Whole-file legality concern: state the packetizer-local issue and direct validation back to `hsb-ip-def`.

## Examples

- `Use hsb-ip-packetizer to choose packetizer fields for two Sensor RX interfaces. RX0 is 512 bits and must split each input word into four 128-bit virtual streams; RX1 is pass-through.` Treat as Standalone SVH and emit only packetizer localparams: enable `[1, 0]`, virtual-port count `[4, 1]`, sort resolution `[512, 2]` for a contiguous split without lane reordering, virtual-port size `[128, 32]`, and cycles `[1, 1]`. Tell the user to run full-file validation with `hsb-ip-def`.
- `hsb-ip-def is generating my HOLOLINK_def.svh and needs packetizer values. RX0 should rearrange 16-bit lanes into four virtual streams; RX1 should stay pass-through.` Treat as Def-Skill Handoff and put the YAML overlay first:

packetizer_profile_overlay
```yaml
sif_rx_packetizer_en: [1, 0]
sif_rx_vp_count: [4, 1]
sif_rx_sort_resolution: [16, 2]
sif_rx_vp_size: [16, 32]
sif_rx_num_cycles: [1, 1]
```

  Then state that `hsb-ip-def` remains responsible for merging this overlay, generating the full `HOLOLINK_def.svh`, and running full validation.
- `Do I need hsb-ip-packetizer if SENSOR_RX_IF_INST is disabled in my HOLOLINK_def.svh?` Explain that packetizer fields apply only to Sensor RX interfaces, emit no packetizer arrays, and avoid runtime APB pattern-RAM configuration.

## Limitations

- Do not generate full `HOLOLINK_def.svh` files.
- Do not validate full `HOLOLINK_def.svh` files.
- Do not generate runtime APB writes or pattern-RAM contents in v1.
- Do not choose packetizer values from corpus frequency or archetype popularity.
