<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Build-Test result rules

This output contract belongs to the skill workflow.
It is not an NVIDIA SDK output format.

Use [the JSON result template](../assets/build-test-result-template.json) for
machine-readable output, as Spec does for its inputs. This Markdown file
explains the contract; it is not a second copy of the template or a JSON Schema.
Replace each placeholder and select one value from each `|`-separated list.
Build-Test checks the result against its stage evidence; no standalone result
validator is shipped. Write the Markdown report from the same final JSON values.

Set `command_exit_code` to `null` when a stage has `NOT_RUN` or
`SKIPPED_OPTIONAL` status. For a stage that ran, record its actual integer exit
code.

Use JSON `null` for an unavailable optional value.
Do not use `null` for a required test result.

When EMU execution runs, record `EMU` in
`emulator.build_mode` and the actual configure-option arguments as strings
in `emulator.configure_options`. When no backend was proved, use `null`
and `[]`. Use correctness environment labels `EMU` and `HARDWARE`, and stage
labels `EMU_BUILD` and `EMU_TEST` for host emulation. The hardware build remains
`PVA_BUILD_MODE=L4T`, using Doctor's proved NVVPU or ASIP/Chess compiler backend.

For `correctness-only`, write an empty `performance` list. Set target clock
fields to `NOT_REQUIRED` or `not required` as the template specifies. Do not
require KPI, clock, profile, or performance artifacts.
