# Handoff Contract

Use this reference when coordinating `hsb-ip-create-top` with `hsb-ip-def`.

## Standalone Prerequisite

When invoked directly, this skill requires a validated `HOLOLINK_def.svh`.

If the user has no defs file:

1. Stop top-generation work.
2. Tell the user to use `hsb-ip-def` to generate and validate `HOLOLINK_def.svh`.
3. Resume only after a valid defs file path or content is available.

If the user has a defs file but validation status is unknown, ask to validate it with `hsb-ip-def` before proceeding.

## Def-Skill Handoff

After `hsb-ip-def` successfully generates and validates a `HOLOLINK_def.svh`, it may ask:

```text
Do you want to create a matching top-level FPGA scaffold with hsb-ip-create-top?
```

When the user says yes, the def skill should pass or summarize:

- validated defs file path or content
- known generated profile values, if available
- selected HSB IP source root, if known
- intended output path, if the user already supplied one

`hsb-ip-create-top` consumes that context and should not re-ask completed defs decisions.

## Output Contract

The create-top skill outputs either:

- a written `FPGA_top.sv` file, when the user asks for a file path, or
- a fenced `systemverilog` scaffold, when the user asks for text.

After output, summarize:

- active signal groups emitted,
- groups omitted because the defs macro is undefined,
- whether live `HOLOLINK_top.sv` or bundled known-rev references supplied the port map,
- the generic note: "This file is a reference/template for hooking up top-level signals and companion IP to the HSB IP."

Do not enumerate what is outside the scaffold's scope in the summary.
