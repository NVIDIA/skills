# Source And Version Policy

Use this reference when locating the HSB IP source or handling IP revision changes.

## Source Authority

Live HSB IP source is authoritative when available. Bundled skill references describe known rev `16'h2604` and backward-compatible rev `16'h2603`; a checked-out `HOLOLINK_top.sv` wins for the user's design.

Known source roots:

| Layout | Root path |
|---|---|
| NVIDIA-internal workspace | `hw/nvcpu_dgx_fpga/vrtl/hololink/` |
| NVIDIA-internal sibling checkout | `hsb_agent/hw/nvcpu_dgx_fpga/vrtl/hololink/` |
| Public release | `fpga/nv_hsb_ip/` |

These are search hints for the agent. Do not present them as radio-button choices to the user.

The top-level module source is:

```text
<hsb-ip-root>/top/HOLOLINK_top.sv
```

## Version Check

Read these localparams from `HOLOLINK_top.sv` when source is present:

```systemverilog
localparam HOLOLINK_REV = 16'h2604;
localparam HOLOLINK_BACKWARD_COMPAT_REV = 16'h2603;
```

If the revision is `16'h2604` or backward-compatible with `16'h2603`, proceed normally. If it is newer, unknown, or locally modified, parse the live module declaration and warn that bundled references may need an update.

If the user's current working directory is not the HSB source checkout, search obvious sibling paths before falling back to bundled references. If the source is still unclear, ask one plain-language question:

```text
Do you have a current HSB IP `HOLOLINK_top.sv` source checkout you want me to use for port extraction, or should I use the skill's bundled known-rev port reference?
```

Accept either a path or a "use bundled reference" answer.

## Port Source Of Truth

For `FPGA_top.sv` generation, derive the HSB-facing signal groups from the live `HOLOLINK_top` module declaration:

- port name
- direction
- packed/unpacked width
- ordering
- comment group
- `ifdef` / `ifndef` gate

Do not rely on example top-level files as IO authority.

## Historical Example Top Files

The fixed output format is baked into `references/fixed-format.md`. Do not read example top files during normal generation just to confirm formatting.

Only inspect historical example top files when the user explicitly asks to compare, explain, or adapt a specific existing project:

- `hw/nvcpu_dgx_fpga/vrtl/bajoran_lite/top/FPGA_top.sv`
- `hw/nvcpu_dgx_fpga/vrtl/bajoran/top/FPGA_top.v`
- `hw/nvcpu_dgx_fpga/vrtl/bajoran_rfsoc/top/FPGA_top.sv`

Do not copy project-specific implementation logic unless the user explicitly requests a project-specific adaptation.
