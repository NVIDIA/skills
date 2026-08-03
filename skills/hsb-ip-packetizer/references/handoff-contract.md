# Packetizer Handoff Contract

Use this reference when emitting output from `hsb-ip-packetizer`.

## Modes

There are two output modes:

- **Standalone SVH**: for a user adding lines to an existing or new `HOLOLINK_def.svh`.
- **Def-skill overlay**: for `hsb-ip-def`, which consumes flat YAML keys and continues full-file generation and validation.

Do not mix both modes in the same answer unless the user explicitly asks for both.

## Standalone SVH Contract

Use `references/packetizer-def-fields.md` as the canonical source for standalone SVH output formatting examples.

Emit one fenced `systemverilog` block containing only packetizer localparams. Do not emit `SENSOR_RX_IF_INST`, `SIF_RX_WIDTH[]`, or non-packetizer macros.

After the block, explain how the emitted settings map to the user's data description. Keep this explanation after the fenced block so the block is easy to copy or consume.

End with a short validation note:

`Run full-file validation with hsb-ip-def after inserting this into HOLOLINK_def.svh.`

## Def-Skill Overlay Contract

When invoked by `hsb-ip-def`, emit a labeled YAML overlay and no standalone SVH block. Put the overlay first, then add a concise explanation of what the settings allow for the user's input data and manipulation needs.

Label line:

`packetizer_profile_overlay`

All-disabled output:

packetizer_profile_overlay
```yaml
sif_rx_packetizer_en: [0, 0]
```

Mixed output:

packetizer_profile_overlay
```yaml
sif_rx_packetizer_en: [1, 0, 1]
sif_rx_vp_count: [4, 1, 2]
sif_rx_sort_resolution: [16, 2, 32]
sif_rx_vp_size: [128, 32, 256]
sif_rx_num_cycles: [1, 1, 3]
```

Uniform enabled output:

packetizer_profile_overlay
```yaml
sif_rx_packetizer_en: [1, 1]
sif_rx_vp_count: [4, 4]
sif_rx_sort_resolution: [16, 16]
sif_rx_vp_size: [128, 128]
sif_rx_num_cycles: [1, 1]
```

The def skill merges these keys into the profile used by `scripts/generate_def.py`. It remains responsible for:

- reconciling with the full design profile,
- emitting the final `HOLOLINK_def.svh`,
- running the full def validator,
- reporting any whole-file errors or warnings.

The explanation may describe:

- which RX interfaces are pass-through and which instantiate packetizer hardware,
- how `sif_rx_vp_count` maps to the requested virtual streams,
- how `sif_rx_vp_size` maps to each stream's width/size,
- how `sif_rx_sort_resolution` maps to the data-rearrangement granularity or disables sorting when equal to the RX width,
- how `sif_rx_num_cycles` maps to the stated single-cycle or multi-cycle manipulation window,
- that disabled-interface peer entries are placeholders ignored by the RTL.

## Placeholder Rule

When any interface is enabled and another interface is disabled, still emit peer arrays at full length. Use placeholders at disabled indices:

```yaml
sif_rx_packetizer_en: [1, 0]
sif_rx_vp_count: [4, 1]
sif_rx_sort_resolution: [16, 2]
sif_rx_vp_size: [128, 32]
sif_rx_num_cycles: [1, 1]
```

These placeholder values match the bypass constants in `top/HOLOLINK_top.sv:1356-1362`.
