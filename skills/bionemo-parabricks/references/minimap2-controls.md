# Minimap2 pipeline controls

These v4.8.0 controls apply only to the commands named in the **Tools** column.
Combine them with the selected tool's option mapping and its scoped rows in
[shared-options.md](shared-options.md) for inputs, intervals, model files,
output, and memory limits. Use the release and execution rules in
[command-conventions.md](command-conventions.md).

| Option | Role | Tools |
| --- | --- | --- |
| `--pbmm2`, `--pbmm2-unmapped` | Compatibility controls for pbmm2-style output and unmapped records. | `minimap2`, `pacbio_germline` |
| `--standalone-bqsr` | Pipeline mode control; for `minimap2`, runs BQSR after sorted BAM generation. | `minimap2`, `ont_germline`, `pacbio_germline` |
| `--nstreams`, `--max-queue-*`, `--chunk-size`, `--estimated-*`, `--stage2-large-chaining-size` | GPU pipeline tuning, including `--max-queue-chunks`, `--max-queue-reads`, `--estimated-read-length`, and `--estimated-anchor-per-read`. For `minimap2`, streams and chunk size default to `auto`, tuned from device memory and preset. Set values only for benchmarked tuning. | `minimap2`, `ont_germline`, `pacbio_germline` |
| `--seeding-chaining-on-gpu on\|off\|auto` | GPU seeding and chaining; `on` may help CPU-bound alignment. `auto` leaves ONT and PacBio presets on CPU; `minimap2` selects GPU only for `splice:sr`. Replaces 4.7.1's bare `--chaining-on-gpu` switch, now a deprecated alias that requires a value. | `minimap2`, `ont_germline`, `pacbio_germline` |
