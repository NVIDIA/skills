# Parabricks pacbio_germline

## First Steps

1. Confirm the data is PacBio germline data and identify HiFi or CLR if
   relevant.
2. Collect required inputs:
   - Reference FASTA.
   - PacBio FASTQ or BAM/CRAM input as supported by the selected version.
   - Output VCF/gVCF or output directory.
   - Model/resource bundle when required.
3. Ask for intervals, sample name, temporary directory, and logs when relevant.
4. For PacBio alignment-only questions, consider
   `pbrun-minimap2.md`; for runtime readiness, use
   `runtime-environment.md`.

## Command Shape

```bash
pbrun pacbio_germline \
  --ref /workdir/<reference.fa> \
  <version-specific-input-options> \
  <version-specific-output-options>
```

Verify exact input mode, model, output, and interval flags against the selected
version.

## minimap2/DeepVariant Option Mapping

Use this mapping when translating a baseline minimap2 plus Google DeepVariant
PacBio germline workflow to `pbrun pacbio_germline`. Parabricks
documents this as a long-read pipeline using minimap2 v2.31 alignment and
DeepVariant calling. The DeepVariant small model is enabled by default; use
`--disable-small-model` to turn it off.

| Baseline option | `pbrun pacbio_germline` equivalent | Notes |
| --- | --- | --- |
| minimap2 `-x map-pbmm2` or PacBio preset | `--preset map-pbmm2` or `--preset map-hifi` | Confirm PacBio subtype and selected version before changing the default. |
| minimap2 `-k`, `-uf`, `-ub`, `--MD`, `--eqx`, `-y` | Matching Parabricks minimap2 flags | Parabricks exposes these minimap2 controls directly. |
| DeepVariant `--model_type PACBIO` | PacBio pipeline default, or `--mode`/model flags where documented | Use PacBio-specific mode/model settings from the selected version. |
| DeepVariant custom model | `--pb-model-file`, `--pb-small-model-file` | Parabricks TensorRT model files. |
| Intervals/regions | `--interval` or `--interval-file` | Parabricks separates inline intervals from interval files. |
| Upstream minimap2/DeepVariant options not listed here | No direct equivalent | Not exposed by current Parabricks docs for `pacbio_germline`. |

## pacbio_germline Options Without minimap2/DeepVariant Equivalents

| `pbrun pacbio_germline` option | Why it has no direct baseline equivalent |
| --- | --- |
| `--pbmm2`, `--pbmm2-unmapped` | Parabricks compatibility controls for pbmm2-style output. |
| `--standalone-bqsr`, `--out-qc-metrics-dir` | Parabricks pipeline mode/output controls. |
| DeepVariant channel/allele-counter flags exposed directly by Parabricks | Parabricks first-class controls for DeepVariant internals. |
| `--nstreams`, `--max-queue-*`, `--chunk-size`, `--chaining-on-gpu` | Parabricks minimap2 GPU pipeline controls. `--chaining-on-gpu` may help CPU-bound alignment. |
| `--disable-small-model` | Opt-out for the default-on DeepVariant small model. |
| `--low-memory` | Parabricks memory-reduction mode. |

## Validation

- Input sequencing technology is PacBio.
- Reference, model/resources, and intervals match.
- Output VCF/gVCF or output directory exists.

## Guardrails

- Do not use for ONT or short-read data.
- Do not infer PacBio subtype or model compatibility from filename alone.
