# Parabricks ont_germline

## First Steps

1. Confirm the data is Oxford Nanopore germline data.
2. Collect required inputs:
   - Reference FASTA.
   - ONT FASTQ or BAM/CRAM input as supported by the selected version.
   - Output VCF/gVCF or output directory.
   - Model/resource bundle when required.
3. Ask for intervals, sample name, temporary directory, and logs when relevant.
4. For ONT alignment-only questions, consider
   `pbrun-minimap2.md`; for runtime readiness, use
   `runtime-environment.md`.

## Command Shape

```bash
pbrun ont_germline \
  --ref /workdir/<reference.fa> \
  <version-specific-input-options> \
  <version-specific-output-options>
```

Verify exact input mode, model, output, and interval flags against the selected
version.

## minimap2/DeepVariant Option Mapping

Use this mapping when translating a baseline minimap2 plus Google DeepVariant
ONT germline workflow to `pbrun ont_germline`. Parabricks documents
this as a long-read pipeline using minimap2 v2.31 alignment and DeepVariant
calling. The DeepVariant small model is enabled by default; use
`--disable-small-model` to turn it off.

| Baseline option | `pbrun ont_germline` equivalent | Notes |
| --- | --- | --- |
| minimap2 `-x map-ont` or ONT preset | `--preset map-ont` or selected ONT preset | Confirm ONT chemistry/use case and selected version before changing defaults. |
| minimap2 `-k`, `-uf`, `-ub`, `--MD`, `--eqx`, `-y` | Matching Parabricks minimap2 flags | Parabricks exposes these minimap2 controls directly. |
| DeepVariant `--model_type ONT` | ONT pipeline default, or `--mode`/model flags where documented | Use ONT-specific mode/model settings from the selected version. |
| DeepVariant custom model | `--pb-model-file`, `--pb-small-model-file` | Parabricks TensorRT model files. |
| Intervals/regions | `--interval` or `--interval-file` | Parabricks separates inline intervals from interval files. |
| Upstream minimap2/DeepVariant options not listed here | No direct equivalent | Not exposed by current Parabricks docs for `ont_germline`. |

## ont_germline Options Without minimap2/DeepVariant Equivalents

| `pbrun ont_germline` option | Why it has no direct baseline equivalent |
| --- | --- |
| `--standalone-bqsr`, `--out-qc-metrics-dir` | Parabricks pipeline mode/output controls. |
| DeepVariant channel/allele-counter flags exposed directly by Parabricks | Parabricks first-class controls for DeepVariant internals. |
| `--nstreams`, `--max-queue-*`, `--chunk-size`, `--chaining-on-gpu` | Parabricks minimap2 GPU pipeline controls. `--chaining-on-gpu` may help CPU-bound alignment. |
| `--disable-small-model` | Opt-out for the default-on DeepVariant small model. |
| `--low-memory` | Parabricks memory-reduction mode. |

## Validation

- Input sequencing technology is ONT.
- Reference, model/resources, and intervals match.
- Output VCF/gVCF or output directory exists.

## Guardrails

- Do not use for PacBio or short-read data.
- Do not infer ONT model/resource compatibility from filename alone.
