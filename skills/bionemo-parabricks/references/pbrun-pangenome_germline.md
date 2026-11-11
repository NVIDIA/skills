# Parabricks pangenome_germline

## First Steps

1. Confirm the user wants pangenome-aware germline calling.
2. Collect required inputs:
   - Reference FASTA and pangenome graph/resource bundle.
   - Paired-end or single-end FASTQ input.
   - Output BAM and VCF/gVCF paths.
   - Any model/resource bundle required by the selected version.
3. For graph alignment only, use `pbrun-giraffe.md`. For prepared BAM input,
   use the standalone caller in `pbrun-pangenome_aware_deepvariant.md`.

## Command Shape

```bash
pbrun pangenome_germline \
  <version-specific-pangenome-resource-options> \
  <version-specific-input-options> \
  <version-specific-output-options>
```

Verify graph/resource, input, and output flags against the selected version.

## vg giraffe/DeepVariant Option Mapping

Use this mapping when translating a baseline pangenome workflow using
`vg giraffe` plus DeepVariant to `pbrun pangenome_germline`. Parabricks
combines graph alignment, preprocessing, and variant calling, so the mapping is
pipeline-level rather than one-to-one.

| Baseline option | `pbrun pangenome_germline` equivalent | Notes |
| --- | --- | --- |
| `vg giraffe --gbz-name`, `-Z` | `--gbz-name` or version-specific graph flag | Required GBZ graph input when documented. |
| `vg giraffe --dist-name`, `-d` | `--dist-name` or version-specific graph flag | Distance index input. |
| `vg giraffe --minimizer-name`, `-m` | `--minimizer-name` or version-specific graph flag | Minimizer index input. |
| `vg giraffe --zipcodes-name`, `-z` | `--zipcodes-name` or version-specific graph flag | Zipcodes input when required. |
| `vg giraffe --ref-paths` | `--ref-paths` or version-specific path-list flag | Path list/dictionary for headers and reference extraction. |
| FASTQ query input | `--in-fq` or selected input flag | Verify paired/single-end syntax for the selected version. |
| Linear reference FASTA for variant calling | `--ref` | Reference must match graph-derived paths/build. |
| DeepVariant custom model | `--pb-model-file` | Optional TensorRT model. Default is the shipped pangenome-aware engine, or the SBX engine when `--sbx` is set. |
| Roche SBX Google case-study settings | `--sbx` | Roche SBX-D / SBX-Fast mode on the pangenome-aware DeepVariant stage. |
| `vg giraffe --max-multimaps`, `-M` | `--max-multimaps` | Max alignments per read (default: 1). |
| `vg giraffe --supplementary` | `--supplementary` | Identify and report supplementary alignments (off by default). |
| Intervals/regions | `--interval` or `--interval-file` | Verify pangenome workflow support for interval restriction. |
| Upstream `vg giraffe` or DeepVariant options not listed here | No direct equivalent | Not exposed by current Parabricks docs for `pangenome_germline`. |

## pangenome_germline Options Without vg giraffe/DeepVariant Equivalents

| `pbrun pangenome_germline` option | Why it has no direct baseline equivalent |
| --- | --- |
| `--sbx` | Roche SBX mode for the pangenome-aware DeepVariant stage. |
| GPU graph-alignment stream, batch, queue, and minimizer controls | Parabricks GPU implementation tuning. |
| `--in-se-cram`, `--in-cram-ref`, `--cram-read-threads` | Single-end CRAM input, as in `giraffe`; the Ultima tag options are giraffe-only. |
| GPU write/sort/storage controls | Parabricks accelerated output pipeline. |
| DeepVariant channel/allele-counter flags exposed directly by Parabricks | Parabricks first-class controls for DeepVariant internals. |

## Validation

- Pangenome resources are complete and compatible with the reference build.
- The graph is GBZ version 1 and the indexes were built with vg v1.75.1
  (v1.70.0 for 4.7.1; see `pbrun-giraffe.md`).
- Output VCF/gVCF or output directory exists.

## Guardrails

- Do not use for standard linear-reference germline calling unless the user
  explicitly wants pangenome-aware analysis.
- Do not invent graph bundle paths or compatibility.
- Use `--sbx` only when the user has Roche SBX-D or SBX-Fast data.
