# Parabricks bamsort

## First Steps

1. Collect required inputs:
   - Input BAM or CRAM path.
   - Output sorted BAM or CRAM path.
   - Reference FASTA when CRAM input or the selected version requires it.
2. Ask whether coordinate sort or another sort order is required, then verify
   support in the selected version.
3. Ask for temporary directory and available storage if the input is large.

## Command Shape

```bash
pbrun bamsort \
  --ref /workdir/<reference.fa> \
  --in-bam /workdir/<input.bam> \
  --out-bam /outputdir/<sorted.bam> \
  --sort-order coordinate
```

Verify exact option names for sort order, reference, temporary directory,
threading, and output format against the selected version.

## SortSam Option Mapping

Use this mapping when translating a GATK/Picard `SortSam` command to
`pbrun bamsort`. Parabricks documents `bamsort` as Picard-compatible by
default and also exposes fgbio-compatible sort modes.

| GATK/Picard option | `pbrun bamsort` equivalent | Notes |
| --- | --- | --- |
| `--OUTPUT`, `-O` | `--out-bam` | Required sorted BAM/CRAM output path. |
| `--REFERENCE_SEQUENCE`, `-R` | `--ref` | Required by current Parabricks docs, especially for CRAM/header handling. |
| `--SORT_ORDER`, `-SO` | `--sort-order` | Parabricks supports `coordinate`, `queryname`, and `templatecoordinate`. |
| Picard-compatible comparator | `--sort-compatibility picard` | Default Parabricks comparator mode. |
| fgbio comparator behavior | `--sort-compatibility fgbio` | Parabricks-specific selector for fgbio-compatible sorting. |
| `--MAX_RECORDS_IN_RAM` | `--max-records-in-ram` | Applies to queryname/template coordinate sort modes. |
| `--arguments_file`, `--VALIDATION_STRINGENCY`, Picard compression/common flags | No direct equivalent | Not exposed by current Parabricks docs. |

## bamsort Options Without SortSam Equivalents

| `pbrun bamsort` option | Why it has no SortSam equivalent |
| --- | --- |
| `--num-zip-threads` | Parabricks compression worker count. |
| `--num-sort-threads` | Parabricks sorting worker count. |
| `--mem-limit` | Parabricks sort/postsort memory limit in GB. |
| `--gpuwrite` | Parabricks GPU-accelerated final BAM/CRAM writing. |
| `--gpusort` | Parabricks GPU-accelerated sorting. |

## Validation

- Output sorted BAM/CRAM is created and is usable by downstream tools.
- Temporary directory has enough free space.

## Guardrails

- Do not claim the output is indexed unless an index was explicitly requested
  and produced.
- Do not change sort order or output format without user intent.
- Do not use this reference for duplicate marking; route to `pbrun-markdup.md`.
