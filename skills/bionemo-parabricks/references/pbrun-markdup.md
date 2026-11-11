# Parabricks markdup

## First Steps

1. Collect required inputs:
   - Input queryname-sorted BAM or CRAM path.
   - Output duplicate-marked BAM or CRAM path.
   - Reference FASTA when CRAM input or the selected version requires it.
2. Ask whether duplicate metrics are needed.
3. Ask about temporary directory, logs, and output index only when relevant and
   supported by the selected version.

## Command Shape

```bash
pbrun markdup \
  --ref /workdir/<reference.fa> \
  --in-bam /workdir/<queryname_sorted.bam> \
  --out-bam /outputdir/<marked.bam> \
  <optional-metrics-output>
```
When duplicate metrics are requested, add
`--out-duplicate-metrics /outputdir/<marked.metrics.txt>`.

Verify exact input, output, metrics, index, and temporary-directory options
against the selected version before finalizing.

## MarkDuplicates Option Mapping

Use this mapping when translating a GATK/Picard `MarkDuplicates` workflow to
`pbrun markdup`. Parabricks documents `markdup` as a duplicate-marking
tool that requires queryname-sorted input, while its default output behavior is
compatible with a coordinate-sort-order `MarkDuplicates` baseline.

| GATK/Picard option | `pbrun markdup` equivalent | Notes |
| --- | --- | --- |
| `SortSam --REFERENCE_SEQUENCE`, `-R` / `MarkDuplicates --REFERENCE_SEQUENCE`, `-R` | `--ref` | Required by Parabricks for CRAM support and BAM/CRAM header verification. |
| `MarkDuplicates --INPUT`, `-I` | `--in-bam` | Required input BAM/CRAM. Parabricks expects queryname-sorted input. |
| `MarkDuplicates --OUTPUT`, `-O` | `--out-bam` | Required duplicate-marked BAM/CRAM output. |
| `MarkDuplicates --METRICS_FILE`, `-M` | `--out-duplicate-metrics` | Duplicate metrics output. |
| `MarkDuplicates --ASSUME_SORT_ORDER coordinate`, `-ASO coordinate` | Default `pbrun markdup` behavior | Parabricks default matches coordinate-sort-order duplicate-marking behavior even though input must be queryname-sorted. |
| `MarkDuplicates --ASSUME_SORT_ORDER queryname`, `-ASO queryname` | `--markdups-assume-sortorder-queryname` | Matches queryname-sort-order duplicate-marking behavior. |
| `MarkDuplicates --OPTICAL_DUPLICATE_PIXEL_DISTANCE` | `--optical-duplicate-pixel-distance` | Same optical duplicate distance role. |
| Single-end duplicate marking by read ends | `--markdups-single-ended-start-end` | Parabricks single-end duplicate-marking control. |
| Ignore read group for single-end duplicate marking | `--ignore-rg-markdups-single-ended` | Must be used with `--markdups-single-ended-start-end`. |
| `SortSam --SORT_ORDER queryname` before marking | No direct `markdup` equivalent | Run `pbrun bamsort --sort-order queryname` first if input is not already queryname-sorted. |
| `SortSam --SORT_ORDER coordinate` after queryname baseline marking | No direct `markdup` equivalent | Use `pbrun bamsort` after `markdup` if the selected workflow requires a separate coordinate-sort step. |
| GATK/Picard `--version` | `--version` | Same version-reporting role. |
| GATK/Picard `--java-options` | No direct equivalent | Java runtime settings do not apply to the Parabricks containerized GPU tool. |
| GATK/Picard `--arguments_file`, validation, compression, and cloud-auth common flags | No direct equivalent | Not exposed by current Parabricks docs for `markdup`. |

## markdup Options Without MarkDuplicates Equivalents

| `pbrun markdup` option | Why it has no MarkDuplicates equivalent |
| --- | --- |
| `--num-zip-threads`, `--num-worker-threads` | Parabricks CPU worker controls. |
| `--mem-limit` | Parabricks memory limit for sorting/postsorting. |
| `--gpuwrite`, `--gpuwrite-deflate-algo`, `--gpusort` | GPU-accelerated write/sort/marking controls. |

## Validation

- Input BAM/CRAM resolves inside the container and is queryname sorted as
  required.
- Output duplicate-marked BAM/CRAM is created.
- Metrics output is present when requested.

## Guardrails

- Do not use `markdup` for sorting; route to `pbrun-bamsort.md` with
  `--sort-order queryname` if sorting is needed first.
- Do not remove duplicates unless the user explicitly requests a tool/version
  option that does that.
- Do not assume metrics are generated unless requested and supported.
