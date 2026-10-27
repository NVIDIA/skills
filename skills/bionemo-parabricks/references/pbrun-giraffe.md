# Parabricks giraffe

## First Steps

1. Confirm the user is doing pangenome graph alignment. If they have ordinary
   short-read DNA FASTQs and no graph resources, route to `pbrun-fq2bam.md`.
2. Collect required inputs for the selected version:
   - FASTQ input paths.
   - Pangenome graph/index/resource files.
   - Reference or auxiliary files required by the workflow.
   - Output BAM/CRAM path and optional metrics/log paths.
3. Ask for read group values when needed.

## Command Shape

```bash
pbrun giraffe \
  <version-specific-graph-input-options> \
  --in-fq /workdir/<sample_R1.fastq.gz> /workdir/<sample_R2.fastq.gz> \
  --out-bam /outputdir/<sample.giraffe.bam>
```

Always verify exact graph resource flags and output options against the selected
version before finalizing. Do not guess graph/index file combinations.

The GBZ graph, distance index, minimizer index, and zipcodes file
are required inputs. Supply `--zipcodes-name` explicitly; do not infer a
compatible zipcodes file from a minimizer filename. Use `--ref-paths` when the
output BAM needs a supplied path order or sequence dictionary.

Giraffe uses vg v1.75.1: generate index files with vg v1.75.1, regenerating any
built with an older vg, and supply a GBZ version 1 graph. vg 1.72.0 and later
write version 2 by default, which fails to load; convert it with
`vg gbwt -Z graph.gbz -g graph-v1.gbz --gbz-v1` (vg 1.74.0 or later). Indexes
built from the same graph stay valid after that conversion. In 4.7.1, giraffe
uses vg v1.70.0; build its indexes with that version.

To produce a **coordinate-sorted BAM without duplicate marking**, use
`--no-markdups`. `--align-only` stops before coordinate sorting and therefore
does not meet that output requirement. Preserve the supplied sample and all
read-group fields using the Giraffe-specific flags in the mapping below.

An `--out-bam` path ending in `.cram` writes CRAM. It requires a
`--ref` FASTA that matches the graph and the paths in `--ref-paths`, cannot be
combined with `--align-only`, and `--run-ref-verification` is recommended the
first time a reference is used. Dropping `--align-only` for CRAM also turns on
duplicate marking; add `--no-markdups` if duplicates should stay unmarked.

## Performance Guidance

Prefer the documented automatic stream selection for general commands: leave
`--nstreams` unset, or set `--nstreams auto` only when making the default
explicit. In current NVIDIA Parabricks documentation, Giraffe auto mode chooses
the number of CUDA streams, batch size, and GPU acceleration options from
available GPU device memory, and can also account for host-memory limits.

Use integer `--nstreams` values only for benchmark-driven or GPU-specific
tuning after confirming the selected Parabricks version's docs. More streams
can improve throughput, but they also increase device and host memory use, so
fixed stream counts should not be part of conservative default command
templates.

## vg giraffe/GATK Option Mapping

Use this mapping when translating a baseline `vg giraffe` plus GATK/Picard
post-processing workflow to `pbrun giraffe`. Parabricks documents
`giraffe` as a GPU pangenome graph aligner that can also sort and mark
duplicates, so not every upstream `vg giraffe` option has a Parabricks
equivalent.

| Baseline option | `pbrun giraffe` equivalent | Notes |
| --- | --- | --- |
| `vg giraffe -Z`, `--gbz-name` | `--gbz-name`, `-Z` | Required GBZ graph input. |
| `vg giraffe -d`, `--dist-name` | `--dist-name`, `-d` | Required distance index. |
| `vg giraffe -m`, `--minimizer-name` | `--minimizer-name`, `-m` | Required minimizer index. |
| `vg giraffe -z`, zipcodes index | `--zipcodes-name`, `-z` | Include when the selected Parabricks version requires a zipcodes file for clustering. |
| `vg giraffe -x`, `--xg-name` | `--xg-name`, `-x` | Optional XG graph used for BAM output. |
| `vg giraffe -g`, `--graph-name` | `--graph-name`, `-g` | Optional GBWTGraph input for mapping. |
| `vg giraffe -H`, `--gbwt-name` | `--gbwt-name`, `-H` | Optional GBWT index input. |
| FASTQ query input | `--in-fq`, `--in-se-fq` | Parabricks paired-end and single-end FASTQ inputs. |
| FASTQ input list | `--in-fq-list`, `--in-se-fq-list` | Parabricks manifest form. |
| `vg giraffe --read-group` | `--read-group` | Read group ID. |
| Read group sample/library/platform/platform-unit fields | `--sample`, `--read-group-library`, `--read-group-platform`, `--read-group-pu` | Parabricks explicit read group tag flags. |
| `vg giraffe --ref-paths` / path list for SAM headers | `--ref-paths` | Path list or HTSlib dictionary for `@SQ` headers. |
| `vg giraffe --prune-low-cplx` | `--prune-low-cplx` | Same low-complexity anchor pruning role. |
| `vg giraffe` fragment length controls | `--max-fragment-length`, `--fragment-mean`, `--fragment-stdev` | Same fragment-distribution role. |
| `vg giraffe --max-multimaps`, `-M` | `--max-multimaps` | Max alignments per read (default: 1). |
| `vg giraffe --copy-comment` | `--copy-comment` | Appends FASTQ comment to BAM output via auxiliary tag. |
| `vg giraffe --supplementary` | `--supplementary` | Identify and report supplementary alignments (off by default). |
| Alignment-only output | `--align-only` | Stops after `vg giraffe` alignment output; does not coordinate-sort. |
| GATK/Picard `SortSam --OUTPUT`, `-O` | `--out-bam` | Final BAM output path, or CRAM for a `.cram` path. |
| GATK/Picard duplicate-marking behavior | `--markdups-*`, `--optical-duplicate-pixel-distance`, `--no-markdups` | Parabricks exposes the duplicate-marking controls documented for this pipeline. |
| GATK/Picard `--version` | `--version` | Same version-reporting role. |
| Upstream `vg giraffe` options not listed here | No direct equivalent | Not exposed by current Parabricks docs for `pbrun giraffe`. |

## giraffe Options Without vg giraffe/GATK Equivalents

| `pbrun giraffe` option | Why it has no direct baseline equivalent |
| --- | --- |
| `--max-read-length`, `--min-read-length` | Parabricks FASTQ filtering/alignment bounds. |
| `--monitor-usage` | Parabricks runtime resource monitoring. |
| `--nstreams`, `--num-cpu-threads-per-gpu`, `--batch-size`, `--write-threads`, `--work-queue-capacity` | Parabricks GPU/CPU pipeline scheduling controls. Prefer `--nstreams auto` for default guidance; use integer stream counts only for benchmarked/manual tuning. |
| `--minimizers-gpu` | Parabricks GPU offload for minimizers/seeds in supported single-end runs. |
| `--memory-limit`, `--low-memory` | Parabricks host/GPU memory controls for the pipeline. |
| `--in-se-cram`, `--in-cram-ref`, `--cram-read-threads` | Single-end CRAM input; aligned reads are re-aligned and need `--in-cram-ref` to decode. Paired-end CRAM is not supported. |
| `--ultima-tags`, `--tp-output-order` | Carry Ultima per-base `tp`/`t0` tags from `--in-se-cram` input to the output. |

## Validation

- Graph resources are from the same build/pangenome bundle expected by the
  workflow.
- Output BAM/CRAM is created.

## Guardrails

- Do not substitute `giraffe` for standard BWA-MEM alignment unless the user
  explicitly wants pangenome graph alignment.
- Do not invent pangenome resource paths or claim compatibility across graph
  bundles without evidence.
- Keep command options version-specific.

## Key References

- <https://docs.nvidia.com/clara/parabricks/get-started/getting-the-best-performance>
