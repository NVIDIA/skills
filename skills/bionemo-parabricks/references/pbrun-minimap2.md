# Parabricks minimap2

## First Steps

1. Confirm the sequencing technology and use case:
   - Oxford Nanopore.
   - PacBio HiFi or CLR.
   - Splice-aware long-read RNA alignment if supported by the selected version.
2. Collect required inputs:
   - Reference FASTA path or minimap2 index path.
   - Input FASTQ or BAM/CRAM path.
   - Output BAM or CRAM path.
3. Ask about presets, known-sites, interval files, read groups, and temporary
   directory only when relevant and supported.

## Command Shape

```bash
pbrun minimap2 \
  --ref /workdir/<reference.fa> \
  --in-fq /workdir/<long_reads.fastq.gz> \
  --out-bam /outputdir/<aligned.bam>
```

Verify exact preset, index, BAM/CRAM input, known-sites, and output options
against the selected version before finalizing.

## minimap2/GATK Option Mapping

Use this mapping when translating a baseline minimap2 plus GATK/Picard
post-processing workflow to `pbrun minimap2`. Parabricks documents
`minimap2` as a GPU long-read alignment workflow compatible with minimap2
v2.31. It can sort, mark duplicates, and optionally run BQSR, so the CLI maps
to several upstream commands rather than one single tool.

| Baseline option | `pbrun minimap2` equivalent | Notes |
| --- | --- | --- |
| `minimap2 <reference>` | `--ref` | Required reference FASTA path. |
| `minimap2 <reference> <R1> <R2>` paired-end FASTQ | `--in-fq <R1> <R2>` | Two files per `--in-fq` are aligned as a pair; one file is single-end. Interleaved FASTQ is not supported. Pairing is new in 4.8.0; do not rely on it in 4.7.1. |
| BAM/CRAM input for preprocessing workflow | `--in-bam` | Parabricks input BAM/CRAM mode. |
| `minimap2 -x` | `--preset` | Parabricks supports documented presets such as `map-pbmm2`, `map-hifi`, `map-ont`, `lr:hq`, `splice`, `splice:hq`, and `splice:sr`. |
| `minimap2 -t` | `--num-threads` | Processing thread count. |
| `minimap2 -k` | `--minimizer-kmer-len`, `-k` | Minimizer k-mer length. |
| `minimap2 -uf` | `--forward-transcript-strand`, `-uf` | Splice preset strand control. |
| `minimap2 -ub` | `--both-strands`, `-ub` | Splice preset strand control. |
| minimap2 splice junction BED options | `--jump-bed`, `-j`; `--junc-bed` | Parabricks splice annotation inputs. |
| `minimap2 --MD` / `--md` | `--md` | Output MD tag. |
| `minimap2 --eqx` | `--eqx` | Write `=/X` CIGAR operators. |
| `minimap2 -y` | `--copy-comment`, `-y` | Append FASTQ comment to BAM output. |
| `minimap2 -R` | `--read-group-*` flags | Parabricks builds read group fields from explicit tags rather than accepting the raw minimap2 string. |
| `minimap2 -a` SAM output piped to sorting | `--out-bam` | Parabricks writes sorted BAM/CRAM output. |
| GATK/Picard `MarkDuplicates -M` | `--out-duplicate-metrics` | Duplicate metrics output after marking duplicates when applicable. |
| GATK `BaseRecalibrator --intervals`, `-L` | `--interval`, `--interval-file` | Parabricks separates inline intervals from interval files. |
| GATK/Picard `--version` | `--version` | Same version-reporting role. |
| Upstream minimap2 options not listed here | No direct equivalent | Not exposed by current Parabricks docs for `pbrun minimap2`. |

## minimap2 Options Without minimap2/GATK Equivalents

Pipeline modes and GPU tuning shared with the long-read callers are in
[minimap2-controls.md](minimap2-controls.md). Memory and output controls are in
[shared-options.md](shared-options.md); use rows scoped to `minimap2`.

| `pbrun minimap2` option | Why it has no direct baseline equivalent |
| --- | --- |
| `--out-qc-metrics-dir` | Parabricks workflow output for QC metrics. |

## Validation

- Output BAM/CRAM is created.
- Selected preset matches sequencing technology and analysis goal.

## Guardrails

- Do not use `minimap2` for ordinary short-read BWA-MEM alignment; route to
  `pbrun-fq2bam.md`.
- Do not infer ONT/PacBio presets from filename alone.
- Do not claim variant-calling completion; this skill only covers alignment and
  related preprocessing available in `minimap2`.
