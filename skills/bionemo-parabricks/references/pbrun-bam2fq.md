# Parabricks bam2fq

## First Steps

1. Collect required inputs:
   - Input BAM or CRAM path.
   - Output FASTQ path or output prefix/directory expected by the selected
     version.
   - Reference FASTA when CRAM input or the selected version requires it.
2. Ask whether reads are paired-end, single-end, interleaved, or should be
   emitted as separate FASTQ files.
3. Ask whether the user needs output compression, read group filtering, or
   unmapped read handling, then verify version-specific support.

## Preserving reads during conversion

For Parabricks 4.7.0, paired reads use `--out-suffixF` and `--out-suffixF2`.
Orphan first mates, orphan second mates, and unpaired reads are **ignored**
unless `--out-suffixO`, `--out-suffixO2`, and `--out-suffixS`, respectively, are
supplied. A single unpaired output does not capture both orphan categories.
Every suffix must end in `.gz`; changing the suffix to `.fastq` does not request
uncompressed output. The paired defaults are `_1.fastq.gz` and `_2.fastq.gz`.

To retain reads marked as QC failures, omit `--remove-qc-failure`. For read-group
splitting, `--rg-tag` accepts only `PU` or `ID`, not `SM` or `LB`; choose the tag
that represents the user's requested grouping. Paired read names acquire `/1`
and `/2` suffixes. Check matching names and counts after conversion, allowing
these suffixes, and account for the separate orphan and unpaired outputs.

When converting CRAM for realignment to a new assembly, `bam2fq --ref` must
match the **source CRAM's** reference. Use the new reference only at the later
alignment step; it cannot decode a CRAM made against a different sequence.
Retaining these read categories still does not recover reads or bases already
removed before the source alignment was produced.

## Command Shape

```bash
pbrun bam2fq \
  --ref /workdir/<reference.fa> \
  --in-bam /workdir/<input.bam> \
  --out-prefix /outputdir/<sample_prefix>
```

For BAM input, `--ref` may be optional depending on the version. For CRAM input,
keep the reference explicit. Check the tool reference before finalizing suffix,
read group splitting, QC filtering, temporary directory, or threading flags.

## SamToFastq Option Mapping

Use this mapping when translating a GATK/Picard `SamToFastq` command to
`pbrun bam2fq`. Parabricks documents `bam2fq` as the GPU counterpart for
converting BAM/CRAM to FASTQ, but it uses an output prefix plus suffix flags
instead of Picard's individual output filenames.

| GATK/Picard option | `pbrun bam2fq` equivalent | Notes |
| --- | --- | --- |
| `--FASTQ`, `-F` | `--out-prefix` plus `--out-suffixF` | Parabricks builds first-in-pair output from prefix and suffix. |
| `--SECOND_END_FASTQ`, `-F2` | `--out-prefix` plus `--out-suffixF2` | Parabricks builds second-in-pair output from prefix and suffix. |
| `--UNPAIRED_FASTQ`, `-FU` | `--out-prefix` plus `--out-suffixS` | Single-end/unpaired output suffix. |
| `--UNPAIRED_FASTQ` / orphan handling | `--out-suffixO`, `--out-suffixO2` | Parabricks can emit orphan first/second reads with separate suffixes. |
| `--REFERENCE_SEQUENCE`, `-R` | `--ref` | Required for CRAM input; optional for BAM depending on version. |
| `--READ_GROUP_TAG` | `--rg-tag` | Parabricks documents `PU` or `ID` for splitting reads into different FASTQ files. |
| `--INCLUDE_NON_PF_READS false` | `--remove-qc-failure` | Partial equivalent: Parabricks removes reads marked as QC failure when set. |
| `--arguments_file`, `--VALIDATION_STRINGENCY`, Picard compression/common flags | No direct equivalent | Not exposed by current Parabricks docs. |

## bam2fq Options Without SamToFastq Equivalents

| `pbrun bam2fq` option | Why it has no SamToFastq equivalent |
| --- | --- |
| `--num-threads` | Parabricks worker-thread count. |

## Validation

- CRAM runs have the required reference available.
- Output FASTQ files are created where expected.
- Paired-end outputs preserve pairing expectations.

## Guardrails

- Do not assume BAM-to-FASTQ reconstructs original raw FASTQs perfectly.
- Do not infer paired-end output mode from filename alone.
- Do not invent output prefixes or compression settings.
