# Parabricks fq2bam_meth

## First Steps

1. Confirm this is a methylation or bisulfite sequencing workflow. If it is
   standard DNA alignment, route to `pbrun-fq2bam.md`.
2. Collect required inputs:
   - Paired or single-end FASTQ paths.
   - Methylation-compatible reference/genome inputs required by the selected
     version.
   - Output BAM or CRAM path.
   - Read group values when needed.
3. Ask about known-sites, duplicate marking, temporary directory, output format,
   and logs only when relevant and supported.

## Command Shape

```bash
pbrun fq2bam_meth \
  --ref /workdir/<reference.fa> \
  --in-fq /workdir/<sample_R1.fastq.gz> /workdir/<sample_R2.fastq.gz> \
  --out-bam /outputdir/<sample.meth.bam>
```

Verify exact methylation-specific reference, index, read group, and output
options against the selected version before finalizing.

## Performance Guidance

See [performance.md](performance.md) for BWA stream defaults and optional
`fq2bam_meth` tuning flags.

## BWA-Meth/GATK Option Mapping

Use this when translating BWA-Meth plus GATK/Picard bisulfite preprocessing to
`pbrun fq2bam_meth`. Shared baseline mappings for `fq2bam_meth` (markdup/BQSR,
intervals, BWA pass-through, GPU controls, and similar) are in
[shared-options.md](shared-options.md).

| Baseline option | `pbrun fq2bam_meth` equivalent | Notes |
| --- | --- | --- |
| BWA-Meth reference / GATK `--reference`, `-R` | `--ref` | Requires the converted `.bwameth.c2t` reference from prior BWA-Meth conversion. |
| Paired-end FASTQ input | `--in-fq <read1> <read2>` | Paired bisulfite FASTQ; repeat for multiple pairs. |
| Single-end FASTQ input | `--in-se-fq` | Single-end bisulfite FASTQ. |
| FASTQ manifest or file list | `--in-fq-list`, `--in-se-fq-list` | Parabricks manifest form. |
| Alignment-only output | `--align-only` | Stops after alignment; no sort/markdup. |
| GATK/Picard `SortSam --SORT_ORDER coordinate` | Default `fq2bam_meth` behavior | Coordinate sorting is part of the normal workflow. |
| GATK/Picard `--arguments_file`, validation, compression, and cloud-auth common flags | No direct equivalent | Not exposed for `fq2bam_meth` in current docs. |

## fq2bam_meth Options Without BWA-Meth/GATK Equivalents

Shared Parabricks pipeline and performance flags for `fq2bam_meth` are in
[shared-options.md](shared-options.md). BWA tuning is in [performance.md](performance.md).

| `pbrun fq2bam_meth` option | Why it has no direct baseline equivalent |
| --- | --- |
| `--set-as-failed` | Bisulfite-specific QC failure flagging for selected strands. |
| `--do-not-penalize-chimeras` | Bisulfite-specific control to disable the chimeric-alignment QC heuristic. |

## Validation

- Output BAM/CRAM is created.
- The output is appropriate for the user’s downstream methylation workflow.

## Guardrails

- Do not use `fq2bam_meth` for ordinary short-read DNA alignment.
- Do not assume standard `fq2bam` flags all apply to `fq2bam_meth`.
- Do not invent bisulfite/methylation reference preparation steps; verify them
  against docs or user-provided pipeline standards.

## Key References

- <https://docs.nvidia.com/clara/parabricks/get-started/getting-the-best-performance>
