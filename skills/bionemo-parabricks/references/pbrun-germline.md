# Parabricks germline

## First Steps

1. Confirm whether the user wants an end-to-end GATK-style germline pipeline or
   a standalone `haplotypecaller` run.
2. Collect required inputs:
   - Reference FASTA.
   - Paired or single-end FASTQs.
   - Output BAM and VCF/gVCF paths.
   - Known-sites/resources for BQSR when used.
3. Ask for read groups, intervals, ploidy, temporary directory, and logs when
   relevant.
4. For preprocessing-only questions, route to
   `pbrun-fq2bam.md` and related FASTQ/BAM references; for runtime readiness, use
   `runtime-environment.md`.
   For prepared BAM/CRAM input, use `pbrun-haplotypecaller.md`; the 4.7.1
   germline pipeline does not document `--in-bam`.

## Command Shape

```bash
pbrun germline \
  --ref /workdir/<reference.fa> \
  --in-fq /workdir/<sample_R1.fastq.gz> /workdir/<sample_R2.fastq.gz> \
  --out-bam /outputdir/<sample.bam> \
  --out-variants /outputdir/<sample.vcf.gz>
```

Verify exact input, known-sites, output, interval, and ploidy flags against the
selected version.

## BWA-MEM/GATK Germline Option Mapping

Use this when translating a baseline BWA-MEM plus GATK germline workflow to
`pbrun germline`. Preprocessing mappings (--ref, FASTQ inputs, BAM output, BWA options,
markdup/BQSR, intervals, and shared execution controls) for `germline` are in
[shared-options.md](shared-options.md).

| Baseline option | `pbrun germline` equivalent | Notes |
| --- | --- | --- |
| GATK `HaplotypeCaller --output`, `-O` | `--out-variants` | VCF/gVCF output. |
| GATK `HaplotypeCaller --emit-ref-confidence GVCF` | `--gvcf` | gVCF mode. |
| GATK `HaplotypeCaller --sample-ploidy` | `--ploidy` | Haploid/diploid where documented. |
| GATK `HaplotypeCaller` annotation/output-mode/pruning controls | `--haplotypecaller-options` | Pass supported HaplotypeCaller options as one string. |
| `--java-options`, GATK engine/common flags | No direct equivalent | Not exposed as GATK engine controls. |

## germline Options Without BWA-MEM/GATK Equivalents

Shared Parabricks pipeline and GPU controls for `germline` are in
[shared-options.md](shared-options.md). BWA tuning is in [performance.md](performance.md).

| `pbrun germline` option | Why it has no direct baseline equivalent |
| --- | --- |
| `--run-partition` | Genome partitions for HaplotypeCaller (one process per partition). Not fq2bam GPU-process partitioning. |
| `--bwa-gpu-num-per-partition` | BWA GPUs per worker when `--run-partition` is set (default: 2; must divide `--num-gpus`). |
| `--num-streams-per-gpu` | HaplotypeCaller streams (default: 1). Not a partition flag. |

## Validation

- Inputs and reference resources match the same build.
- Read groups are explicit when FASTQs are used.
- Output VCF/gVCF exists and is indexed when requested.

## Guardrails

- Do not use `germline` for somatic calling.
- Do not invent known-sites or ploidy settings.
- Do not pass `--sample-sex`, `--range-male`, `--range-female`, or
  `--use-GRCh37-regions` (removed in v4.7.1).
