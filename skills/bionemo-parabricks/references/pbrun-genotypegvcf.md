# Parabricks genotypegvcf

## First Steps

1. Confirm the input is GVCF. If the user has BAM/CRAM or FASTQ, route to
   variant-calling references in this skill first.
2. Collect required inputs:
   - Input GVCF path or cohort input expected by the selected version.
   - Reference FASTA if required.
   - Output VCF path.
3. Ask about intervals, sample/cohort structure, temporary directory, and logs
   only when relevant.
4. If the GVCF is not indexed and the workflow requires an index, route to
   `pbrun-indexgvcf.md`.

## Command Shape

```bash
pbrun genotypegvcf \
  <version-specific-gvcf-input-options> \
  <version-specific-output-options>
```

Verify exact GVCF input, reference, interval, and output flags against the
selected version.

## GenotypeGVCFs Option Mapping

Use this mapping when translating a GATK `GenotypeGVCFs` command to
`pbrun genotypegvcf`. Parabricks documents this as an accelerated
GATK GenotypeGVCFs counterpart for converting one or more gVCF inputs to VCF.

| GATK option | `pbrun genotypegvcf` equivalent | Notes |
| --- | --- | --- |
| `--variant`, `-V` | `--in-gvcf` | Required input gVCF/gVCF.GZ. |
| `--output`, `-O` | `--out-vcf` | Required output VCF. |
| GATK intervals, annotation controls, cloud flags, `--arguments_file`, and Java options | No direct equivalent | Not exposed by current Parabricks docs for `genotypegvcf`. |

## genotypegvcf Options Without GenotypeGVCFs Equivalents

| `pbrun genotypegvcf` option | Why it has no GATK GenotypeGVCFs equivalent |
| --- | --- |
| `--num-threads` | Parabricks worker-thread count. |

## Validation

- Required GVCF index and reference files exist.
- Output VCF exists and is indexed when requested.

## Guardrails

- Do not present `genotypegvcf` as a read-level variant caller.
- Do not invent cohort/sample assumptions.
- Treat multi-sample behavior and input list formats as version-sensitive.
