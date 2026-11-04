# Parabricks dbsnp

## First Steps

1. Confirm the task is dbSNP annotation or processing of existing variant data.
   If the user needs to call variants from reads, route to
   variant-calling references in this skill.
2. Collect required inputs for the selected version:
   - Input VCF path.
   - dbSNP/resource VCF path.
   - Reference FASTA or reference build if required.
   - Output VCF path.
3. Ask about compression, indexing, intervals, and logs only when relevant.

## Command Shape

```bash
pbrun dbsnp \
  <version-specific-input-options> \
  <version-specific-dbsnp-resource-options> \
  <version-specific-output-options>
```

Verify exact input, dbSNP resource, reference, and output flags against the
selected version.

## dbSNP Annotation Option Mapping

Use this mapping when translating a VCF dbSNP annotation step to `pbrun dbsnp`.
Parabricks documents `dbsnp` as a VCF annotation tool using a dbSNP VCF
resource.

| Baseline option | `pbrun dbsnp` equivalent | Notes |
| --- | --- | --- |
| Input VCF | `--in-vcf` | Required input VCF. |
| dbSNP/resource VCF | `--in-dbsnp-file` | Required dbSNP VCF.GZ with tabix index. |
| Output annotated VCF | `--out-vcf` | Required output VCF. |
| Docker volume/workdir options | Docker `--volume` / `--workdir` outside `pbrun` | Container launch options, not `pbrun dbsnp` flags. |
| GATK/BCFtools annotation flags not listed here | No direct equivalent | Not exposed by current Parabricks docs for `dbsnp`. |

## Validation

- dbSNP resource matches the same reference build as the input VCF.
- Output VCF exists and is indexed when requested.

## Guardrails

- Do not invent dbSNP resource paths or builds.
- Do not use this skill for BAM/FASTQ variant calling.
- Verify annotation semantics in the selected release's manual.
