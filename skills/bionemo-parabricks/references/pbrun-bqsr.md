# Parabricks bqsr

## First Steps

1. Collect required inputs:
   - Reference FASTA path.
   - Input BAM or CRAM path.
   - One or more known-sites VCF paths.
   - Output recalibration report path.
2. Ask about optional interval files, temporary directory, logs, and GPU count
   only when relevant.
3. If the user wants to modify BAM qualities after report generation, route the
   next step to `pbrun-applybqsr.md`.

## Command Shape

```bash
pbrun bqsr \
  --ref /workdir/<reference.fa> \
  --in-bam /workdir/<input.bam> \
  --knownSites /workdir/<known-sites.vcf.gz> \
  --out-recal-file /outputdir/<sample.recal.txt>
```

Verify exact known-sites, interval, threading, logging, and temporary-directory
flags against the selected version before finalizing.

## BaseRecalibrator Option Mapping

Use this when translating GATK `BaseRecalibrator` to `pbrun bqsr`. Shared
reference, BAM, and interval mappings for `bqsr` are in
[shared-options.md](shared-options.md).

| GATK option | `pbrun bqsr` equivalent | Notes |
| --- | --- | --- |
| `--known-sites` | `--knownSites` | Required known-sites VCF; can be repeated. |
| `--output`, `-O` | `--out-recal-file` | Required recalibration report output. |
| `--arguments_file`, GATK engine/read-filter flags, covariate-control flags | No direct equivalent | Not exposed in current docs. |

Parabricks execution and wrapper controls for `bqsr` are in
[shared-options.md](shared-options.md) and
[command-conventions.md](command-conventions.md#shared-wrapper-controls).

## Validation

- Known-sites files match the reference build.
- Output recalibration report is created.

## Guardrails

- Do not invent known-sites files or reference builds.
- Do not present the recalibration report as an updated BAM; applying it is a
  separate step handled by `applybqsr`.
- Ask whether BQSR is appropriate for the organism/reference when the user is
  not working with a standard human reference workflow.
