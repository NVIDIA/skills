# Parabricks applybqsr

## First Steps

1. Confirm the user already has a BQSR recalibration report. If not, route to
   `pbrun-bqsr.md`.
2. Collect required inputs:
   - Reference FASTA path.
   - Input BAM or CRAM path.
   - Input recalibration report path.
   - Output BAM or CRAM path.
3. Ask about optional interval files, temporary directory, logs, and GPU count
   only when relevant.

## Command Shape

```bash
pbrun applybqsr \
  --ref /workdir/<reference.fa> \
  --in-bam /workdir/<input.bam> \
  --in-recal-file /workdir/<input.recal.txt> \
  --out-bam /outputdir/<recalibrated.bam>
```

Verify option names against the tool reference for the selected version before
finalizing interval, threading, logging, or temporary-directory flags.

## ApplyBQSR Option Mapping

Use this when translating GATK `ApplyBQSR` to `pbrun applybqsr`. Shared
alignment and interval mappings for `applybqsr` are in
[shared-options.md](shared-options.md).

| GATK option | `pbrun applybqsr` equivalent | Notes |
| --- | --- | --- |
| `--bqsr-recal-file` | `--in-recal-file` | Recalibration report from `pbrun bqsr` or GATK `BaseRecalibrator`. |
| `--arguments_file` | No direct equivalent | Argument-file expansion is not documented for `pbrun applybqsr`. |
| `--read-index`, `--read-validation-stringency`, common GATK engine flags | No direct equivalent | GATK engine controls are not exposed in current docs. |

Parabricks execution and wrapper controls for `applybqsr` are in
[shared-options.md](shared-options.md) and
[command-conventions.md](command-conventions.md#shared-wrapper-controls).

## Validation

- Output BAM/CRAM is created in the mounted output directory.
- The output is used downstream instead of the original unrecalibrated input.

## Guardrails

- Do not generate a recalibration report here; use `bqsr` for that step.
- Do not apply a recalibration report from a different reference/sample unless
  the user explicitly confirms compatibility.
- Do not invent intervals, output names, or runtime estimates.
