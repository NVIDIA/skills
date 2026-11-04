# Parabricks deepvariant_germline

## First Steps

1. Confirm whether input is FASTQ or prepared BAM/CRAM for the selected
   pipeline mode.
2. Collect required inputs:
   - Reference FASTA.
   - Input FASTQ pairs or BAM/CRAM, depending on mode.
   - Output VCF/gVCF or output directory.
   - Model/resource settings required by the selected version.
3. Ask for read groups, known-sites/resources, intervals, temporary directory,
   and logs when relevant.
4. For alignment-only questions, route to
   `pbrun-fq2bam.md` and related FASTQ/BAM references; for runtime readiness, use
   `runtime-environment.md`.

## Command Shape

```bash
pbrun deepvariant_germline \
  --ref /workdir/<reference.fa> \
  <version-specific-input-options> \
  <version-specific-output-options>
```

Verify exact input mode, model, output, and preprocessing flags against the
selected version.

## BWA-MEM/GATK/DeepVariant Option Mapping

Use this when translating BWA-MEM plus Google DeepVariant germline workflows to
`pbrun deepvariant_germline`. Preprocessing and shared DeepVariant mappings for
`deepvariant_germline` are in [shared-options.md](shared-options.md). The small
model is enabled by default; use `--disable-small-model` to turn it off. For WES,
`--use-wes-model` also disables it, matching Google DeepVariant; in 4.7.1, also
pass `--disable-small-model`.

| Baseline option | `pbrun deepvariant_germline` equivalent | Notes |
| --- | --- | --- |
| DeepVariant `--proposed_variants` | `--proposed-variants` | Candidate/importer VCF input. |
| DeepVariant make-examples options | Matching explicit Parabricks flags such as `--vsc-*`, `--variant-caller`, `--min-*`, `--channel-*` | Parabricks exposes many make-examples options directly. |
| `--java-options`, DeepVariant/GATK engine flags not listed here | No direct equivalent | Not exposed for this pipeline in current docs. |

## deepvariant_germline Options Without BWA-MEM/GATK/DeepVariant Equivalents

Shared pipeline and GPU controls for `deepvariant_germline` are in
[shared-options.md](shared-options.md). BWA tuning is in [performance.md](performance.md).

| `pbrun deepvariant_germline` option | Why it has no direct baseline equivalent |
| --- | --- |
| `--disable-use-window-selector-model`, `--disable-small-model` | Model/compatibility controls; small model is on by default. |

## Validation

- Input mode matches the command options.
- Reference, known resources, model/resources, and intervals match.
- Output VCF/gVCF or output directory is created.

## Guardrails

- Do not confuse this end-to-end/pipeline skill with standalone `deepvariant`.
- Do not invent read groups, model files, or known-sites resources.
