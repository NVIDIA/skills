# Parabricks haplotypecaller

## First Steps

1. Confirm the user has aligned BAM/CRAM input. If starting from FASTQ and
   wanting a pipeline, consider `pbrun-germline.md`.
2. Collect required inputs:
   - Reference FASTA.
   - Input BAM/CRAM.
   - Output VCF or gVCF.
3. Ask for intervals, ploidy, emit mode, gVCF mode, and logs only when relevant.

## Command Shape

```bash
pbrun haplotypecaller \
  --ref /workdir/<reference.fa> \
  --in-bam /workdir/<input.bam> \
  --out-variants /outputdir/<sample.vcf.gz>
```

Verify exact VCF/gVCF, interval, ploidy, and output flags against the selected
version.

## HaplotypeCaller Option Mapping

Use this mapping when translating a GATK `HaplotypeCaller` command to
`pbrun haplotypecaller`. Parabricks documents this as a
GPU-accelerated HaplotypeCaller counterpart, but the CLI is not one-to-one:
some GATK flags are direct Parabricks flags, while other supported original
HaplotypeCaller options must be passed through `--haplotypecaller-options`.
Use the spelling documented for that inner parser, which differs from GATK's
top-level CLI for some options. For example, 4.7.1 accepts one quoted value
`'-min-pruning 4 -pcr-indel-model NONE'`; those are single-hyphen inner flags.
`--in-recal-file` applies the supplied recalibration table during calling;
creating that table earlier did not itself rewrite the input BAM. A separate
ApplyBQSR BAM is unnecessary when using this caller input.

| GATK option | `pbrun haplotypecaller` equivalent | Notes |
| --- | --- | --- |
| `--input`, `-I` | `--in-bam` | Required BAM/CRAM input. |
| `--output`, `-O` | `--out-variants` | Required VCF/gVCF output. |
| `--bqsr-recal-file` via prior `ApplyBQSR` | `--in-recal-file` | Optional BQSR report input; Parabricks applies updated qualities internally. |
| `--intervals`, `-L` | `--interval` or `--interval-file` | Parabricks separates inline intervals from interval files. |
| `--exclude-intervals`, `-XL` | `--exclude-intervals`, `-XL` | Same exclude-interval role. |
| `--emit-ref-confidence GVCF` | `--gvcf` | Generate gVCF output. |
| `--sample-ploidy` | `--ploidy` | Parabricks currently documents haploid and diploid support. |
| `--annotation`, `-A`; `--annotations-to-exclude`, `-AX`; `--output-mode`; selected assembly/calling knobs | `--haplotypecaller-options` | Pass supported original HaplotypeCaller options as one string. |
| `--annotation-group`, `-G` | `--annotation-group`, `-G` | Supported annotation group output. |
| `--gvcf-gq-bands`, `-GQB` | `--gvcf-gq-bands`, `-GQB` | Reference-confidence GQ bands. |
| `--dont-use-soft-clipped-bases` | `--dont-use-soft-clipped-bases` | Same role. |
| `--minimum-mapping-quality` | `--minimum-mapping-quality` | Same read filtering role. |
| `--mapping-quality-threshold-for-genotyping` | `--mapping-quality-threshold-for-genotyping` | Same genotyping threshold role. |
| `--min-base-quality-score` | `--min-base-quality-score` | Same base quality role. |
| `--max-alternate-alleles` | `--max-alternate-alleles` | Same genotyping cap role. |
| `--disable-read-filter` | `--disable-read-filter` | Limited to filters documented by the selected Parabricks version. |
| `--native-pair-hmm-threads`, `--java-options`, GATK engine/common flags | No direct equivalent | Not exposed as GATK engine controls by current Parabricks docs. |

## haplotypecaller Options Without HaplotypeCaller Equivalents

| `pbrun haplotypecaller` option | Why it has no GATK HaplotypeCaller equivalent |
| --- | --- |
| `--htvc-bam-output` | Parabricks output for assembled haplotypes. |
| `--htvc-alleles` | Parabricks force-call VCF input naming for the HTVC path. |
| `--rna` | Parabricks RNA-optimized mode. |
| `--adaptive-pruning` | Parabricks-exposed graph pruning control. |
| `--force-call-filtered-alleles` | Parabricks force-calling behavior tied to its documented allele input. |
| `--filter-reads-too-long`, `--no-alt-contigs` | Parabricks read/contig filtering conveniences. |
| `--htvc-low-memory`, `--num-htvc-threads`, `--num-streams-per-gpu`, `--run-partition`, `--gpu-num-per-partition` | Parabricks GPU/partition performance controls. `--num-streams-per-gpu` default 1. |

## Validation

- BAM/CRAM and reference match.
- Output VCF/gVCF exists and is indexed when requested.

## Guardrails

- Do not use for somatic tumor/normal calling.
- Do not claim BQSR was performed unless input was already recalibrated or the
  selected pipeline did so.
