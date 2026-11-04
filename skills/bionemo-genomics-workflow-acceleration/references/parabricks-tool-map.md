# Parabricks tool map (framework-agnostic)

Map each step's biological intent and CPU tool to the Parabricks equivalent (`pbrun`
subcommand). Confirm flags and I/O against [NVIDIA Parabricks documentation](https://docs.nvidia.com/clara/parabricks/latest).

| Typical CPU / GATK-style step | Parabricks tool | Notes |
|------------------------------|-----------------|-------|
| BWA-MEM + sort + mark duplicates (+ optional BQSR) | `pbrun fq2bam` | Often replaces align + sort + dedup (+ BQSR) in one step |
| Apply BQSR | `pbrun applybqsr` | After fq2bam when BQSR table produced |
| GATK HaplotypeCaller (germline) | `pbrun haplotypecaller` | Germline SNV/indels |
| DeepVariant | `pbrun deepvariant` | Germline variant calling |
| Index gVCF | `pbrun indexgvcf` | gVCF indexing |
| Bisulfite alignment | `pbrun fq2bam_meth` | Methylation workflows |
| Mutect2 / somatic SNV | `pbrun mutectcaller` | Somatic — confirm version support in docs |
| RNA-seq alignment | `pbrun rna_fq2bam` | Use `parabricks` skill for RNA-specific flags |
| No Parabricks equivalent | — | **Keep original CPU step**; document gap in report |

## How this relates to nf-core

| Framework | Preferred integration |
|-----------|----------------------|
| **Nextflow / nf-core** | [nf-core Parabricks modules](https://nf-co.re/modules/) — see [nf-core-parabricks-map.md](nf-core-parabricks-map.md) |
| **Snakemake, WDL, Python, shell** | Wrap `pbrun` in rules/tasks/scripts; use nf-core module docs as **I/O reference** only |

Do not require converting Snakemake/WDL/Python pipelines to Nextflow unless the user asks.

## Inspection signals (grep / read)

| Signal | Likely tool / step |
|--------|-------------------|
| `bwa mem`, `bwa-mem2` | Alignment → fq2bam candidate |
| `gatk MarkDuplicates`, `picard MarkDuplicates` | Dedup — may fold into fq2bam |
| `gatk BaseRecalibrator`, `ApplyBQSR` | BQSR chain |
| `gatk HaplotypeCaller`, `HaplotypeCaller` | haplotypecaller |
| `deepvariant` | deepvariant |
| `pbrun` already present | Note version and which subcommands |

## Step consolidation (after 1:1 mapping)

A 1:1 swap may leave **redundant** GPU-branch stages. Review merges using
[step-consolidation.md](step-consolidation.md). Canonical example: **one** `fq2bam`
instead of separate align + MarkDuplicates + BQSR on the GPU branch.

## When no mapping exists

- State clearly in the acceleration report and `ACCELERATION.md`.
- Keep the original CPU step as the only path for that stage.
- Do not force an unsuitable GPU substitution or invent tool names.
