# Parabricks tool index

This is the shared directory of tool purposes and official manuals; tool names
link to NVIDIA's documentation. Use it when selecting a tool or locating its
manual. Individual references contain the command details and constraints.
Read only the selected tool reference and its relevant shared mappings.

## Tool Reference Index

### FASTQ/BAM Processing

| Parabricks Tool | Skill Reference | Use when |
|------|-----------|----------|
| [`applybqsr`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/applybqsr) | [pbrun-applybqsr.md](pbrun-applybqsr.md) | Apply a BQSR recalibration table to aligned BAM/CRAM after `bqsr`. |
| [`bam2fq`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/bam2fq) | [pbrun-bam2fq.md](pbrun-bam2fq.md) | Convert BAM input to FASTQ output. |
| [`bamsort`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/bamsort) | [pbrun-bamsort.md](pbrun-bamsort.md) | Standalone BAM sort (CRAM v3.1 read support documented for recent releases). |
| [`bqsr`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/bqsr) | [pbrun-bqsr.md](pbrun-bqsr.md) | Generate base quality score recalibration data for later `applybqsr`. |
| [`fq2bam`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/fq2bam) | [pbrun-fq2bam.md](pbrun-fq2bam.md) | Raw paired short-read DNA FASTQ → aligned BAM/CRAM with common preprocessing (align/sort/dedup/BQSR path). |
| [`fq2bam_meth`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/fq2bam_meth) | [pbrun-fq2bam_meth.md](pbrun-fq2bam_meth.md) | Bisulfite or methylation FASTQ workflows — not standard DNA `fq2bam`. |
| [`giraffe`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/giraffe) | [pbrun-giraffe.md](pbrun-giraffe.md) | Pangenome graph alignment with vg giraffe; produce aligned BAM for pangenome-aware variant calling. |
| [`markdup`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/markdup) | [pbrun-markdup.md](pbrun-markdup.md) | Mark duplicate reads in aligned BAM/CRAM when not folded into `fq2bam`. |
| [`minimap2`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/minimap2) | [pbrun-minimap2.md](pbrun-minimap2.md) | Long-read FASTQ alignment (CRAM v3.1 read support documented for recent releases). |

### Variant Calling

| Parabricks Tool | Skill Reference | Use when |
|------|-----------|----------|
| [`deepsomatic`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/deepsomatic) | [pbrun-deepsomatic.md](pbrun-deepsomatic.md) | Tumor/normal or tumor-only somatic calling with DeepSomatic (CRAM v3.1 reads documented for recent releases). |
| [`deepvariant`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/deepvariant) | [pbrun-deepvariant.md](pbrun-deepvariant.md) | DeepVariant from BAM/CRAM; short-read germline recall alternative to `haplotypecaller`. |
| [`deepvariant_germline`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/deepvariant_germline) | [pbrun-deepvariant_germline.md](pbrun-deepvariant_germline.md) | FASTQ-in DeepVariant germline pipeline when the user wants DeepVariant end-to-end from reads. |
| [`germline`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/germline) | [pbrun-germline.md](pbrun-germline.md) | FASTQ-in GATK-style germline pipeline; do not route existing BAM/CRAM here without version-verified docs. |
| [`haplotypecaller`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/haplotypecaller) | [pbrun-haplotypecaller.md](pbrun-haplotypecaller.md) | GATK HaplotypeCaller-compatible germline calling from BAM/CRAM (primary short-read variant recall path). |
| [`mutectcaller`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/mutectcaller) | [pbrun-mutectcaller.md](pbrun-mutectcaller.md) | Tumor/normal or tumor-only somatic calling with Mutect2-compatible caller (CRAM v3.1 reads documented for recent releases). |
| [`ont_germline`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/ont_germline) | [pbrun-ont_germline.md](pbrun-ont_germline.md) | Oxford Nanopore long-read germline workflow. |
| [`pacbio_germline`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/pacbio_germline) | [pbrun-pacbio_germline.md](pbrun-pacbio_germline.md) | PacBio long-read germline workflow. |
| [`pangenome_aware_deepvariant`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/pangenome_aware_deepvariant) | [pbrun-pangenome_aware_deepvariant.md](pbrun-pangenome_aware_deepvariant.md) | Pangenome-aware DeepVariant; for Roche SBX-D/SBX-Fast BAMs use `--sbx` (verify in docs/MCP). CRAM v3.1 reads documented for recent releases. |
| [`pangenome_germline`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/pangenome_germline) | [pbrun-pangenome_germline.md](pbrun-pangenome_germline.md) | Pangenome-aware germline workflow with graph resources. |
| [`postpon`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/postpon) | [pbrun-postpon.md](pbrun-postpon.md) | Annotate Mutectcaller variants using an existing panel of normals (PON). |
| [`prepon`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/prepon) | [pbrun-prepon.md](pbrun-prepon.md) | Prepare an existing PON for `mutectcaller --pon`. |
| [`somatic`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/somatic) | [pbrun-somatic.md](pbrun-somatic.md) | Tumor/normal somatic variant calling pipeline (caller family alternative to `mutectcaller` / `deepsomatic`). |

### RNA

| Parabricks Tool | Skill Reference | Use when |
|------|-----------|----------|
| [`rna_fq2bam`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/rna_fq2bam) | [pbrun-rna_fq2bam.md](pbrun-rna_fq2bam.md) | RNA-seq FASTQ → splice-aware aligned BAM (STAR-based workflow). `--ref` and the STAR `--genome-lib-dir` must be the same assembly (contig renaming cannot fix a mismatch); STAR 2.7.2a compatibility baseline. |
| [`starfusion`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/starfusion) | [pbrun-starfusion.md](pbrun-starfusion.md) | Fusion detection with STAR-Fusion from chimeric junction input and genome library. |

### Quality Control

| Parabricks Tool | Skill Reference | Use when |
|------|-----------|----------|
| [`bammetrics`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/bammetrics) | [pbrun-bammetrics.md](pbrun-bammetrics.md) | BAM metrics and QC on existing alignments — not variant calling or coverage repair. |
| [`collectmultiplemetrics`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/collectmultiplemetrics) | [pbrun-collectmultiplemetrics.md](pbrun-collectmultiplemetrics.md) | Multiple Picard/GATK-style alignment metrics — not variant recall from BAM/CRAM. |

### Variant and GVCF Processing

| Parabricks Tool | Skill Reference | Use when |
|------|-----------|----------|
| [`dbsnp`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/dbsnp) | [pbrun-dbsnp.md](pbrun-dbsnp.md) | dbSNP annotation or variant file processing support. |
| [`genotypegvcf`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/genotypegvcf) | [pbrun-genotypegvcf.md](pbrun-genotypegvcf.md) | Joint-genotype GVCF input(s) into VCF after indexing. |
| [`indexgvcf`](https://docs.nvidia.com/clara/parabricks/tool-reference/tools/indexgvcf) | [pbrun-indexgvcf.md](pbrun-indexgvcf.md) | Index GVCF before `genotypegvcf` or consolidation workflows. Writes `<input>.tbi` beside the input (no `--out`); for read-only inputs, index a writable copy after checking for existing files. |

**CRAM v3.1 reads:** Documented for `bamsort`, `deepsomatic`, `deepvariant`,
`haplotypecaller`, `minimap2`, `mutectcaller`, and `pangenome_aware_deepvariant`
for recent Parabricks releases. Verify in NVIDIA docs or MCP before routing
CRAM v3.1 inputs to other tools.


## Routing Heuristics

- Raw paired FASTQ to aligned BAM/CRAM: start with `fq2bam`.
- Raw RNA-seq FASTQ to aligned BAM: consider `rna_fq2bam`.
- Methylation FASTQ workflows: consider `fq2bam_meth`.
- Long-read FASTQ alignment: consider `minimap2`.
- Pangenome graph alignment: consider `giraffe`.
- Short-read germline variant calling from FASTQ: consider `germline` or
  `deepvariant_germline` depending on the desired caller.
- Short-read germline variant calling from BAM: consider `haplotypecaller` or
  `deepvariant`.
- Tumor/normal or tumor-only somatic calling: consider `somatic`,
  `mutectcaller`, or `deepsomatic` depending on the caller requested.
- PacBio germline data: consider `pacbio_germline`.
- Oxford Nanopore germline data: consider `ont_germline`.
- Pangenome-aware alignment or calling: consider `giraffe`,
  `pangenome_germline`, or `pangenome_aware_deepvariant`.
- Mutect panel-of-normals preparation and annotation: use `prepon` before
  calling with `--pon`, and `postpon` for subsequent PON annotation.
- Existing BAM QC: consider `bammetrics` or `collectmultiplemetrics`.
- GVCF consolidation or genotyping: consider `indexgvcf` and `genotypegvcf`.
- dbSNP annotation or variant processing: consider `dbsnp`.

## Key References

- Tool index (4.7.0 baseline): <https://archive.docs.nvidia.com/clara/parabricks/4.7.0/ToolReference.html>
- Performance notes (4.7.0):
  <https://archive.docs.nvidia.com/clara/parabricks/4.7.0/GettingStarted/BestPerformance.html>
- Getting started and deployment (4.7.0):
  <https://archive.docs.nvidia.com/clara/parabricks/4.7.0/GettingStarted.html>
- Other releases: <https://docs.nvidia.com/clara/parabricks/about-parabricks/release-notes>
