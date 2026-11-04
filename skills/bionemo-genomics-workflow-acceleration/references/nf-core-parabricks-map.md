# Nextflow Parabricks module map (Nextflow only)

For Snakemake, WDL, or Python, use [parabricks-tool-map.md](parabricks-tool-map.md) and
`pbrun` in the native framework.

Load the [Nextflow agent skill](https://github.com/nextflow-io/agent-skills) when
installing or wiring nf-core modules. Ask the user to install it if unavailable.

Confirm I/O against [nf-co.re/modules](https://nf-co.re/modules/) before implementing.

| Typical CPU / GATK-style step | nf-core module | Notes |
|------------------------------|----------------|-------|
| BWA-MEM + sort + mark duplicates (+ optional BQSR) | [parabricks_fq2bam](https://nf-co.re/modules/parabricks_fq2bam/) | **Files must be copied into work dir, not symlinked** |
| Apply BQSR (after fq2bam table) | [parabricks_applybqsr](https://nf-co.re/modules/parabricks_applybqsr/) | Chains with fq2bam BQSR table |
| GATK HaplotypeCaller (germline) | [parabricks_haplotypecaller](https://nf-co.re/modules/parabricks_haplotypecaller/) | GPU germline SNV/indels |
| DeepVariant | [parabricks_deepvariant](https://nf-co.re/modules/parabricks_deepvariant/) | GPU DeepVariant-equivalent |
| Index gVCF | [parabricks_indexgvcf](https://nf-co.re/modules/parabricks_indexgvcf/) | GPU gVCF indexing |
| Bisulfite / methylation alignment | [parabricks_fq2bammeth](https://nf-co.re/modules/parabricks_fq2bammeth/) | Methylation-specific |

## Discovery workflow

1. Grep the workflow for process names and tool images (`bwa`, `gatk`, `HaplotypeCaller`, etc.).
2. Search [nf-co.re/modules](https://nf-co.re/modules/) for `parabricks_` + tool name.
3. Read module **Input/Output** channels — tuples must match accelerated process wiring.
4. Check `compatible_versions.yml` on modules when present.

## In-place integration

- Install modules in the **same pipeline repo**: `nf-core modules install parabricks/fq2bam`
- Add optional processes with `when: params.use_parabricks`; keep CPU processes for default path.
- Assign `label 'gpu'` / `accelerator 1` on Parabricks processes only.
- Use `stageInMode 'copy'` on fq2bam processes when symlink staging would break inputs.

Example include:

```nextflow
include { FQ2BAM } from './modules/nf-core/parabricks/fq2bam/main'
```

## Gaps (no direct nf-core Parabricks module)

Document in the report; do not silently skip:

- RNA-seq quantification, single-cell, structural variants (unless a module exists at inspection time)
- Custom in-house processes
- Symlink-heavy staging without `stageInMode 'copy'`

State "no nf-core Parabricks module found" and link the search URL rather than inventing a module name.

## Consolidation on GPU branch

`parabricks_fq2bam` often replaces **multiple** CPU modules in one process on the GPU
branch. After wiring, review whether separate BWA/MarkDuplicates Parabricks processes
can merge into one fq2bam when `params.use_parabricks` is true. See
[step-consolidation.md](step-consolidation.md).

## References

- [Nextflow agent skills](https://github.com/nextflow-io/agent-skills)
