# Toggle-off vs toggle-on comparison checklist

Compare the **same workflow** with the runtime toggle **off** (CPU path) vs **on**
(GPU path) using the **same** inputs.

## Run setup

- [ ] Same samplesheet, reference genome build, and interval lists
- [ ] Distinct output dirs (e.g. `results-cpu/` vs `results-gpu/`) — avoid overwriting
- [ ] Document workflow revision, container digests, Parabricks version (`pbrun version`), and **both run commands** in `ACCELERATION.md`
- [ ] GPU run logged: device type, driver, and where it ran (local / HPC / cloud)

Example commands (same entrypoint, different toggle):

| Framework | Toggle off (CPU) | Toggle on (GPU) |
|-----------|------------------|-----------------|
| Nextflow | `nextflow run . -params-file cpu.config` | `nextflow run . -params-file accelerated.config` |
| Snakemake | `snakemake` | `snakemake --config use_parabricks=true` |
| WDL | `cromwell run pipeline.wdl -i inputs-cpu.json` | `cromwell run pipeline.wdl -i inputs-gpu.json` |
| Python | `python pipeline.py` | `python pipeline.py --use-parabricks` |

## Outputs to compare

| Stage | Suggested checks |
|-------|------------------|
| Alignment (fq2bam) | Flagstat, insert size, duplicate rate; spot-check chr depth |
| BQSR | Compare recalibration tables if emitted |
| Variants (HC / DeepVariant) | VCF concordance (e.g. bcftools `isec`); review indel-region differences |
| Runtime | Wall time, GPU utilization (informational, not sole correctness gate) |

## Acceptance

- Define tolerances **with the user** (VCF concordance %, etc.).
- Treat first GPU-enabled run as **validation**, not production cutover.
- File gaps in `ACCELERATION.md` for steps with no Parabricks mapping.

## Reporting template

Record in `ACCELERATION.md` (required). Optional HTML/markdown table if the user
requested automation — see SKILL.md §9.

```markdown
## A/B comparison — [date]

| Metric | Toggle off (CPU) | Toggle on (GPU) | Pass? |
|--------|------------------|-----------------|-------|
| Wall time | | | |
| VCF concordance | | | |
| Notes | | | |
```
