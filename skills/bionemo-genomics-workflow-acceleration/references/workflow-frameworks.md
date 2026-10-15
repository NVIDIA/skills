# Workflow framework detection and in-place acceleration

Supports **Nextflow**, **Snakemake**, **WDL** (Cromwell, miniwdl, Terra-style), and
**pure Python**. Same flow everywhere: inspect → map to Parabricks → add optional
GPU steps **in place** with a runtime toggle (default off) → compare toggle off vs on.

**Generic triggers:** "Make this pipeline faster", "improve price/performance", or
"convert to GPUs" are in scope when the user provides a workflow path. Ask for the
path if missing; do not invent a pipeline.

## 1. Detect framework

| Framework | Typical markers | Inventory focus |
|-----------|-----------------|-----------------|
| **Nextflow** | `main.nf`, `nextflow.config`, `modules/`, `include {` | Processes, channels, nf-core paths |
| **Snakemake** | `Snakefile`, `rules/`, `config.yaml` | Rules, `wrapper:` / `container:` / `conda:`, `resources:` |
| **WDL** | `*.wdl`, `workflow {`, `call`, `import` | Tasks, runtime blocks, docker strings |
| **Python** | `*.py`, `pyproject.toml`, pipeline entrypoints | Subprocess/shell calls, tool CLIs, config dicts |
| **Mixed** | Multiple markers in one repo | List each sub-pipeline; accelerate per subtree |

If ambiguous, ask which entrypoint is canonical.

## 2. Map steps (all frameworks)

1. List computational steps (not config-only files).
2. For each step, identify the **tool** (shell block, container, or conda env).
3. Look up [parabricks-tool-map.md](parabricks-tool-map.md).
4. **Nextflow only:** prefer [nf-core-parabricks-map.md](nf-core-parabricks-map.md).

## 3. Implement in place (by framework)

Default: **one workflow tree** with a toggle. Do not copy to `*-accelerated/` unless
the user explicitly requests a sibling copy (see [workflow-layout.md](workflow-layout.md)).

### Nextflow / nf-core

- Add optional Parabricks processes/modules with `when: params.use_parabricks`.
- Keep existing CPU processes for `when: !params.use_parabricks` (default).
- Prefer nf-core `parabricks/*` modules; `nf-core modules install parabricks/fq2bam` in the same repo.
- GPU `label` / `accelerator` in config for accelerated processes only.
- fq2bam: respect [no symlink staging](https://nf-co.re/modules/parabricks_fq2bam/); use `stageInMode 'copy'` when needed.
- Optional `accelerated.config` or profile: `params.use_parabricks = true`.

### Snakemake

- Add parallel GPU rules alongside existing CPU rules; branch in `rule all` on `config["use_parabricks"]`.
- Default `use_parabricks: false` in `config.yaml`; optional `config.accelerated.yaml` overlay.
- GPU rules example:

```python
rule fq2bam_gpu:
    resources:
        gpu=1
    container:
        "nvcr.io/nvidia/clara/clara-parabricks:VERSION"  # pin with user
    shell:
        "pbrun fq2bam ..."
```

- Do not remove or rewrite CPU rules when first adding acceleration.

### WDL

- Branch on workflow input `Boolean use_parabricks = false`.
- GPU runtime only on Parabricks tasks:

```wdl
runtime {
  docker: "nvcr.io/nvidia/clara/clara-parabricks:VERSION"
  gpuCount: 1
}
```

- Cromwell vs Terra backends differ — ask which executor if GPU keys vary.

### Pure Python

- Add `--use-parabricks` (default off) or `USE_PARABRICKS` env.
- Branch subprocess calls; keep original CPU functions as default path.

```python
def run_haplotypecaller(bam, ref, out_vcf, *, use_parabricks: bool) -> None:
    if use_parabricks:
        run(["pbrun", "haplotypecaller", ...])
    else:
        run(["gatk", "HaplotypeCaller", ...])
```

## 4. Acceleration report columns

| Column | Content |
|--------|---------|
| Step ID | Rule name / task name / function |
| Framework | nextflow / snakemake / wdl / python |
| Current tool | e.g. bwa mem + gatk MarkDuplicates |
| Parabricks | pbrun subcommand and/or nf-core module |
| Integration | module install / rule branch / WDL if / Python flag |
| GPU | Resource requirements |
| Parity risk | Notes for A/B comparison |

## 5. When nf-core modules do not apply

Snakemake/WDL/Python do not import nf-core modules directly:

1. Use **pbrun** (or official Parabricks container) with equivalent flags.
2. Cite nf-core module pages as **optional I/O templates**.
3. Record "no nf-core wrapper" in `ACCELERATION.md` — not a blocker.

## 6. Comparison (same workflow, toggle off vs on)

See [comparison-checklist.md](comparison-checklist.md). Document in `ACCELERATION.md`:

| Framework | CPU run (toggle off) | GPU run (toggle on) |
|-----------|----------------------|---------------------|
| Nextflow | `nextflow run .` | `nextflow run . -params-file accelerated.config` |
| Snakemake | `snakemake` | `snakemake --configfile config.accelerated.yaml` |
| WDL | `cromwell run pipeline.wdl -i inputs-cpu.json` | `cromwell run pipeline.wdl -i inputs-gpu.json` |
| Python | `python pipeline.py` | `python pipeline.py --use-parabricks` |

Use distinct `--outdir` values (e.g. `results-cpu/` vs `results-gpu/`) to avoid overwriting.
