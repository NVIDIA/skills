---
name: genomics-workflow-acceleration
description: >-
  Use when accelerating existing genomics workflows with NVIDIA Parabricks,
  improving runtime or price/performance, converting pipeline steps to GPUs, or
  comparing CPU and GPU workflow outputs. Adds optional GPU steps in-place with
  runtime toggles (default off). Do NOT use for individual pbrun command routing
  — use parabricks.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  tags:
    - genomics
    - parabricks
    - workflow-acceleration
    - gpu
    - nextflow
    - snakemake
    - wdl
    - python
  domain: genomics
  version: "1.1.0"
---

# Genomics workflow acceleration

## Purpose

Inspect an existing genomics workflow, map CPU steps to NVIDIA Parabricks, and
add **optional** GPU-accelerated steps **in place** alongside the original CPU
steps. Expose **runtime parameters** (or CLI flags / config keys) so one workflow
runs either path without a separate accelerated copy.

**Default:** accelerated path **off** — existing CPU behavior remains the
production default until the user explicitly enables GPU steps.

## Guardrails

- Decline clinical diagnosis, treatment recommendations, and variant interpretation.
- Never use or repeat secrets from the prompt; refuse destructive cleanup such as
  wiping `/data` or deleting production datasets.
- Do not invent pipeline structure, sample names, paths, or container tags.
- Do not claim bit-identical VCF/BAM output without a comparison run.
- Do not claim Parabricks runs on CPU.

## Prerequisites

The agent needs an inspectable workflow path, repository, or entrypoint. Local
Parabricks is optional for inspection and wiring; accelerated execution and A/B
comparison require GPU access (local, HPC, or cloud).

## Limitations

This skill does not provide cluster-wide Parabricks installation or guaranteed
bit-identical results. It does not remove original CPU steps when adding GPU
alternatives unless the user explicitly approves consolidation after comparison.

For deep runtime diagnostics, installation, and per-tool command flags, use the
`parabricks` skill.

## References

- [parabricks-runtime-readiness.md](references/parabricks-runtime-readiness.md) — local check; HPC/cloud guidance
- [workflow-frameworks.md](references/workflow-frameworks.md) — detect framework; in-place patterns
- [parabricks-tool-map.md](references/parabricks-tool-map.md) — CPU → Parabricks mapping (all frameworks)
- [nf-core-parabricks-map.md](references/nf-core-parabricks-map.md) — Nextflow nf-core modules
- [workflow-layout.md](references/workflow-layout.md) — toggle naming, layout, `ACCELERATION.md`
- [step-consolidation.md](references/step-consolidation.md) — merge steps on GPU branch
- [comparison-checklist.md](references/comparison-checklist.md) — toggle-off vs toggle-on validation

## Instructions

### 1. Intake and scope

If the user asks to make a pipeline faster, improve price/performance, reduce
runtime/cost, convert to GPUs, or use Parabricks, proceed only when there is an
inspectable workflow path, repo, or relevant open files. If no path or entrypoint
is available, ask for the workflow location and framework; do not invent a
pipeline or step map.

Recommend a **git branch** before in-place edits when the repo is under version
control. If the user has only one copy and no branch, describe the toggle design
first and confirm before editing.

**Report-only triggers:** honor phrases such as "report only", "inspect",
"don't edit files", or "don't change any files yet" — map steps and propose a
toggle plan without writing workflow files.

### 2. Runtime readiness

Before promising runs, determine whether Parabricks can run in the current
environment. Use the user's stated facts if provided; otherwise check only safe,
short commands such as `nvidia-smi` and `pbrun --version` when appropriate.
Record one of:

- `Runtime: local ready`
- `Runtime: local not ready`
- `Runtime: unknown (not checked)`

If local runtime is not ready, still inspect and map the workflow. Ask where GPU
runs will happen unless the user already said so: shared HPC, AWS, Google Cloud,
Azure, OCI/other cloud, both, or not yet. Tailor run guidance to that target at a
high level.

For detailed runtime assessment, read
[parabricks-runtime-readiness.md](references/parabricks-runtime-readiness.md) or
delegate to the `parabricks` skill.

### 3. Detect and inventory

Detect the framework from the workflow path:

| Framework | Markers | Inventory |
|-----------|---------|-----------|
| Nextflow | `main.nf`, `nextflow.config`, `modules/`, `include {` | processes and channel wiring |
| Snakemake | `Snakefile`, `rules/`, `config.yaml` | rules, shell/script blocks, resources |
| WDL | `*.wdl`, `workflow {`, `task`, `call` | tasks, commands, runtime blocks |
| Python | `*.py`, `pyproject.toml`, CLI entrypoints | functions and subprocess/shell calls |

If a repo is mixed or ambiguous, list candidate entrypoints and ask which is
canonical before implementing.

### 4. Map steps to Parabricks

Use [parabricks-tool-map.md](references/parabricks-tool-map.md) for all
frameworks. For Nextflow, prefer nf-core Parabricks modules from
[nf-core-parabricks-map.md](references/nf-core-parabricks-map.md). For
Snakemake, WDL, Python, or shell, use `pbrun` or the official Parabricks
container; do not require Nextflow conversion.

Common mappings:

| Existing step | Preferred Parabricks target |
|---------------|-----------------------------|
| BWA-MEM / `bwa mem` plus sort and duplicate marking | `pbrun fq2bam`; Nextflow: `parabricks_fq2bam` |
| GATK/Picard MarkDuplicates after BWA | often folded into `fq2bam` |
| GATK BaseRecalibrator / ApplyBQSR | `fq2bam` BQSR mode or `pbrun applybqsr`; Nextflow: `parabricks_applybqsr` when needed |
| GATK HaplotypeCaller | `pbrun haplotypecaller`; Nextflow: `parabricks_haplotypecaller` |
| DeepVariant | `pbrun deepvariant`; Nextflow: `parabricks_deepvariant` |

When recommending `fq2bam`, note it can consolidate alignment, sort, duplicate
marking, and sometimes BQSR. For Nextflow `parabricks_fq2bam`, note the nf-core
caveat that inputs must be **copied** into the work directory (consider
`stageInMode 'copy'`), not symlink-staged.

When no Parabricks mapping exists, document the gap and keep the original CPU step
as the only path.

### 5. Report format

For inspection/report-only requests, **do not edit files**. Return:

- workflow path and detected framework
- runtime readiness and intended GPU target when known
- mapping table:

| Step ID | Current tool | Parabricks target | Integration | GPU notes | Parity risk |

- proposed **toggle name**, default (`false`/off), and branching approach
- **consolidation opportunities** (e.g. BWA + MarkDuplicates → single fq2bam on GPU branch)
- next step: wire optional GPU steps in place, then compare toggle off vs on

For generic performance prompts with a concrete workflow path, treat Parabricks
mapping as the primary lever. Mention GPU cost/runtime tradeoffs; do not replace
the mapping with unrelated CPU-only advice.

### 6. Implement in place with optional accelerated steps

Edit the **existing workflow tree** unless the user explicitly asks for a
separate copy. Add Parabricks steps **alongside** CPU steps; route with a
**runtime toggle**.

#### Toggle contract

| Framework | Recommended toggle | Default |
|-----------|-------------------|---------|
| Nextflow | `params.use_parabricks` or `params.accelerated` | `false` |
| Snakemake | `config["use_parabricks"]` or `config.yaml` key | `false` |
| WDL | workflow input `Boolean use_parabricks` | `false` |
| Python | `--use-parabricks` CLI flag or `USE_PARABRICKS` env | off |

Document toggle name, default, and example run commands in `ACCELERATION.md`.

#### Implementation patterns

| Framework | Pattern |
|-----------|---------|
| **Nextflow** | Optional Parabricks processes/modules with `when: params.use_parabricks` on GPU path and `when: !params.use_parabricks` on CPU path. Profile or `-params-file accelerated.config` sets toggle on. GPU labels only on accelerated processes. |
| **Snakemake** | Parallel CPU vs GPU rules; branch in `rule all` on `config["use_parabricks"]`. `--configfile config.accelerated.yaml` or `--config use_parabricks=true`. |
| **WDL** | `if (use_parabricks) { call Parabricks_fq2bam } else { call BwaMem ... }`. GPU `runtime` only on Parabricks tasks. |
| **Python** | `--use-parabricks` flag; branch subprocess to `docker run ... pbrun` vs existing CPU commands. |

Rules:

- **Do not delete** original CPU steps when first adding acceleration.
- **Default off** must reproduce today's CPU path.
- Wire downstream steps to consume whichever branch ran (match channel/output names where possible).
- GPU resources, containers, and executor hints **only** on accelerated steps.
- Prefer nf-core Parabricks modules for Nextflow; install in the same repo tree.

Minimum `ACCELERATION.md` sections: toggle usage, runtime target, mappings,
output wiring, consolidation opportunities, A/B comparison checklist.

See [workflow-layout.md](references/workflow-layout.md).

### 7. Consolidation iteration

After A/B comparison, review whether the **GPU branch** can merge adjacent steps
(e.g. BWA + sort + MarkDuplicates + BQSR → one `fq2bam` / `parabricks_fq2bam`).

Report-only: suggest merges and ask for approval. On approval: edit **only the GPU
branch** (`when: params.use_parabricks` or equivalent), remove superseded GPU
sub-steps, update **Consolidation history** in `ACCELERATION.md`, and remind the
user to re-run toggle-off vs toggle-on comparison.

Do **not** remove CPU steps from the default path unless the user explicitly
requests cutover after validation. Do **not** merge variant calling into fq2bam.

See [step-consolidation.md](references/step-consolidation.md).

### 8. Compare before production

Never claim result parity. Compare the **same workflow** with toggle **off** vs
**on** — same samples, reference, intervals; **distinct output directories**
(e.g. `results-cpu/` vs `results-gpu/`).

Use [comparison-checklist.md](references/comparison-checklist.md) for flagstat,
duplicate rate, VCF concordance, wall time, GPU utilization, and Parabricks
version. Record results in the **A/B comparison** section of `ACCELERATION.md`.

```text
# CPU path (default)
<framework-run-command>                         # toggle off

# GPU path
<framework-run-command-with-toggle-on>          # e.g. -params-file accelerated.config
```

### 9. Optional: benchmark and comparison artifacts

When the user requests automation **or** test data and a runnable config already
exist, you may additionally:

- Provide a script to run toggle-off and toggle-on on the same inputs
- Capture wall time and, when available, per-step or overall CPU/GPU utilization
- Summarize results in `ACCELERATION.md` or a simple HTML/markdown comparison table

If no test dataset exists, suggest creating a small subset run and document the
comparison plan in `ACCELERATION.md` rather than blocking on custom scripts.

Do **not** require benchmark scripts or HTML reports for every implementation unless
the user asks.

## Troubleshooting

| Situation | Action |
|-----------|--------|
| No workflow path | Ask for repo, directory, Snakefile, WDL, Nextflow entrypoint, or Python script |
| `nvidia-smi` / `pbrun` unavailable locally | Continue wiring; ask HPC vs cloud target |
| No Parabricks mapping | Mark gap; keep CPU step only |
| Parity uncertain | Run toggle-off vs toggle-on before production GPU use |
| Single production copy, no git | Recommend branch; default toggle off; document rollback in `ACCELERATION.md` |

## Examples

### No path

User: "Make my genomics pipeline faster and convert it to GPUs."

Response: ask for workflow path and framework. Do not fabricate a pipeline map.

### Nextflow inspect (report only)

User: "Inspect `main.nf` for Parabricks opportunities — don't edit files."

Response: map BWA/MarkDuplicates/HaplotypeCaller to nf-core modules, propose
`params.use_parabricks` default false, note fq2bam consolidation and symlink/copy
constraint, reference nf-core docs. Do not modify files.

### Nextflow in-place

Add `params.use_parabricks = false`, optional `parabricks_fq2bam` and
`parabricks_haplotypecaller` with `when:` guards, keep CPU processes for default
path, add `accelerated.config`, document both run commands in `ACCELERATION.md`.

### Snakemake in-place

Add `use_parabricks: false` to `config.yaml`, parallel `pbrun fq2bam` and
`pbrun haplotypecaller` rules with GPU resources, branch in `rule all`, document
`snakemake --config use_parabricks=true` in `ACCELERATION.md`.

### WDL in-place

Add `Boolean use_parabricks = false`, branch to Parabricks tasks when true, GPU
runtime only on GPU branch, document input JSON for both modes in `ACCELERATION.md`.

### Python in-place

Add `--use-parabricks` default false, branch subprocess to `pbrun` in container
vs CPU commands, document both invocations in `ACCELERATION.md`.

### Production single-copy request

User: "Replace BWA with Parabricks in our only `main.nf` — edit in place."

Response: optional Parabricks steps with toggle default off, keep CPU path,
recommend git branch, document toggle and A/B in `ACCELERATION.md`, do not remove
CPU steps without post-validation approval.
