# Step consolidation (GPU branch review)

After optional GPU steps exist (even a first-pass **1:1** swap on the GPU branch),
prompt the user to **review for merged steps**. Parabricks often **subsumes**
multiple CPU/GATK stages in one `pbrun` call or nf-core module.

Include opportunities in every acceleration report and `ACCELERATION.md` under
**Consolidation opportunities**.

## When to suggest

- After toggle-off vs toggle-on comparison, or at end of inspect-only report.
- When the GPU branch still chains align → sort → dedup → BQSR as separate steps.
- When the first transform only replaced tools **per step** without removing redundant GPU stages.

## When the user approves — implement on GPU branch only

If the user says yes (e.g. "consolidate", "merge those steps", "do the fq2bam merge"):

1. Edit **only the GPU branch** (`when: params.use_parabricks`, `if use_parabricks`, etc.).
2. **Do not** remove or alter CPU/default-path steps.
3. Merge redundant GPU sub-steps into one `fq2bam` / `parabricks_fq2bam`; rewire channels/outputs.
4. Update `ACCELERATION.md` **Consolidation history** (iteration, removed step names).
5. Remind user to re-run **toggle-off vs toggle-on** comparison on a subset.

Do not only suggest — **make the edits** unless the user asked report-only ("don't edit yet").

## Common consolidation patterns

| Original GPU-branch steps | Consolidated Parabricks | nf-core (Nextflow) |
|---------------------------|-------------------------|---------------------|
| BWA-MEM + sort + MarkDuplicates (+ BQSR) | `pbrun fq2bam` | `parabricks_fq2bam` |
| ApplyBQSR only (table from fq2bam) | `pbrun applybqsr` | `parabricks_applybqsr` — only if not fully from fq2bam |

**Example:** separate BWA + MarkDuplicates Parabricks processes on the GPU branch →
**one** `parabricks_fq2bam` when I/O aligns.

## What to deliver

### After suggestion (user has not decided)

1. Merge-candidate table in report / `ACCELERATION.md`.
2. Ask: "Should I apply these merges on the GPU branch?"

### After user approves

1. Edited workflow files (GPU branch only).
2. Before/after table of removed GPU sub-step names.
3. Channel/I/O notes for dropped intermediates.
4. Validation reminder: toggle-off vs toggle-on on a subset.

## Framework notes

| Framework | Action |
|-----------|--------|
| Nextflow | Replace multiple GPU processes with one `parabricks_fq2bam`; rewire in `main.nf` on GPU branch |
| Snakemake | Merge GPU rules; drop superseded GPU rules only |
| WDL | Combine Parabricks tasks in GPU branch |
| Python | Collapse GPU-path function calls only |

## What not to do

- Remove CPU steps from the default (toggle-off) path during consolidation.
- Merge variant calling into fq2bam.
- Remove QC the user still needs without equivalent Parabricks outputs.
- Edit only `evals/fixtures/` or other read-only eval paths when the user's tree is elsewhere.

## ACCELERATION.md snippets

**Opportunities:**

```markdown
## Consolidation opportunities

| GPU-branch steps | Suggested merge | Benefit |
|------------------|-----------------|---------|
| BWA_MEM, GATK_MARKDUPLICATES (when use_parabricks) | parabricks_fq2bam | Fewer passes, less disk I/O |

**Next step:** Approve to apply merges on the GPU branch only.
```

**History:**

```markdown
## Consolidation history

| Iteration | Change | Removed GPU-branch steps |
|-----------|--------|------------------------|
| 2 | Merged alignment + dedup into parabricks_fq2bam | BWA_MEM_PB, GATK_MARKDUPLICATES_PB |

Re-run toggle-off vs toggle-on comparison before production.
```
