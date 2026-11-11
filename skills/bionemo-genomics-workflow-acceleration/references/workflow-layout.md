# In-place optional acceleration layout

## Rules

1. **Edit the existing workflow tree** — add optional GPU steps alongside CPU steps.
2. Expose a **runtime toggle** (parameter, config key, WDL input, or CLI flag) that
   selects original vs accelerated execution.
3. **Default the toggle off** so current CPU behavior is unchanged.
4. Record all changes in `ACCELERATION.md` at the workflow root. Include
   **Runtime target**, **Toggle usage**, **Consolidation opportunities**, and
   after user-approved merges on the GPU branch, **Consolidation history**.

## Toggle naming (prefer consistency)

See the **Toggle contract** table in `SKILL.md` for the recommended toggle
name and default per framework.

Document example commands for toggle off (CPU) and toggle on (GPU) in
`ACCELERATION.md`.

## Recommended layout (single tree)

**Nextflow**

```
my-pipeline/
├── ACCELERATION.md
├── main.nf                     # branches on params.use_parabricks
├── nextflow.config             # default params.use_parabricks = false
├── accelerated.config          # optional: params.use_parabricks = true
├── modules/
│   ├── local/                  # existing CPU modules (unchanged default path)
│   └── parabricks/             # optional nf-core or local GPU modules
└── ...
```

**Snakemake**

```
my-pipeline/
├── ACCELERATION.md
├── Snakefile                   # branches on config["use_parabricks"]
├── config.yaml                 # use_parabricks: false
├── config.accelerated.yaml     # optional overlay: use_parabricks: true
└── rules/
    ├── align_cpu.smk           # existing
    └── align_parabricks.smk    # optional GPU rules
```

**WDL**

```
my-pipeline/
├── ACCELERATION.md
├── pipeline.wdl                # if (use_parabricks) { ... } else { ... }
└── inputs/
    ├── cpu.json                # use_parabricks: false
    └── gpu.json                # use_parabricks: true
```

**Python**

```
my-pipeline/
├── ACCELERATION.md
├── pipeline.py                 # --use-parabricks flag, default false
└── parabricks_wrappers.py      # optional: pbrun/docker helpers
```

## Branching strategy

- **First iteration:** 1:1 optional Parabricks steps parallel to CPU steps; toggle
  selects which branch runs.
- **Later iteration:** consolidate steps **on the GPU branch only** (e.g. single
  fq2bam replacing BWA + sort + markdup when `use_parabricks` is true).
- Never remove CPU steps from the default path without explicit user approval
  after A/B validation.

## Output wiring

- Downstream steps should consume outputs from whichever branch ran.
- Prefer matching output file names/paths between CPU and GPU branches when
  feasible so later stages need no duplicate logic.
- Use distinct `--outdir` or output tags when running A/B comparison (toggle off
  vs on) to avoid overwriting.

## Git

- Recommend a feature branch (e.g. `feat/parabricks-optional`) before in-place edits.
- Never force-push or rewrite production history without explicit user request.

## Legacy sibling-copy pattern

If the user explicitly requests a separate `*-accelerated` tree instead of
in-place toggles, create a sibling copy — but the default for this skill is
**in-place optional steps with a runtime toggle**.
