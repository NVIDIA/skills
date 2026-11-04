---
name: complexa-evaluate-pdbs
description: >
  Agent runbook for evaluating an existing PDB directory through the separately
  installed first-party Proteina-Complexa CLI. Use to refold or score designs,
  compute interface pAE, i_pLDDT, scRMSD, motif RMSD, or designability, run
  complexa analysis, or assess third-party BindCraft, AlphaProteo, or RFdiffusion
  outputs. SKILL.md selects and invokes the evaluation configuration; bundled
  executables only inspect local readiness and write a run manifest.
compatibility: "complexa CLI installed (pip install -e .); CUDA GPU; AF2_DIR (colabdesign) or RF3_CKPT_PATH+RF3_EXEC_PATH (rf3_latest); ESMFold weights for monomer paths"
allowed-tools: Bash, Read, Write, Env, AskUserQuestion
---


# Complexa Evaluate-PDBs Skill

Score a directory of pre-existing PDB files against the same metrics Proteina-Complexa uses internally. Wraps `complexa analysis <evaluate_config> ++sample_storage_path=<dir>`: the CLI runs the `evaluate` step (refold + interface metrics + monomer metrics) and then the `analyze` step (success thresholds, diversity, pass-rate CSVs). Do **not** run `complexa generate` here — the inputs already exist.

## What this skill enables

- Re-fold a directory of designed PDBs with AF2 (`colabdesign`), RF3 (`rf3_latest`), ESMFold (`esmfold`), or Boltz2 (`boltz2_default`).
- Compute binder interface metrics: `i_pAE`, `min_ipAE`, `i_pTM`, `pLDDT`, binder/complex scRMSD.
- Compute monomer **designability** (ProteinMPNN-redesigned scRMSD) and **codesignability** (original sequence refold scRMSD).
- For motif inputs: motif RMSD (CA + all-atom), motif-region designability/codesignability, sequence recovery.
- Aggregate into per-PDB CSVs plus pass-rate summaries using the default thresholds for the `result_type`.

## Step 1: Pre-flight

Always check GPU / disk / tool binaries before launching a refold job. RF3 and ColabDesign-AF2 are large.

```bash
bash scripts/preflight.sh
```

Surface from `preflight.json`:

- `gpu.available` and `gpu.vram_gb` — colabdesign/RF3 need ≥40 GB; ESMFold tolerates ≥24 GB.
- `env.missing_required` — must include the keys for the chosen folding backend:
  - `colabdesign` → `AF2_DIR`
  - `rf3_latest` → `RF3_CKPT_PATH`, `RF3_EXEC_PATH`
  - `esmfold` → ESMFold weights resolvable
- `tools.{foldseek,mmseqs}` — required by `aggregation.compute_diversity` / `compute_mmseqs_diversity` (both default `true`).

If any required key is missing, route the user to the `complexa-setup` skill.

## Step 2: Identify the design type

Like `complexa design`, evaluation has one default flow (protein binder) and two extensions (ligand binder, AME). The evaluate config you pass to `complexa analysis` decides everything else (which metrics, which refolder defaults, which thresholds the analyze step applies).

### Default — protein binder

```bash
complexa analysis configs/evaluate_from_pdb_dir.yaml \
    ++sample_storage_path=/abs/path/to/pdbs \
    ++dataset.task_name=02_PDL1 \
    ++result_type=protein_binder \
    ++metric.binder_folding_method=colabdesign \
    ++metric.inverse_folding_model=soluble_mpnn \
    ++run_name=eval_pdl1_af2
```

Use this when the user's PDBs are protein-binder designs (multi-chain, binder is the last chain) or third-party outputs from BindCraft / AlphaProteo / RFdiffusion. Pulls thresholds for `protein_binder` (`i_pAE * 31 ≤ 7.0`, `pLDDT ≥ 0.9`, `scRMSD_ca < 1.5 Å`).

### Extensions — pick the matching config

| Design type | Use the protein-binder default? | Evaluate config | Analyze config | `result_type` | Default backend |
|---|---|---|---|---|---|
| Protein binder | **Yes (default)** | `configs/evaluate_from_pdb_dir.yaml` | `configs/analyze.yaml` | `protein_binder` | `colabdesign` (AF2) |
| Ligand binder (binder + small-molecule) | Same evaluate config, swap 3 overrides | `configs/evaluate_from_pdb_dir.yaml` | `configs/analyze.yaml` | `ligand_binder` | `rf3_latest` |
| AME / motif + ligand (enzyme outputs) | No — needs motif-aware config | `configs/evaluate_ame_from_pdb_dir.yaml` | `configs/analyze_motif_binder.yaml` | `motif_ligand_binder` | `rf3_latest` |
| Motif protein binder (standalone) | No — no `_from_pdb_dir` variant | `configs/evaluate_motif_binder.yaml` + `++input_mode=pdb_dir` | `configs/analyze_motif_binder.yaml` | `motif_protein_binder` | `colabdesign` / `rf3_latest` |

**Extending to ligand binder** (same evaluate config as default, three override swaps):

```bash
complexa analysis configs/evaluate_from_pdb_dir.yaml \
    ++sample_storage_path=/abs/path/to/pdbs \
    ++dataset.task_name=39_7V11_LIGAND \
    ++result_type=ligand_binder \
    ++metric.binder_folding_method=rf3_latest \
    ++metric.inverse_folding_model=ligand_mpnn \
    ++run_name=eval_v11_rf3
```

**Extending to AME** (different config; ligand auto-completion gotcha — see Step 4):

```bash
complexa analysis configs/evaluate_ame_from_pdb_dir.yaml \
    ++sample_storage_path=/abs/path/to/pdbs \
    ++dataset.task_name=M0096_1chm \
    ++run_name=eval_ame_chm
```

See `references/eval_configs.md` for the full matrix (every `result_type`, every threshold default, every supported folding backend, motif-protein-binder variant).

## Step 3: Gather inputs (AskUserQuestion)

Ask in one batched `AskUserQuestion`:

1. **`pdb_dir`** — absolute path to the directory of PDBs to evaluate.
2. **Design type** — protein binder / ligand binder / AME (motif + ligand).
3. **Folding backend** — `colabdesign` (AF2, protein binders), `rf3_latest` (ligand / AME), `boltz2_default` (alternative).
4. **Target / task name** — must match a key in `configs/targets/targets_dict.yaml`, `configs/targets/ligand_targets_dict.yaml`, or `configs/design_tasks/ame_dict_v2.yaml`. Required for binder + AME evaluation (needed to identify the target reference and, for AME, the motif contigs).
5. **AME-only**: confirm ligand residue name is already renamed to `L:0` in every PDB (see Troubleshooting). If not, do that rename first.

## Step 4: Run evaluate → analyze

Prefer `complexa analysis` (the evaluate→analyze chain) — it reuses the same config for both steps and writes a single log dir.

```bash
# Protein binder PDB dir, AF2 refold
complexa analysis configs/evaluate_from_pdb_dir.yaml \
  ++sample_storage_path=/abs/path/to/pdbs \
  ++dataset.task_name=02_PDL1 \
  ++metric.binder_folding_method=colabdesign \
  ++metric.inverse_folding_model=soluble_mpnn \
  ++result_type=protein_binder \
  ++run_name=eval_pdl1_af2
```

For ligand binders flip `binder_folding_method=rf3_latest`, `inverse_folding_model=ligand_mpnn`, `result_type=ligand_binder`. For AME use `configs/evaluate_ame_from_pdb_dir.yaml` — see `references/eval_configs.md` for full worked examples.

If you need to inspect output between stages, run them separately. The configs above are shared between `evaluate` and `analyze`:

```bash
complexa evaluate configs/evaluate_from_pdb_dir.yaml ++sample_storage_path=/abs/path/to/pdbs ++run_name=eval_pdl1_af2
complexa analyze  configs/evaluate_from_pdb_dir.yaml ++run_name=eval_pdl1_af2
```

Dry-run first if the user is unsure (no GPU work happens; the planned file walk + invocation prints):

```bash
complexa analysis configs/evaluate_from_pdb_dir.yaml ++sample_storage_path=/abs/path/to/pdbs ++dryrun=true
```

### Direct module invocation (debug fallback)

`complexa evaluate` / `analyze` are subprocess wrappers around the Hydra
modules with logging + parallel job splitting bolted on. To attach a debugger
or run under a profiler, invoke the module directly:

```bash
python -m proteinfoundation.evaluate \
    --config-path "$(realpath configs)" \
    --config-name evaluate_from_pdb_dir \
    ++sample_storage_path=/abs/path/to/pdbs \
    ++dataset.task_name=02_PDL1 \
    ++metric.binder_folding_method=colabdesign \
    ++run_name=eval_debug
```

For normal one-shot runs prefer `complexa analysis` — you get the shared log
dir and a single replayable invocation, instead of having to thread the same
overrides through two `python -m` calls.

## Step 5: Parse results

Output lands under `./evaluation_results/${run_name}/`:

- Per-PDB metrics CSV — `*_results_*.csv` (one row per input PDB × `sequence_types`).
- Pass-rate summaries — written by the `analyze` step (e.g. `res_designability.csv`, `res_filter_ligand_pass_*.csv`, `success_criteria_*.json`).
- Diversity output — FoldSeek/MMseqs2 cluster files when `aggregation.compute_diversity=true` (default).

Summarize to the user:

- Per-PDB row count and number of successful designs vs total.
- Default-threshold pass rate by `result_type` (e.g. for `protein_binder`: `i_pAE*31 <= 7.0 AND pLDDT >= 0.9 AND scRMSD_ca < 1.5`).
- Top 5 designs by primary metric (`i_pAE` for protein, `min_ipAE` for ligand, `motif_rmsd_pred_all` for motif binders).

## Step 6: Emit manifest

Capture the resolved invocation + outputs for replay.

```bash
python3 scripts/write_manifest.py \
  --output-dir ./evaluation_results/${run_name} \
  --command "complexa analysis configs/evaluate_from_pdb_dir.yaml ++sample_storage_path=<dir> ++dataset.task_name=<task> ++metric.binder_folding_method=<backend> ++run_name=<run>" \
  --skill complexa-evaluate-pdbs \
  --out ./eval_manifest.json
```

The manifest pins: resolved config, git SHA, ckpt SHA-256s, the result CSV paths, and the user-stated `result_type` for replay.

## Most common overrides

| Override                                       | Effect                                                                |
|------------------------------------------------|-----------------------------------------------------------------------|
| `++sample_storage_path=<dir>`                  | The directory of PDBs to evaluate (required).                         |
| `++dataset.task_name=<name>`                   | Target / AME task name (binders, AME). Resolves target PDB + contigs. |
| `++metric.binder_folding_method=<backend>`     | `colabdesign` / `rf3_latest` / `boltz2_default`.                      |
| `++metric.inverse_folding_model=<model>`       | `protein_mpnn` / `soluble_mpnn` / `ligand_mpnn`.                      |
| `++metric.sequence_types=[self,mpnn,mpnn_fixed]` | Which sequence flavors to refold.                                   |
| `++metric.num_redesign_seqs=N`                 | ProteinMPNN/LigandMPNN redesign count.                                |
| `++metric.compute_pre_refolding_metrics=true`  | Add bioinformatics/TMOL/HBPLUS metrics on the input structures.       |
| `++metric.keep_folding_outputs=true`           | Save the refolded PDBs (large, but useful for inspection).            |
| `++result_type=<type>`                         | Override default thresholds: `protein_binder` / `ligand_binder` / `motif_protein_binder` / `motif_ligand_binder`. |
| `++aggregation.success_thresholds.<…>`         | Tighten or loosen specific thresholds (see `references/eval_configs.md`). |
| `++eval_njobs=N`                               | Parallel GPUs for the evaluate step.                                  |
| `++dryrun=true`                                | Plan without running any folding.                                     |
| `++file_limit=N`                               | Cap input PDBs (handy for first-pass smoke tests).                    |

## Hardware

- **GPU**: ≥1 CUDA GPU. AF2 (`colabdesign`) and RF3 (`rf3_latest`) need ≥40 GB VRAM (A100/H100/L40S). ESMFold runs on ≥24 GB. Multi-GPU via `++eval_njobs=N`.
- **CPU/disk**: 24 CPUs default (`ncpus_: 24`). Each refolded PDB + intermediate output is ~1–5 MB; `keep_folding_outputs=true` can balloon to tens of GB for thousands of inputs.
- See `references/hardware.md` for per-backend wall-clock and VRAM tables.

## Troubleshooting

- **`Error: Config file not found`** — paths are relative to the repo root; `cd` to the repo before invoking `complexa analysis`.
- **`compute_motif_binder_metrics=True` but `result_type=protein_binder`** — `result_type` and the underlying `compute_*_metrics` must agree. Use `configs/evaluate_ame_from_pdb_dir.yaml` for AME inputs rather than mutating `evaluate_from_pdb_dir.yaml`.
- **RF3 shape errors on AME PDBs** — RF3 tries to auto-complete the ligand atoms from CCD. Rename the ligand residue to `L:0` in every input PDB before evaluation; see the snippet in `README.md` (`atom_array.res_name[ligand_mask] = "L:0"`).
- **Diversity step fails (`foldseek not found`)** — `FOLDSEEK_EXEC` not on `PATH`. Either fix `.env` (preferred) or disable: `++aggregation.compute_diversity=false ++aggregation.compute_mmseqs_diversity=false`.
- **All pass-rates are 0%** — check the `binder_folding_method` matches the target type (RF3 for ligand, AF2 for protein) and that `++dataset.task_name` resolves to the correct reference PDB (`complexa target show <name>` to verify).

## Reference

Full evaluate/analyze config matrix, every supported `result_type`, per-threshold defaults, and three worked examples (protein binder / ligand binder / AME): see `references/eval_configs.md`.
