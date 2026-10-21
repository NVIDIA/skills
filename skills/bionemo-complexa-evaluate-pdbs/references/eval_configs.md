# Evaluate-from-PDB-Dir Config Reference

Companion to `SKILL.md`. Every `evaluate_*_from_pdb_dir.yaml` and its paired `analyze_*.yaml`, with the `result_type` they emit, the `dataset.*` keys they require, and the `metric.*` / `aggregation.*` knobs they accept. Three worked examples at the bottom.

## 1. Design type → config matrix

| Design type            | Evaluate config                              | Analyze config              | `result_type`            | Folding backends         | Inverse-folding default |
|------------------------|----------------------------------------------|-----------------------------|--------------------------|--------------------------|-------------------------|
| Protein binder         | `configs/evaluate_from_pdb_dir.yaml`         | `configs/analyze.yaml`      | `protein_binder`         | `colabdesign`, `rf3_latest`, `esmfold`, `boltz2_default`, `protenix_base_default_v0.5.0` | `soluble_mpnn` |
| Ligand binder          | `configs/evaluate_from_pdb_dir.yaml`         | `configs/analyze.yaml`      | `ligand_binder`          | `rf3_latest` (default), `boltz2_default` | `ligand_mpnn` |
| AME / motif ligand binder | `configs/evaluate_ame_from_pdb_dir.yaml`  | `configs/analyze_motif_binder.yaml` | `motif_ligand_binder` | `rf3_latest`            | `ligand_mpnn`           |
| Motif protein binder   | `configs/evaluate_motif_binder.yaml` (no `_from_pdb_dir` variant; set `input_mode=pdb_dir`) | `configs/analyze_motif_binder.yaml` | `motif_protein_binder` | `colabdesign`, `rf3_latest` | `protein_mpnn` / `soluble_mpnn` |

Notes:
- The protein-binder and ligand-binder cases share `evaluate_from_pdb_dir.yaml`; switch behavior by setting `result_type`, `metric.binder_folding_method`, and `metric.inverse_folding_model` on the CLI.
- There is no shipped `evaluate_motif_protein_binder_from_pdb_dir.yaml`; reuse `configs/evaluate_motif_binder.yaml` with `++input_mode=pdb_dir ++sample_storage_path=<dir>` and an appropriate `dataset/motif_target_dict_cfg` override.

## 2. Evaluate config schema

All `evaluate_*` configs share a top-level shape (run identification, `input_mode`, `protein_type`, `sample_storage_path`, `output_dir`, `eval_njobs`, `seed`, `ncpus_`, `dataset.*`, `metric.*`). Below: the differences that matter when running from a PDB directory.

### `evaluate_from_pdb_dir.yaml`

- `protein_type: binder` (one config handles both protein and ligand binders).
- `input_mode: pdb_dir` (already set; never override back to `generated`).
- `defaults: - generation/targets_dict@dataset` — resolves `dataset.task_name` against `configs/targets/targets_dict.yaml` (and ligand_targets_dict via task naming).
- Required `dataset.*`:
  - `dataset.task_name` — target key (e.g. `02_PDL1`, `39_7V11_LIGAND`).
- Key `metric.*` fields:
  - `compute_binder_metrics: true`.
  - `binder_folding_method` — picks the refolding backend (table above).
  - `sequence_types: [self|mpnn|mpnn_fixed]` — which sequence(s) to refold.
  - `num_redesign_seqs` — MPNN sequence count (default 8 here).
  - `interface_cutoff` — Å cutoff for interface residue detection (default 8.0 protein, 6.0 motif).
  - `inverse_folding_model` — `soluble_mpnn` / `protein_mpnn` (protein) or `ligand_mpnn` (ligand).
  - `compute_pre_refolding_metrics`, `pre_refolding.{bioinformatics,tmol,hbplus}` — optional pre-refold interface metrics.
  - `compute_refolded_structure_metrics`, `refolded.{bioinformatics,tmol,hbplus}` — optional post-refold.
  - `compute_monomer_metrics`, `compute_designability`, `compute_codesignability`, `designability_modes`, `codesignability_modes` — monomer-on-binder eval.
  - `compute_co_sequence_recovery`, `compute_ss` — sequence/secondary-structure metrics.
  - `compute_novelty_{pdb,afdb,afdb_rep_v4}` — FoldSeek novelty against known DBs.
  - `keep_folding_outputs` — keep refolded PDBs.
- File walk control:
  - `ignore_generated_pdb_suffix: "_binder.pdb"` (default) — drop intermediate binder-only PDBs from the walk.
  - `file_limit: N` — cap input count.
- `result_type` is set inline (`ligand_binder` or `protein_binder`) and propagates to the paired analyze step.

### `evaluate_ame_from_pdb_dir.yaml`

- `protein_type: motif_binder`.
- `defaults: - /design_tasks/ame_dict_v2@dataset` — resolves AME tasks (`M0024_1nzy`, `M0096_1chm`, etc.).
- Required `dataset.task_name` — must match a key in `ame_dict_v2`.
- Key `metric.*`:
  - `compute_motif_binder_metrics: True`.
  - `binder_folding_method: rf3_latest` (only RF3 makes sense for ligand motif binders).
  - `inverse_folding_model: ligand_mpnn`.
  - `sequence_types: [mpnn_fixed, self]` (default — `mpnn_fixed` keeps the motif residues constant).
  - `interface_cutoff: 6.0`.
  - `compute_binder_metrics`, `compute_monomer_metrics` — optional add-ons; `compute_motif_metrics` is incompatible with `motif_binder`.
- `result_type: motif_ligand_binder`.

### Shared reference configs (no `_from_pdb_dir` suffix)

These ship for completeness; the `from_pdb_dir` variants are derived from them with `input_mode=pdb_dir` baked in.

- `configs/evaluate.yaml` — unified binder evaluation, defaults `input_mode: generated`. Pass `++input_mode=pdb_dir ++sample_storage_path=<dir>` to evaluate an external directory.
- `configs/evaluate_motif_binder.yaml` — motif binder (protein + ligand variants). Set `++result_type=motif_protein_binder` (with AF2/ColabDesign + ProteinMPNN/SolubleMPNN) or rely on the default `motif_ligand_binder` (RF3 + LigandMPNN). Use this when you have motif-protein-binder PDBs — there is no dedicated `_from_pdb_dir` variant.

## 3. Analyze config schema

### `analyze.yaml`

- `result_type: protein_binder` (default) or `ligand_binder` (override).
- `aggregation.analysis_modes` — default `[binder, monomer]` for binder result types.
- `aggregation.success_thresholds` — `null` (defaults) or a dict of `{metric: {threshold, op, scale, column_prefix}}` entries.
- Built-in defaults:
  - `protein_binder`: `i_pAE * 31 <= 7.0`, `pLDDT >= 0.9`, `scRMSD_ca < 1.5`.
  - `ligand_binder`: `min_ipAE * 31 < 2.0`, `scRMSD_ca < 2.0`, `ligand_scRMSD_aligned_allatom < 5.0`.
- Monomer-mode thresholds (when `[monomer]` is in `analysis_modes`):
  - `aggregation.designability_thresholds` — `{mode: {model: {threshold, op}}}`; default 2.0 Å.
  - `aggregation.ca_codesignability_thresholds` — default 2.0 Å.
  - `aggregation.allatom_codesignability_thresholds` — default 2.0 Å.
  - `aggregation.require_all_thresholds: false` — `false` = OR across modes/models, `true` = AND.

### `analyze_motif_binder.yaml`

- `result_type: motif_protein_binder` (default in the YAML) or `motif_ligand_binder` (override on CLI).
- `aggregation.analysis_modes` — default `[motif_binder, binder, monomer]`.
- `aggregation.motif_binder_success_thresholds`:
  - **`motif_protein_binder` defaults** — binder: `i_pAE*31 <= 7.0`, `pLDDT >= 0.8`, `scRMSD_ca < 2.0`; motif: `motif_rmsd_pred_all < 2.0`, `correct_motif_sequence_all >= 1.0`.
  - **`motif_ligand_binder` defaults** — binder: `scRMSD_bb3 <= 2.0`; motif: `motif_rmsd_pred_all <= 1.5`, `correct_motif_sequence_all >= 1.0`, `has_ligand_clashes_all < 0.5`.
- A sample is successful when **at least one redesign index** passes **every** binder criterion AND **every** motif criterion jointly. Per-redesign joint evaluation, not pooled.
- `aggregation.success_thresholds` (binder-only mode) and the three monomer threshold dicts work identically to `analyze.yaml`.

## 4. `motif_protein_binder` vs `motif_ligand_binder`

Both come from `evaluate_motif_binder.yaml` family + `analyze_motif_binder.yaml`. The split is target-driven:

| Property                    | `motif_protein_binder`           | `motif_ligand_binder` (AME)        |
|-----------------------------|----------------------------------|------------------------------------|
| Target                      | Protein receptor                 | Small molecule ligand              |
| Folding (`binder_folding_method`) | `colabdesign` (AF2) or `rf3_latest` | `rf3_latest`                  |
| Inverse folding             | `protein_mpnn` / `soluble_mpnn`  | `ligand_mpnn`                      |
| Motif criteria              | RMSD + seq recovery              | RMSD + seq recovery + ligand clashes |
| Default binder threshold    | `i_pAE*31 <= 7.0`, `pLDDT >= 0.8`, `scRMSD_ca < 2.0` | `scRMSD_bb3 <= 2.0` |
| Default motif threshold     | `motif_rmsd_pred_all < 2.0`, `correct_motif_sequence_all >= 1.0` | `motif_rmsd_pred_all <= 1.5`, seq recovery, `has_ligand_clashes_all < 0.5` |
| `evaluate_*_from_pdb_dir`   | none — use `evaluate_motif_binder.yaml` + `++input_mode=pdb_dir` | `evaluate_ame_from_pdb_dir.yaml` |

When in doubt: if every PDB has a small-molecule ligand chain, it is `motif_ligand_binder`. If the second chain is another protein receptor, it is `motif_protein_binder`.

## 5. Ligand `L:0` rename for AME RF3 evaluation

Lifted from `README.md` "Evaluating AME Designs with Ligand Targets (RF3)":

> When running RF3 evaluation on AME-generated PDB files that contain a ligand (small molecule on chain A), RF3 will attempt to add missing atoms based on the ligand's CCD code. This can cause shape errors in downstream RMSD calculations and provide the incorrect structure.
>
> **Solution:** rename the ligand residue to `L:0` before passing to RF3. This tells RF3 to treat it as a generic ligand and skip atom completion.

```python
from atomworks.io import load_any, to_pdb_file
atom_array = load_any("my_design.pdb")
ligand_mask = atom_array.chain_id == "A"
atom_array.res_name[ligand_mask] = "L:0"
to_pdb_file(atom_array, "my_design_rf3_ready.pdb")
```

Apply this transform once over the whole input directory before invoking
`complexa analysis configs/evaluate_ame_from_pdb_dir.yaml ...`. If the PDBs
came out of `complexa generate` with `protein_type=motif_binder`, the rename is
already done.

## 6. Worked examples

### Example A — protein binder PDB directory, AF2 refold

User: "Re-fold these 200 PDL1 binders with AlphaFold2 and tell me what fraction passes the default AlphaProteo thresholds."

```bash
complexa analysis configs/evaluate_from_pdb_dir.yaml \
  ++sample_storage_path=/data/pdl1_designs \
  ++dataset.task_name=02_PDL1 \
  ++metric.binder_folding_method=colabdesign \
  ++metric.inverse_folding_model=soluble_mpnn \
  ++metric.sequence_types=[self,mpnn_fixed] \
  ++metric.num_redesign_seqs=8 \
  ++result_type=protein_binder \
  ++eval_njobs=2 \
  ++run_name=pdl1_pdb_dir_af2
```

Pass-rate filter (applied by analyze):
`mpnn_complex_i_pAE * 31 <= 7.0 AND mpnn_complex_pLDDT >= 0.9 AND mpnn_binder_scRMSD_ca < 1.5`.

### Example B — AME PDB directory, RF3 refold (motif ligand binder)

User: "Score this folder of AME 1nzy designs — joint binder + motif success please."

```bash
# First, rename ligand to L:0 in every PDB (one-time).
python scripts/rename_ligand_to_L0.py /data/ame_1nzy_designs

complexa analysis configs/evaluate_ame_from_pdb_dir.yaml \
  ++sample_storage_path=/data/ame_1nzy_designs \
  ++dataset.task_name=M0024_1nzy \
  ++metric.binder_folding_method=rf3_latest \
  ++metric.inverse_folding_model=ligand_mpnn \
  ++metric.sequence_types=[mpnn_fixed,self] \
  ++metric.num_redesign_seqs=2 \
  ++result_type=motif_ligand_binder \
  ++run_name=ame_m0024_pdb_dir
```

Joint success criterion (from `analyze_motif_binder.yaml`, `motif_ligand_binder` defaults): at least one redesign index satisfies binder `scRMSD_bb3 <= 2.0` AND motif `motif_rmsd_pred_all <= 1.5` AND `correct_motif_sequence_all >= 1.0` AND `has_ligand_clashes_all < 0.5`.

### Example C — ligand binder PDB directory, RF3 refold

User: "Score this folder of 7V11 ligand-binder designs with RF3 and LigandMPNN."

```bash
complexa analysis configs/evaluate_from_pdb_dir.yaml \
  ++sample_storage_path=/data/7v11_designs \
  ++dataset.task_name=39_7V11_LIGAND \
  ++metric.binder_folding_method=rf3_latest \
  ++metric.inverse_folding_model=ligand_mpnn \
  ++metric.sequence_types=[self,mpnn_fixed] \
  ++metric.num_redesign_seqs=8 \
  ++result_type=ligand_binder \
  ++eval_njobs=2 \
  ++run_name=7v11_pdb_dir_rf3
```

Pass-rate filter (applied by analyze): `min_ipAE * 31 < 2.0 AND scRMSD_ca < 2.0 AND ligand_scRMSD_aligned_allatom < 5.0`. Override via `++aggregation.success_thresholds.min_ipAE.threshold=…` etc.

## 7. Pointers

- Per-metric output column names: [Evaluation & Analysis Guide](EVALUATION_METRICS.md), "Result CSV Reference".
- Default thresholds and Python filter examples: [Evaluation & Analysis Guide](EVALUATION_METRICS.md), "Success Criteria" and "Reading Results in Python".
- Hardware tables (VRAM per backend, wall-clock per N PDBs): `references/hardware.md`.
