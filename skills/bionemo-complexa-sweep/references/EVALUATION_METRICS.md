# Evaluation & Analysis Guide

Metric definitions, result CSV structure, motif scaffolding, and Python analysis for the Proteina-Complexa pipeline.

> **Documentation Map**
> - Running a design? See [Inference Guide](INFERENCE.md).
> - Tuning YAML configs? See [Configuration Guide](CONFIGURATION_GUIDE.md).
> - Parameter sweeps? See [Sweep System](SWEEP.md).
> - Search metadata? See [Search Metadata](SEARCH_METADATA.md).

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Protein Types & Result Types](#protein-types--result-types)
3. [Evaluation Step (`evaluate`)](#evaluation-step)
   - [Protein Binder Evaluation](#protein-binder-evaluation)
   - [Ligand Binder Evaluation](#ligand-binder-evaluation)
   - [Monomer Evaluation](#monomer-evaluation)
   - [Motif Evaluation (Monomer Motif)](#motif-evaluation-monomer-motif)
   - [Motif Binder Evaluation](#motif-binder-evaluation)
4. [Motif Scaffolding Deep Dive](#motif-scaffolding-deep-dive)
   - [Tip Atoms vs All-Atom](#tip-atoms-vs-all-atom)
   - [Motif Alignment Modes: Indexed vs Unindexed](#motif-alignment-modes-indexed-vs-unindexed)
   - [Contig Strings](#contig-strings)
5. [Analysis Step (`analyze`)](#analysis-step)
6. [Result CSV Reference](#result-csv-reference)
   - [Protein Binder Result Columns](#protein-binder-result-columns)
   - [Ligand Binder Result Columns](#ligand-binder-result-columns)
   - [Monomer Result Columns](#monomer-result-columns)
   - [Motif Result Columns](#motif-result-columns)
   - [Motif Binder Result Columns](#motif-binder-result-columns)
7. [Reading Results in Python](#reading-results-in-python)
8. [Success Criteria](#success-criteria)
9. [Metric Interpretation Cheat Sheet](#metric-interpretation-cheat-sheet)
10. [Environment Variables](#environment-variables)

---

## Pipeline Overview

The evaluation workflow has two stages:

```
evaluate  →  analyze
  │              │
  │  Per-sample  │  Aggregate, filter,
  │  metrics     │  success rates, diversity
  │              │
  ▼              ▼
  CSV per job    Summary CSVs + organized output
```

**Evaluate** computes per-sample metrics (RMSD, designability, folding confidence, etc.)
and writes one CSV per parallel job.

**Analyze** loads all per-job CSVs, merges them, computes pass rates, diversity,
and saves organized output.

```bash
# Run both stages (standalone)
complexa evaluate configs/evaluate.yaml
complexa analyze  configs/evaluate.yaml

# Or as part of a full design pipeline (evaluate + analyze run automatically)
complexa design configs/search_binder_local_pipeline.yaml          # Protein binder
complexa design configs/search_ligand_binder_local_pipeline.yaml   # Ligand binder
complexa design configs/search_ame_local_pipeline.yaml             # AME motif scaffolding
```

---

## Protein Types & Result Types

Every evaluation is configured by two keys: **`protein_type`** (set in evaluation config, controls which metrics are computed) and **`result_type`** (set in analysis config, controls default success thresholds).

| `protein_type` | `result_type` | What is evaluated | Pipeline config | Default thresholds |
|----------------|---------------|-------------------|----------------|--------------------|
| `binder` | `protein_binder` | Binder + protein target | `search_binder_local_pipeline.yaml` | i_pAE*31 <= 7.0, pLDDT >= 0.9, scRMSD_ca < 1.5 |
| `binder` | `ligand_binder` | Binder + ligand target | `search_ligand_binder_local_pipeline.yaml` | min_ipAE*31 < 2.0, scRMSD_ca < 2.0, ligand scRMSD < 5.0 |
| `monomer` | `monomer` | Single-chain protein | — | designability scRMSD < 2.0 |
| `monomer_motif` | `monomer_motif` | Single-chain + motif region | — | motif RMSD + codesignability |
| `motif_binder` | `motif_protein_binder` | Binder + protein target + motif | standalone `evaluate_motif_binder.yaml` | Joint binder + motif |
| `motif_binder` | `motif_ligand_binder` | Binder + ligand target + motif | `search_ame_local_pipeline.yaml` | Joint binder + motif + clash |

Each base binder type has a **motif counterpart** that adds motif RMSD, motif sequence recovery, and (for ligand targets) ligand clash detection on top of the standard binder metrics.

Pipeline configs live under `configs/pipeline/`. The top-level pipeline YAML (e.g. `search_binder_local_pipeline.yaml`) composes stage configs via Hydra defaults. The motif protein binder evaluation can also be run standalone against outputs from any protein binder pipeline.

---

## Evaluation Step

All evaluation uses `protein_type: binder` for binder designs. The difference between protein and ligand binders is in the folding method, inverse folding model, and which columns appear in the output.

### Protein Binder Evaluation

Evaluates protein-protein binder designs. Uses AlphaFold2 (ColabDesign) for refolding and SolubleMPNN for sequence redesign.

```yaml
protein_type: binder

metric:
  compute_binder_metrics: true
  binder_folding_method: colabdesign   # AF2 multimer
  sequence_types: [self, mpnn, mpnn_fixed]  # pipeline default: [self]
  num_redesign_seqs: 8
  inverse_folding_model: soluble_mpnn
```

> **Note:** The pipeline default (`binder_evaluate.yaml`) uses `sequence_types: [self]` for speed. Add `mpnn` and/or `mpnn_fixed` for ProteinMPNN redesign evaluation.

**Workflow per sample:**

1. **Inverse folding** -- ProteinMPNN redesigns the binder sequence (`mpnn`, `mpnn_fixed`) or keeps the original (`self`)
2. **Structure prediction** -- Refold the complex with AF2 (ColabDesign)
3. **Metrics** -- Compute binding confidence (i_pAE, i_pTM, pLDDT) and structural consistency (scRMSD)

**Key metrics:**

| Metric | Column pattern | What it measures | Good value |
|--------|---------------|------------------|------------|
| Interface PAE | `{seq}_complex_i_pAE` | Binding confidence | < 10 (< 5 excellent) |
| Interface pTM | `{seq}_complex_i_pTM` | Interface quality | > 0.5 |
| pLDDT | `{seq}_complex_pLDDT` | Structure confidence | > 70 |
| Binder scRMSD | `{seq}_binder_scRMSD` | Design preserved after refold | < 2.0 A |

### Ligand Binder Evaluation

Evaluates small-molecule binder designs. Uses RF3 for refolding and LigandMPNN for sequence redesign. Produces additional ligand-specific RMSD columns.

```yaml
protein_type: binder

metric:
  compute_binder_metrics: true
  binder_folding_method: rf3_latest    # RoseTTAFold3
  sequence_types: [self, mpnn]
  num_redesign_seqs: 2
  inverse_folding_model: ligand_mpnn
```

**Workflow per sample:**

1. **Inverse folding** -- LigandMPNN redesigns the binder sequence (ligand-aware)
2. **Structure prediction** -- Refold the complex with RF3
3. **Metrics** -- Compute RF3 confidence metrics (min_ipAE, pLDDT, ipSAE) and both binder and ligand scRMSD

**Key metrics (in addition to protein binder metrics above):**

| Metric | Column pattern | What it measures | Good value |
|--------|---------------|------------------|------------|
| min_ipAE | `{seq}_complex_min_ipAE` | Minimum interface PAE (primary for ligand) | < 2.0 (after * 31) |
| Binder scRMSD (CA) | `{seq}_binder_scRMSD_ca` | Binder backbone preserved (CA only) | < 2.0 A |
| Ligand scRMSD | `{seq}_ligand_scRMSD` | Ligand position preserved after refold | < 5.0 A |
| Ligand scRMSD (aligned) | `{seq}_ligand_scRMSD_aligned_allatom` | Ligand position after aligning binder | < 5.0 A |
| ipSAE | `{seq}_complex_min_ipSAE` | Interface pSAE (higher is better) | > 0.5 |

**Key differences from protein binder evaluation:**

- `binder_folding_method: rf3_latest` instead of `colabdesign`
- `inverse_folding_model: ligand_mpnn` instead of `soluble_mpnn`
- Produces ligand RMSD columns (`ligand_scRMSD`, `ligand_scRMSD_aligned_allatom`, `ligand_scRMSD_aligned_bb`)
- RF3 provides additional metrics: `min_ipAE`, `min_ipSAE`, `avg_ipSAE`, `max_ipSAE`, `ranking_score`, `has_clash`
- Default ranking uses `min_ipAE` (minimize) instead of `i_pAE`

### Monomer Evaluation

Evaluates single-chain structural quality via fold-and-compare.

```yaml
metric:
  compute_monomer_metrics: true
  monomer_folding_models: [esmfold]
  designability_modes: [ca]
  codesignability_modes: [ca, all_atom]
  compute_ss: true
```

**Workflow per sample:**

1. **Designability** — ProteinMPNN redesigns the sequence → fold with ESMFold → compute scRMSD vs generated structure
2. **Codesignability** — Fold the *original* PDB sequence → compute scRMSD vs generated structure
3. **Secondary structure** — Compute alpha/beta/coil fractions via biotite
4. **Sequence recovery** — Compare ProteinMPNN output to original sequence

**Key metrics:**

| Metric | Column pattern | What it measures | Good value |
|--------|---------------|------------------|------------|
| Designability | `_res_scRMSD_{mode}_{model}` | Best scRMSD over MPNN sequences | < 2.0 A |
| Single-seq designability | `_res_scRMSD_single_{mode}_{model}` | scRMSD of first MPNN sequence | < 2.0 A |
| Codesignability | `_res_co_scRMSD_{mode}_{model}` | scRMSD using original sequence | < 2.0 A |
| Best MPNN sequence | `_res_mpnn_best_sequence` | Sequence with lowest scRMSD | — |
| Secondary structure | `_res_ss_alpha`, `_res_ss_beta`, `_res_ss_coil` | Structure composition | — |
| pLDDT | `_res_pLDDT` | Refolded confidence | > 0.7 |

> **Note:** `{mode}` is `ca`, `bb3o`, or `all_atom`. `{model}` is `esmfold` or `colabfold`. The number of MPNN redesign sequences is controlled by `designability_num_seq` (default 8).

### Motif Evaluation (Monomer Motif)

Evaluates motif scaffolding by measuring how well the generated scaffold preserves the target motif. This is for **monomer** motif scaffolding (no binder/target complex).

```yaml
protein_type: monomer_motif

dataset:
  motif_task_name: 1QJG_AA_TIP
  unindexed: true

metric:
  compute_motif_metrics: true
  compute_monomer_metrics: true   # also run monomer metrics on the full structure
  motif_rmsd_modes: [ca, all_atom]
  designability_modes: [ca]
  codesignability_modes: [ca, all_atom]
```

**Workflow per sample:**

1. **Motif alignment** — Align the ground-truth motif into the generated structure (indexed or unindexed)
2. **Direct motif RMSD** — Compare generated structure at motif positions vs ground truth
3. **Motif sequence recovery** — Compare generated sequence at motif positions vs ground truth
4. **Designability** — ProteinMPNN redesign → fold → compute full-structure + motif-region scRMSD
5. **Codesignability** — Fold original sequence → compute full-structure + motif-region scRMSD
6. **Secondary structure** — Compute alpha/beta/coil fractions

**Key metrics (motif-specific):**

| Metric | Column pattern | What it measures | Good value |
|--------|---------------|------------------|------------|
| Motif RMSD | `_res_motif_rmsd_{mode}` | Generated vs ground-truth motif | < 1.0 A (CA), < 2.0 A (all-atom) |
| Motif sequence recovery | `_res_motif_seq_rec` | Fraction of motif residues matching | 1.0 (perfect) |
| Motif-region designability | `_res_des_motif_scRMSD_{mode}_{model}` | Motif region preserved after MPNN+refold | < 1.0 A |
| Motif-region codesignability | `_res_co_motif_scRMSD_{mode}_{model}` | Motif region preserved after refold | < 1.0 A |

### Motif Binder Evaluation

Evaluates designs that combine **binder refolding** with **motif preservation**. The generated protein must both bind a target and preserve a functional motif. There are two variants depending on whether the target is a protein or a ligand:

| Variant | `result_type` | Folding | Inverse Folding | Extra motif criteria |
|---------|---------------|---------|-----------------|---------------------|
| Motif Protein Binder | `motif_protein_binder` | ColabDesign / RF3 | ProteinMPNN / SolubleMPNN | motif RMSD, seq recovery |
| Motif Ligand Binder (AME) | `motif_ligand_binder` | RF3 | LigandMPNN | motif RMSD, seq recovery, ligand clashes |

#### Motif Protein Binder

Adds motif constraints on top of standard protein binder evaluation. Use when the designed protein must preserve specific structural motifs while binding a protein target.

```yaml
protein_type: motif_binder

metric:
  compute_motif_binder_metrics: true
  binder_folding_method: colabdesign   # or rf3_latest
  sequence_types: [self, mpnn_fixed]
  inverse_folding_model: soluble_mpnn  # protein-only
```

Standalone evaluation:

```bash
python -m proteinfoundation.evaluate --config-name evaluate_motif_binder \
    dataset.task_name=MY_TASK \
    metric.binder_folding_method=colabdesign \
    metric.inverse_folding_model=soluble_mpnn
```

#### Motif Ligand Binder (AME)

Adds motif constraints on top of ligand binder evaluation, with ligand clash detection. This is the evaluation mode used by the AME pipeline.

```yaml
protein_type: motif_binder

metric:
  compute_motif_binder_metrics: true
  binder_folding_method: rf3_latest
  sequence_types: [self, mpnn_fixed]
  inverse_folding_model: ligand_mpnn

  # Pre/post-refolding interface metrics
  compute_pre_refolding_metrics: true
  pre_refolding:
    bioinformatics: true
    tmol: true

  compute_refolded_structure_metrics: true
  refolded:
    bioinformatics: true
    tmol: true

  # Optional monomer metrics on the binder chain
  compute_monomer_metrics: false
  monomer_folding_models: [esmfold]
```

**Workflow per sample (both variants):**

1. **Binder refolding** — Same as standard binder evaluation: inverse folding redesigns the binder sequence, then refold and compute binding confidence metrics (i_pAE, i_pTM, pLDDT, scRMSD)
2. **Motif overlay** — Compute motif RMSD and motif sequence recovery on the predicted structure, measuring how well the functional motif is preserved after refolding
3. **Ligand clash detection** (ligand variant only) — Check for steric clashes between the refolded structure and the ligand
4. **Interface metrics** — Pre- and post-refolding bioinformatics, force field, and hydrogen bond analysis
5. **Optional monomer metrics** — If enabled, run designability/codesignability on the binder chain alone

**Key metrics (combines binder + motif):**

| Metric | What it measures | Good value |
|--------|------------------|------------|
| `{seq}_complex_i_pAE` | Binding confidence | < 10 (protein), varies (ligand) |
| `{seq}_complex_pLDDT` | Structure confidence | > 0.8 |
| `{seq}_binder_scRMSD` | Design preserved after refold | < 2.0 A |
| `{seq}_motif_rmsd_pred` | Motif RMSD in predicted structure | < 2.0 A (protein), < 1.5 A (ligand) |
| `{seq}_correct_motif_sequence` | Motif residues match ground truth | >= 1.0 (perfect) |
| `{seq}_has_ligand_clashes` | Steric clashes with ligand (ligand only) | < 0.5 (no clashes) |

**Joint per-redesign success:** A sample is "successful" when at least one redesign passes ALL binder criteria AND ALL motif criteria simultaneously. This is evaluated per redesign index, ensuring the same predicted structure satisfies both sets of requirements.

**Comparison across all evaluation types:**

| Aspect | Binder Eval | Monomer Motif Eval | Motif Protein Binder | Motif Ligand Binder |
|--------|-------------|--------------------|-----------------------|----------------------|
| `protein_type` | `binder` | `monomer_motif` | `motif_binder` | `motif_binder` |
| `result_type` | `protein_binder` / `ligand_binder` | `monomer_motif` | `motif_protein_binder` | `motif_ligand_binder` |
| Folding model | AF2 / RF3 | ESMFold | AF2 / RF3 | RF3 (ligand-aware) |
| Inverse folding | ProteinMPNN / SolubleMPNN | ProteinMPNN | ProteinMPNN / SolubleMPNN | LigandMPNN |
| Binder metrics | Yes | No | Yes | Yes |
| Motif RMSD | No | Yes (direct + refolded) | Yes (on predicted) | Yes (on predicted) |
| Ligand context | No / Yes | No | No | Yes |
| Ligand clashes | No | No | No | Yes |

### Ranking Criteria (Best Redesign Selection)

For binder and motif binder evaluation, multiple redesigned sequences are evaluated per sample. The "best" redesign is selected using a **composite score** from ranking criteria.

**Default ranking:**
- Protein binder: minimize `i_pAE` (scale 1.0)
- Ligand binder: minimize `min_ipAE` (scale 1.0)

**Custom ranking** via `metric.ranking_criteria`:

```yaml
metric:
  ranking_criteria:
    i_pAE:
      scale: 1.0
      direction: minimize
    pLDDT:
      scale: 0.5
      direction: maximize
```

The composite score is `sum(metric_value * scale * direction_sign)` where `direction_sign` is +1 for minimize, -1 for maximize. The redesign with the lowest composite score is selected as best.

For motif binder evaluation, use `metric.motif_ranking_criteria` with the same format.

### Optional ESM Metrics

ESM pseudo-perplexity can be computed alongside binder evaluation to assess sequence plausibility:

```yaml
metric:
  compute_esm_metrics: true
  esm_model: facebook/esm2_t33_650M_UR50D   # default
```

**Columns produced** (per sequence type):

| Column | Description |
|--------|-------------|
| `{seq}_esm_pseudo_perplexity` | ESM pseudo-perplexity (lower = more natural) |
| `{seq}_esm_log_likelihood` | ESM log-likelihood |

Uses `ESM_DIR` or `CACHE_DIR` environment variable for model cache.

### Job Parallelization

Evaluation can be split across parallel jobs for large datasets:

```yaml
job_id: 0          # This job's index (0-based)
eval_njobs: 8      # Total number of parallel jobs
```

Each job evaluates a shard of the input PDBs. Output CSVs are named `{eval_type}_results_{config_name}_{job_id}.csv` (e.g. `binder_results_evaluate_0.csv`). The `analyze` step automatically discovers and merges all per-job CSVs.

### Evaluating External PDB Files

To evaluate PDB files not generated by the pipeline (e.g. from external models), use the `evaluate_from_pdb_dir` configs:

```bash
# Protein/ligand binder evaluation on external PDBs
complexa evaluate configs/evaluate_from_pdb_dir.yaml \
    dataset.sample_storage_path=/path/to/pdbs \
    dataset.task_name=MY_TARGET

# AME motif binder evaluation on external PDBs
complexa evaluate configs/evaluate_ame_from_pdb_dir.yaml \
    dataset.sample_storage_path=/path/to/pdbs \
    dataset.task_name=MY_TASK
```

Key config differences from pipeline evaluation:

| Setting | Pipeline (`binder_evaluate.yaml`) | External PDB (`evaluate_from_pdb_dir.yaml`) |
|---------|----------------------------------|---------------------------------------------|
| `input_mode` | `generated` | `pdb_dir` |
| `sample_storage_path` | Set by pipeline | User provides path to PDB directory |
| `ignore_generated_pdb_suffix` | — | `_binder.pdb` (strips suffix to find target) |

---

## Motif Scaffolding Deep Dive

### Tip Atoms vs All-Atom

The `atom_selection_mode` in the motif task definition controls which atoms are used when defining the motif region and computing motif RMSD:

| Mode | Atoms included | When to use |
|------|----------------|-------------|
| `all_atom` | All heavy atoms in each motif residue (N, CA, C, O, CB, side chain) | When the full atomic detail of the motif matters (e.g. active sites with precise side-chain geometry) |
| `tip_atoms` | Only the terminal/tip atoms of each side chain | When the backbone is flexible but side-chain endpoint placement matters (e.g. functional groups that must reach specific coordinates) |
| `ca` | Only C-alpha atoms | Backbone-only motif matching |
| `bb3o` | N, CA, C, O backbone atoms | Backbone with oxygen orientation |

#### Where tip atoms are defined

The per-residue tip atom definitions live in `src/proteinfoundation/utils/constants.py` as `SIDECHAIN_TIP_ATOMS`. This dictionary maps each amino acid to its functional/terminal atoms:

```python
SIDECHAIN_TIP_ATOMS = {
    "ALA": ["CA", "CB"],
    "ARG": ["CD", "CZ", "NE", "NH1", "NH2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "ASN": ["CB", "CG", "ND2", "OD1"],
    "CYS": ["CA", "CB", "SG"],
    "GLU": ["CG", "CD", "OE1", "OE2"],
    "GLN": ["CG", "CD", "NE2", "OE1"],
    "GLY": [],                              # no side chain
    "HIS": ["CB", "CG", "CD2", "CE1", "ND1", "NE2"],
    "LYS": ["CE", "NZ"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    # ... etc for all 20 amino acids
}
```

At evaluation time, `src/proteinfoundation/utils/motif_utils.py::_select_motif_atoms()` reads this dictionary to build the atom mask for each motif residue:

```python
# For each residue in the motif, only these atoms are included in the mask
tip_atom_names = SIDECHAIN_TIP_ATOMS.get(residue_name, [])
selected = [atom_order[name] for name in tip_atom_names
            if name in atom_order and atom_order[name] in available_atoms]
```

This means:
- **Tip atoms mode** focuses on where the side chains *end up* (functional groups, charged tips, ring systems) rather than backbone placement.
- **GLY has no tip atoms** (empty list), so glycine motif residues contribute nothing to tip-atom RMSD.
- The atom indices map into the 37-slot `UNIFIED_ATOM37_ENCODING` also defined in `constants.py`, which provides the canonical atom ordering for all residue types.

#### How `atom_selection_mode` affects metrics

When `atom_selection_mode` is not `all_atom` (e.g. `tip_atoms`), the motif mask contains only the selected atoms. This means CA-based RMSD modes would have an empty atom intersection. To keep columns consistent, the evaluation auto-fills CA-based motif metrics with `0.0`:

```
atom_selection_mode = "all_atom":
  _res_motif_rmsd_ca        → computed normally
  _res_motif_rmsd_all_atom  → computed normally

atom_selection_mode = "tip_atoms":
  _res_motif_rmsd_ca        → auto-filled with 0.0 (no CA atoms in mask)
  _res_motif_rmsd_all_atom  → computed on tip atoms only
```

This auto-fill ensures that success criteria columns always exist and CA-based threshold checks pass automatically for non-CA motifs.

#### Configuration

The `atom_selection_mode` is set per task in the motif task dictionary (e.g. `design_tasks/motif_dict.yaml`):

```yaml
1QJG_AA_TIP:
  contig: "15/A45-65/20"
  pdb_path: /path/to/1QJG.pdb
  atom_selection_mode: tip_atoms
  motif_only: true

1QJG_AA_NATIVE:
  contig: "15/A45-65/20"
  pdb_path: /path/to/1QJG.pdb
  atom_selection_mode: all_atom
  motif_only: true
```

The evaluation config references the task by name:

```yaml
dataset:
  motif_task_name: 1QJG_AA_TIP   # Uses tip_atoms mode
```

### Motif Alignment Modes: Indexed vs Unindexed

Controls how the ground-truth motif is aligned into the generated structure.

```yaml
dataset:
  motif_task_name: 1QJG_AA_TIP
  unindexed: true    # or false (default)
```

#### Indexed Mode (`unindexed: false`)

The motif occupies **known positions** in the generated structure, determined by the contig string.

```
Contig: "15/A45-65/20"
         ^^            ^^
         scaffold       scaffold
              ^^^^^^^^^
              motif (residues 45-65 of chain A)

Generated structure:  [---scaffold---][===MOTIF===][---scaffold---]
                      positions 1-15   16-36        37-56

Motif residues are placed at positions 16-36 → direct comparison.
```

**When to use:** When the generative model explicitly places motif residues at contig-specified positions (the standard case for most generation pipelines).

**Variable scaffolds:** If different samples have different scaffold lengths, provide a per-sample contig CSV. You can set it explicitly or rely on auto-discovery.

**Option 1 — Explicit path:**

```yaml
metric:
  motif_info_csv: /path/to/motif_info.csv
```

**Option 2 — Auto-discovery:** If `motif_info_csv` is not set, the evaluator looks for a file named `{motif_task_name}_{job_id}_motif_info.csv` in:

1. The evaluation output directory (where `copy_motif_csvs` copies CSVs)
2. `sample_storage_path`
3. The parent directory of `sample_storage_path`

Indexed mode **requires** this CSV to be present; if none is found, evaluation raises `FileNotFoundError` with instructions.

**CSV format:** The CSV must have a `contig` column. For reliable per-sample matching, include a `filename` column (extensionless PDB stem). If the column is present, matching falls back to `sample_num` (parsed from filenames like `job_0_id_10_motif_TASK`) or to row order (same order as PDBs).


#### Unindexed Mode (`unindexed: true`)

The motif residues are **not** at known positions. The evaluation uses **greedy coordinate matching** to find them.

```
Generated structure:  [------full structure------]
Ground-truth motif:   [==MOTIF==]

Algorithm:
  For each motif residue:
    1. Compute RMSD to every generated residue (using overlapping atoms)
    2. Optionally filter by amino acid type match
    3. Select the closest unmatched residue
```

**When to use:**

- When samples are refolded structures (not direct model outputs) — refolding may reorder or renumber residues
- When the generation method does not preserve contig indexing
- When evaluating external model outputs where motif positions are unknown

**Important:** The motif coordinates must be centered (near the origin) for unindexed matching to work correctly, as the algorithm searches for the best coordinate overlap.

### Contig Strings

A contig string encodes the structure layout as alternating scaffold and motif segments, separated by `/`:

```
"15/A45-65/20/A20-30/10"
 ↑   ↑       ↑   ↑      ↑
 scaffold  scaffold  scaffold
     motif       motif
     (chain A,   (chain A,
      res 45-65)  res 20-30)
```

**Parsing rules:**

| Part | Format | Meaning |
|------|--------|---------|
| Integer | `15` | Scaffold segment of length 15 |
| Chain + range | `A45-65` | Motif from chain A, residues 45 to 65 (inclusive) |
| Chain + single | `A33` | Single motif residue from chain A, position 33 |

**Example dissection:**

```
Contig: "8/A9-16/17/A34-41/8"

Segment 1: scaffold, length 8    → positions 1-8
Segment 2: motif A9-16, length 8 → positions 9-16
Segment 3: scaffold, length 17   → positions 17-33
Segment 4: motif A34-41, length 8 → positions 34-41
Segment 5: scaffold, length 8    → positions 42-49

Total length: 49 residues, 16 motif residues in 2 segments
```

---

## Analysis Step

The analysis step loads evaluation CSVs, computes aggregate metrics, and organizes output.

```bash
complexa analyze configs/evaluate.yaml
# or
python -m proteinfoundation.analyze --config-name evaluate
```

### What it does

1. **Load results** — Reads all `{result_type}_results_*.csv` files from the output directory
2. **Merge monomer results** — For `monomer_motif` and binder result types, merges monomer metrics into the primary results (handles column conflicts with `_monomer` suffix)
3. **Compute pass rates** — Filters by thresholds and computes success rates, grouped by experimental parameters
4. **Compute diversity** — FoldSeek (structural) and MMseqs2 (sequence) diversity on full and filtered subsets
5. **Secondary structure** — Aggregate SS fractions across groups
6. **Organize output** — Moves result files into categorized subdirectories

### Analysis Modes

Controlled by `aggregation.analysis_modes`:

| Mode | What it computes | Valid for |
|------|------------------|-----------|
| `binder` | Success rates, interface metrics, binder diversity | `protein_binder`, `ligand_binder` |
| `monomer` | Designability/codesignability pass rates, monomer diversity | all result types |
| `motif` | Motif RMSD pass rates, motif success criteria, motif diversity | `monomer_motif` |
| `motif_binder` | Joint binder + motif success filtering, per-task pass rates, diversity | `motif_protein_binder`, `motif_ligand_binder` |

Defaults (from code, configs may override):
- `protein_binder` / `ligand_binder`: `[binder, monomer]`
- `monomer`: `[monomer]`
- `monomer_motif`: `[motif, monomer]`
- `motif_protein_binder` / `motif_ligand_binder`: `[motif_binder]`

The pipeline configs may add additional modes. For example, `ame_analyze.yaml` uses `[motif_binder, monomer]` to also compute monomer designability metrics.

### Monomer Merging for Combined Result Types

When `result_type` is `monomer_motif` or a binder type, both motif/binder results and monomer results are evaluated separately by `evaluate`. The `analyze` step automatically merges them:

```
motif_results_*.csv  +  monomer_results_*.csv  →  merged DataFrame
```

**Column conflict resolution:** If a column exists in both CSVs (e.g. `_res_scRMSD_ca_esmfold`), the monomer version is renamed with a `_monomer` suffix:

```
motif CSV column:   _res_scRMSD_ca_esmfold           (kept as-is)
monomer CSV column: _res_scRMSD_ca_esmfold           → _res_scRMSD_ca_esmfold_monomer
```

Columns unique to the monomer CSV (e.g. `_res_scRMSD_single_ca_esmfold`) are added directly.

The analysis functions transparently handle this via `resolve_monomer_column()`, which looks for the canonical column name first, then falls back to the `_monomer` variant.

### Output Directory Structure

After `analyze`, results are organized into subdirectories:

```
evaluation_results/{run_name}/
├── motif_metrics/                    # Motif pass rates, success rates
│   └── motif_results_*_combined.csv
├── monomer_metrics/                  # Designability, codesignability
├── filter_results/                   # Filtered subsets (success, RMSD thresholds)
├── diversity/                        # FoldSeek + MMseqs cluster results
│   ├── foldseek_clusters_*/
│   └── mmseqs_clusters_*/
├── amino_acid_distribution/          # Residue type distributions
├── secondary_structure/              # SS fractions
└── RAW_*.csv                         # Combined raw results (all columns)
```

### Diversity Computation

**FoldSeek (structural diversity)** — Clusters samples by TM-score similarity:

| Mode | Scope | Available for |
|------|-------|---------------|
| `complex` | Full complex structure | binder |
| `binder` | Binder chain only | binder, motif_binder |
| `interface` | Interface residues only | binder |
| `monomer` | Monomer structure | monomer, motif |

Alignment type 1 (structure only) and 2 (structure + sequence) are both computed.

**MMseqs2 (sequence diversity)** — Clusters samples by sequence identity:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_seq_id` | 0.1 | Minimum sequence identity for clustering |
| `coverage` | 0.7 | Minimum alignment coverage |

Both diversity methods are run on **all samples** and on **successful samples** (filtered subset per sequence type).

### Analysis Output Files

| File | Description |
|------|-------------|
| `res_nsamples.csv` | Sample counts per run |
| `res_designability.csv` | Designability pass rates (2 Å threshold) |
| `res_codesignability_{mode}.csv` | Codesignability pass rates per mode |
| `res_filter_binder_pass_{suffix}.csv` | Protein binder success rates |
| `res_filter_ligand_pass_{suffix}.csv` | Ligand binder success rates |
| `res_filter_motif_binder_pass_{suffix}.csv` | Motif binder joint success rates |
| `res_motif_binder_per_task_pass_rates.csv` | Per-task motif binder pass rates |
| `res_ss_biot_{suffix}.csv` | Secondary structure fractions |
| `res_type_prop_{suffix}.csv` | Residue type proportions |
| `res_aa_distribution_{suffix}.csv` | AA distribution |
| `res_div_foldseek_{mode}_{suffix}.csv` | FoldSeek diversity |
| `res_div_mmseqs_{suffix}.csv` | MMseqs diversity |
| `success_criteria_*.json` | Saved thresholds used for filtering |

---

## Result CSV Reference

### Identifier Columns (all result types)

| Column | Description |
|--------|-------------|
| `id_gen` | Sample identifier (unique per generated structure) |
| `pdb_path` | Path to the generated PDB file |
| `L` | Protein length (residue count) |
| `run_name` | Experiment name from config |
| `ckpt_path` | Checkpoint path used for generation |

### Protein Binder Result Columns

Written to `binder_results_*.csv` by binder evaluation with `binder_folding_method: colabdesign`.

**Per sequence type** (`self`, `mpnn`, `mpnn_fixed`):

| Column | Description |
|--------|-------------|
| `{seq}_complex_i_pAE` | Best interface PAE (lower = better binding) |
| `{seq}_complex_i_pTM` | Best interface pTM (higher = better) |
| `{seq}_complex_pLDDT` | Best overall pLDDT (higher = more confident) |
| `{seq}_complex_pTM` | Best overall pTM score |
| `{seq}_binder_scRMSD` | Best binder backbone RMSD after refolding |
| `{seq}_binder_scRMSD_ca` | Binder CA-only RMSD after refolding |
| `{seq}_binder_scRMSD_bb3` | Binder BB3 (N, CA, C) RMSD after refolding |
| `{seq}_binder_scRMSD_bb3o` | Binder BB3O (N, CA, C, O) RMSD after refolding |
| `{seq}_binder_scRMSD_allatom` | Binder all-atom RMSD after refolding |
| `{seq}_complex_scRMSD` | Best full-complex RMSD after refolding |
| `{seq}_complex_scRMSD_ca` | Complex CA-only RMSD after refolding |
| `{seq}_binder_pLDDT` | Binder-only pLDDT |
| `{seq}_complex_pdb_path` | Path to best refolded structure |
| `{seq}_sequence` | Best sequence selected by ranking |
| `{seq}_complex_i_pAE_all` | All i_pAE values (list, one per MPNN sequence) |
| `{seq}_binder_scRMSD_all` | All scRMSD values (list) |
| `{seq}_aa_counts` | Amino acid composition (dict) |
| `{seq}_aa_interface_counts` | Interface residue AA composition (dict) |

**Optional force field metrics** (if `compute_pre_refolding_metrics` or `compute_refolded_structure_metrics` enabled):

| Column | Description |
|--------|-------------|
| `generated_n_interface_hbonds_tmol` | H-bond count (generated structure) |
| `generated_total_interface_hbond_energy_tmol` | H-bond energy (kcal/mol) |
| `refolded_{seq}_n_interface_hbonds_tmol` | H-bond count (refolded structure) |
| `generated_binder_interface_sc` | Shape complementarity |
| `generated_binder_interface_dSASA` | Buried surface area (A^2) |

### Ligand Binder Result Columns

Written to `binder_results_*.csv` by binder evaluation with `binder_folding_method: rf3_latest`. Includes all protein binder columns above, plus:

**Additional RF3 confidence metrics** (per sequence type):

| Column | Description |
|--------|-------------|
| `{seq}_complex_min_ipAE` | Minimum interface PAE (primary ranking metric for ligand binders) |
| `{seq}_complex_min_ipSAE` | Minimum interface pSAE (higher = better) |
| `{seq}_complex_avg_ipSAE` | Average interface pSAE |
| `{seq}_complex_max_ipSAE` | Maximum interface pSAE |
| `{seq}_complex_ranking_score` | RF3 composite ranking score |
| `{seq}_complex_has_clash` | 1.0 if clash detected, 0.0 otherwise |

**Ligand-specific RMSD metrics** (per sequence type):

| Column | Description |
|--------|-------------|
| `{seq}_binder_scRMSD_ca` | Binder CA-only RMSD after refolding |
| `{seq}_binder_scRMSD_allatom` | Binder all-atom RMSD after refolding |
| `{seq}_ligand_scRMSD` | Direct ligand RMSD (generated vs refolded) |
| `{seq}_ligand_scRMSD_aligned_bb` | Ligand RMSD after aligning binder backbone |
| `{seq}_ligand_scRMSD_aligned_allatom` | Ligand RMSD after aligning binder all-atoms |

### Monomer Result Columns

Written to `monomer_results_*.csv` by monomer evaluation.

| Column | Description |
|--------|-------------|
| `_res_scRMSD_{mode}_{model}` | Best designability scRMSD (min over N MPNN sequences) |
| `_res_scRMSD_{mode}_{model}_all` | All designability scRMSD values (list) |
| `_res_scRMSD_single_{mode}_{model}` | scRMSD of first MPNN sequence only |
| `_res_co_scRMSD_{mode}_{model}` | Best codesignability scRMSD |
| `_res_co_scRMSD_{mode}_{model}_all` | All codesignability scRMSD values (list) |
| `_res_mpnn_sequences` | All MPNN-generated sequences (list) |
| `_res_mpnn_best_sequence` | Best MPNN sequence (by RMSD) |
| `_res_co_sequence_recovery` | Sequence recovery (MPNN vs original) |
| `_res_ss_alpha` | Alpha helix fraction |
| `_res_ss_beta` | Beta sheet fraction |
| `_res_ss_coil` | Coil fraction |

**Mode/model substitutions:**

- `{mode}`: `ca` (C-alpha only), `all_atom` (all heavy atoms), `bb3o` (N, CA, C, O)
- `{model}`: `esmfold`, `colabfold`

**Example column names:**
- `_res_scRMSD_ca_esmfold` — Best CA designability with ESMFold
- `_res_co_scRMSD_all_atom_esmfold` — Best all-atom codesignability with ESMFold
- `_res_scRMSD_single_ca_esmfold` — Single-sequence CA designability

### Motif Result Columns

Written to `motif_results_*.csv` by motif evaluation. Includes all monomer columns above plus:

| Column | Description |
|--------|-------------|
| `contig_string` | Contig used for this sample's motif alignment |
| `_res_motif_rmsd_ca` | Direct motif RMSD, CA atoms |
| `_res_motif_rmsd_all_atom` | Direct motif RMSD, all atoms |
| `_res_motif_seq_rec` | Motif sequence recovery (0.0 - 1.0) |
| `_res_scRMSD_{mode}_{model}` | Best full-structure designability scRMSD (at motif argmin) |
| `_res_scRMSD_{mode}_{model}_all` | All full-structure designability scRMSD values |
| `_res_des_motif_scRMSD_{mode}_{model}` | Best motif-region designability scRMSD |
| `_res_des_motif_scRMSD_{mode}_{model}_all` | All motif-region designability scRMSD values |
| `_res_co_scRMSD_{mode}_{model}` | Best full-structure codesignability scRMSD (at motif argmin) |
| `_res_co_scRMSD_{mode}_{model}_all` | All full-structure codesignability scRMSD values |
| `_res_co_motif_scRMSD_{mode}_{model}` | Best motif-region codesignability scRMSD |
| `_res_co_motif_scRMSD_{mode}_{model}_all` | All motif-region codesignability scRMSD values |
| `_res_mpnn_sequences` | MPNN-generated sequences |
| `_res_mpnn_best_sequence` | Best MPNN sequence (by motif-region RMSD) |

**Important:** For motif evaluation, "best" means the sequence whose **motif-region** scRMSD is lowest (argmin of motif scRMSD), not the full-structure minimum. The full-structure scRMSD reported is the value at that same index.

### Motif Binder Result Columns

Written to `motif_binder_results_*.csv` by motif binder evaluation. Used by both `motif_protein_binder` and `motif_ligand_binder` result types. Combines binder and motif columns.

**Metadata columns:**

| Column | Description |
|--------|-------------|
| `task_name` | Motif task name |
| `result_type` | `motif_protein_binder` or `motif_ligand_binder` |

**Binder columns** (same as binder results, per sequence type):

| Column | Description |
|--------|-------------|
| `{seq}_complex_i_pAE` | Best interface PAE (lower = better binding) |
| `{seq}_complex_i_pTM` | Best interface pTM |
| `{seq}_complex_pLDDT` | Best overall pLDDT |
| `{seq}_binder_scRMSD` | Best binder backbone RMSD after refolding |
| `{seq}_binder_scRMSD_ca` | CA-only binder scRMSD |
| `{seq}_binder_scRMSD_bb3` | BB3 (N, CA, C) binder scRMSD |
| `{seq}_complex_pdb_path` | Path to best refolded structure |

**Generated-structure motif columns** (computed on the generated structure before refolding):

| Column | Description |
|--------|-------------|
| `motif_rmsd_gen` | Motif RMSD in the generated (pre-refolding) structure |
| `motif_seq_rec_gen` | Motif sequence recovery in generated structure |
| `correct_motif_sequence_gen` | Whether motif sequence is fully recovered in generated structure |
| `has_ligand_clashes_gen` | Ligand clash flag in generated structure (ligand binder only) |

**Predicted-structure motif columns** (per sequence type, with `_all` list variants for per-redesign evaluation):

| Column | Description |
|--------|-------------|
| `{seq}_motif_rmsd_pred` | Best motif RMSD in the predicted/refolded structure |
| `{seq}_motif_rmsd_pred_all` | Motif RMSD per redesign (list) |
| `{seq}_motif_seq_rec` | Motif sequence recovery fraction |
| `{seq}_correct_motif_sequence` | Whether motif sequence is fully recovered (best) |
| `{seq}_correct_motif_sequence_all` | Motif sequence recovery per redesign (list) |
| `{seq}_has_ligand_clashes` | Ligand clash flag (ligand binder only, best) |
| `{seq}_has_ligand_clashes_all` | Ligand clash flag per redesign (list, ligand only) |

**Interface metric columns** (if pre/post-refolding metrics enabled):

| Column | Description |
|--------|-------------|
| `generated_n_interface_hbonds_tmol` | H-bond count (generated structure) |
| `generated_total_interface_hbond_energy_tmol` | H-bond energy (kcal/mol) |
| `refolded_{seq}_n_interface_hbonds_tmol` | H-bond count (refolded structure) |
| `generated_binder_interface_sc` | Shape complementarity |
| `generated_binder_interface_dSASA` | Buried surface area (A^2) |

---

## Reading Results in Python

### Loading and Basic Inspection

```python
import pandas as pd

# Load motif results
df = pd.read_csv("evaluation_results/my_run/motif_results_my_config_combined.csv")

# Check available columns
print(df.columns.tolist())

# Basic stats
print(f"Samples: {len(df)}")
print(f"Mean motif RMSD (CA): {df['_res_motif_rmsd_ca'].mean():.2f}")
print(f"Mean motif seq rec:   {df['_res_motif_seq_rec'].mean():.2f}")
```

### Working with List Columns

Some columns store lists of values (one per MPNN sequence). After loading from CSV, these are strings that need parsing:

```python
import ast

# Parse list columns
df["_res_scRMSD_ca_esmfold_all"] = df["_res_scRMSD_ca_esmfold_all"].apply(ast.literal_eval)

# Get the number of sequences evaluated per sample
df["n_seqs"] = df["_res_scRMSD_ca_esmfold_all"].apply(len)
```

### Filtering Successful Motif Scaffolds

```python
# Direct motif RMSD filter
good_motif = df[
    (df["_res_motif_rmsd_all_atom"] < 2.0) &
    (df["_res_motif_seq_rec"] >= 1.0)
]
print(f"Motif RMSD pass rate: {len(good_motif) / len(df) * 100:.1f}%")

# Full success criteria (motif + codesignability)
success = df[
    (df["_res_motif_seq_rec"] >= 1.0) &
    (df["_res_motif_rmsd_ca"] < 1.0) &
    (df["_res_motif_rmsd_all_atom"] < 2.0) &
    (df["_res_co_scRMSD_all_atom_esmfold"] < 2.0)
]
print(f"Motif success rate: {len(success) / len(df) * 100:.1f}%")
```

### Filtering Successful Protein Binders

```python
# Standard protein binder success
success = df[
    (df["mpnn_complex_i_pAE"] * 31 <= 7.0) &
    (df["mpnn_complex_pLDDT"] >= 0.9) &
    (df["mpnn_binder_scRMSD"] < 1.5)
]
print(f"Protein binder success rate: {len(success) / len(df) * 100:.1f}%")
```

### Filtering Successful Ligand Binders

```python
# Default ligand binder success thresholds
success = df[
    (df["mpnn_complex_min_ipAE"] * 31 < 2.0) &
    (df["mpnn_binder_scRMSD_ca"] < 2.0) &
    (df["mpnn_ligand_scRMSD_aligned_allatom"] < 5.0)
]
print(f"Ligand binder success rate: {len(success) / len(df) * 100:.1f}%")
```

### Filtering Successful Motif Protein Binders

```python
# Load motif binder results
df = pd.read_csv("evaluation_results/my_run/motif_binder_results_combined.csv")

# Joint binder + motif success (protein target)
success = df[
    (df["mpnn_fixed_complex_i_pAE"] * 31 <= 7.0) &
    (df["mpnn_fixed_complex_pLDDT"] >= 0.8) &
    (df["mpnn_fixed_binder_scRMSD_ca"] < 2.0) &
    (df["mpnn_fixed_motif_rmsd_pred"] < 2.0) &
    (df["mpnn_fixed_correct_motif_sequence"] >= 1.0)
]
print(f"Motif protein binder success rate: {len(success) / len(df) * 100:.1f}%")
```

### Filtering Successful Motif Ligand Binders (AME)

```python
# Load motif binder results
df = pd.read_csv("evaluation_results/my_ame_run/motif_binder_results_combined.csv")

# Joint binder + motif success (ligand target — includes clash check)
success = df[
    (df["mpnn_fixed_binder_scRMSD_bb3"] <= 2.0) &
    (df["mpnn_fixed_motif_rmsd_pred"] <= 1.5) &
    (df["mpnn_fixed_correct_motif_sequence"] >= 1.0) &
    (df["mpnn_fixed_has_ligand_clashes"] < 0.5)
]
print(f"Motif ligand binder success rate: {len(success) / len(df) * 100:.1f}%")
```

### Comparing Across Experiments

```python
# Load multiple experiments
dfs = []
for run in ["run_A", "run_B", "run_C"]:
    d = pd.read_csv(f"evaluation_results/{run}/motif_results_*_combined.csv")
    d["experiment"] = run
    dfs.append(d)

df_all = pd.concat(dfs, ignore_index=True)

# Compare by experiment
summary = df_all.groupby("experiment").agg(
    motif_rmsd_mean=("_res_motif_rmsd_all_atom", "mean"),
    motif_rmsd_median=("_res_motif_rmsd_all_atom", "median"),
    seq_rec_mean=("_res_motif_seq_rec", "mean"),
    des_rmsd_mean=("_res_scRMSD_ca_esmfold", "mean"),
)
print(summary)
```

---

## Success Criteria

| Result Type | Key Thresholds (defaults) |
|-------------|--------------------------|
| `protein_binder` | i_pAE*31 <= 7.0, pLDDT >= 0.9, scRMSD_ca < 1.5 A |
| `ligand_binder` | min_ipAE*31 < 2.0, scRMSD_ca < 2.0 A, ligand scRMSD < 5.0 A |
| `monomer` | designability scRMSD < 2.0 A |
| `monomer_motif` | motif RMSD + codesignability (see below) |
| `motif_protein_binder` | binder (i_pAE, pLDDT, scRMSD) + motif RMSD < 2.0 + seq recovery >= 1.0 |
| `motif_ligand_binder` | binder scRMSD_bb3 <= 2.0 + motif RMSD <= 1.5 + seq recovery + no clashes |

All thresholds are customizable via `aggregation.success_thresholds` in the analysis config. Details for each type below.

### Protein Binder Success

Default thresholds (`result_type: protein_binder`), based on AlphaProteo criteria:

| Metric | Column | Threshold | Direction |
|--------|--------|-----------|-----------|
| i_pAE | `{seq}_complex_i_pAE` | * 31 <= 7.0 | Lower is better |
| pLDDT | `{seq}_complex_pLDDT` | >= 0.9 | Higher is better |
| scRMSD_ca | `{seq}_binder_scRMSD_ca` | < 1.5 A | Lower is better |

A sample passes if **all three** thresholds are met for at least one redesigned sequence.

### Ligand Binder Success

Default thresholds (`result_type: ligand_binder`):

| Metric | Column | Threshold | Direction |
|--------|--------|-----------|-----------|
| min_ipAE | `{seq}_complex_min_ipAE` | * 31 < 2.0 | Lower is better |
| scRMSD_ca | `{seq}_binder_scRMSD_ca` | < 2.0 A | Lower is better |
| scRMSD_aligned_allatom | `{seq}_ligand_scRMSD_aligned_allatom` | < 5.0 A | Lower is better |

A sample passes if **all three** thresholds are met for at least one redesigned sequence.

### Customizing Binder Thresholds

Both protein and ligand binder thresholds can be overridden in the analysis config via `success_thresholds`. Each entry has a metric name, threshold, comparison operator, optional scale factor, and column prefix:

```yaml
# In binder_analyze.yaml or ligand_binder_analyze.yaml
aggregation:
  success_thresholds:
    i_pAE:
      threshold: 10.0       # less strict than default 7.0
      op: "<="
      scale: 31.0
      column_prefix: complex
    pLDDT:
      threshold: 0.8        # less strict than default 0.9
      op: ">="
      scale: 1.0
      column_prefix: complex
    scRMSD_ca:
      threshold: 2.0        # less strict than default 1.5
      op: "<"
      scale: 1.0
      column_prefix: binder
```

If `success_thresholds` is `null` (the default), the built-in defaults for the `result_type` are used (`DEFAULT_PROTEIN_BINDER_THRESHOLDS` or `DEFAULT_LIGAND_BINDER_THRESHOLDS`).

### Monomer Success

| Metric | Default threshold | Direction |
|--------|-------------------|-----------|
| `_res_scRMSD_ca_{model}` | < 2.0 A | Lower is better |
| `_res_co_scRMSD_ca_{model}` | < 2.0 A | Lower is better |
| `_res_co_scRMSD_all_atom_{model}` | < 2.0 A | Lower is better |

### Motif Success (Built-in Presets)

Two presets are evaluated automatically during `analyze`:

**`motif_success`** — Direct motif quality:
1. `_res_motif_seq_rec` >= 1.0 (perfect sequence recovery)
2. `_res_motif_rmsd_ca` < 1.0 A (CA motif RMSD)
3. `_res_motif_rmsd_all_atom` < 2.0 A (all-atom motif RMSD)
4. `_res_co_scRMSD_all_atom_{model}` < 2.0 A (full-structure codesignability)

**`refolded_motif_success`** — Refolded motif quality:
1. `_res_motif_seq_rec` >= 1.0
2. `_res_co_motif_scRMSD_ca_{model}` < 1.0 A (motif-region codesign CA scRMSD)
3. `_res_co_motif_scRMSD_all_atom_{model}` < 2.0 A (motif-region codesign all-atom scRMSD)
4. `_res_co_scRMSD_all_atom_{model}` < 2.0 A (full-structure codesignability)

**Custom success criteria** can be added in the config:

```yaml
aggregation:
  motif_success_criteria:
    - column: "_res_motif_rmsd_ca"
      threshold: 0.5
      op: "<"
    - column: "_res_motif_seq_rec"
      threshold: 1.0
      op: ">="
```

> **Tip atoms note:** For tasks with `atom_selection_mode: tip_atoms`, CA-based motif metrics are auto-filled with `0.0`, so CA threshold checks in success criteria pass automatically. The all-atom thresholds evaluate the actual tip atom RMSD.

### Motif Protein Binder Success

A sample is "successful" when at least one redesign passes ALL binder AND ALL motif criteria jointly:

**Binder criteria:**

| Metric | Default threshold | Direction |
|--------|-------------------|-----------|
| `{seq}_complex_i_pAE` (scaled by 31) | <= 7.0 | Lower is better |
| `{seq}_complex_pLDDT` | >= 0.8 | Higher is better |
| `{seq}_binder_scRMSD_ca` | < 2.0 A | Lower is better |

**Motif criteria (evaluated on the same redesign):**

| Metric | Default threshold | Direction |
|--------|-------------------|-----------|
| `{seq}_motif_rmsd_pred` | < 2.0 A | Lower is better |
| `{seq}_correct_motif_sequence` | >= 1.0 (perfect recovery) | Higher is better |

### Motif Ligand Binder Success (AME Pipeline)

Same joint per-redesign evaluation, but with ligand-specific criteria including clash detection:

**Binder criteria:**

| Metric | Default threshold | Direction |
|--------|-------------------|-----------|
| `{seq}_binder_scRMSD_bb3` | <= 2.0 A | Lower is better |

**Motif criteria (evaluated on the same redesign):**

| Metric | Default threshold | Direction |
|--------|-------------------|-----------|
| `{seq}_motif_rmsd_pred` | <= 1.5 A | Lower is better |
| `{seq}_correct_motif_sequence` | >= 1.0 (perfect recovery) | Higher is better |
| `{seq}_has_ligand_clashes` | < 0.5 (no clashes) | Lower is better |

### Customizing Motif Binder Thresholds

Both binder and motif criteria can be overridden in the analysis config. The `motif_binder_success_thresholds` has two sub-keys: `binder` (same format as standard binder thresholds) and `motif` (list of column/threshold/op dicts):

```yaml
# In analyze_motif_binder.yaml or ame_analyze.yaml
aggregation:
  motif_binder_success_thresholds:
    binder:
      i_pAE:
        threshold: 8.0
        op: "<="
        scale: 31.0
        column_prefix: complex
      scRMSD:
        threshold: 2.0
        op: "<"
        scale: 1.0
        column_prefix: binder
    motif:
      - column: "{seq_type}_motif_rmsd_pred_all"
        threshold: 1.5
        op: "<"
      - column: "{seq_type}_correct_motif_sequence_all"
        threshold: 1.0
        op: ">="
```

The `{seq_type}` placeholder in motif column names is automatically resolved to the active sequence type (e.g. `self`, `mpnn_fixed`) at analysis time.

---

## Metric Interpretation Cheat Sheet

### RMSD Metrics (Lower is Better)

| Value | Interpretation |
|-------|----------------|
| < 1.0 A | Excellent — near-identical structure |
| 1.0 - 2.0 A | Good — minor deviations, design likely preserved |
| 2.0 - 4.0 A | Moderate — noticeable structural change |
| > 4.0 A | Poor — structure not preserved |

### Binding Confidence (i_pAE, Lower is Better)

| Value | Interpretation |
|-------|----------------|
| < 5 | Excellent binding confidence |
| 5 - 8 | Good, likely successful |
| 8 - 15 | Moderate, needs validation |
| > 15 | Poor prediction |

### Structure Confidence (pLDDT, Higher is Better)

| Value | Interpretation |
|-------|----------------|
| > 90 | Very high confidence |
| 70 - 90 | Confident |
| 50 - 70 | Low confidence |
| < 50 | Very low / disordered |

### Folding Models

**Binder refolding** (`binder_folding_method`):

| Value | Model | Best for |
|-------|-------|----------|
| `colabdesign` | AlphaFold2 Multimer (ColabDesign) | Protein-protein binder refolding |
| `rf3_latest` | RoseTTAFold3 | Protein-ligand, high accuracy |
| `protenix_base_default_v0.5.0` | Protenix (base) | Alternative structure prediction |
| `protenix_mini_default_v0.5.0` | Protenix (mini) | Faster Protenix variant |

> ColabDesign does **not** support ligand targets. Use RF3 or Protenix for ligand binder evaluation.

**Monomer refolding** (`monomer_folding_models`):

| Value | Model | Best for |
|-------|-------|----------|
| `esmfold` | ESMFold | Fast monomer designability screening |
| `colabfold` | ColabFold (AF2 monomer) | Higher accuracy monomer refolding |

**Inverse folding** (`inverse_folding_model`):

| Value | Model | Best for |
|-------|-------|----------|
| `protein_mpnn` | ProteinMPNN | Standard protein redesign |
| `soluble_mpnn` | SolubleMPNN | Protein binder redesign (solubility-aware) |
| `ligand_mpnn` | LigandMPNN | Ligand-aware binder redesign |

RF3 requires environment variables (see [Environment Variables](#environment-variables)). Protenix uses its own checkpoint management.

---

## Environment Variables

These variables are used by the evaluation and generation pipelines when external tools are involved.

### RF3 (RoseTTAFold3)

Required when using RF3 for refolding (e.g. `binder_folding_method: rf3_latest` or RF3 reward during generation):

| Variable | Description |
|----------|-------------|
| `RF3_CKPT_PATH` | Path to the RF3 checkpoint file (e.g. `rf3_latest.pt`). |
| `RF3_EXEC_PATH` | Path to the RF3 executable. |

If either is unset, RF3 reward/refolding will fail at initialization with a clear error.

**RF3 output directory:** Predictions are written to an output directory passed at call time. In config you can set `search.rf3_dump_dir` (e.g. under the generation config) to control where RF3 writes results; if unset, the reward runner uses a default of `./rf3_outputs` (relative to the process working directory). The output directory is **not** created at reward initialization—it is created only when a prediction is actually run (e.g. by the binder evaluation pipeline or by the code that calls `reset_dump_dir` with a concrete path). This avoids leaving an empty `rf3_outputs` directory when RF3 is never used.

### Interface Metrics

Optional pre- and post-refolding interface metrics can be enabled in config:

```yaml
metric:
  pre_refolding:
    bioinformatics: true   # SC, SASA, hydrophobicity
    tmol: true             # H-bonds, electrostatics (requires TMOL)
  refolded:
    bioinformatics: true
    tmol: true
```

### ESM Model Cache

| Variable | Description |
|----------|-------------|
| `ESM_DIR` | Directory for ESM model cache |
| `CACHE_DIR` | Fallback cache directory if `ESM_DIR` is unset |

### Ligand Evaluation

| Variable | Description | Default |
|----------|-------------|---------|
| `LIGAND_CLASH_THRESHOLD` | Distance threshold (Å) for ligand clash detection | `1.5` |

### Novelty Computation

Novelty is computed using FoldSeek against reference databases. Enable via config:

| Config key | Database |
|------------|----------|
| `compute_novelty_pdb` | PDB |
| `compute_novelty_afdb` | AlphaFold DB |
| `compute_novelty_afdb_rep_v4` | AlphaFold DB representative v4 |
| `compute_novelty_afdb_rep_v4_geniefilters_maxlen512` | AlphaFold DB rep v4, filtered (max 512 residues) |

---

## Configuration Guide

For complete evaluation and analysis configuration examples -- including binder, monomer, combined, external PDB, custom ranking criteria, custom success thresholds, and training configs -- see the [Configuration Guide](CONFIGURATION_GUIDE.md).

### Minimal Config Quick Reference

**Protein binder** (`protein_type: binder`, `result_type: protein_binder`):

```yaml
protein_type: binder
metric:
  compute_binder_metrics: true
  binder_folding_method: colabdesign
  sequence_types: [mpnn]
  inverse_folding_model: soluble_mpnn
```

**Ligand binder** (`protein_type: binder`, `result_type: ligand_binder`):

```yaml
protein_type: binder
metric:
  compute_binder_metrics: true
  binder_folding_method: rf3_latest
  sequence_types: [mpnn]
  inverse_folding_model: ligand_mpnn
```

**Monomer** (`protein_type: monomer`, `result_type: monomer`):

```yaml
protein_type: monomer
metric:
  compute_monomer_metrics: true
  monomer_folding_models: [esmfold]
```

**Motif** (`protein_type: monomer_motif`, `result_type: monomer_motif`):

```yaml
protein_type: monomer_motif
dataset:
  motif_task_name: 1QJG_AA_TIP
  unindexed: true
metric:
  compute_motif_metrics: true
  compute_monomer_metrics: true
  monomer_folding_models: [esmfold]
```

### Common Evaluation Options

| Config key | Default | Description |
|------------|---------|-------------|
| `input_mode` | `generated` | `generated` (pipeline output) or `pdb_dir` (external PDBs) |
| `dryrun` | `false` | Show what would be evaluated without running |
| `show_progress` | `true` | Display progress bar |
| `file_limit` | `null` | Limit number of PDBs to evaluate (useful for testing) |
| `job_id` | `0` | Parallel job index (0-based) |
| `eval_njobs` | `1` | Total parallel jobs for splitting |
| `metric.num_redesign_seqs` | `8` | Number of MPNN redesign sequences per sample |
| `metric.interface_cutoff` | `8.0` | Distance cutoff (Å) for defining interface residues |
| `metric.compute_esm_metrics` | `false` | Compute ESM pseudo-perplexity |
| `metric.compute_pre_refolding_metrics` | `false` | Compute interface metrics on generated structure |
| `metric.compute_refolded_structure_metrics` | `false` | Compute interface metrics on refolded structure |
| `metric.keep_folding_outputs` | `true` | Retain intermediate folding output files |
| `metric.designability_num_seq` | `8` | Number of MPNN sequences for monomer designability |
