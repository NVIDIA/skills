# Inference and Search Guide

How to run protein design with Proteina-Complexa: local execution, SLURM cluster deployment, custom targets, and troubleshooting.

> **Documentation Map**
> - Tuning YAML configs? See [Configuration Guide](CONFIGURATION_GUIDE.md).
> - Understanding metrics? See [Evaluation Guide](EVALUATION_METRICS.md).
> - Parameter sweeps? See [Sweep System](SWEEP.md).
> - Search metadata? See [Search Metadata](SEARCH_METADATA.md).

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Design Pipeline Types](#design-pipeline-types)
3. [Configuration Architecture](#configuration-architecture)
4. [Running Locally](#running-locally)
5. [SLURM Cluster Execution](#slurm-cluster-execution)
6. [Defining Custom Targets](#defining-custom-targets)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

---

## Pipeline Overview

All design pipelines share the same four-stage structure:

| Stage | Module | Config Section | Description |
|-------|--------|----------------|-------------|
| 1. Generate | `proteinfoundation.generate` | `generation.*` | Sample structures using flow matching + reward scoring |
| 2. Filter | `proteinfoundation.filter` | `generation.filter.*` | Filter samples by reward scores |
| 3. Evaluate | `proteinfoundation.evaluate` | `metric.*` | Redesign sequences and validate with structure prediction |
| 4. Analyze | `proteinfoundation.analyze` | `aggregation.*` | Aggregate metrics, compute success rates, diversity |

---

## Design Pipeline Types

There are three main design pipelines, each targeting a different design task:

### Protein Binder Pipeline

Design protein binders for target proteins. Uses AF2 as the primary reward model and ColabDesign (AF2) or RF3 for evaluation refolding.

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=my_binder ++generation.task_name=02_PDL1
```

| Aspect | Setting |
|--------|---------|
| Model | Protein model (`complexa.ckpt`) |
| Generation reward | AF2 folding (TMOL, bioinformatics optional) |
| Inverse folding | SolubleMPNN |
| Evaluation folding | ColabDesign (default), RF3 |
| Evaluation type | `protein_type: binder`, `result_type: protein_binder` |
| Analysis modes | `[binder, monomer]` |

### Ligand Binder Pipeline

Design proteins that bind small-molecule ligands. Uses RF3 for both reward and evaluation since it can handle protein-ligand complexes.

```bash
complexa design configs/search_ligand_binder_local_pipeline.yaml \
    ++run_name=my_ligand_binder ++generation.task_name=39_7V11_LIGAND
```

| Aspect | Setting |
|--------|---------|
| Model | Ligand model with LoRA (`complexa_ligand.ckpt`) |
| Generation reward | RF3 folding |
| Inverse folding | LigandMPNN |
| Evaluation folding | RF3 |
| Evaluation type | `protein_type: binder`, `result_type: ligand_binder` |
| Analysis modes | `[binder, monomer]` |

### AME Pipeline (Motif + Ligand Binder)

Scaffold functional motifs with ligand context. Combines motif features (atom-spec mode) and ligand features. Uses RF3 for reward and evaluation.

```bash
complexa design configs/search_ame_local_pipeline.yaml \
    ++run_name=my_ame ++generation.task_name=M0096_1chm
```

| Aspect | Setting |
|--------|---------|
| Model | AME model with LoRA (`complexa_ame.ckpt`) |
| Generation reward | RF3 folding |
| Inverse folding | LigandMPNN |
| Evaluation folding | RF3 |
| Evaluation type | `protein_type: motif_binder`, `result_type: motif_ligand_binder` |
| Analysis modes | `[motif_binder, binder, monomer]` |
| Extra metrics | Motif RMSD, motif sequence recovery, ligand clash detection |

### Motif Binder Evaluation (Standalone)

Each base binder type (protein, ligand) has a **motif counterpart** that adds motif preservation metrics on top of the standard binder evaluation. The AME pipeline automatically uses `motif_ligand_binder`, but you can also run motif binder evaluation standalone on outputs from any binder pipeline:

```bash
# Motif protein binder evaluation (on outputs from protein binder pipeline)
complexa evaluate configs/evaluate_motif_binder.yaml \
    ++dataset.task_name=MY_MOTIF_TASK \
    ++metric.binder_folding_method=colabdesign \
    ++metric.inverse_folding_model=soluble_mpnn

# Analysis (set result_type to match)
complexa analyze configs/analyze_motif_binder.yaml \
    ++result_type=motif_protein_binder
```

| Variant | `result_type` | Binder thresholds | Motif thresholds |
|---------|---------------|-------------------|------------------|
| Motif Protein Binder | `motif_protein_binder` | i_pAE*31 <= 7.0, pLDDT >= 0.8, scRMSD_ca < 2.0 | motif_rmsd < 2.0, seq_recovery >= 1.0 |
| Motif Ligand Binder | `motif_ligand_binder` | scRMSD_bb3 <= 2.0 | motif_rmsd <= 1.5, seq_recovery >= 1.0, no ligand clashes |

---

## Configuration Architecture

The pipeline uses a modular config system. Each top-level pipeline config composes stage-specific sub-configs via Hydra defaults:

```
configs/search_binder_local_pipeline.yaml
├── pipeline/binder/binder_generate.yaml    → generation.*
├── pipeline/binder/binder_evaluate.yaml    → metric.*
└── pipeline/binder/binder_analyze.yaml     → aggregation.*

configs/search_ligand_binder_local_pipeline.yaml
├── pipeline/ligand_binder/ligand_binder_generate.yaml    → generation.*
├── pipeline/ligand_binder/ligand_binder_evaluate.yaml    → metric.*
└── pipeline/ligand_binder/ligand_binder_analyze.yaml     → aggregation.*

configs/search_ame_local_pipeline.yaml
├── pipeline/ame/ame_generate.yaml    → generation.*
├── pipeline/ame/ame_evaluate.yaml    → metric.*
└── pipeline/ame/ame_analyze.yaml     → aggregation.*
```

For the full config structure, pipeline YAML examples, and every configurable parameter, see the [Configuration Guide](CONFIGURATION_GUIDE.md).

> **Note**: You can also run individual modules directly with `python -m proteinfoundation.generate`, `python -m proteinfoundation.evaluate`, etc. The `complexa` CLI wraps these with additional validation and logging.

---

## Running Locally

### Quick Start

```bash
# Protein binder design
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=my_binder ++generation.task_name=02_PDL1

# Ligand binder design
complexa design configs/search_ligand_binder_local_pipeline.yaml \
    ++run_name=my_ligand_binder ++generation.task_name=39_7V11_LIGAND

# AME motif scaffolding
complexa design configs/search_ame_local_pipeline.yaml \
    ++run_name=my_ame ++generation.task_name=M0096_1chm
```

### Validate Before Running

```bash
complexa validate design configs/search_binder_local_pipeline.yaml
complexa validate design configs/search_ligand_binder_local_pipeline.yaml
complexa validate design configs/search_ame_local_pipeline.yaml
```

### Individual Stages

```bash
complexa generate configs/search_binder_local_pipeline.yaml
complexa filter configs/search_binder_local_pipeline.yaml
complexa evaluate configs/search_binder_local_pipeline.yaml
complexa analyze configs/search_binder_local_pipeline.yaml
```

The same stage commands work for all pipeline configs -- just substitute the config path.

### CLI Options

```bash
# Verbose mode (show output instead of logging to file)
complexa design configs/search_binder_local_pipeline.yaml --verbose

# Override any config parameter with ++key=value
complexa design configs/search_binder_local_pipeline.yaml \
    ++generation.args.nsteps=200 \
    ++metric.binder_folding_method=rf3_latest
```

### Common Overrides

```bash
# Change target
++generation.task_name=33_TrkA

# Change search algorithm
++generation.search.algorithm=beam-search
++generation.search.beam_search.beam_width=8

# Change sampling steps (fewer = faster, lower quality)
++generation.args.nsteps=200

# Change folding model for evaluation
++metric.binder_folding_method=rf3_latest

# Change reward weights (protein binder -- AF2)
++generation.reward_model.reward_models.af2folding.reward_weights.i_pae=-2.0

# Change reward weights (ligand binder / AME -- RF3)
++generation.reward_model.reward_models.rf3folding.reward_weights.min_ipAE=-2.0

# Change success thresholds for analysis
++aggregation.success_thresholds.i_pAE.threshold=5.0

# Change filter settings
++generation.filter.filter_samples_limit=500
++generation.filter.reward_threshold=0.5
```

For the full list of configurable parameters, see the [Configuration Guide](CONFIGURATION_GUIDE.md#common-overrides-cheat-sheet).

### Quick Local Test

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=quick_test \
    ++generation.task_name=02_PDL1 \
    ++generation.args.nsteps=100 \
    ++generation.dataloader.dataset.nres.nsamples=2
```

| Config | Use case |
|--------|----------|
| `search_binder_local_pipeline.yaml` | Protein-protein binder design (local) |
| `search_ligand_binder_local_pipeline.yaml` | Small-molecule binder design (local) |
| `search_ame_local_pipeline.yaml` | AME motif scaffolding (local) |
| `search_binder_pipeline.yaml` | Protein binder design (SLURM cluster) |

---

## SLURM Cluster Execution

### Setup

1. Create user configuration:

```bash
cp slurm_utils/.user_info_example slurm_utils/.user_info
```

2. Edit `.user_info` with your cluster credentials:

```bash
USER="your_username"
REMOTE="cluster.example.com"
ROOT_REMOTE="/path/to/workspace"
ENV_BINDER="complexa"
PYTHON_BINDER="/path/to/python"
MAMBA_EXEC="/path/to/mamba"
```

### Launch Scripts

**From local machine (SSH to cluster):**

```bash
# Single target
bash slurm_utils/launch_protein_binder_search_from_local_conda.sh 02_PDL1

# Single target with run number
bash slurm_utils/launch_protein_binder_search_from_local_conda.sh 02_PDL1 2

# All targets from config
bash slurm_utils/launch_protein_binder_search_from_local_conda.sh

# Multiple targets via loop
bash slurm_utils/launch_protein_binder_search_target_loop_conda.sh
```

**Directly on cluster:**

```bash
./slurm_utils/launch_protein_binder_search_from_slurm_conda.sh 02_PDL1
./slurm_utils/launch_protein_binder_search_from_slurm_conda.sh  # all targets
```

### What the Launcher Does

1. **Config generation** -- Creates per-run configs in `configs/inference_configs/` and `configs/eval_configs/`
2. **Code sync** -- Rsyncs code to cluster (local launcher only)
3. **Job submission** -- Submits SLURM array jobs for each stage
4. **Monitoring** -- Waits for each stage to complete before proceeding
5. **Result download** -- Downloads results locally (local launcher only)

### Job Parallelism

Use `gen_njobs` and `eval_njobs` to parallelize across samples:

```bash
complexa evaluate configs/search_binder_pipeline.yaml \
    ++eval_njobs=20 \
    ++job_id=$SLURM_ARRAY_TASK_ID
```

Set `eval_njobs` to match `gen_njobs` so each eval job processes one generation job's outputs.

---

## Defining Custom Targets

### Protein Targets

Add entries to `configs/targets/targets_dict.yaml`:

```yaml
target_dict_cfg:
  my_target:
    source: my_targets           # Subfolder in $DATA_PATH/target_data/
    target_filename: my_protein  # PDB filename (without .pdb)
    target_input: A1-150         # Chain and residue range
    hotspot_residues: [A45, A67, A89]
    binder_length: [60, 120]     # [min, max] binder length
    pdb_id: 1abc                 # Optional: PDB ID reference
```

Then run:

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++generation.task_name=my_target
```

### Ligand Targets

Add entries to `configs/targets/ligand_targets_dict.yaml` with additional fields:

```yaml
target_dict_cfg:
  my_ligand_target:
    source: my_targets
    target_filename: my_complex
    target_input: A1-200
    binder_length: [60, 120]
    pdb_id: 2xyz
    res_name: LIG               # Ligand residue name in PDB
    ligand_only: false           # Whether the target is ligand-only
    SMILES: "CCO"               # SMILES string for the ligand
    use_bonds_from_file: true    # Use bond topology from PDB
```

### Target Input Format

The `target_input` field specifies which residues to use:

- `A1-150` -- Chain A, residues 1-150
- `A1-100,B1-50` -- Multiple chains/ranges
- `A` -- Entire chain A

### Hotspot Residues

Interface residues the binder should contact:

```yaml
hotspot_residues: [A45, A67, A89, A102]
```

---

## Troubleshooting

### Import Errors

Set PYTHONPATH if running without installation:

```bash
export PYTHONPATH=/path/to/project/src:/path/to/project/community_models:$PYTHONPATH
```

### Missing Model Weights

```bash
complexa download --status
```

### Config Validation Failures

```bash
complexa validate design configs/search_binder_local_pipeline.yaml --verbose
```

### RF3 Environment Variables

RF3 requires `RF3_CKPT_PATH` and `RF3_EXEC_PATH` to be set. Add them to your `.env` or export them:

```bash
export RF3_CKPT_PATH=/path/to/rf3_latest.pt
export RF3_EXEC_PATH=/path/to/rf3
```

### SLURM Job Failures

```bash
ls slurm_run_outputs/inf/   # Generation logs
ls slurm_run_outputs/eval/  # Evaluation logs
```

### Memory Issues

Reduce batch size or parallelization:

```bash
++generation.dataloader.batch_size=8
++gen_njobs=1
++eval_njobs=1
```

---

## Examples

### Production Beam Search (Protein Binder)

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=pdl1_beam_v1 \
    ++generation.task_name=02_PDL1 \
    ++generation.search.algorithm=beam-search \
    ++generation.search.beam_search.beam_width=8 \
    ++generation.search.beam_search.n_branch=4
```

### Multiple Targets on SLURM

```bash
# Launch all targets defined in config
bash slurm_utils/launch_protein_binder_search_target_loop_conda.sh

# Or specific targets
for target in 02_PDL1 33_TrkA 24_SpCas9; do
    bash slurm_utils/launch_protein_binder_search_from_local_conda.sh $target
done
```

### High-Quality Protein Binder Campaign

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=high_quality_campaign \
    ++generation.search.algorithm=beam-search \
    ++generation.search.beam_search.beam_width=8 \
    ++metric.binder_folding_method=rf3_latest \
    ++metric.num_redesign_seqs=16
```

To also enable TMOL rewards during generation, uncomment the `tmol` sub-model in `binder_generate.yaml` or add it via CLI:

```bash
++generation.reward_model.reward_models.tmol._target_=proteinfoundation.rewards.tmol_reward.TmolRewardModel \
++generation.reward_model.reward_models.tmol.enable_hbond=true \
++generation.reward_model.weights.tmol=0.5
```

### Custom Success Criteria

```bash
# Protein binder with stricter thresholds
complexa analyze configs/search_binder_local_pipeline.yaml \
    ++aggregation.success_thresholds.i_pAE.threshold=5.0 \
    ++aggregation.success_thresholds.scRMSD.threshold=1.0

# AME motif binder with custom thresholds
complexa analyze configs/search_ame_local_pipeline.yaml \
    ++aggregation.motif_binder_success_thresholds.motif_rmsd_pred.threshold=1.0 \
    ++aggregation.motif_binder_success_thresholds.motif_seq_recovery.threshold=0.8
```

### Custom Filter Settings

```bash
complexa filter configs/search_binder_local_pipeline.yaml \
    ++generation.filter.filter_samples_limit=200 \
    ++generation.filter.reward_threshold=0.3
```
