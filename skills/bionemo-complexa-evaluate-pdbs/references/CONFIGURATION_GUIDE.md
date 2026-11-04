# Pipeline Configuration Guide

The single reference for all YAML configuration in the design pipeline: search/generation, reward models, evaluation, analysis, and training.

> **Documentation Map**
> - Running a design? See [Inference Guide](INFERENCE.md).
> - Understanding metrics? See [Evaluation Guide](EVALUATION_METRICS.md).
> - Parameter sweeps? See [Sweep System](SWEEP.md).
> - Search metadata? See [Search Metadata](SEARCH_METADATA.md).

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
   - [Common Overrides Cheat Sheet](#common-overrides-cheat-sheet)
2. [Search & Generation Configs](#search--generation-configs)
   - [Config Architecture](#config-architecture)
   - [Protein Binder Pipeline](#protein-binder-pipeline)
   - [Ligand Binder Pipeline](#ligand-binder-pipeline)
   - [Search Algorithms](#search-algorithms)
   - [Refinement](#refinement)
   - [Reward Models](#reward-models)
   - [CompositeRewardModel](#compositerewardmodel)
   - [AF2 Reward (Protein Targets)](#af2-reward-protein-targets)
   - [RF3 Reward (Ligand or Protein Targets)](#rf3-reward-ligand-or-protein-targets)
   - [Interface Reward Models](#interface-reward-models)
   - [Model-Level Weights](#model-level-weights)
   - [Model Sampling Parameters](#model-sampling-parameters)
   - [Running the Pipeline](#running-the-pipeline)
3. [Evaluation Configs](#evaluation-configs)
   - [Evaluation Config Reference](#evaluation-config-reference)
   - [External PDB Directory](#external-pdb-directory)
4. [Analysis Configs](#analysis-configs)
   - [Protein Binder Analysis](#protein-binder-analysis)
   - [Ligand Binder Analysis](#ligand-binder-analysis)
   - [Monomer Analysis](#monomer-analysis)
   - [Custom Success Thresholds](#custom-success-thresholds)
   - [Custom Ranking Criteria](#custom-ranking-criteria)
5. [Training Configs](#training-configs)
6. [Common Patterns](#common-patterns)
   - [Config Composition (Hydra Defaults)](#config-composition-hydra-defaults)
   - [Command-Line Overrides](#command-line-overrides)
   - [Job Parallelism](#job-parallelism)

---

## Pipeline Overview

The design pipeline runs four stages sequentially:

```
generate  →  filter  →  evaluate  →  analyze
```

| Stage | What it does | Config section |
|-------|-------------|----------------|
| **Generate** | Sample binder structures via flow matching, optionally guided by search algorithms and reward models | `generation.*` |
| **Filter** | Rank and filter generated samples by reward scores | `generation.search.*` |
| **Evaluate** | Redesign sequences (ProteinMPNN) and validate with structure prediction (AF2/RF3) | `metric.*` |
| **Analyze** | Aggregate per-sample metrics into summary CSVs with success rates and diversity | `aggregation.*` |

A single pipeline config file composes all four stages via Hydra defaults.

### Common Overrides Cheat Sheet

| What to change | CLI override | Example |
|----------------|--------------|---------|
| Target | `++generation.task_name=...` | `02_PDL1` |
| Search algorithm | `++generation.search.algorithm=...` | `beam-search` |
| Beam width | `++generation.search.beam_search.beam_width=...` | `8` |
| Sampling steps | `++generation.args.nsteps=...` | `200` |
| Batch size | `++generation.dataloader.batch_size=...` | `8` |
| Folding method | `++metric.binder_folding_method=...` | `rf3_latest` |
| Filter limit | `++generation.filter.filter_samples_limit=...` | `500` |
| Reward threshold | `++generation.filter.reward_threshold=...` | `0.3` |
| Success threshold | `++aggregation.success_thresholds.i_pAE.threshold=...` | `5.0` |
| Redesign count | `++metric.num_redesign_seqs=...` | `16` |

---

## Search & Generation Configs

### Config Architecture

The pipeline uses a modular config system. A top-level pipeline config composes sub-configs for each stage:

```
configs/search_binder_local_pipeline.yaml          # Protein binder pipeline
configs/search_ligand_binder_local_pipeline.yaml    # Ligand binder pipeline
│
├── pipeline/binder/binder_generate.yaml          → generation.*
│   ├── pipeline/binder/model_sampling.yaml       → generation.args.*, generation.model.*
│   └── pipeline/binder/base_gen_data.yaml        → generation.dataloader base
│
├── pipeline/binder/binder_evaluate.yaml          → metric.*
└── pipeline/binder/binder_analyze.yaml           → aggregation.*
```

### Protein Binder Pipeline

The main protein binder pipeline config. This is what you run for protein-protein binder design.

```yaml
# configs/search_binder_local_pipeline.yaml
defaults:
  - pipeline/binder/binder_generate@generation      # Search, rewards, targets → generation.*
  - pipeline/binder/binder_evaluate@_global_        # Evaluation metrics → metric.*
  - pipeline/binder/binder_analyze@_global_         # Analysis thresholds → aggregation.*
  - _self_

run_name: search_binder_local
ckpt_path: /path/to/checkpoints
ckpt_name: complexa.ckpt
autoencoder_ckpt_path: /path/to/checkpoints/complexa_ae.ckpt

ncpus_: 24
seed: 5
gen_njobs: 2
eval_njobs: 2
```

Run the full pipeline:

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=my_experiment \
    ++generation.task_name=02_PDL1
```

Or run individual stages:

```bash
complexa generate configs/search_binder_local_pipeline.yaml
complexa filter configs/search_binder_local_pipeline.yaml
complexa evaluate configs/search_binder_local_pipeline.yaml
complexa analyze configs/search_binder_local_pipeline.yaml
```

### Ligand Binder Pipeline

For small-molecule binder design. Uses a separate checkpoint (ligand model), RF3 as the folding reward, and `LigandFeatures` for conditional generation.

```yaml
# configs/search_ligand_binder_local_pipeline.yaml
defaults:
  - pipeline/ligand_binder/ligand_binder_generate@generation
  - pipeline/ligand_binder/ligand_binder_evaluate@_global_
  - pipeline/ligand_binder/ligand_binder_analyze@_global_
  - _self_

run_name: search_ligand_binder_local
ckpt_path: /path/to/checkpoints
ckpt_name: complexa_ligand.ckpt
autoencoder_ckpt_path: /path/to/checkpoints/complexa_ligand_ae.ckpt

# LoRA is required for the paper ligand model checkpoint
lora:
  r: 32
  lora_alpha: 64.0
  lora_dropout: 0.0
  train_bias: none

ncpus_: 24
seed: 5
gen_njobs: 2
eval_njobs: 2
```

**Key differences from protein pipeline:**

- Uses `ligand_binder_generate` (RF3 reward, `LigandFeatures` conditioning)
- Uses `ligand_binder_evaluate` (`rf3_latest` folding, `ligand_mpnn` inverse folding)
- Requires LoRA config matching the ligand checkpoint
- Target definitions come from `configs/targets/ligand_targets_dict.yaml`

### Search Algorithms

Configured in `pipeline/binder/binder_generate.yaml` (or `ligand_binder/ligand_binder_generate.yaml`) under the `search:` key. CLI path: `++generation.search.*`.

```yaml
search:
  algorithm: best-of-n     # single-pass, best-of-n, beam-search, fk-steering, mcts
  reward_threshold: null

  step_checkpoints: [0, 100, 200, 300, 400]   # Denoising steps where rewards are computed

  best_of_n:
    replicas: 2             # Number of independent generation runs per batch element

  beam_search:
    n_branch: 4             # Candidates to generate at each checkpoint
    beam_width: 4           # Top candidates to keep at each checkpoint
    keep_lookahead_samples: true

  fk_steering:
    n_branch: 4
    beam_width: 4
    temperature: 0.1        # Boltzmann temperature for selection
    keep_lookahead_samples: true

  mcts:
    n_simulations: 20
    exploration_prob: 0.5
    exploration_constant: 1.0
    keep_lookahead_samples: true
```

| Algorithm | Description | When to use |
|-----------|-------------|-------------|
| `single-pass` | Standard generation, no search | Baseline, fast sampling |
| `best-of-n` | Generate N replicas, keep the best | Simple, embarrassingly parallel |
| `beam-search` | Branch and prune at checkpoints | Highest quality, more compute |
| `fk-steering` | Temperature-weighted selection | Balance exploration/exploitation |
| `mcts` | Monte Carlo tree search | Exploration-heavy campaigns |

**CLI examples:**

```bash
# Switch to beam search with wider beam
++generation.search.algorithm=beam-search \
++generation.search.beam_search.beam_width=8 \
++generation.search.beam_search.n_branch=8

# Increase best-of-n replicas
++generation.search.algorithm=best-of-n \
++generation.search.best_of_n.replicas=10
```

### Post-Generation Filtering

Configured under the `filter:` key (CLI path: `++generation.filter.*`). Runs after generation to rank and prune samples.

```yaml
filter:
  filter_samples_limit: 1000        # Max samples to keep (top-N by reward)
  delete_non_top_n_samples: false   # true = delete unselected dirs, false = move to filtered_out_samples/
  dedup_sequence: true              # Deduplicate identical sequences before ranking
  reward_threshold: null            # Drop samples below this reward before top-N selection
```

### Refinement

Refinement is an optional post-search optimisation step that improves binder sequences using ColabDesign's AlphaFold2 design pipeline. It runs **after** the search algorithm on the final selected samples and **before** reward scoring.

```
search  -->  refinement (optional)  -->  reward scoring  -->  output
```

Set `refinement.algorithm` to `null` (default) to skip refinement, or `sequence_hallucination` to enable it.

```yaml
# In pipeline/binder_generate.yaml (CLI path: ++generation.refinement.*)
refinement:
  algorithm: null                     # null or sequence_hallucination
  refine_targets: final               # "final" (default) or "all" (final + lookahead)
  save_pre_refinement: none           # "none", "final", or "all" -- keep unrefined copies

  # Stage toggles
  enable_soft_optimization: false     # Stages 2+3: softmax + one-hot optimisation
  enable_greedy_optimization: true    # Stage 4: PSSM semigreedy optimisation

  # Iteration counts
  n_temp_iters: 45                    # Softmax temperature annealing iterations (stage 2)
  n_hard_iters: 5                     # One-hot optimisation iterations (stage 3)
  n_greedy_iters: 15                  # Semigreedy hard iterations (stage 4)
  n_recycles: 3                       # AF2 recycle count
  greedy_percentage: 1                # % of residues to try per greedy step

  # Loss weights for ColabDesign AF2 optimisation
  loss_weights:
    pae: 0.4
    plddt: 0.1
    i_pae: 0.1
    con: 1.0
    i_con: 1.0
    dgram_cce: 0.0
    rg: 0.3
    i_ptm: 0.05
    helix_binder: -0.3
    nc_termini: 0.0
    alignment_bb_ca: 0.0
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `algorithm` | `null` (off) or `sequence_hallucination` | `null` |
| `refine_targets` | Which samples are replaced with refined versions: `final` or `all` (final + lookahead). Non-targeted samples pass through unchanged. | `final` |
| `save_pre_refinement` | Also output pre-refinement copies alongside the refined ones: `none`, `final`, or `all` | `none` |
| `enable_soft_optimization` | Run softmax + one-hot stages (higher quality, slower) | `false` |
| `enable_greedy_optimization` | Run PSSM semigreedy stage | `true` |
| `n_temp_iters` | Softmax annealing iterations | `45` |
| `n_hard_iters` | One-hot iterations | `5` |
| `n_greedy_iters` | Semigreedy hard iterations | `15` |
| `n_recycles` | AF2 recycle count | `3` |
| `greedy_percentage` | % of binder residues to try per greedy step (higher = more aggressive) | `1` |

**Loss weight reference:**

| Weight | What it controls | Recommended range |
|--------|-----------------|-------------------|
| `pae` | Predicted aligned error | 0.1 - 1.0 |
| `plddt` | Predicted LDDT confidence | 0.05 - 0.5 |
| `i_pae` | Interface PAE | 0.05 - 0.5 |
| `con` | Intra-chain contact loss | 0.5 - 2.0 |
| `i_con` | Interface contact loss | 0.5 - 2.0 |
| `dgram_cce` | Distogram cross-entropy | 0.0 (usually off) |
| `rg` | Radius of gyration (binder compactness) | 0.1 - 0.5 |
| `i_ptm` | Interface pTM score | 0.01 - 0.1 |
| `helix_binder` | Helicity (negative = encourage helices) | -0.5 to 0.0 |
| `nc_termini` | N-C termini distance penalty | 0.0 - 0.3 |
| `alignment_bb_ca` | Backbone CA alignment to input | 0.0 - 0.3 |

**When to use refinement:**

- **Hard targets** where search alone does not produce good binders: enable both soft and greedy with `greedy_percentage: 5`
- **Quick polish** on easy targets: greedy-only (default) with `greedy_percentage: 1`
- **Maximum quality**: enable soft + greedy, increase `n_temp_iters` to 60-80

**Controlling what gets refined and saved:**

Only samples targeted by `refine_targets` are replaced with their refined versions. Everything else passes through unchanged -- for example, with `refine_targets: final` lookaheads are always kept as-is.

`save_pre_refinement` optionally saves copies of the structures *before* refinement so you can compare side-by-side. Rewards are computed for all saved samples.

| `refine_targets` | `save_pre_refinement` | What happens |
|-------------------|-----------------------|-------------|
| `final` | `none` | Finals are refined and replace originals. Lookaheads pass through unchanged. |
| `final` | `final` | Same, but also keeps the unrefined finals (`final_unrefined`). |
| `all` | `none` | Finals and lookaheads are both refined, replacing originals. |
| `all` | `final` | Both refined, plus unrefined finals kept for comparison. |
| `all` | `all` | Both refined, plus unrefined copies of both kept. |

**Error handling:** If refinement fails for an individual sample (e.g. residue count mismatch from ColabDesign), the original unrefined structure is kept and processing continues for remaining samples.

**CLI examples:**

```bash
# Enable sequence hallucination with greedy-only (fast)
++generation.refinement.algorithm=sequence_hallucination

# Enable both stages for hard targets
++generation.refinement.algorithm=sequence_hallucination \
  ++generation.refinement.enable_soft_optimization=true \
  ++generation.refinement.greedy_percentage=5

# Tune loss weights
++generation.refinement.loss_weights.rg=0.5 \
  ++generation.refinement.loss_weights.nc_termini=0.2

# Refine all samples, keep unrefined copies for comparison
++generation.refinement.refine_targets=all \
  ++generation.refinement.save_pre_refinement=all

# Refine finals only, but keep unrefined finals to compare
++generation.refinement.save_pre_refinement=final

# Disable refinement
++generation.refinement.algorithm=null
```

### Reward Models

Reward models score generated structures during search. They are configured under the `reward_model:` key in the generation config.

### CompositeRewardModel

All pipeline configs use `CompositeRewardModel`, which combines multiple reward sub-models. Each sub-model computes its own `total_reward`, and the composite sums them (optionally weighted).

```yaml
reward_model:
  _target_: "proteinfoundation.rewards.base_reward.CompositeRewardModel"
  reward_models:
    model_name_1:
      _target_: "..."
      # model-specific config
    model_name_2:
      _target_: "..."
      # model-specific config

  # Optional model-level weights (default 1.0 if omitted)
  # weights:
  #   model_name_1: 1.0
  #   model_name_2: 0.5
```

**Execution order:**

1. **Folding models** run first (AF2, RF3) and produce refolded structures
2. **Interface models** run second (TMOL, Bioinformatics) and can optionally use refolded structures via `structure_source`

**Final reward:** `total_reward = sum(weight_i * model_i.total_reward)`

Component rewards are prefixed with the model name in output CSVs (e.g., `af2folding_i_pae`, `tmol_hbond_count`).

### AF2 Reward (Protein Targets)

Primary reward for protein-protein binder design. Runs AlphaFold2 Multimer to predict the complex and scores binding confidence.

```yaml
reward_model:
  _target_: "proteinfoundation.rewards.base_reward.CompositeRewardModel"
  reward_models:
    af2folding:
      _target_: "proteinfoundation.rewards.alphafold2_reward.AF2RewardModel"
      protocol: "binder"
      use_multimer: true
      af_params_dir: ${oc.env:AF2_DIR}
      num_recycles: 3
      use_initial_guess: true
      use_initial_atom_pos: false
      seed: 0
      device_id: null         # Auto-detects current CUDA device
      reward_weights:
        i_pae: -1.0           # Interface PAE (primary, lower is better → negative weight)
        con: 0.0              # Intra-binder confidence
        dgram_cce: 0.0        # Distance gram cross-entropy
        min_ipae: 0.0         # Minimum interface PAE
        min_ipsae: 0.0        # ipSAE metrics
        avg_ipsae: 0.0
        max_ipsae: 0.0
        min_ipsae_10: 0.0     # ipSAE with 10A cutoff
        max_ipsae_10: 0.0
        avg_ipsae_10: 0.0
```

**Reward weight convention:** Set the weight to `0.0` to disable a metric. Use negative weights for metrics where lower is better (e.g., `i_pae: -1.0`), positive for metrics where higher is better.

**CLI:**

```bash
# Increase weight on i_pae
++generation.reward_model.reward_models.af2folding.reward_weights.i_pae=-2.0

# Also reward pLDDT
++generation.reward_model.reward_models.af2folding.reward_weights.plddt=0.5
```

### RF3 Reward (Ligand or Protein Targets)

Primary reward for ligand binder design. Runs RoseTTAFold3 via CLI to predict the complex. Can also be used for protein-protein targets.

```yaml
reward_model:
  _target_: "proteinfoundation.rewards.base_reward.CompositeRewardModel"
  reward_models:
    rf3folding:
      _target_: "proteinfoundation.rewards.rf3_reward.RF3RewardRunner"
      ckpt_path: ${oc.env:RF3_CKPT_PATH}
      rf3_path: ${oc.env:RF3_EXEC_PATH}
      normalize_pae: true      # Divide PAE-family metrics by 31 for 0-1 scale
      reward_weights:
        min_ipAE: -1.0         # Minimum interface PAE (primary, lower → negative weight)
        plddt: 0.0             # Structure confidence (higher is better)
        ipAE: 0.0              # Mean interface PAE
        mean_min_ipAE: 0.0
        mean_ipAE: 0.0
        min_mean_ipAE: 0.0
        pAE: 0.0               # Overall PAE
        ipTM: 0.0              # Interface pTM score
        pTM: 0.0               # Overall pTM score
        ranking_score: 0.0     # RF3 composite ranking
        has_clash: 0.0         # 1.0 if clash, 0.0 if none; use negative weight to penalize
        min_ipSAE: 0.0         # Interface pSAE metrics (higher is better)
        max_ipSAE: 0.0
        avg_ipSAE: 0.0
```

**`normalize_pae`:** When `true` (default), PAE-family metrics (`ipAE`, `min_ipAE`, `pAE`, etc.) are divided by 31.0 before applying weights. This normalizes them to the 0-1 range so you can use simple weights like `-1.0` instead of `-1/31`.

**`has_clash`:** Converted from boolean to numeric (1.0/0.0). To penalize clashes, set a negative weight (e.g., `has_clash: -5.0`).

**Environment variables required:** `RF3_CKPT_PATH` and `RF3_EXEC_PATH` must be set.

**CLI:**

```bash
# Use RF3 as reward in a protein binder pipeline
++generation.reward_model.reward_models.rf3folding._target_="proteinfoundation.rewards.rf3_reward.RF3RewardRunner" \
++generation.reward_model.reward_models.rf3folding.ckpt_path='${oc.env:RF3_CKPT_PATH}' \
++generation.reward_model.reward_models.rf3folding.rf3_path='${oc.env:RF3_EXEC_PATH}' \
++generation.reward_model.reward_models.rf3folding.normalize_pae=true \
++generation.reward_model.reward_models.rf3folding.reward_weights.min_ipAE=-1.0
```

### Interface Reward Models

These non-folding models score the interface quality of generated (or refolded) structures. They run after folding models and can optionally use refolded structures via `structure_source`.

**TMOL (force field):**

```yaml
    tmol:
      _target_: "proteinfoundation.rewards.tmol_reward.TmolRewardModel"
      enable_hbond: true
      enable_elec: false
      hbond_weight: 1.0
      elec_weight: 1.0
      reward_type: "interaction_count"
      energy_threshold: -0.6
```

**Bioinformatics (shape complementarity, SASA, hydrophobicity):**

```yaml
    bioinformatics:
      _target_: "proteinfoundation.rewards.bioinformatics_reward.BioinformaticsRewardModel"
      reward_weights:
        surface_hydrophobicity: 0.0
        interface_sc: 1.0
        interface_dSASA: 0.0
        interface_fraction: 0.0
        interface_hydrophobicity: 1.0
        interface_nres: 0.0
      reward_thresholds:
        interface_sc: 0.55
        interface_nres: 7
      structure_source: null    # null = generated structure. Set to a folding model key (e.g. "af2folding") to use its refolded structure.
```

**Using refolded structures:** `structure_source` must match the key name of a folding reward model defined in the same `reward_models:` block (e.g. `af2folding` or `rf3folding`). When set, this interface model scores the refolded structure produced by that folding model instead of the raw generated structure.

### Model-Level Weights

Control the relative contribution of each sub-model to the composite reward. If omitted, all models default to weight 1.0.

```yaml
reward_model:
  _target_: "proteinfoundation.rewards.base_reward.CompositeRewardModel"
  reward_models:
    af2folding: { ... }
    tmol: { ... }
    bioinformatics: { ... }

  weights:
    af2folding: 1.0
    tmol: 0.5
    bioinformatics: 1.0
```

### Model Sampling Parameters

Configured in `pipeline/model_sampling.yaml`. Controls the diffusion sampling process.

```yaml
args:
  nsteps: 400              # Number of denoising steps
  self_cond: true           # Self-conditioning
  guidance_w: 1.0           # Classifier-free guidance weight
  save_trajectory_every: 0  # 0 = don't save intermediate structures

model:
  bb_ca:                    # Backbone CA coordinates
    schedule:
      mode: log
      p: 2.0
    simulation_step_params:
      sampling_mode: sc
      sc_scale_noise: 0.1
      sc_scale_score: 1.0

  local_latents:            # Local latent features (side chains, sequence)
    schedule:
      mode: power
      p: 2.0
    simulation_step_params:
      sampling_mode: sc
      sc_scale_noise: 0.1
      sc_scale_score: 1.0
```

**CLI examples:**

```bash
# Fewer denoising steps (faster, lower quality)
++generation.args.nsteps=200

# Increase guidance weight
++generation.args.guidance_w=2.0

# Reduce batch size to save memory
++generation.dataloader.batch_size=8
```

### Running the Pipeline

**Full pipeline (all 4 stages):**

```bash
complexa design configs/search_binder_local_pipeline.yaml

# With overrides
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=pdl1_beam_v1 \
    ++generation.task_name=02_PDL1 \
    ++generation.search.algorithm=beam-search \
    ++generation.search.beam_search.beam_width=8
```

**Individual stages:**

```bash
complexa generate configs/search_binder_local_pipeline.yaml
complexa filter configs/search_binder_local_pipeline.yaml
complexa evaluate configs/search_binder_local_pipeline.yaml
complexa analyze configs/search_binder_local_pipeline.yaml
```

**Ligand binder pipeline:**

```bash
complexa design configs/search_ligand_binder_local_pipeline.yaml \
    ++run_name=ligand_test \
    ++generation.task_name=39_7V11_LIGAND
```

**Quick local test (reduced samples):**

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=quick_test \
    ++generation.task_name=02_PDL1 \
    ++generation.args.nsteps=100 \
    ++generation.dataloader.dataset.nres.nsamples=2
```

**Verbose mode (output to terminal instead of log file):**

```bash
complexa design configs/search_binder_local_pipeline.yaml --verbose
```

---

## Evaluation Configs

All evaluation configs are run with:

```bash
complexa evaluate configs/<config_name>.yaml
# or
python -m proteinfoundation.evaluate --config-name <config_name>
```

### Evaluation Config Reference

All evaluation configs share the same structure. The difference between binder-only, monomer-only, and combined evaluation is which boolean flags are enabled.

```yaml
defaults:
  - /generation/targets_dict@dataset
  - _self_

run_name: my_eval
ckpt_path: Complexa
ckpt_name: Complexa_ckpt

protein_type: binder           # binder, monomer, monomer_motif, or motif_binder
input_mode: generated          # generated (from pipeline) or pdb_dir (flat directory)

sample_storage_path: ./inference/search_binder_TARGET
output_dir: ./evaluation_results/my_eval

ncpus_: 24
seed: 5
eval_njobs: 20                 # Match to gen_njobs from generation
job_id: 0                      # Set via SLURM array or loop

dataset:
  task_name: 32_PDL1_ALPHA_REPACK

metric:
  # -- Binder metrics (full complex refolding) --
  compute_binder_metrics: true
  binder_folding_method: colabdesign   # colabdesign (AF2), rf3_latest, protenix_base_default_v0.5.0
  sequence_types: [self, mpnn, mpnn_fixed]
  num_redesign_seqs: 8
  interface_cutoff: 8.0
  inverse_folding_model: soluble_mpnn  # soluble_mpnn, protein_mpnn, ligand_mpnn

  # -- Pre/post refolding interface metrics --
  compute_pre_refolding_metrics: true
  pre_refolding:
    bioinformatics: true       # Shape complementarity, SASA, hydrophobicity
    tmol: true                 # TMOL force field (H-bonds, electrostatics)

  compute_refolded_structure_metrics: true
  refolded:
    bioinformatics: true
    tmol: true

  # -- Monomer metrics (binder chain designability) --
  compute_monomer_metrics: true
  monomer_folding_models: [esmfold]    # esmfold, colabfold, chai1
  compute_designability: true
  designability_modes: [ca, bb3o]      # ca, bb3o, all_atom
  compute_codesignability: true
  codesignability_modes: [ca, all_atom]
  compute_co_sequence_recovery: true
  compute_ss: true

  # -- Novelty --
  compute_novelty_pdb: true
  compute_novelty_afdb: false          # Requires AFDB index

  keep_folding_outputs: false
```

**Evaluation presets** -- toggle these flags to get common evaluation modes:

| Preset | `protein_type` | `compute_binder_metrics` | `compute_monomer_metrics` | `compute_pre_refolding_metrics` | `compute_refolded_structure_metrics` |
|--------|---------------|--------------------------|---------------------------|---------------------------------|--------------------------------------|
| Binder-only | `binder` | `true` | `false` | `false` | `false` |
| Monomer-only | `monomer` | `false` | `true` | `false` | `false` |
| Binder + Monomer | `binder` | `true` | `true` | `true` | `false` |
| All benchmarks | `binder` | `true` | `true` | `true` | `true` |

When `protein_type: binder` and `compute_monomer_metrics: true`, monomer evaluation automatically extracts and evaluates only the binder chain.

**Key options:**

- `binder_folding_method`: `colabdesign` (AF2, protein-protein), `rf3_latest` (protein-ligand or high accuracy)
- `sequence_types`: `self` (original sequence), `mpnn` (ProteinMPNN redesigned), `mpnn_fixed` (MPNN with fixed target)
- `inverse_folding_model`: `soluble_mpnn`, `protein_mpnn`, `ligand_mpnn`
- `monomer_folding_models`: `esmfold` (fast), `colabfold`, `chai1`
- `compute_novelty_afdb`: requires AFDB index; set to `false` for quick runs

> ColabDesign does **not** support ligand targets. Use RF3 or Protenix for ligand binder evaluation.

### External PDB Directory

For evaluating PDB files from external sources (BindCraft, AlphaProteo, etc.) that do not have the `job_X_*` directory structure.

```yaml
defaults:
  - /generation/targets_dict@dataset
  - _self_

run_name: external_binder_eval
ckpt_path: external
ckpt_name: external_ckpt

protein_type: binder
input_mode: pdb_dir   # Raw PDB files from any flat directory

sample_storage_path: ./pdb_samples/external_binders
output_dir: ./evaluation_results/external_binder_eval

ignore_generated_pdb_suffix: "_binder.pdb"   # Skip PDB files matching this suffix

ncpus_: 24
seed: 5
eval_njobs: 1
job_id: 0

dataset:
  task_name: 32_PDL1_ALPHA_REPACK

metric:
  compute_binder_metrics: true
  binder_folding_method: colabdesign
  sequence_types: [self]
  num_redesign_seqs: 8
  interface_cutoff: 8.0
  inverse_folding_model: soluble_mpnn

  compute_pre_refolding_metrics: false
  compute_refolded_structure_metrics: false
  compute_monomer_metrics: false
  compute_designability: false
  compute_codesignability: false
  compute_co_sequence_recovery: false
  compute_novelty_pdb: false
  compute_novelty_afdb: false
```

**Key difference:** `input_mode: pdb_dir` expects a flat directory of PDB files rather than the `job_X_*` directory structure from `generate`. Use `ignore_generated_pdb_suffix` to skip auxiliary PDB files.

---

## Analysis Configs

All analysis configs are run with:

```bash
complexa analyze configs/<config_name>.yaml
# or
python -m proteinfoundation.analyze --config-name <config_name>
```

The analysis step loads evaluation CSVs, computes aggregate metrics (success rates, diversity), and organizes output files.

### Protein Binder Analysis

Uses default AlphaProteo-style success thresholds: `i_pAE * 31 <= 7.0`, `pLDDT >= 0.9`, `binder_scRMSD < 1.5`.

```yaml
defaults:
  - /analyze@_here_
  - _self_

result_type: protein_binder
```

Run with:

```bash
python -m proteinfoundation.analyze \
    --config-name analyze \
    results_dir=./evaluation_results/my_binder_run \
    config_name=search_binder
```

### Ligand Binder Analysis

Uses default ligand binder thresholds: `min_ipAE * 31 < 2.0`, `binder_scRMSD_ca < 2.0`, `ligand_scRMSD_aligned_allatom < 5.0`.

```yaml
defaults:
  - /analyze@_here_
  - _self_

result_type: ligand_binder
```

Run with:

```bash
python -m proteinfoundation.analyze \
    --config-name analyze \
    results_dir=./evaluation_results/my_ligand_run \
    config_name=search_ligand_binder
```

### Monomer Analysis

Uses default designability/codesignability thresholds (2.0 A, auto-detected from result columns).

```yaml
defaults:
  - /analyze@_here_
  - _self_

result_type: monomer
```

Run with:

```bash
python -m proteinfoundation.analyze \
    --config-name analyze \
    results_dir=./evaluation_results/my_monomer_run \
    config_name=evaluate
```

### Custom Success Thresholds

Override the default success thresholds for any result type. Each threshold specifies a column prefix, a scaling factor, a threshold value, and a comparison operator.

**Custom binder thresholds:**

```yaml
defaults:
  - /analyze@_here_
  - _self_

result_type: protein_binder

aggregation:
  success_thresholds:
    i_pAE:
      threshold: 8.0         # More relaxed than default 7.0
      op: "<="
      scale: 31.0            # Column stores normalized values; multiply back
      column_prefix: complex
    pLDDT:
      threshold: 0.85        # More relaxed than default 0.9
      op: ">="
      scale: 1.0
      column_prefix: complex
    scRMSD:
      threshold: 2.0         # More relaxed than default 1.5
      op: "<"
      scale: 1.0
      column_prefix: binder
```

**Custom monomer thresholds (stricter):**

```yaml
defaults:
  - /analyze@_here_
  - _self_

result_type: monomer

aggregation:
  require_all_thresholds: true

  designability_thresholds:
    ca:
      esmfold:
        threshold: 1.5       # Stricter than default 2.0
        op: "<="
    all_atom:
      esmfold:
        threshold: 2.5
        op: "<="

  codesignability_thresholds:
    ca:
      esmfold:
        threshold: 2.0
        op: "<="
    all_atom:
      esmfold:
        threshold: 2.5
        op: "<="
```

### Custom Ranking Criteria

During **evaluation**, the `ranking_criteria` config controls how the best refolded sample is selected from multiple inverse-folding sequences. The ranking system computes a composite score where lower is better.

The default ranking criteria are:
- **Protein binders:** `i_pAE` (minimize)
- **Ligand binders:** `min_ipAE` (minimize)

To use custom ranking criteria (e.g., incorporating ipSAE or pLDDT), add them to the `metric` section of your **evaluation** config:

```yaml
metric:
  compute_binder_metrics: true
  binder_folding_method: rf3_latest

  # Custom ranking: pick the sample with the best combination of min_ipAE and ipSAE
  ranking_criteria:
    min_ipAE:
      scale: 1.0
      direction: minimize     # Lower ipAE is better
    min_ipSAE:
      scale: 1.0
      direction: maximize     # Higher ipSAE is better
```

Any metric present in the refolding output can be used as a ranking criterion. For RF3, the available metrics include: `pLDDT`, `i_pAE`, `min_ipAE`, `pAE`, `min_ipSAE`, `max_ipSAE`, `avg_ipSAE`.

If `ranking_criteria` is not specified, the defaults above are used. If a metric name in the criteria is not found in the results, a warning is logged and that criterion is skipped.

---

## Training Configs

Training configs use the same Hydra composition system. Three dataloader types are supported.

### PyG Dataloader (default)

Standard PyG/foldcomp dataloader, matching the original training pipeline.

```yaml
run_name: pyg_training
dataloader_type: pyg

defaults:
  - nn: local_latents_score_nn_160M
  - generation: validation_local_latents
  - dataset: afdb_fromraw/genie2
  - _self_

# ... (model, loss, optimizer, and training config)
```

### Atomworks Dataloader

Uses the atomworks dataloader pipeline for ligand-aware training.

```yaml
run_name: atomworks_training
dataloader_type: atomworks

defaults:
  - nn: local_latents_score_nn_160M_ligand_chainbreak
  - generation: validation_local_latents
  - dataset: atomworks/plinder
  - _self_

dataloader:
  train:
    dataloader_params:
      batch_size: 5
      num_workers: 10
      prefetch_factor: 2

# ... (model, loss, optimizer, and training config)
```

### Combined Dataloader

Mixes both atomworks and PyG dataloaders for multi-source training.

```yaml
run_name: combined_training
dataloader_type: combined

defaults:
  - nn: local_latents_score_nn_160M_ligand_chainbreak
  - generation: validation_local_latents
  - dataset: atomworks/plinder
  - dataset/afdb_fromraw/genie2@dataset_pyg
  - _self_

# ... (model, loss, optimizer, and training config)
```

---

## Common Patterns

### Config Composition (Hydra Defaults)

Configs use Hydra's `defaults` list to compose from shared config fragments:

```yaml
defaults:
  - /generation/targets_dict@dataset   # Load target definitions into 'dataset' key
  - /analyze@_here_                    # Load analyze defaults at the current level
  - _self_                             # Apply this file's values last (highest priority)
```

- `/path@key` loads a config group and places it under the specified key
- `@_here_` places the loaded config at the root level
- `_self_` ensures the current file's values override any defaults

### Command-Line Overrides

Use Hydra `++` syntax to override any config value from the command line:

```bash
# Override dataset target
complexa evaluate configs/evaluate.yaml ++dataset.task_name=02_PDL1

# Override multiple values
complexa evaluate configs/evaluate.yaml \
    ++run_name=my_run \
    ++metric.binder_folding_method=rf3_latest \
    ++metric.sequence_types="[self,mpnn]"

# Override nested values
complexa analyze configs/analyze.yaml \
    ++aggregation.success_thresholds.i_pAE.threshold=10.0
```

### Job Parallelism

Evaluation is designed for embarrassingly parallel execution across samples. Use `eval_njobs` and `job_id` to split work:

```bash
# Sequential (all samples in one job)
complexa evaluate configs/evaluate.yaml ++eval_njobs=1 ++job_id=0

# Parallel with SLURM array
# In your SLURM script:
complexa evaluate configs/evaluate.yaml \
    ++eval_njobs=20 \
    ++job_id=$SLURM_ARRAY_TASK_ID
```

Set `eval_njobs` to match `gen_njobs` from generation so each eval job processes the outputs of one generation job.
