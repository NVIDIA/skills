# Project README excerpts

Source: repository README.md, Quick Start and CLI Reference sections.

## Quick Start

> **GPU parallelism**: The pipeline configs default to `gen_njobs: 1` and `eval_njobs: 1`, which uses a single GPU. If you have multiple GPUs, increase these to run generation and evaluation in parallel — each job uses one GPU. For example, with 4 GPUs:
>
> ```bash
> complexa design configs/search_binder_local_pipeline.yaml \
>     ++gen_njobs=4 ++eval_njobs=4 ...
> ```
>
> Or edit `gen_njobs` / `eval_njobs` directly in the pipeline YAML.

```bash
# 1. Setup
source .venv/bin/activate
complexa init
complexa download --all

# 2. Validate configuration
complexa validate design configs/search_binder_local_pipeline.yaml

# 3. Design binders for PDL1
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=pdl1_test \
    ++generation.task_name=02_PDL1

# 4. Check results
complexa status configs/search_binder_local_pipeline.yaml
```

Other pipeline types:

```bash
# Ligand binder design
complexa design configs/search_ligand_binder_local_pipeline.yaml \
    ++run_name=ligand_test \
    ++generation.task_name=39_7V11_LIGAND

# AME motif + ligand binder scaffolding
complexa design configs/search_ame_local_pipeline.yaml \
    ++run_name=ame_test \
    ++generation.task_name=M0024_1nzy_v3

# Monomer motif scaffolding (indexed mode). Note motif targets not provided
complexa design configs/search_motif_local_pipeline.yaml \
    ++run_name=motif_test \
    ++generation.task_name=1YCR_AA
```

> **Known limitation: TMOL reward not supported for ligand binder / AME pipelines**
>
> The TMOL force-field reward model currently does not work with protein-ligand complexes. The TMOL reward section is commented out by default in the ligand binder and AME generate configs. If you enable it, TMOL scores will be fail and only the other reward models (e.g. RF3) will contribute to the reward.

## CLI Reference

The `complexa` CLI provides a unified interface for all operations.

```bash
complexa --help      # Show all commands
```

| Command | Description |
|---------|-------------|
| `complexa init` | Initialize environment configuration (.env file) |
| `complexa download` | Download model weights (interactive wizard) |
| `complexa validate` | Validate configuration before running |
| `complexa design` | Run full pipeline: generate → filter → evaluate → analyze |
| `complexa generate` | Generate binder structures |
| `complexa filter` | Filter samples by reward scores |
| `complexa evaluate` | Evaluate with structure prediction |
| `complexa analyze` | Aggregate and analyze results |
| `complexa analysis` | Run evaluate → analyze pipeline (for evaluating PDB files) |
| `complexa target` | Target management (list, add, show) |
| `complexa status` | Check pipeline status and outputs |
| `complexa demo` | Show usage examples and explanations |

### Design Pipelines

There are four design pipelines, each with its own config and model:

| Pipeline | Config | Model | Use Case |
|----------|--------|-------|----------|
| Protein Binder | `search_binder_local_pipeline.yaml` | Protein model | Design binders for protein targets |
| Ligand Binder | `search_ligand_binder_local_pipeline.yaml` | Ligand model (LoRA) | Design binders for small-molecule ligands |
| AME (Motif + Ligand) | `search_ame_local_pipeline.yaml` | AME model (LoRA) | Scaffold functional motifs with ligand context |
| Monomer Motif | `search_motif_local_pipeline.yaml` | AME model (LoRA) | Scaffold structural motifs into monomer proteins |

Each pipeline runs four stages: **generate → filter → evaluate → analyze**. Run the full pipeline with `complexa design`, or run stages individually (`complexa generate`, `complexa filter`, etc.). See the [Inference Guide](docs/INFERENCE.md) for individual stages, and advanced usage.
