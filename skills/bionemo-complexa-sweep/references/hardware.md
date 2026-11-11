# Proteina-Complexa Hardware Reference

Shared hardware reference for every `complexa-*` skill. Tables only — see each
skill's `SKILL.md` "Hardware" section for the user-facing prose. Numbers marked
`(empirical)` are not pulled from any doc and represent conservative defaults
based on the configs in this repo.

## Per-pipeline GPU requirements

| Pipeline           | Config                                            | Min VRAM (GB) | Recommended VRAM (GB) | Supported SKUs            | Single-GPU only |
|--------------------|---------------------------------------------------|--------------:|----------------------:|---------------------------|-----------------|
| Protein Binder     | `search_binder_local_pipeline.yaml`               |  24 (empirical) |  40 (empirical)       | H100, A100-80, L40S       | Yes             |
| Ligand Binder      | `search_ligand_binder_local_pipeline.yaml`        |  32 (empirical) |  48 (empirical)       | H100, A100-80, L40S       | Yes             |
| AME (motif+ligand) | `search_ame_local_pipeline.yaml`                  |  32 (empirical) |  48 (empirical)       | H100, A100-80, L40S       | Yes             |

Notes:
- All three inference pipelines are single-GPU (the `complexa generate` stage
  is one process per `gen_njobs` slot; jobs do not shard a single design
  across GPUs).
- AME requires `USE_V2_COMPLEXA_ARCH=True`, set via `env_vars:` in
  `configs/search_ame_local_pipeline.yaml` (no runtime VRAM impact).
- Multi-GPU hosts run multiple pipelines / stages in parallel by bumping
  `gen_njobs` / `eval_njobs` — each job takes one GPU.

## Per-evaluation-backend requirements

| Backend             | Min VRAM (GB) | Extra packages / env             | Wall-clock per sample      |
|---------------------|--------------:|----------------------------------|----------------------------|
| ColabDesign / AF2   |            16 | JAX + ColabFold; `AF2_DIR`       | ~30–60 s (empirical)       |
| RoseTTAFold3 (rf3)  |            24 | `RF3_EXEC_PATH`, `RF3_CKPT_PATH` | ~60–180 s (empirical)      |
| ESMFold             |            16 | `fair-esm`, internet/cache OK    | ~5–15 s (empirical)        |

Selected via `++metric.binder_folding_method=colabdesign|rf3_latest|esmfold`.

## Search settings that affect resource use

Select the algorithm with `++generation.search.algorithm=<value>`. Values use
hyphens; nested configuration keys use underscores. These names come from
`configs/pipeline/*/*_generate.yaml` and `src/proteinfoundation/search/search_factory.py`.

| Algorithm value | Candidate-budget keys under `generation.search` |
|-----------------|--------------------------------------------------|
| `single-pass` | No search expansion |
| `best-of-n` | `best_of_n.replicas` |
| `beam-search` | `beam_search.beam_width`, `beam_search.n_branch` |
| `fk-steering` | `fk_steering.beam_width`, `fk_steering.n_branch` |
| `mcts` | `mcts.n_simulations` |

For example, use `++generation.search.algorithm=best-of-n` with
`++generation.search.best_of_n.replicas=4`. More candidates increase work, but
wall-clock time and peak VRAM depend on batching, search checkpoints, the reward
backend, and hardware; candidate counts are not measured timing multipliers.

## CPU / RAM / disk

Defaults pulled from `configs/search_*_local_pipeline.yaml`:

| Pipeline           | `ncpus_` | `gen_njobs` | `eval_njobs` | RAM (rec.) | Output disk / 100 designs |
|--------------------|---------:|------------:|-------------:|-----------:|--------------------------:|
| Protein Binder     |       24 |           1 |            1 |  32 GB (empirical) | ~10–20 GB (empirical) |
| Ligand Binder      |       24 |           1 |            1 |  32 GB (empirical) | ~15–30 GB (empirical) |
| AME                |       24 |           1 |            1 |  32 GB (empirical) | ~20–50 GB (empirical) |

`keep_folding_outputs=true` (eval default) roughly doubles the output disk
footprint — set to `false` if disk is tight.

## When you hit OOM

Try these in order — cheapest mitigations first:

- Reduce `++generation.dataloader.batch_size` (often the biggest VRAM lever).
- Reduce `++gen_njobs` (frees one inference process worth of memory).
- Reduce `++generation.args.nsteps` (less VRAM tied up in trajectory buffers
  when using search algorithms that retain steps).
- Reduce `++generation.search.beam_search.beam_width` /
  `++generation.search.beam_search.n_branch`.
- Set `++metric.keep_folding_outputs=false` to free fold-stage RAM/disk
  pressure (helps when an OOM lands during evaluation).
- Switch fold backend: `++metric.binder_folding_method=esmfold` is the
  cheapest; `rf3_latest` is the heaviest.
- For AME: confirm `USE_V2_COMPLEXA_ARCH=True` matches the AME checkpoint —
  loading the wrong arch wastes ~10–20% VRAM (empirical).
- Multi-GPU host: set `CUDA_VISIBLE_DEVICES=<idx>` to pin the run to a single
  card and avoid PyTorch placing tensors on a busy peer GPU.
