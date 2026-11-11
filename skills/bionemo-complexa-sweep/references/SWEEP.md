# Sweep System

How to run parameter sweeps and override experiment settings without modifying source code.

> **Documentation Map**
> - Running a design? See [Inference Guide](INFERENCE.md).
> - Tuning YAML configs? See [Configuration Guide](CONFIGURATION_GUIDE.md).
> - Understanding metrics? See [Evaluation Guide](EVALUATION_METRICS.md).
> - Search metadata? See [Search Metadata](SEARCH_METADATA.md).

## Overview

The sweep system has two mechanisms that work together:

- **`--sweeper FILE`** loads a YAML file defining one or more parameter axes.
  All combinations are generated as a cartesian product.
- **`--override KEY=VAL [KEY=VAL ...]`** pins individual parameters to scalar
  values, applied to every generated config.

Both can be used independently or combined. When a key appears in both the
sweeper file and the CLI overrides, the override wins and that axis is
collapsed to a single value.

## Quick Start

```bash
# 1. Single experiment with a specific target (no sweep)
./slurm_utils/launch_protein_binder_search_from_local_docker.sh \
    --override generation.task_name=22_DerF21

# 2. Sweep beam widths for a single target
./slurm_utils/launch_protein_binder_search_from_local_docker.sh \
    --sweeper configs/sweeps/beam_width.yaml \
    --override generation.task_name=22_DerF21

# 3. Sweep beam widths with nsteps pinned to 400
./slurm_utils/launch_protein_binder_search_from_local_docker.sh \
    --sweeper configs/sweeps/beam_width.yaml \
    --override generation.task_name=22_DerF21 generation.args.nsteps=400

# 4. Dry run to preview config generation
python script_utils/generate_inference_configs.py \
    --config_name search_binder_pipeline \
    --sweeper configs/sweeps/beam_width.yaml \
    --override generation.task_name=22_DerF21 \
    --dryrun
```

## Sweep YAML Format

Sweep files live in `configs/sweeps/`. Each key is a dot-notation config path
and each value is a list of values to sweep over:

```yaml
# configs/sweeps/beam_width.yaml
generation.search.beam_search.beam_width:
  - 1
  - 2
  - 4
  - 8
```

Multiple axes produce a cartesian product:

```yaml
# 3 beam widths x 2 nsteps = 6 configs
generation.search.beam_search.beam_width:
  - 2
  - 4
  - 8

generation.args.nsteps:
  - 200
  - 400
```

Long values (e.g., checkpoint paths) benefit from YAML block-list style:

```yaml
ckpt_path:
  - /path/to/checkpoints/model_v1/epoch_100.ckpt
  - /path/to/checkpoints/model_v2/epoch_200.ckpt
  - /path/to/checkpoints/model_v3/epoch_50.ckpt
```

A scalar value (not a list) is automatically wrapped in a single-element list,
so it pins that parameter without adding a sweep dimension:

```yaml
generation.args.self_cond: true
```

## `--override` Format

Override arguments are `KEY=VALUE` strings where:

- The key uses dot notation (e.g., `generation.args.nsteps`)
- The value is auto-typed: integers, floats, booleans (`true`/`false`),
  null (`null`/`none`), or strings

```bash
--override generation.task_name=22_DerF21 generation.args.nsteps=400 generation.args.self_cond=true
```

Multiple overrides are space-separated after a single `--override` flag.

## How It Works

### Config Generation Pipeline

```
                    +-----------------+
                    | Sweeper YAML    |  (optional)
                    +-----------------+
                            |
                            v
+-----------+      +------------------+      +-------------------+
| Pipeline  | ---> | generate_infer-  | ---> | N inf_*.yaml +    |
| Config    |      | ence_configs.py  |      | N eval_*.yaml     |
+-----------+      +------------------+      +-------------------+
                            ^
                            |
                    +-----------------+
                    | --override      |  (optional)
                    | KEY=VAL ...     |
                    +-----------------+
```

1. The base pipeline config (e.g., `search_binder_pipeline.yaml`) is loaded
   via Hydra.
2. If a `--sweeper` file is provided, its axes are combined into a cartesian
   product.
3. For each combination, the sweep values are merged into the base config.
4. `--override` values are applied on top of every config (overriding sweep
   values if they conflict).
5. Each config gets a unique `root_path` and is saved as a pair:
   `inf_{idx}_{run}.yaml` and `eval_{idx}_{run}.yaml`.

### Config Naming

Generated config filenames follow the pattern:

```
inf_{index}_{run_name}.yaml
eval_{index}_{run_name}.yaml
```

The index is sequential (0, 1, 2, ...) and the run name comes from the
`--run_name` argument. Task name is intentionally NOT in the filename -- it
lives inside the config content. This keeps filenames predictable so the bash
launcher can construct Hydra `--config-name` paths without parsing YAML.

### Directory Structure

On the **remote cluster** (and on-cluster direct launches), configs always live
at the canonical paths:

```
configs/
  inference_configs/
    inf_0_my_run.yaml
    inf_1_my_run.yaml
  eval_configs/
    eval_0_my_run.yaml
    eval_1_my_run.yaml
```

#### Local-to-remote flow (with rsync)

When launching from a local machine, configs are first generated into a unique
temporary directory (`configs/_gen_XXXXXX/`) so that concurrent launches never
collide.  Inside the rsync lock, the temp directory contents are **staged**
(moved) into the canonical `configs/inference_configs/` and
`configs/eval_configs/` paths, rsync'd to the cluster, and then cleaned up
locally.

```
generate_configs()        →  configs/_gen_abc123/{inference_configs,eval_configs}/
stage_configs_for_sync()  →  configs/{inference_configs,eval_configs}/   (inside lock)
rsync                     →  $RUN_DIR/configs/{inference_configs,eval_configs}/
cleanup_staged_configs()  →  local canonical dirs removed                (inside lock)
```

#### Direct on-cluster flow (no rsync)

When running directly on the cluster, no temp directory is needed.  Configs are
generated straight into the canonical paths and used in-place by the SLURM
array jobs.

## Integration with SLURM Scripts

All three launcher scripts accept `--sweeper` and `--override`:

```bash
./slurm_utils/launch_protein_binder_search_from_local_docker.sh \
    --sweeper configs/sweeps/beam_width.yaml \
    --override generation.task_name=22_DerF21
```

The on-cluster script works identically (no rsync needed):

```bash
./slurm_utils/launch_protein_binder_search_conda.sh \
    --sweeper configs/sweeps/beam_width.yaml \
    --override generation.task_name=22_DerF21
```

The multi-target loop wrapper also passes these through:

```bash
./slurm_utils/launch_protein_binder_search_target_loop_conda.sh \
    --sweeper configs/sweeps/beam_width.yaml \
    ./my_targets.txt
```

### Legacy Positional Arguments

The `TARGET_TASK` positional argument is still supported for backward
compatibility. Internally it is converted to
`--override generation.task_name=$TARGET_TASK`.

## Running Concurrent Experiments

The system is designed for safe concurrent launches:

- **Local config isolation**: Each launch generates configs in a unique
  `mktemp -d configs/_gen_XXXXXX` directory.  Before rsync (inside the lock),
  configs are staged to the canonical `configs/inference_configs/` and
  `configs/eval_configs/` paths, rsync'd, then cleaned up.  This means the
  generation phase is fully parallel, and the staging+rsync phase is serialised
  by the lock.
- **Remote directory uniqueness**: The run name on the cluster includes the
  sweeper file basename (e.g., `my_run-search-beam_width-22_DerF21`), making
  collisions nearly impossible for different sweeps.
- **Lock file for rsync**: The `.lock` file prevents simultaneous rsync
  operations from corrupting the code copy.

Example: launching two experiments in parallel from different terminals:

```bash
# Terminal 1: sweep beam widths for target A
./slurm_utils/launch_protein_binder_search_from_local_docker.sh \
    --sweeper configs/sweeps/beam_width.yaml \
    --override generation.task_name=22_DerF21

# Terminal 2: sweep replicas for target B (safe to run simultaneously)
./slurm_utils/launch_protein_binder_search_from_local_docker.sh \
    --sweeper configs/sweeps/search_replicas.yaml \
    --override generation.task_name=02_PDL1
```

## Running Individual Pipeline Stages

The core Python modules (`generate`, `filter`, `evaluate`, `analyze`) can
still be run independently with Hydra `++key=value` overrides:

```bash
# Run just evaluation on existing inference outputs
python -m proteinfoundation.evaluate \
    --config-path /path/to/eval_configs \
    --config-name eval_0_22_DerF21_my_run \
    ++eval_njobs=4
```

The sweep system only affects config generation and the bash pipeline
orchestration. It does not change how individual modules consume configs.

## Migration from Old System

| Old                                        | New                                              |
|--------------------------------------------|--------------------------------------------------|
| Edit `sweeper = {...}` dict in Python      | Create a YAML file in `configs/sweeps/`          |
| `--target_task_override 22_DerF21`         | `--override generation.task_name=22_DerF21`      |
| `--unified` flag                           | Removed (always unified pipeline mode)           |
| `--eval_config_name evaluate`              | Removed (eval derived from pipeline config)      |
| `--infer_config_name search_binder`        | `--config_name search_binder_pipeline`           |
| `generate_binder_inference_configs.py`     | `generate_inference_configs.py`                  |
| Hardcoded `ALPHA_PROTEO_TARGETS` list      | Removed (use sweep or override)                  |
| Manually delete generated configuration files | Automatic: temp dir + staging (concurrent-safe) |

## Tests

The sweep system is covered by `tests/test_sweep.py` (92 tests):

```bash
pytest tests/test_sweep.py -v
```

Test categories:
- `TestParseScalar` / `TestParseOverride`: Value parsing and type inference
- `TestLoadSweeperFile`: YAML loading and validation
- `TestBuildSweeper`: Merge logic and conflict resolution
- `TestDotKeyDictToNestedDict`: Dot-notation conversion
- `TestGetTaskNameFromConfig` / `TestCreateEvalConfig`: Config helpers
- `TestApplySweeperAndSaveConfigs`: End-to-end cartesian product, value
  propagation, unique paths, filename conventions
- `TestConcurrentSafety`: Parallel generation without collisions
- `TestBuildParser`: CLI argument parser validation
- `TestCreateEvalConfigNjobs`: eval_njobs priority regression tests
- `TestCreateEvalConfigPaths`: Path derivation edge cases
- `TestEmptySweepAxis`: Zero-config guard regression tests
- `TestConfigFilenameContract`: Filename contract between Python and bash
- `TestEvalPathEdgeCases`: Eval path derivation with adversarial inputs
