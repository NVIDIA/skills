---
name: complexa-sweep
description: Agent runbook for Proteina-Complexa parameter sweeps through the separately installed first-party CLI and config generator. Use for cartesian-product hyperparameter scans, sweeper YAML authoring, configuration comparisons, ablations, tuning, success-rate ranking, or Pareto searches over quality and wall-clock cost. Do not use for a single existing-job launch or SLURM submission template. SKILL.md performs the orchestration; bundled executables only inspect local readiness and write a run manifest.
allowed-tools: Bash, Read, Write, Env, AskUserQuestion
metadata:
  author: Ohad Mosafi <omosafi@nvidia.com>
---


# complexa-sweep

Run cartesian-product parameter sweeps over Proteina-Complexa design pipelines. Pick or author a sweeper YAML in `configs/sweeps/`, expand it to N inference configs with `script_utils/generate_inference_configs.py`, loop `complexa design` over those configs, then aggregate per-config success metrics into a ranked summary CSV plus a manifest.

> **Important:** The `complexa design` CLI does **NOT** accept `--sweeper` directly. Sweeps are driven by `script_utils/generate_inference_configs.py`, which writes one `configs/inference_configs/inf_{idx}_{run_name}.yaml` per sweep combination. You then loop `complexa design` over those generated configs.

## Instructions

Sweeper authoring, Cartesian enumeration, cost estimates, and analysis of supplied
CSVs need only the relevant configs and data. Complete those artifacts without
setting up inference. Read the relevant axis section in
[sweep_axes.md](references/sweep_axes.md), then check the matching keys in the
selected source config for current defaults. Summary CSVs with supplied counts
and objective columns do not require the full inference or metric guides.
For a partial snapshot,
enumerate combinations locally and provide the future generator command;
running that command requires the full project and its dependencies.

For a requested launch, check readiness and the budget below. A single existing
job or a SLURM template does not require a parameter sweep.

## Step 1: Pre-flight

Before GPU execution, or when the request conditions its workflow on hardware,
set `SKILL_DIR` to the directory containing this manifest and run preflight once.
Use a separate shell assignment; an inline environment assignment cannot set
`$SKILL_DIR`'s expansion in the same command.
Reuse an existing probe from the same request if the environment is unchanged:

```bash
bash "$SKILL_DIR"/scripts/preflight.sh
```

Read `./complexa_setup/preflight.json`. A sweep multiplies GPU time by the number
of configs and targets. Compute the budget before launch and obtain confirmation
if the user has not already authorized that scope and cost:

> "This sweep produces N configs × ~M minutes per config ≈ TOTAL GPU-hours. OK to proceed? (y / reduce / cancel)"

If `gpu.available=false`, stop before GPU execution. Sweeper authoring,
Cartesian-product enumeration, cost estimates, and analysis of supplied CSVs
can run on CPU. For those requests, finish the requested artifacts and record
the launch prerequisites. Label synthetic input data as synthetic and keep
estimated GPU time separate from measured runtime.

For a hardware-conditional request, verify the allocated hardware before
selecting the CPU fallback. Missing weights, dependencies, or sufficient VRAM
for an allocated GPU are launch blockers, not evidence of a CPU-only environment.
Report probe or execution failures without reclassifying the selected branch.
The shared preflight's `gpu.available=false` also covers failed or missing
`nvidia-smi`; check device/allocation evidence before treating it as GPU absence.
In containers, `/proc/driver/nvidia/gpus` and loaded kernel modules may describe
the host, not this task's allocation. Check container device exposure and any
available scheduler allocation information together. A CPU allocation can share
a GPU host; an allocated GPU with inaccessible devices is blocked. If allocation
remains unclear, report it as undetermined.

## Step 2: Pick the pipeline + target

Use the user's supplied pipeline, targets, and run name. Protein, ligand, and AME
runs use `search_binder_local_pipeline`, `search_ligand_binder_local_pipeline`,
and `search_ame_local_pipeline`, respectively. Ask only for missing values:

- `pipeline_config_name` — e.g. `search_binder_local_pipeline` (default), `search_ligand_binder_local_pipeline`, `search_ame_local_pipeline`.
- `task_name` — e.g. `02_PDL1`, `22_DerF21`, `39_7V11_LIGAND`. Passed as `--override generation.task_name=<task>`.
- `run_name` — short tag for output dir naming.

## Examples

### Step 3: Pick or author the sweeper YAML

Sweeper YAMLs live in `configs/sweeps/`. Each key is a dot-notation Hydra path; each value is a list. The cartesian product becomes N configs.

#### Canned sweepers

| File | Axis | Values | Configs |
|---|---|---|---|
| `configs/sweeps/beam_width.yaml` | `generation.search.beam_search.beam_width` | 1, 2, 4, 8 | 4 |
| `configs/sweeps/bb_ca_temperature.yaml` | `generation.model.bb_ca.simulation_step_params.sc_scale_noise` | 0.1, 0.4 | 2 |
| `configs/sweeps/search_replicas.yaml` | `generation.search.best_of_n.replicas` | 1, 4, 16, 64 | 4 |
| `configs/sweeps/example.yaml` | beam_width × nsteps | (2,4) × (200,400) | 4 |

Use a matching sweeper, then check that its axes affect the selected algorithm.
The binder config defaults to `best-of-n`: a beam-width sweep must also pin
`generation.search.algorithm: [beam-search]`. A replicas sweep pins
`generation.search.algorithm: [best-of-n]`. For AME, reward-guided search also
requires a configured reward model; its default is `null`.

#### Authoring a new sweeper

Minimal multi-axis example (saved to `configs/sweeps/my_sweep.yaml`):

```yaml
# 3 beam widths × 2 nsteps = 6 configs
generation.search.algorithm: [beam-search]

generation.search.beam_search.beam_width:
  - 2
  - 4
  - 8

generation.args.nsteps:
  - 200
  - 400
```

Rules (from `script_utils/generate_inference_configs.py:load_sweeper_file`):

- Top-level mapping only. Keys are dot-notation Hydra paths.
- Keep budgets, launch status, and other planning metadata in the launch plan, outside the sweeper. Every top-level entry becomes an override axis; a separate `targets` list would multiply the grid again.
- Values must be **lists**. A scalar is auto-wrapped into a single-element list (which pins a value without adding a dimension).
- For a list-valued parameter, nest each candidate list: `metric.sequence_types: [[self, mpnn]]` pins one list. The flat form sweeps two strings; `--override` parses scalars only.
- Cartesian product: total configs = product of list lengths. Multiply by the number of targets unless target names are already one of the axes; do not count them twice.
- If a key appears in both the sweeper file and an `--override`, the override wins and that axis collapses.
- Reject empty axes before launch: an axis of `[]` produces zero configurations.

See [references/sweep_axes.md](references/sweep_axes.md) for the full catalogue of swept keys (typical ranges, cost multipliers, what improves/regresses).

#### Dry-run preview before generating

With the full project available, confirm the config count first:

```bash
python script_utils/generate_inference_configs.py \
    --config_name search_binder_local_pipeline \
    --sweeper configs/sweeps/my_sweep.yaml \
    --override generation.task_name=22_DerF21 \
    --run_name my_sweep \
    --dryrun
```

The output lists every axis + value list and prints `DRY RUN — would generate N config pair(s)`.
This previews enumeration only; it returns before Hydra composition and does
not validate the pipeline, parameter types, or launch prerequisites.

When the full project is available, inspect the materialized inference configs
before launching. Check the resolved target, selected search algorithm, and
requested parameter values across the generated grid. The generator permits
unknown keys during merging, so a successful write alone does not establish
that a misspelled or misplaced axis changes the intended parameter. With only
a partial snapshot, check each dot-path against its source config and report
that materialization remains unverified.

## Step 4: Generate configs + loop `complexa design`

Once the dry-run looks right, drop `--dryrun` to materialize `inf_{idx}_{run_name}.yaml` + `eval_{idx}_{run_name}.yaml` pairs under `configs/inference_configs/` and `configs/eval_configs/`. Then loop `complexa design` over the inference configs.

For multiple targets, put `generation.task_name` in the sweeper as an axis, or
give each target's generator invocation a distinct `--run_name`. Indices restart
at zero on each invocation and filenames omit the target; reusing the same run
name and output directories overwrites earlier configs and reuses result paths.

```bash
# 1. Generate one inf_*.yaml + one eval_*.yaml per combination.
python script_utils/generate_inference_configs.py \
    --config_name search_binder_local_pipeline \
    --sweeper configs/sweeps/my_sweep.yaml \
    --override generation.task_name=22_DerF21 \
    --run_name my_sweep

# 2. Loop complexa design over each generated inference config.
for cfg in configs/inference_configs/inf_*_my_sweep.yaml; do
    complexa design "$cfg" || echo "FAILED: $cfg"
done
```

This serialises the sweep on a single GPU — be honest with the user that wall-clock = N × per-run.

### Multi-GPU host (optional speed-up)

On a host with K GPUs, shard the config list K ways and launch one loop per GPU in parallel. Each `complexa design` call uses exactly one GPU at default `gen_njobs=1` / `eval_njobs=1`, so pinning via `CUDA_VISIBLE_DEVICES` keeps them from colliding:

```bash
CONFIGS=(configs/inference_configs/inf_*_my_sweep.yaml)
N=${#CONFIGS[@]}; K=4   # 4 GPUs
for gpu in $(seq 0 $((K-1))); do
    (
        for i in $(seq $gpu $K $((N-1))); do
            CUDA_VISIBLE_DEVICES=$gpu complexa design "${CONFIGS[$i]}" \
                || echo "FAILED on GPU $gpu: ${CONFIGS[$i]}"
        done
    ) &
done
wait
```

Drop `gen_njobs` / `eval_njobs` overrides into the loop only if you intentionally want each `complexa design` invocation to consume multiple GPUs (and you have a way to keep them out of each other's way).

## Step 5: Collect results

Each swept config writes to its own `./inference/inf_{idx}_{run_name}/` directory; the analyze step writes `./evaluation_results/eval_{idx}_{run_name}/results_*.csv`. After the sweep finishes:

```bash
ls -d ./inference/inf_*_my_sweep/
ls ./evaluation_results/eval_*_my_sweep/results_*.csv
```

Inspect the supplied CSV schema before aggregating. If it already contains one
row per configuration with `n_passed` and `n_designs`, compute
`success_rate = n_passed / n_designs`; flag zero denominators and invalid counts.
Do not treat those summary rows as individual designs. For per-design rows,
use existing pass flags, parsing string booleans explicitly. If flags are absent,
use the run's analysis thresholds, `result_type`, column names, and units from
[EVALUATION_METRICS.md](references/EVALUATION_METRICS.md). PAE may be normalized
by 31 and pLDDT may use different scales; do not substitute generic thresholds.

## Step 6: Rank configs

Emit `sweep_summary.csv` to the run directory. One row per config:

| Column | How to compute |
|---|---|
| `config_id` | The `{idx}` from `inf_{idx}_{run_name}` |
| `<axis_1>`, `<axis_2>`, ... | The swept value at this combination (read from the per-config `inf_*.yaml`) |
| `n_samples` | `n_designs` for a supplied config summary, otherwise per-design row count |
| `success_rate` | `n_passed / n_designs` for a config summary, otherwise mean of parsed per-design pass flags |
| `mean_i_pae` | Preserve the supplied config mean, or average per-design values in consistent units (lower = better) |
| `mean_i_plddt` | `i_plddt.mean()` (higher = better) |
| `diversity_score` | Unique sequence count / `n_samples` (or use TM-score clustering if available) |
| `wall_clock_min` | From the per-config log timestamps |

Populate only metrics present in the inputs or computable from them. Missing
diversity or timing data stays unavailable.

Then report:

- **Best config** = argmax of `success_rate`. Tie-break on `mean_i_pae` (lower).
- **Pareto frontier**: choose the requested objectives and directions. For time
  versus success, minimize time and maximize success; for time versus iPAE,
  minimize both. A point is dominated when another is no worse on every
  objective and strictly better on at least one. Keep distinct config IDs at
  equal nondominated coordinates; report invalid/missing measurements separately.

Save the full table to `sweep_summary.csv`, then reopen the saved results and
print a compact summary: row count, best config and metric values, and frontier
IDs with their objective values. Report the CSV paths and keep the hardware
probe and configuration dumps separate from this result summary.

## Step 7: Emit manifest

Capture the resolved invocation + outputs for replay. The shared helper takes a single `--output-dir` (it walks for CSVs and pulls the Hydra config from the run's `.hydra/`), so call it against the best-config's evaluation directory and pass the parent `complexa design` command for that run:

```bash
BEST_ID=4   # from Step 6 ranking
python3 "$SKILL_DIR"/scripts/write_manifest.py \
    --output-dir ./evaluation_results/eval_${BEST_ID}_my_sweep \
    --command "python script_utils/generate_inference_configs.py --config_name search_binder_local_pipeline --sweeper configs/sweeps/my_sweep.yaml --override generation.task_name=22_DerF21 --run_name my_sweep && for cfg in configs/inference_configs/inf_*_my_sweep.yaml; do complexa design \"\$cfg\"; done" \
    --skill complexa-sweep \
    --out ./sweep_runs/my_sweep/sweep_manifest.json
```

Alongside the manifest, save the ranked `sweep_summary.csv` from Step 6 to `./sweep_runs/my_sweep/sweep_summary.csv` and surface both paths to the user.

## Recommended sweep recipes

| Symptom | Sweep this | Why |
|---|---|---|
| Quality not good enough | `beam_width × nsteps` (use `example.yaml` as a starting point) | Both raise compute → quality; find the cheapest combination that lands. |
| Too slow / want speed-up | `nsteps` downward (e.g. `[100, 200, 400]`) with fixed `beam_width=4` | Find the smallest nsteps that retains success rate. |
| Mode collapse / low diversity | `generation.model.bb_ca.simulation_step_params.sc_scale_noise` (use `bb_ca_temperature.yaml`) | Higher noise → more diverse backbones. |
| Reward over-fitting (high reward, bad metrics) | `generation.reward_model.reward_models.af2folding.reward_weights.{i_pae, plddt}` ratio | Re-balance composite reward. |
| Broaden the best-of-N candidate pool | `search_replicas.yaml` (`best_of_n.replicas`) | More candidates per search; use separate seeds/runs to measure repeatability. |
| Algorithm shoot-out | `generation.search.algorithm` over `[single-pass, best-of-n, beam-search, fk-steering]` | Compare search regimes; pin everything else. |

## Hardware

Total GPU-time for a sweep = `N_configs × per_run_GPU_time`. A single `search_binder_local_pipeline` run on one A100 is roughly 30–90 min (binder length + nsteps dependent). A 4-axis × 4-value sweep = 256 configs × ~60 min = ~256 GPU-hours — at that scale, plan on a multi-GPU host with the Step-4 sharding pattern.

Refer to [`references/hardware.md`](references/hardware.md) for the per-run baseline + VRAM minima.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Sweeper file not found` from `generate_inference_configs.py` | Path resolved from wrong CWD | Use a path relative to the repo root; or pass an absolute path. |
| `Sweeper YAML must be a mapping` | Top-level YAML is a list or scalar | Rewrite as `key: [v1, v2]` mapping. |
| Dry-run reports zero config pairs | One of the value lists is empty `[]` | Supply at least one value before generating or launching; the generator does not reject an empty product. |
| `complexa design` rejects `--sweeper` | Confusion between entrypoints | The CLI does not accept `--sweeper`. Use `generate_inference_configs.py` first, then loop `complexa design` over the generated configs. |
| Override silently collapses a sweep axis | `--override key=v` shadowed a sweep key | Drop either the override OR the matching key from the sweeper file. |
| One config in the loop fails, sweep keeps going | The `\|\| echo "FAILED…"` in Step 4 swallows the error | Re-run the failed `inf_*.yaml` standalone; check the per-config log under `./logs/`. Skip failed `config_id` when ranking. |

---

For per-axis reference (typical ranges, cost, what gets better/worse), see [references/sweep_axes.md](references/sweep_axes.md).

For the user-facing sweep system overview (config generation, output layout), see [`references/SWEEP.md`](references/SWEEP.md).
