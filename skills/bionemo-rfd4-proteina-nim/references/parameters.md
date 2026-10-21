# Tasks and design controls

Grounded in the public [model card](https://build.nvidia.com/ipd/rfd4-proteina/modelcard)
and [API schema](https://build.nvidia.com/ipd/rfd4-proteina), version 1.0.0.
The running deployment remains authoritative for capability and resource limits.

## Task selection

| Task | Use |
| --- | --- |
| `generation` | Generate designable regions while preserving supplied conditions; includes monomers and conditioned protein binders. |
| `motif_scaffolding` | Build a scaffold around indexed motif segments. |
| `enzyme_design` | Scaffold an unindexed catalytic site. |
| `small_molecule_binder` | Design around a ligand held in the input. |
| `antibody_design` | Redesign selected CDRs on a fixed framework, optionally with an antigen. |
| `partial_diffusion` | Refine a supplied design by restarting partway through generation. |
| `inverse_folding` | Hold a backbone and generate sequence and side chains. |

Use one of these seven published names. Protein binders use `generation` with
appropriate target/binder conditions; a task name does not encode an epitope or
hold a target by itself. Preserve framework, motif, ligand, and hotspot
annotations actually supplied by the user.

## Sampling and reproducibility

| Field | Public contract |
| --- | --- |
| `sampling.num_replicates` | Positive integer; the deployment and GPU still limit practical request size. |
| `sampling.batch_size` | Positive integer or omitted. Omission batches all requested replicates. Lower values trade throughput for memory and replay behavior. |
| `sampling.seed` | Nonnegative base seed. With batch size 1, replicate i uses `(seed+i) % 2**63`. |
| `sampling.num_steps` | Optional step count. Explicit steps without recycling request flat sampling; the loaded checkpoint can impose a higher minimum than the schema. |
| `sampling.recycle_strategy` | Named recycle schedule. |
| `sampling.recycle_schedule` | Explicit `[step_count, depth]` bands; cannot be combined with a named strategy. |

| Named strategy | Steps | Trunk passes |
| --- | ---: | ---: |
| `thinking_low` | 60 | 110 |
| `thinking_medium` | 60 | 185 |
| `thinking_high` | 60 | 295 |

Use the public example's `thinking_medium` unless the user selects another
schedule. The checkpoint default is also `thinking_medium`. Supplying a step
count with a named strategy rescales its shape, and the response warns about
the change; quality of such modified schedules is not established by this skill.
For custom bands, an explicit step count must equal their sum. Record the
returned schedule and warnings rather than assuming the request ran unchanged.

A batch uses a single random stream: designs can share a seed while differing
from one another. Replay the whole batch with the same input, weights, backend,
runtime, and batch size, including any smaller final batch. GPU determinism
across runtimes is not guaranteed. Use batch size 1 when individual ordinary
sampling replicates must be replayable.

## Partial diffusion

`partial` is required for `partial_diffusion` and rejected for other tasks.
Use `trajectory_fraction_steps` in `(0,1]`, or explicit `t_backbone` and
`t_latent` values in `(0,1)`. A fraction is mutually exclusive with explicit
modality times. `couple_partial_t` allows the missing modality time to be
derived from the supplied one.

Supply an appropriate complete starting backbone. A motif-only template is
not a substitute for a complete design. Inverse folding preserves backbone
geometry and cannot repair every defect in the input.

The public response contract describes executed steps and recycle bands after
partial scaling. Retain the requested fraction/times and returned schedule.
If they disagree, report that discrepancy and avoid making unsupported compute
or adherence claims from the metadata alone.

## Search

Search branches and prunes trajectories at checkpoints. Asking for ordinary
replicates does not enable search. An example from the published schema's
reward fields, with a schedule consistent with `thinking_medium`, is:

```json
{
  "step_checkpoints": [0, 30, 60],
  "n_branch": 2,
  "beam_width": 2,
  "reward_metrics_tags": ["interchain_interactions"],
  "weights": {"interchain_interactions.hbond_atom_pair_count_coi": 1.0},
  "clip_values": {"interchain_interactions.hbond_atom_pair_count_coi": {"max": 6.0}}
}
```

Checkpoints start at 0, strictly increase, and end at the resolved integration
step count. Pin the schedule when preparing search; partial scaling can change
that endpoint. The schema permits 2–21 checkpoints and 1–16 branches and beam
width. Use batch size 1 for the skill's reproducible search examples and inspect
the deployed service's validation if combining optional settings.

Reward weights name `metric.output_key` entries. Negative weights penalize;
clip bounds limit their contribution before weighting. `reward_metrics_tags`
enables scoring families independently of `metrics.tags`, which controls final
reporting. Missing/null rewards are not zero, and the metric's chain context
must match the intended design.

`checkpoint_overrides` changes rewards at chosen checkpoints.
`intermediate_save_min_reward` retains qualifying non-survivors in addition to
the final beam. `scoring_chunk_size` bounds decoded candidates per scoring chunk.
Keep `search_final` and `search_intermediate` separate when counting results.

Approximate work is
`beam_width * n_branch * (len(step_checkpoints)-1) * num_replicates` rollouts.
Read the returned search report for actual work. Seeds of individual search
branches are provenance, not independently replayable requests; replay the
entire search with the original configuration.

## Metrics and deployment limits

Start with the public example's `metrics.tags=["design"]` for geometry metrics.
The API also lists condition-specific families such as `hotspot_conditioning`,
`interchain_interactions`, `3d_coordinate_conditioning`, `unindexed`, and `cms`.
Select families relevant to the input and supported by the deployment.
An empty tag list requests diagnostics without those metric families.

The default product deployment serves non-oracle metrics. Refolding metrics
require an operator-configured oracle; `model.oracle` identifies what actually
ran. A published tag name does not guarantee the deployment has its dependencies
or that a value is defined for the supplied structure.

The public model card lists a 1024-token design limit. Read the actual response's
`runtime.limits` for token, step, trunk-pass, request-size, and search-rollout
limits. GPU memory depends on input size, batching, and decoding as well as the
nominal number of designs. Server limits can be stricter than schema bounds.
