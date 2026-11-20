# Public model plans

Generate a plan against the confirmed query and snapshot. This inspects metadata and does not
launch training. For supplied weights, pass the same `TrainingTableSpec` as `train_table_spec=`.

```python
from kumoapi.model_plan import RunMode

plan = pq.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
print(plan)
```

Run modes include FAST, NORMAL, BEST and DEBUG. They change suggested search settings; names
are not quality guarantees or permission for larger workloads. Keep the requested mode.
`ModelPlan` has five public sections:

| Section | Configuration examples |
| --- | --- |
| `training_job` | `num_experiments`, `metrics`, `tune_metric`, `refit_trainval`, `refit_full`, `disable_explain` |
| `column_processing` | `encoder_overrides` by `table.column`; `na_strategy` by semantic type |
| `neighbor_sampling` | `num_neighbors`, `adaptive_sampling`, static-task `sample_from_entity_table` |
| `optimization` | `max_epochs`, step limits, `base_lr`, `batch_size`, supported loss/optimizer choices |
| `model_architecture` | Task-compatible `model` candidates such as GNNModelPlan and GraphTransformerModelPlan |

Some tunable fields take lists of candidates, not scalar values. Public dataclasses validate
assignments, and task validation further restricts options. Do not copy arbitrary fields from
an internal plan. Generation settings, split plans, lag settings and `forecast_length` belong
to `TrainingTableGenerationPlan`, not `ModelPlan`.

For a user-requested one-experiment, three-epoch smoke test without refits:

```python
plan.training_job.num_experiments = 1
plan.training_job.refit_trainval = False
plan.training_job.refit_full = False
plan.optimization.max_epochs = 3
```

This limits functional validation, not production modeling quality. For requested candidate
optimization and two-hop sampling, assignment shapes are:

```python
plan.optimization.base_lr = [0.001]
plan.optimization.batch_size = [256]
plan.neighbor_sampling.num_neighbors = [[16, 8]]
```

For a requested GNN architecture candidate, use the public nested model shape:

```python
from kumoapi.model_plan import GNNModelPlan

plan.model_architecture.model = [GNNModelPlan(channels=[128])]
```

Apply these values only when chosen for the task. Candidate count is not experiment count;
`num_experiments` controls trials, and refits add work. Explain all three before execution.

For an explicitly requested numerical encoder on an existing numerical feature:

```python
from kumoapi.encoder import Numerical

plan.column_processing.encoder_overrides = {
    "customers.account_balance": Numerical(),
}
```

Merge into existing overrides when present instead of erasing unrelated choices. The override
key must match the graph's actual table/column and semantic type. Do not use encoder choices
to hide source type errors or remove task-required label columns.

Metric names belong to `plan.training_job.metrics` and `tune_metric`. Choose names supported
by the actual task and keep tuning/evaluation direction explicit: AUROC/AP and ranking
precision are maximized; RMSE/MAE are minimized. Preserve averaging, label, K and horizon.
No universal metric set or arbitrary Python metric callable is supported here.

`refit_trainval=True` fits the selected configuration on train plus validation; `refit_full=True`
includes test data too. Full refit metrics are not an independent test estimate. Explanation
artifacts may use the selected train/validation refit. Record the selected trial and refit
when comparing results. Preserve the resolved `plan.as_json()` with run metadata, without
credential material. Execute exactly the requested experiments, then compare using diagnostics.
