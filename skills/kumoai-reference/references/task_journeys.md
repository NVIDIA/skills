# Complete a task journey

Start with a confirmed graph, query, source scope, population, evaluation protocol and workload.
The examples below share the [SDK setup](kumo_finetuning_sdk.md). Symbols in uppercase are
user-selected inputs; table/column names in PQL are illustrative and must match the actual
graph. Each journey validates the query, generates populations, suggests a model plan, fits,
predicts and inspects output. Do not execute every example as one notebook.

## Classification and regression

Choose one query from [PQL forms](pql_writing.md), then set `EXPECTED_TASK` to its confirmed
public `TaskType`, imported from `kumoapi.task`. `IS_TEMPORAL` comes from the confirmed query;
`SCORING_TIME` is the requested naive datetime in UTC. `PREDICT_OPTIONS` contains only supported
options for that task, as listed below.

```python
import kumo.engine as kumo
from kumoapi.model_plan import RunMode

pq = kumo.PredictiveQuery(graph=graph, query=QUERY)
pq.validate()
if pq.get_task_type() != EXPECTED_TASK:
    raise ValueError("Resolved query task differs from confirmed intent")
snapshot = graph.snapshot()
training_table = pq.generate_training_table(snapshot)
plan = pq.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
# Apply only previously requested workload/configuration choices here.
trainer = kumo.Trainer(plan)
training_result = trainer.fit(training_table, snapshot=snapshot)
anchors = {"prediction_time": SCORING_TIME} if IS_TEMPORAL else {}
population = pq.generate_prediction_table(snapshot, **anchors)
predictions = trainer.predict(population, snapshot=snapshot, **PREDICT_OPTIONS)
print(training_result.metrics)
print(predictions.num_rows, predictions.files)
```

| Journey | Task confirmation, scoring options and interpretation |
| --- | --- |
| Static or temporal binary | Confirm BINARY_CLASSIFICATION and positive condition. Supply `binary_classification_threshold=THRESHOLD` in [0,1]. `True_PROB`/`False_PROB` are probabilities and `TARGET_PRED` is the thresholded label. Compare AUROC/AP and threshold metrics only when recorded |
| Static or temporal multiclass | Confirm MULTICLASS_CLASSIFICATION and class meaning. No binary threshold. Optional `num_classes_to_return` limits returned classes only when requested. Long rows carry CLASS, SCORE and PREDICTED with entity metadata. Compare compatible class/averaging metrics and per-class cohorts when present |
| Static or temporal multilabel | Confirm MULTILABEL_CLASSIFICATION (or explicitly intended MULTILABEL_RANKING) and label set. No explicit binary threshold for multilabel prediction output: the API rejects it. Long CLASS/SCORE/PREDICTED rows represent separate label outcomes; do not interpret scores as exclusive class probabilities |
| Static or temporal regression | Confirm REGRESSION and units. No threshold or top-class count. Inspect TARGET_PRED, MAE/RMSE when recorded, and matching temporal/split semantics |

Static tasks omit prediction_time entirely, and a static query scores only rows whose target is
NULL; to score rows that already have one, use the population table confirmed at the task gate
(`kumoai-reference/references/kumo_finetuning_sdk.md` §7.1). Supplied exact populations use
[custom populations](../../kumo-custom-populations/SKILL.md) instead of generated populations;
never attach evaluation labels to a scoring table. For generated static tasks, verify which
rows are eligible before claiming all entities were scored.

## Links and ranking

Use the same journey with a confirmed FK target, for example
`PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 10 FOR EACH customers.customer_id`.
For static links use `LIST_DISTINCT(pairs.product_id)` without a time window and omit the
scoring timestamp. Confirm TEMPORAL_LINK_PREDICTION or STATIC_LINK_PREDICTION respectively.
Set `PREDICT_OPTIONS = {"num_classes_to_return": N}`, where N is the confirmed number of
results per entity: the query's `RANK TOP K` unless the user asked for a different count
(1–500; `kumoai-reference/references/kumo_finetuning_sdk.md` §9). No binary threshold.
Preserve the item universe, positive interaction, eligibility and evaluation sampling. Inspect
entity/CLASS and available SCORE/time columns. Returned rows can be fewer than N when
candidates are unavailable. Compare recorded precision/MAP/recall at the same K.

For a non-FK category ranking, confirm the distinct MULTILABEL_RANKING task rather than treating
it as a link to a product table. Do not create a graph edge solely to force a task type.

## Forecasting

Confirm interval, number of steps, target units, required history and evaluation horizon.
Use SUM, COUNT or COUNT_DISTINCT over that interval. The initial query can resolve as
REGRESSION before generation settings promote it to forecasting; use the generation plan in this workflow, not a `ModelPlan.training_table_generation` field.

This example forecasts `FORECAST_STEPS` consecutive windows for the confirmed temporal query:

```python
from kumoapi.model_plan import TrainingTableGenerationPlan, RunMode
from kumoapi.task import TaskType

pq = kumo.PredictiveQuery(graph=graph, query=FORECAST_QUERY)
pq.validate()
if pq.get_task_type() not in (TaskType.REGRESSION, TaskType.FORECASTING):
    raise ValueError("Forecast query must have a supported numerical temporal target")
snapshot = graph.snapshot()
generation = TrainingTableGenerationPlan(forecast_length=FORECAST_STEPS)
training_table = pq.generate_training_table(snapshot, plan=generation)
plan = pq.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
trainer = kumo.Trainer(plan)
training_result = trainer.fit(training_table, snapshot=snapshot)
population = pq.generate_prediction_table(
    snapshot, prediction_time=SCORING_TIME,
    plan=trainer.training_table_generation_plan,
)
predictions = trainer.predict(population, snapshot=snapshot)
print(trainer.training_table_generation_plan)
print(training_result.metrics, predictions.num_rows, predictions.files)
```

Inspect effective generation metadata to confirm the intended forecast length, lags and
intervals. Prediction must reuse the fitted generation settings. Outputs expand into one row
per entity/forecast timestamp with TARGET_PRED and ANCHOR_TIMESTAMP; configured quantiles can
add q_* columns. Compare horizon-aligned errors and temporal splits, not one-window regression
scores as if they established all horizons. Column analysis is not supported for forecasting.

## Finish each journey

Use [outputs](outputs.md) for requested write-back and a separate embedding call.
Inspect selected Parquet schemas before interpreting columns; do not hardcode optional fields.
Use [diagnostics](../../kumo-training-diagnostics/SKILL.md) for actual metrics and trial histories,
and [explanations](../../kumo-model-explanations/SKILL.md) for available attribution/cohorts.
Missing explanations do not imply training failed. Retain requested artifacts via
[persistence](../../kumo-model-persistence/SKILL.md) before notebook teardown.
