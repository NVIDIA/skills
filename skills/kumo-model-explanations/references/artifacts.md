# Explanation artifact workflow

## Open the selected run

In the existing notebook, reuse the result of the requested fit:

```python
import kumo.engine as kumo

print("Kumo runtime:", kumo.__version__)
explanations = training_result.explanations
print("Run:", explanations.run_root)
print("Selected TEST trial:", explanations.trial_root)
print(explanations.availability)
```

To reopen an explicitly retained run instead, replace the assignment with
`explanations = kumo.TrainingExplanations.load("/path/to/retained/run")`.
Do not change `kumo.configure`, search other runs, or train while inspecting existing files.
If these APIs are absent from the executing build, report the runtime version and unavailable
interface rather than calling private trainers.

The loader resolves one `models/<task>/automl/best_trainer.json` within the supplied run and
reads the selected trial's `config.json` refit state. TEST explanations use the recorded
train+validation refit (`_refit_trainval`), otherwise the original trial; they never select the
full-data refit. Directory presence alone does not select a refit. Copied runs resolve within
the supplied root instead of following old absolute paths. Invalid or ambiguous selection
raises `kumo.InvalidRunError`; resolve the intended run rather than choosing the newest match.
Retain this selection metadata as well as the artifacts to reopen a run through the API.

Training availability keys are `gradient`, `cohorts`, and `samples`. Each entry reports `available`,
`path`, `reason`, and `generation_failure`. Availability checks presence, not contents;
a generation failure can leave readable partial artifacts. Missing files are not zero-valued
explanations. Reopening cannot recover generation status captured only during fit, so missing
artifacts may have an unknown cause. Readers raise `kumo.ExplanationUnavailableError` for
missing artifacts and `kumo.InvalidExplanationError` for malformed data.

Published model bundles exclude these files. `Trainer.load(bundle)` leaves explanations
unavailable with reason `model_only`; retain `explanations.run_root` separately.

## Inspect a batch prediction run

For a requested prediction, opt in per call:

```python
prediction = trainer.predict(snapshot, explanations=True)
explanations = prediction.explanations
print(explanations.run_root)
print(explanations.run_id)  # recorded ID, independent of the directory name
print(explanations.availability)
print(explanations.coverage)  # typed observed counts and checkpoint
# Later, reopen the complete retained run without inference:
explanations = kumo.PredictionExplanations.load("/path/to/retained/prediction/run")
```

The default `explanations=False` skips explanation computation. Supported outputs are
column cohorts and sampled graphs, using the current prediction inputs and published checkpoint.
To cap each population independently, pass
`explanations=kumo.PredictionExplanationOptions(cohort_rows=500, sample_rows=100)`.
These are independent positive row limits, including the final batch. With
`B = max(eval_batch_size // 4, 1)`, the default cohort cap is `max(50 * B, 1000)` rows rounded
up to whole batches: `ceil(max(50 * B, 1000) / B) * B`. Sampled graphs cover up to the first
1,000 prediction rows. A smaller population processes fewer rows. Generation starts after
predictions are saved and any configured export and tracking finish.
There are no prediction gradient scores, RFM/history readers or observed-target cohorts.
Node-level task support follows the cohort producer; supported link models have sampled graphs
and no cohorts. Unsupported tasks remain marked `unsupported` even if another producer fails.

Retain the compact `xai/manifest.json` status with the run. It is the authoritative explanation
record: actual prediction run ID, enabled/completion state, independent producer outcomes and
observed coverage. It does not duplicate the full prediction manifest or enumerate graph paths.
Loading requires valid status; merely copying artifact directories is insufficient. Copied runs
resolve files beneath their new root and keep their recorded run ID. Inspect recorded errors
rather than inferring success from JSON presence. Failed producers' partial files cannot be read through the normal API;
an independently completed producer remains readable. Completed predictions survive XAI failure.

Prediction cohorts live at `xai/column_analysis/pred/column_analysis.json`; graphs and mappings
live under `xai/subgraph_collector/pred/`. Use the same `cohort_keys()`, `cohort()`, `classes()`,
`plot_cohort()`, `samples()` and `sample()` methods described below, selecting only available
outputs. Plots use PRED stage and omit observed targets. Sampled graphs cover selected task rows,
so repeated entity IDs at different prediction times remain distinct. Read `explanations.coverage`,
a `PredictionExplanationCoverage` value with `cohort_rows_processed`, `sampled_graphs`, and
`checkpoint_path` relative to the published model root. `None` means unresolved; zero means
observed empty. Cohort counts include failed feature callbacks. These are generation counts,
not the number of files still available after cleanup or a partial copy. Sample membership
comes from `strategy_mappings.json`; unmapped files do not become prediction samples.
Check availability separately, and do not describe a bounded prefix as the entire prediction
population.

## Read and plot gradient scores

Training runs only. `PredictionExplanations` has no `gradient` availability entry and no
`gradient_scores()` or `plot_gradient()`; for a prediction run, say gradient attribution is not
produced for PRED and continue with cohorts or samples.

```python
from IPython.display import display

gradient = explanations.availability.get("gradient")
if gradient is None:
    print("Gradient attribution is not produced for prediction (PRED) runs.")
elif gradient.available:
    scores = explanations.gradient_scores().to_pandas()  # table, column, score; descending
    if scores.empty:
        print("No gradient rows were recorded.")
    else:
        print("Recorded score sum:", scores["score"].sum())
        display(scores.head(20))
        figure, axis = explanations.plot_gradient(top_k=20)
        axis.set_title(f"TEST gradient scores — {explanations.trial_root}")
        display(figure)
else:
    print(gradient)
```

The reader uses `explain_test/scores.csv` from the selected trial and returns the shared
`GraphGradientScore` schema. It validates the CSV rather than treating extra columns or
malformed rows as features. The producer aggregates gradients within examples using
gradient-based weights, averages across examples, then normalizes across all table-column
scores when the total is positive. Scores are written with six decimal places; a positive
recorded total is approximately one, while all-zero output remains zero.

`plot_gradient()` rereads that file, sorts before selecting the top N, labels bars
`table.column`, and preserves recorded values. Do not type values copied from another output,
renormalize a top-20 subset, or present scores as percentages. Report the run and TEST stage.
The helpers return figures and axes without calling `show()`. In notebooks use `display(figure)`;
in scripts use `figure.savefig("training-importance.png", bbox_inches="tight")` when saving is
requested, or `matplotlib.pyplot.show()` for a window. Plotting requires Matplotlib
(`kumo-engine[plots]`).

## Read and plot column cohorts

The reader validates `column_analysis/test/column_analysis.json` and returns a shared
`ColumnAnalysisOutput`. Discover the recorded task, complete `(table_name, hop, col_name)` keys,
and class labels before selecting a chart:

```python
if explanations.availability["cohorts"].available:
    analysis = explanations.column_analysis()
    print("Recorded task:", analysis.task_type)
    print("Cohort keys:", explanations.cohort_keys())
else:
    print(explanations.availability["cohorts"])
```

After choosing a key from that output, inspect its labels:

```python
key = ("users", 0, "age")  # replace with the table/hop/column you chose
print("Classes:", explanations.classes(key))
```

For binary positive-class or regression vectors, `classes(key)` is empty:

```python
cohort = explanations.cohort(key)
figure, (prediction_axis, population_axis) = explanations.plot_cohort(key)
display(figure)
```

For class-keyed multiclass or multilabel outputs, use this alternative with an explicitly
chosen label from `classes(key)`:

```python
class_name = "True"  # replace with your chosen recorded class label
cohort = explanations.cohort(key, class_name=class_name)
figure, (prediction_axis, population_axis) = explanations.plot_cohort(
    key, class_name=class_name)
display(figure)
```

`cohort()` returns `ColumnCohorts`; class selection validates the label while retaining the
full class dictionaries. The upper plot compares mean predictions and observed targets when
recorded; the lower plot shows population shares separately. These are relative shares, not
integer row counts. Do not derive counts without the correct denominator or fabricate a
missing negative-class vector. Preserve producer order, including `N/A` (missing values),
`OTHER(*)` (remaining categories), and `COUNT(*)` (relationship counts); none is a literal
user feature to remove automatically. Scores measure weighted between-cohort standard
deviation before separate normalization, not predictive accuracy or gradient attribution.

## Sampled subgraphs

The same post-training step writes per-entity subgraphs to `subgraph_collector/test/` in the
selected trainer directory: `strategy_mappings.json` plus `subgraph_json/<ordinal>.json`, one file
per sampled test entity. The API resolves the selected trainer from `best_trainer.json` and
its recorded `_refit_trainval` state as described above. TEST explanations use the
train+validation refit when recorded, otherwise the original fit. If the published bundle uses a
full-data refit, these artifacts describe the earlier fit, not the published model's parameters.
Do not glob every `version_*` directory or every run; inspect only the selected TEST trial.
The folder also holds other collector formats (`nested_json/`, `output/`, `rfm/`) and
`global_aggr_feature/`; read `subgraph_json/` only.

The files cover a bounded sample, not the test set. For each selection strategy (for binary
classification: `random`, `unsure`, true/false positive/negative, `old_entity`, `recent_entity`)
up to 50 entities are drawn from the processed test batches. `strategy_mappings.json` holds
`mappings` (strategy to ordinals, the file stems, as strings) and `orig_mappings` (strategy to
entity primary keys). An entity can appear under several strategies.

Each file is a `kumoapi.explain.Subgraph`:

- `subgraph_store`: `root_id` (the entity's node), `task_type`, `seed_time`, `nodes`, `edges`,
  `tables`. `nodes` maps node id to `{id, pkey_name, pkey}`. `edges` maps a source node id to
  `{relation: [destination node ids]}`; `tables` maps table name to node ids.
- `feature_store.features`: node id to `{column: {table_name, name, value, stype, score}}`.
- At most 16 neighbours are kept per relation. The rest become one aggregated node with a negative
  id, `count`, `start_time`, `end_time` and `pkey_name`; its features have `value` null and a
  `score` summed after normalization, so it can exceed 1 and is not on the same scale as a single
  node's.
- `score` is a gradient magnitude divided by the largest one in that subgraph, so it is
  comparable within one file, not across files or with `scores.csv`. It may be null.

List sampled graph references without loading their bodies:

```python
try:
    samples = explanations.samples()
except kumo.ExplanationUnavailableError as error:  # disabled, unsupported, failed, model_only
    print("Sampled graphs unavailable:", error)
    samples = ()
else:
    if not samples:
        print("No sampled graphs recorded:", explanations.availability.get("samples"))
for reference in samples:
    print(reference.artifact_id, reference.categories, reference.original_ids,
          reference.available)
```

`samples()` lists references without loading graph bodies. Collector-local `artifact_id` values
are not entity primary keys; `categories` and optional `original_ids` preserve the sampling
metadata. A reference may name a missing graph, so check `available`. Training graphs can also
be read without mapping metadata; prediction readers require membership in the saved mappings.
Arbitrary original-key lookup is not provided.

Choose an artifact ID from this inventory; do not silently select the first sample:

```python
artifact_id = "0"  # replace with your chosen available collector ID
reference = next((item for item in samples if item.artifact_id == artifact_id), None)
if reference is None:
    print("Sample ID was not recorded:", artifact_id)
elif reference.available:
    graph = explanations.sample(reference.artifact_id)  # typed kumoapi Subgraph
    print(graph.subgraph_store, graph.feature_store)
else:
    print("Sample unavailable:", reference.path)
```

The typed stores use integer node keys, unlike raw JSON's string object keys. Either store can
be absent. The API preserves structure, feature values, scores, timestamps, and aggregated
nodes; it does not render an interactive sampled graph.

Absent or empty `subgraph_json/` is not "no neighbourhood": report the path and recorded
availability reason; for training also check `disable_explain` and post-training errors, and follow *If artifacts are missing*. The collector skips forecasting, multilabel ranking, and
link-prediction models other than the hybrid, ID and shallow-RHS modules. These files carry
entity primary keys and raw feature values of real rows. Review them before sharing, and keep
them out of shared locations unless the user asks.

## If artifacts are missing

For a **training** run, check, in order: the selected run/path and retained files; runtime
generation support; `disable_explain` in the actual run plan; task/module support; and
post-training warning/error logs.

For a **prediction** run, check, in order: the selected `run_root` and its retained
`xai/manifest.json`; whether that `predict` call passed `explanations=True` or
`PredictionExplanationOptions` (the default `explanations=False` records reason `disabled`);
each producer's recorded outcome and error (`unsupported` follows task/module support); and the
prediction warning/error logs. `disable_explain` does not affect prediction.

A successful fit or prediction alone does not prove explanations succeeded. An older run cannot acquire
files merely by changing its plan now. Report malformed or empty artifacts explicitly instead
of interpreting them as zero importance.

For an already requested **new fit**, the public path is:

```python
# Existing model_plan, train_table and snapshot belong to the requested workflow.
model_plan.training_job.disable_explain = False
trainer = kumo.Trainer(model_plan)
training_result = trainer.fit(train_table, snapshot=snapshot)
```

This runs training; it is not a read-only regeneration command. There is no public standalone
explanation regeneration API here. `Trainer.load` does not restore missing explanation files.
An authorized new prediction can request `explanations=True`, including after model loading.
Do not invoke internal `best_trainer.explain` as customer API code.

## Evidence and compatibility

Source: Kumo modules, as shipped in the runtime:

- `kumo.engine.explain.results.TrainingExplanations`: lazy availability, typed readers,
  sample references and plotting; `kumo.engine.explain._artifacts`: retained-run selection
  and artifact validation; `kumo.engine.explain.plotting`: plotting semantics.

- `kumo.engine.trainer.trainer.Trainer.fit`: post-training dispatch, disable flag, error
  handling, local result fields and run directory.
- `kumo.engine.trainer.trainer.Trainer.predict`, `kumo.engine.explain.results.PredictionExplanations`
  and `kumo_core.prediction_explanations`: prediction replay, retained outcomes, support and
  independent producer completion.
- `kumo_core.train.trainer.Trainer.set_explainers`, `kumo_core.train.trainer.Trainer.explain`,
  and the explanation directory methods: task/module gates, test-stage refit selection and file
  roots.
- `kumoapi.explain.gradient`: CSV schema and normalization.
- `kumoapi.explain.column_analysis`: cohort schema and separate score normalization.
- `kumo_core.explain.subgraph_collector.SubgraphCollector.format_subgraph_json` and
  `kumoapi.explain.Subgraph`: `subgraph_json/` layout, fields, the 16-neighbour cap
  (`kumo_core.training.runtime.LocalExplainRuntime`) and score normalization.
- `kumo_core.training.explain_selection.select_input_nodes`: selection strategies, the 50-entity
  cap and `strategy_mappings.json`.

Source: Kumo ML producer modules:

- `kumoml.enterprise.explain.gradient`: `scores.csv` writer and aggregation.
- `kumoml.common.explain.column_analysis`: `column_analysis/test/column_analysis.json`
  writer, cohort semantics and supported task list.

These are source contracts, not a Snowflake runtime certification. The ML producer revision
must be checked against the executing build before expanding support claims. Local scratch
artifacts can disappear with the notebook runtime and are not in the model publication
allowlist. See [retention](../../kumo-model-persistence/references/retention.md) to
preserve requested files.
