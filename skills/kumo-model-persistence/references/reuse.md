# Publish and restore

## Publish during the requested fit

Inputs: a confirmed `model_plan`, `training_table` and its compatible `snapshot`, plus a
Snowflake connector scoped to the selected registry database/schema. Choose a new explicit
model version. Check the destination and privileges first; do not overwrite an existing
version or silently let the registry choose its default on reload.

```python
import kumo.engine as kumo

location = kumo.ModelRegistryLocation(model_name=MODEL_NAME, version=MODEL_VERSION)
trainer = kumo.Trainer(model_plan)
training_result = trainer.fit(
    training_table, snapshot=snapshot,
    export_config=kumo.ModelOutputConfig(connector=registry_connector, location=location),
)
published = training_result.published_model
if published is None:
    raise RuntimeError("Fit returned no publication record")
print(published.name, published.version)
```

Record the returned name/version, registry database/schema, query, graph-building code,
column/key/time metadata, source scope, package versions and intended evaluation protocol.
Keep credentials outside this record. `ModelOutputConfig` accepts exactly one of `path` or
`location`; Snowflake uses `location`, not `path`. Do not run a second fit just to inspect or
retrieve an existing publication. An exception during publication is not evidence training
never ran; inspect retained results before any retry.

## Restore and prepare scoring inputs

```python
loaded = kumo.Trainer.load_from_connector(
    registry_connector,
    location=kumo.ModelRegistryLocation(model_name=MODEL_NAME, version=MODEL_VERSION),
)
```

For an explicitly retained complete local bundle, use this alternative:

```python
loaded = kumo.Trainer.load(RETAINED_BUNDLE_DIRECTORY)
```

Load only trusted publications. The loader checks bundle integrity and training metadata;
hashes detect corruption, not publisher trust. Missing `training-spec.json`, incompatible
spec versions or failed integrity checks are actionable restore errors. Retrieve the complete
correct bundle; do not fabricate metadata or bypass validation.

Reconstruct the Snowflake sources, immutable Tables, keys/types/time columns and Edges from
the recorded graph definition, then materialize `graph.snapshot()`. Preserve compatible
feature schema, topology, entity/target contract and encodings. New scoring data is possible;
an old snapshot ID is provenance, not a substitute for current compatible graph data. Do not
force compatibility by editing bundle internals.

With that reconstructed `graph` and recorded `TRAINED_QUERY`, and a user-selected naive UTC
`SCORING_TIME` for a temporal query:

```python
pq = kumo.PredictiveQuery(graph=graph, query=TRAINED_QUERY)
pq.validate()
print(pq.get_task_type())
snapshot = graph.snapshot()
population = pq.generate_prediction_table(
    snapshot, prediction_time=SCORING_TIME,
    plan=loaded.training_table_generation_plan,
)
# This example is binary classification; preserve the chosen threshold.
scores = loaded.predict(
    population, snapshot=snapshot, binary_classification_threshold=THRESHOLD,
)
print(scores.num_rows, scores.files)
```

For static queries omit `prediction_time`; for other tasks use their supported scoring
arguments in [task journeys](../../kumoai-reference/references/task_journeys.md). Forecast scoring must
reuse `loaded.training_table_generation_plan`. Exact populations use
`PredictionTable.from_source_table`, not an entity filter silently replacing supplied rows.
The loader restores model state; scoring still needs explicit population and snapshot.

## Pick the version and rebuild from its record

A registry holds several versions, and the default is not necessarily the newest. Take the
version the user named. If they said "latest" or "newest", read `SHOW VERSIONS` and use the
newest `created_on`, say which version that is, and pass it explicitly. If they named none and
several exist, ask; `version=None` loads the registry default, not the newest. A model that was
never published has no version to load; say so instead of choosing another.

The registry records only `spec_version`, the predictive query and `snapshot_id` as searchable
metadata, so the table and column lists come from the team's record, not from `SHOW VERSIONS`.
Rebuild the graph from the team's training record for that exact version: its tables, keys,
time columns, edges, query and threshold, not from the live schema and not from another
version's notebook. A column the live source gained is left out. A recorded column the live
source no longer has stays in the graph definition: do not drop it quietly. Name it in a
markdown cell, and before building any table, compare the recorded columns with the live ones
and raise an error naming each missing column per table. Binary scoring needs an explicit
threshold even for a loaded model.

A retained local bundle path from the training host may not exist in this session. List the
filesystem, use the bundle directory the user chose, and prefer the complete one over a stale
neighbour.

## What survives publication

| Content | Lifetime and retrieval |
| --- | --- |
| Selected checkpoint, encoders/target encoder, trainer config and resolved plan | Complete portable bundle |
| Graph/task/input manifests, metrics, optional per-epoch search-trial history, prediction schema, training specification | Bundle metadata, not source rows or graph storage |
| Graph index, staged source export, generated training and scoring tables | Separate local working state; recreate or explicitly retain as appropriate |
| Per-trial results, refit epochs, progress, attribution and column analysis | Separate run files; retain selected diagnostics before session teardown |
| Predictions and embeddings | Separate selected Parquet outputs or explicit Snowflake writes |

A bundle alone cannot provide an absent history (older bundles), refit epochs, or explanation files. Local scratch paths
are disposable unless their storage lifetime has been established. Keep retention separate
from a reusable index cache, which can be rebuilt and is not a portable model bundle.
