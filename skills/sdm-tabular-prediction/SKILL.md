---
name: sdm-tabular-prediction
description: Runs single-table classification and regression with NVIDIA Structured Data Models (SDM), a PyTorch-native library. Defaults to KumoTabular, a pretrained tabular foundation model. Use for in-context prediction from labeled tables, including holdout evaluation, batch scoring, and exporting predictions.
license: Apache-2.0
---

# Tabular prediction with SDM

Build an in-context task from labeled rows (`x_context`, `y_context`) and the features of the rows to predict (`x_query`). Use the requested model from `sdm.models`. For a new baseline, try `KumoTabular` if the installed version includes it. Check the [model API](https://nvidia.github.io/structured-data-models/api/models) for model-specific limits and outputs.

The package is [`structured-data-models`](https://nvidia.github.io/structured-data-models/install) and the import is `sdm`. Pretrained constructors can fetch weights on first use. TabFM requires acceptance of its noncommercial weights license for pretrained inference.

## Prepare context and query rows

- Preserve the supplied split. When creating an evaluation split, reflect how predictions will be used, such as holding out later rows for a time-dependent task. Keep query labels for scoring and exclude the target and any features unavailable at prediction time from both feature tables.
- Select one target column. Use the `categorical` semantic type for classification, including integer class labels, and `numerical` for regression. For classification, include the classes the model is expected to predict in the labeled context.
- Infer feature semantic types on context rows and share the resulting column mapping with the query. Override ambiguous types with `sdm.infer_stypes(..., overrides=...)`, such as integer category codes or measurements with few distinct values. Start with the model's default recipe and fit any additional data-dependent preprocessing on context rows only.

This classification example assumes `context_df` and `query_df` are pandas DataFrames, `target` is the label column name, and `feature_columns` lists the selected features.

```python
import torch

import sdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_features = context_df[feature_columns]
feature_stypes = sdm.infer_stypes(context_features)
x_context = sdm.TableTensor.from_pandas(
    df=context_features, stypes=feature_stypes, device=device
)
y_context = sdm.TableTensor.from_pandas(
    df=context_df[[target]], stypes={target: "categorical"}, device=device
)
x_query = sdm.TableTensor.from_pandas(
    df=query_df[feature_columns], stypes=feature_stypes, device=device
)

model = sdm.models.KumoTabular(task="classification", size="small", device=device)
predictions = model(x_context=x_context, y_context=y_context, x_query=x_query)
```

For regression with KumoTabular, use `task="regression"` and set the target semantic type to `numerical`.

## Reuse context across query batches

For repeated queries, pass an iterable of query `TableTensor` batches as `x_query_batches`. Build each batch with the same feature columns and semantic types as `x_context`.

```python
model.fit(x=x_context, y=y_context)
try:
    batch_predictions = [model.predict(batch) for batch in x_query_batches]
finally:
    model.clear()
```

`fit` caches context for inference without updating model weights. Refit if the context or fitted recipe changes.

## Interpret and check predictions

- Preserve query row order or keep a separate row key when joining predictions back to the original data. Check that the number of outputs matches the number of query rows.
- For classification, class labels identify the probability columns. Build a categorical `y_query` in the same row order and use `sdm.evaluation.to_class_indices(predictions, y_query)` to align scores to the target's category order. For binary metrics, use `sdm.evaluation.to_binary_class(predictions, y_query, positive_class=...)` with the actual positive label. If a held-out class has no score column, inspect context class coverage before reporting metrics.
- For regression, inspect the output columns. KumoTabular and TabICLv2 expose the median as `predictions["q500"].numerical` with the default recipe. TabFM returns point predictions in `predictions["pred"].numerical`.

See the [tabular quickstart](https://github.com/NVIDIA/structured-data-models/blob/main/examples/tabular/quickstart.py) for the one-shot and cached APIs, and the [in-context learning guide](https://nvidia.github.io/structured-data-models/icl) for ensembling and context selection.
