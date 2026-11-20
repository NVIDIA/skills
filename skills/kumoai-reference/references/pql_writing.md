# PQL — Predictive Query Language

Constructs marked ``Source: shipped package (`<python.module.path>`)`` are verified against the Kumo package source and are ground truth. The rest of this page is the PQL surface as previously compiled; it is **not** re-verifiable against the published Kumo documentation, which is stale for this version and is not a source here — do not cite it, fetch it, or reconcile against it. Anything explicitly not verified is marked with a `>` note.

**Never emit a construct that is not on this page.** If the user's request needs one that is absent, say so rather than reaching for the docs or memory.

PQL is the query language; the surrounding Python is in `kumoai-reference/references/graph_construction.md` and `kumoai-reference/references/kumo_finetuning_sdk.md`.

**Never invent PQL syntax.** If the user's request cannot be expressed with the constructs on this page, say so and ask for a reformulation.

**Scope rule:** KumoRFM is a different product with a different validator (`FOR customers.customer_id=42`, `RANK TOP K` capped at 20, no joined static filters). Do not mix it in. Platform/fine-tuning PQL uses `FOR EACH`.

---

## Grammar

```text
PREDICT <target_expression>
FOR EACH <entity_table>.<primary_key>
[WHERE <condition>]
[ASSUMING <condition>]
```

- `PREDICT` and `FOR EACH` are **required**. `WHERE` and `ASSUMING` are optional.
- `FOR EACH` accepts **only a primary key column**. The entity table must be directly connected to the target table by a PK→FK relationship.
- `RANK TOP <K>` / `CLASSIFY` are suffixes on the target expression, inside the `PREDICT` clause.

> `RANK TOP <K>` has been seen both on its own line between `PREDICT` and `FOR EACH` and inline at the end of the `PREDICT` line. Prefer the inline form.

### Target forms

```text
PREDICT <table>.<column>                                                  -- static / column target
PREDICT <AGG>(<table>.<column>, <start>, <end>[, <unit>])                 -- temporal aggregation target
PREDICT <target_expression> <comparison_operator> <constant>              -- conditional -> classification
PREDICT LIST_DISTINCT(<table>.<fk_column>, <start>, <end>) RANK TOP <K>   -- ranking / link prediction
PREDICT LIST_DISTINCT(<table>.<column>, <start>, <end>) CLASSIFY          -- per-class binary classification
```

---

## Aggregation functions

Valid in the target and in temporal filters.

| Function | Applies to |
|---|---|
| `SUM` | numerical columns |
| `AVG` | numerical columns |
| `MAX` | numerical, timestamp |
| `MIN` | numerical, timestamp |
| `COUNT` | all columns (use `TABLE.*` to count rows regardless of N/A) |
| `COUNT_DISTINCT` | all columns |
| `LIST_DISTINCT` | all columns; recommended only if fewer than ~1000 possible values |
| `LAST` | all columns **except foreign keys** |
| `FIRST` | all columns **except foreign keys** |

Forecasting targets support **only** `SUM`, `COUNT`, `COUNT_DISTINCT` (`AVG` is undefined when an entity has no fact rows in a timeframe).

### Null / empty-window behavior (critical, differs per function)

- `SUM`, `COUNT`, `COUNT_DISTINCT`: if no fact rows exist in the window, the training label is **0**.
- `AVG`, `MIN`, `MAX`, `FIRST`, `LAST`: if no fact rows exist, the training example is **dropped**. This implicitly assumes every predicted entity has at least one fact row in the horizon. Use `IS NULL` / `IS NOT NULL` if you need to predict periods with no interactions.

This is the difference between "will spend $0" (`SUM`/`COUNT`, keeps inactive entities) and "what will they buy next" (`FIRST`/`LAST`, silently drops inactive entities).

---

## Time window / horizon

The horizon is expressed **inside the aggregation function**, not as a separate clause. There is no `WITHIN` keyword.

```text
<AGG>(<table>.<column>, <start>, <end>, <unit>)
```

- `start` is **exclusive**, `end` is **inclusive**.
- Both must be **non-negative integers**; `end` must be strictly greater than `start`.
- In the **target**, the window is forward-looking (positive). In a **temporal filter**, use negative offsets for the past.
- `unit` defaults to `days`. Valid: `minutes`, `hours`, `days`, `months` (uppercase also accepted). **Not** accepted: `seconds`, `weeks`, `years`, or any singular form (`day`, `week`).

Worked example: predicting on `2020-01-01 00:00:00` with `SUM(TRANSACTIONS.PRICE, 10, 30)` aggregates rows with timestamp `t` where `2020-01-11 00:00:00 < t <= 2020-01-31 00:00:00`.

```text
PREDICT COUNT(PURCHASES.*, 0, 48, hours) > 20
PREDICT SUM(Purchases.amount, 0, 6, months) > 500
```

Guidance: the fact table should hold history at least an order of magnitude longer than the prediction horizon (e.g. 1 year of rows for a 30-day horizon). At least 3 anchor times are needed for a healthy train/val/test split, so very large horizons can starve the splits.

---

## Comparison operators

Used to turn a target into a conditional (classification) target, and in filters.

| Operator | Applies to |
|---|---|
| `>` `<` `>=` `<=` | numeric, timestamp |
| `=` `!=` | all column types |
| `STARTS WITH` | text |
| `ENDS WITH` | text |
| `CONTAINS` / `NOT CONTAINS` | text |
| `LIKE` / `NOT LIKE` | text, `%` wildcard |
| `IN` | e.g. `LOAN.STATUS IN ('A', 'C')` |
| `IS NULL` / `IS NOT NULL` | all column types (`= NULL` / `!= NULL` also shown) |

Logical combinators: `AND`, `OR`, `NOT`. Each operand must itself be a filter operation. Supported in all filter fields and in targets.

```text
LAST(LOAN.STATUS, 0, 30) CONTAINS 'SUCCESS' AND LAST(LOAN.AMOUNT, 0, 30) > 30
NOT LAST(LOAN.AMOUNT, 0, 30) > 30
ARTICLES.DESCRIPTION NOT CONTAINS 'polyester'
LAST(LOAN.STATUS, 0, 30) LIKE '%SUCCESS'
```

---

## Filters

### `WHERE` — entity / temporal filter

Restricts which entities (and therefore which training examples) are included.

```text
PREDICT SUM(transactions.price, 0, 30)
FOR EACH customers.customer_id
WHERE COUNT(transactions.*, -30, 0) > 0
```

```text
PREDICT fraud_reports.is_fraud
FOR EACH transaction.id
WHERE transaction.type = "bank transfer"
```

Temporal filters use negative offsets (look-back). Static filters compare a column to a constant.

### Inline `WHERE` inside an aggregation

Restricts the rows that the aggregation counts, rather than the entities.

```text
PREDICT LIST_DISTINCT(
	transactions.article_id
  WHERE transactions.price > 100,
0, 30, days)
RANK TOP 25
FOR EACH customers.customer_id
```

Note the syntax: the inline `WHERE` goes **after the column reference and before the `start, end, unit` arguments**. It may also be written on one line:

```text
COUNT_DISTINCT(Ratings.rating WHERE Ratings.rating < 3, 0, 60, days)
```

Source: shipped package (`kumopql.grammar`)

**Inline vs trailing `WHERE` is the same keyword doing two different jobs** — inline selects rows for the aggregation, trailing selects eligible entities. A single query may use both; see "Both `WHERE`s in one query" under the worked examples.

### `ASSUMING` — hypothetical / future condition

Followed by one or more future-looking temporal aggregations assumed true at prediction time.

```text
PREDICT SUM(TRANSACTIONS.PRICE, 0, 30)
FOR EACH CUSTOMERS.CUSTOMER_ID
ASSUMING COUNT(NOTIFICATIONS.*, 0, 7) > 2
```

Also usable as a way to drop zero-heavy training rows in forecasting:

```text
PREDICT SUM(Sales.quantity, START_DAYS, END_DAYS)
FOR EACH Groups.group_id
ASSUMING SUM(Sales.quantity, START_DAYS, END_DAYS) > 0
```

`ASSUMING` is not a causal estimate. Do not present a difference between assumed and unassumed predictions as proof of causation.

---

## Date and time literals in filters (gotcha)

Timestamp constants are written as **bare ISO 8601, unquoted** — no quotes, no `DATE '...'` wrapper, no `TIMESTAMP` cast. Hours/minutes/seconds may be included and default to `0`.

```text
PREDICT COUNT(transactions.*, 0, 10)
FOR EACH customers.customer_id
WHERE customers.date_joined < 2022-01-01 12:45:30
```

```text
PREDICT CUSTOMERS.industry = 'IT' AND CUSTOMERS.date_of_birth <= 1990-01-01
```

String literals in filters appear both single-quoted (`'bank transfer'`, `'IT'`) and double-quoted (`"bank transfer"`).

---

## Task type is inferred from query shape

Kumo determines the task type automatically. `pquery.get_task_type()` returns it — always call it to confirm intent.

| Query shape | Task type |
|---|---|
| `PREDICT <table>.<numeric_column>` | Regression (static) |
| `PREDICT SUM/AVG/MIN/MAX/COUNT(...)` with no comparison | Regression (temporal) |
| Any target with a conditional operator (`=`, `>`, `<`, …) | Binary classification |
| `PREDICT FIRST/LAST(<categorical>, …)` over a high-cardinality categorical | Multiclass classification |
| `LIST_DISTINCT` over a **non-foreign-key** categorical column | Multilabel classification |
| `LIST_DISTINCT` over a **foreign key** column (+ `RANK TOP K`) | Link prediction / ranking |
| `LIST_DISTINCT(...) CLASSIFY` | Separate binary classification per class |
| Temporal regression meant as a forecast | Regression: `get_task_type()` reads the query alone, so it does not change when the generation plan sets `forecast_length`; that plan is what makes it a forecast (*Forecasting* below) |

Static vs temporal: a query without a time window is **static** (default 80/10/10 random split, no time column required). A query aggregating over a window is **temporal** (time-based splits, anchor times sampled back through history).

---

## Worked examples by task type

### Binary classification — churn

```text
PREDICT COUNT(transactions.*, 0, 90, days) = 0
FOR EACH customers.customer_id
WHERE COUNT(transactions.*, -60, 0, days) > 0
```

Predicts whether a customer active in the last 60 days makes zero transactions in the next 90. `=` makes it classification; swapping to `>` changes which class is positive. Output is the probability of the positive class.

### Binary classification — will a *qualifying* event occur (temporal, filtered)

Use this shape for "will entity X have at least one event of a particular kind in the next N days" — fraud, conversion, unsubscribe, churn-by-event. The condition that defines *which* rows count goes **inside** the aggregation.

```text
PREDICT COUNT(transactions.* WHERE transactions.is_fraud = 1, 0, 30, days) > 0
FOR EACH accounts.account_id
```

When the label is already a numeric 0/1 flag, summing it is equivalent:

```text
PREDICT SUM(fraud_reports.LABEL, 0, 30, days) > 0
FOR EACH accounts.account_id
```

**Do not** express the event condition as an entity-level `WHERE`. These mean different things and only the first is correct here:

| Query | Meaning |
|---|---|
| `PREDICT COUNT(t.* WHERE t.is_fraud = 1, 0, 30, days) > 0` | will this account have a **fraudulent** transaction |
| `PREDICT COUNT(t.*, 0, 30, days) > 0 ... WHERE t.is_fraud = 1` | will this account have **any** transaction, training only on accounts that already had fraud |

The second leaks the label into entity selection and predicts the wrong thing. If the target condition describes the *future event*, it belongs inline; if it describes *which entities are eligible*, it belongs in the trailing `WHERE`.

Also do not reach for the static form (`PREDICT fraud_reports.is_fraud`) when the request names a time window — static queries have no horizon and cannot answer "in the next 30 days".

#### Both `WHERE`s in one query — the canonical example

Source: shipped package (`kumopql.grammar`, `kumo.engine.pquery.predictive_query`)

This query parses under the PQL grammar (`PredictiveQuery.validate()` parses and validates the query string against the graph) and uses **both** jobs of the `WHERE` keyword at once. Copy this shape when a request has a row-level target condition *and* an entity-level cohort condition.

```text
PREDICT COUNT_DISTINCT(Ratings.rating WHERE Ratings.rating < 3, 0, 60, days) > 0
FOR EACH Movies.MovieID
WHERE AVG(Ratings.rating, -60, 0, days) > 3
```

Read it clause by clause:

| Clause | Job |
|---|---|
| `WHERE Ratings.rating < 3` (inline, inside `COUNT_DISTINCT`) | selects **which rows** the aggregation counts — only low ratings |
| `0, 60, days` | forward target window |
| `> 0` | makes it binary classification: at least one low rating occurs |
| `WHERE AVG(Ratings.rating, -60, 0, days) > 3` (trailing) | selects **which entities** are eligible — only movies well rated over the past 60 days |

Same keyword, two different jobs, in one query: "will a currently well-rated movie start getting low ratings?". Moving the inline condition to the trailing `WHERE` would instead train only on movies that already had low ratings and predict whether they get *any* rating — a different and wrong question.

Note also the offsets: the target window is non-negative (`0, 60`), the entity filter looks back (`-60, 0`).

### Binary classification — static / non-temporal

```text
PREDICT fraud_reports.is_fraud
FOR EACH transaction.id
WHERE transaction.type = "bank transfer"
```

```text
PREDICT customer_item_pairs.likes = 1
FOR EACH customer_item_pairs.uuid
```

Rows where the label is NULL are dropped from training and become the prediction set.

### Multiclass classification — next purchase category

```text
PREDICT FIRST(purchase.type, 0, 7)
FOR EACH user.user_id
```

`FIRST` is undefined when a user made no purchase in the window, so those examples are dropped — every prediction assumes the user buys something.

### Regression — static

```text
PREDICT customers.age
FOR EACH customers.customer_id
```

### Regression — temporal (LTV)

```text
PREDICT SUM(transactions.price, 0, 30)
FOR EACH customers.customer_id
WHERE COUNT(transactions.*, -30, 0) > 0
```

### Temporal link prediction / recommendation

```text
PREDICT LIST_DISTINCT(transactions.article_id, 0, 30, days)
RANK TOP 25
FOR EACH customers.customer_id
```

```text
PREDICT LIST_DISTINCT(transactions.article_id, 0, 7) RANK TOP 10
FOR EACH customers.customer_id
```

With an inline filter on candidate items:

```text
PREDICT LIST_DISTINCT(
	transactions.article_id
  WHERE transactions.price > 100,
0, 30, days)
RANK TOP 25
FOR EACH customers.customer_id
```

Output is K rows per entity by default; `num_classes_to_return` at prediction time can return a different count (1–500 for link prediction; `kumoai-reference/references/kumo_finetuning_sdk.md` §9). Ranking tasks have a limit of 10 million target items. `RANK` on a foreign key is the only case where new targets may be added at batch-prediction time (by adding rows to the item table).

### Static link prediction — item-to-item similarity

```text
PREDICT LIST_DISTINCT(copurchase.dst_item_id) RANK TOP 10
FOR EACH src_item.item_id
```

`LIST_DISTINCT` with **no window arguments** → static link prediction.

### Multilabel via CLASSIFY

```text
PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID, 0, 30) CLASSIFY
FOR EACH CUSTOMERS.CUSTOMER_ID
```

Produces a separate binary classification per article ID. Without `TOP K`, classes are generated for all available classes up to the limit.

### Forecasting

Forecasting is **not a distinct PQL keyword**. Write a temporal regression query, set `forecast_length` on the `TrainingTableGenerationPlan` passed to `generate_training_table(snapshot, plan=...)`, and pass the fitted trainer's `trainer.training_table_generation_plan` to `generate_prediction_table(snapshot, plan=...)` — both take a `TrainingTableGenerationPlan`, not a `PredictionTableGenerationPlan`, and never the model plan (Source: shipped package (`kumoapi.model_plan`, `kumo.engine.pquery.predictive_query`)); `forecast_length=3` on a 7-day window forecasts the next three 7-day windows. The flow is in `kumoai-reference/references/task_journeys.md` *Forecasting*.

```text
PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 7, days)
FOR EACH CUSTOMERS.CUSTOMER_ID
```

```text
PREDICT COUNT(Sales.*, 0, 28)
FOR EACH Items.item_id
```

```text
-- 6th week out
PREDICT SUM(Sales.quantity, 35, 42)
FOR EACH Groups.group_id
```

Targets restricted to `SUM`, `COUNT`, `COUNT_DISTINCT`. `forecast_type` (`mlp` default, `implicit`, `autoregressive`) is a model-plan option. Treat forecasting as **experimental**.

---

## Evaluation metrics per task type

| Task type | Metrics Kumo reports |
|---|---|
| Binary classification | Accuracy, AUROC, AUPRC, confusion matrix, gain chart |
| Multiclass classification | Accuracy |
| Multilabel classification | AUPRC, AUROC, AP (macro / micro / per-label); also F1, Hamming score |
| Link prediction | F1@K, MAP@K, Precision@K, Recall@K for K = 1, 10, 100 |
| Regression | MAE, MSE, RMSE, MAPE, SMAPE, prediction-distribution histogram |

Additional selectable metrics (model-plan `metrics` field): binary adds `ap`, `f1`, `ndcg[@k]`, `precision[@k]`, `recall[@k]`; multiclass adds `f1`, `precision`, `recall`; link prediction adds `mrr@k`, `hit_ratio@k`, `coverage@k`, `avg_popularity@k`, `personalization@k`, `diversity[col]@k`; forecasting adds `neg_binomial`, `normal`, `lognormal`.

**Defaults:** the SDK's default model plan sets `metrics` and `tune_metric` per task type, for
every run mode:

| Task type | Reported metrics | Tuned on |
|---|---|---|
| Binary classification | `auroc`, `ap`, `auprc` | `auroc` |
| Multiclass classification | `acc` | `acc` |
| Multilabel classification | `ap`, `auroc`, `auprc`, each `_macro` / `_micro` / `_per_label` | `ap_macro` |
| Regression, forecasting | `mae`, `mse`, `rmse` | `mae` |
| Ranking, link prediction | `map`, `precision`, `recall`, `f1`, each `@1` / `@10` / `@100` | `map@10` |
| Link prediction with a training-table weight column | `weighted_ndcg`, `weighted_recall`, `precision`, each `@1` / `@10` / `@100` | `weighted_ndcg@10` |

Source: shipped package (`kumo_core.model_plan`)

---

## Gotchas and malformed-query traps

1. **No `WITHIN` / `ON` / `TOP_K` / `NEXT_ITEM` keywords.** The horizon lives in the aggregation arguments; the entity clause is `FOR EACH`; ranking is `RANK TOP K`.
2. **Do not mix static conditions and temporal aggregations in one target.** Temporal aggregations depend on anchor time, static conditions do not; they need different sampling strategies. Documented as invalid practice.
3. **`start` must be non-negative and `end` strictly greater than `start`** in the target. Negative offsets only in temporal filters.
4. **Unit names must be plural and from the allowed set.** `day`, `week`, `weeks`, `years`, `seconds` are rejected.
5. **Date literals are unquoted ISO 8601.** See the date/time section.
6. **`FOR EACH` requires a primary key.** If the entity table has duplicate rows, pre-build a `SELECT DISTINCT` view first.
7. **FK direction matters.** The entity table must connect to the target table via PK→FK. In the fraud example, `transaction` holds the FK to `fraud_reports`, not the reverse — otherwise multiple reports could link to one transaction.
8. **`LAST`/`FIRST` cannot be applied to foreign key columns.** Use `LIST_DISTINCT` + `RANK` for FK targets.
9. **`LIST_DISTINCT` over a non-FK column silently becomes multilabel classification, not ranking.** Check `get_task_type()`.
10. **`LIST_DISTINCT` beyond ~1000 distinct values is not recommended.**
11. **Horizon vs. split health:** too large a window leaves fewer than 3 anchor times; too small a window on a long history generates too many examples (mitigate with `train_start_offset` on the `TrainingTableGenerationPlan` passed to `generate_training_table(snapshot, plan=...)`; the model plan has no such field) and, for link prediction, too few positives.
12. **Always validate before generating tables:** `pquery.validate()` for syntax, `pquery.get_task_type()` for semantics. Required — see "Required pre-flight" under SDK usage.
13. **Inline `WHERE` vs trailing `WHERE` are not interchangeable.** A row-level condition on the predicted event goes inside the aggregation; an entity-eligibility condition goes in the trailing `WHERE`. Swapping them produces a query that validates and trains on the wrong question.

---

## SDK usage

`PredictiveQuery(graph=graph, query=<str>)` binds the graph at construction. Call `validate()` and `get_task_type()` with no arguments.

```python
import kumo.engine as kumo

pquery = kumo.PredictiveQuery(
    graph=graph,
    query=(
        "PREDICT MAX(transaction.Quantity, 0, 30, days)\n"
        "FOR EACH customer.CustomerID\n"
        "ASSUMING SUM(transaction.UnitPrice, 0, 7, days) > 15"
    ),
)

pquery.validate()
print(pquery.get_task_type())
```

Build multi-line query strings with a triple-quoted string and `.strip()`. Explicit `\n` separators also work.

```python
QUERY = """
PREDICT COUNT_DISTINCT(Ratings.rating WHERE Ratings.rating < 3, 0, 60, days) > 0
FOR EACH Movies.MovieID
WHERE AVG(Ratings.rating, -60, 0, days) > 3
""".strip()
```

Table names used in PQL are the **source table names** — `Graph(tables=[...])` has no keys. See `kumoai-reference/references/graph_construction.md`.

Source: shipped package (`kumo.engine.pquery.predictive_query`)

### Required pre-flight, before any training step

Source: shipped package (`kumo.engine.pquery.predictive_query`)

Always emit these two calls, in this order, immediately after constructing the `PredictiveQuery` and before generating training tables or fitting:

```python
pquery.validate()                      # syntax + schema; call with no arguments
print('task type:', pquery.get_task_type())
```

`validate()` catches syntax and graph-mismatch errors; `get_task_type()` is the semantic check: the task type is never written in the query, only inferred from its shape, so print it and confirm it matches the user's intent before spending a training run. A query that validates can still be the wrong task (see the `LIST_DISTINCT` multilabel-vs-ranking trap below).

> There is no `KumoFineTuning` class. Do not emit `kumo.init()` — hosted-edition only, out of scope.

> `RANK BY` appears in one stray example elsewhere but is defined nowhere and contradicts the `LIST_DISTINCT(...) RANK TOP K` form used here. Treat `RANK BY` as unverified; **do not emit it.**

---

## Mapping a validated request to PQL

| Validated field | PQL element |
|---|---|
| Prediction target | `PREDICT` expression — pick the aggregation from the target column type |
| Primary entity | `FOR EACH <entity_table>.<primary_key>` |
| Prediction horizon | `start`, `end`, `unit` arguments inside the aggregation |
| Eligibility / cohort constraint (which **entities**) | trailing `WHERE` (temporal look-back or static column filter) |
| Condition on the predicted event (which **rows** count) | inline `WHERE` inside the aggregation, before the window arguments |
| Hypothetical future condition | `ASSUMING` |
| Number of recommendations | `RANK TOP <K>` |
| Task type | Not written explicitly — inferred from query shape; confirm with `get_task_type()` |
