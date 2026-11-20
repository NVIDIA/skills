# Examples: Requests that ASK BACK or REJECT

## 1. Missing required fields → ASK BACK

> "Use my sales data to figure out which customers are risky."

### Node 1a — Required fields: ASK BACK
- `source_scope`: "sales data" does not resolve to a database and schema ✗
- `prediction_entity`: customers ✓
- `prediction_target`: "risky" is not an observable outcome ✗
- `time_window`: `missing` — reads as temporal, no window ✗

**Not missing, not to be asked for:** schema, column types, join keys, output table.

Response to user:

> "To proceed I need three things:
>
> 1. **Where the data lives** — for example a database and schema, or a folder of files.
> 2. **What 'risky' means** — an observable outcome, e.g. 'a chargeback is filed' or 'no purchase'.
> 3. **The prediction window** — e.g. 'in the next 30 days'."

Why the reply stops here: the atomic check and preflight need a target and a source. Asking for
schema details now would be asking the user to do the agent's job.

---

## 2. Only the window is missing → ASK BACK for one field

> "On `USER_DATA.E_COMMERCE`, build a notebook that flags customers who will have a fraud event."

- `source_scope` ✓ `prediction_entity` ✓ `prediction_target` ✓ `time_window`: `missing` ✗

> "One thing: over what window should 'will have a fraud event' be judged — e.g. the next 30 days?"

Nothing else is asked. Not the output table, not the columns.

---

## 3. Source does not resolve → ASK BACK

> "Train on `USER_DATA.AD_CLICKS`, recommend paths per cookie."

Preflight: schema not found in `USER_DATA`.

> "I can't find a schema named `AD_CLICKS` in `USER_DATA`. Could you check the database and schema name? I won't guess a similar one."

---

## 5. Full data-science request → REJECT

> "Here is our Snowflake semantic view. Figure out what we should predict, choose the best target
> and horizon, build the best graph, tune the model, and deploy it."

> "I can't choose the predictive problem or optimise the model on your behalf. To generate training
> code I need: where the data lives, the entity that receives a prediction, the observable outcome
> (or interaction), and the time window. I can infer tables, keys and links from there and show
> them to you for confirmation. Which single question would you like to start with?"

No artifact is generated. The agent does not propose a target, horizon, or graph.

---

## 6. Bundled goals → REJECT

> "Predict which customers will churn AND which products to recommend to retain them."

Two tasks, two graphs, two PQL statements — one session covers one predictive task. Ask the user to submit them separately.
