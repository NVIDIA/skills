# Input Contract — What a request must contain

The contract is keyed by **operation**. Decide the operation first, then apply that operation's
table. The tables below are the complete contract: a field that is not in them is not required.

Three kinds of field exist, and the distinction is the whole point of this file:

| Kind | Who supplies it | If absent |
|---|---|---|
| **Required** | the user | ASK BACK — ask for exactly these, nothing else |
| **Inferred, then confirmed** | the agent, by read-only preflight (`accept/preflight.md`) | never asked; shown as `[inferred]` and confirmed at the graph gate — except a key that preflight could only fail to disqualify, which is shown `[inferred, unverified]` and cannot be used until confirmed |
| **Optional** | the user, if they want | never asked; defaults apply; if supplied it must be preserved verbatim |

**Infer structure, never infer intent.** Column names, keys, time columns and links are structure
— measure them. Entity, outcome, positive-label condition, interaction, item, window and
destination are intent — the user's, and only theirs. No outcome definition is ever invented.

A preflight result or schema listing **never fills a required field.** Seeing a `sales_events`
table does not tell you the positive interaction is "a sale"; seeing a `target_label` column does
not tell you the positive-label condition. If the user did not state it, it is missing — ask,
even when the schema makes the answer look obvious. Preflight may resolve *which table holds*
what the user named; it may not supply what they did not name.

**The quote test — is a required field present?** A required field is present only if you can
quote the user's own words for it, from any turn, in any vocabulary: "each customer", "a
cancelled booking", "at least one fraud event", "next 30 days". A table name, a column name, or "the only
entity in the schema" is not a quote — the field is missing, and the reply is an ask-back, not a
graph proposal.

**Named intent, inferred mapping — never ask the user to map their words onto columns.** Once the
user has named a field, turning their words into a physical table, column, or value is structure.
Preflight infers it — "a cancelled booking" → `bookings` where `status` marks a cancellation —
and it is shown as `[inferred]` in the proposal for the user to confirm or correct. Do
not ask which column or value implements something the user already named, and do not ask them to
re-define it more precisely. Inferred values are fine as long as the complete proposal is shown
and confirmed; questions are only for missing, ambiguous, or contradictory inputs.

The two rules are one rule seen from both sides: the **user's words** decide *whether* a field is
present; the **data** decides *how* a present field is implemented.

---

## Operation

`train` or `batch_inference`. Read it from the request: "train / build a model / notebook that
predicts" is `train`; "use my trained model / score / run inference with job X" is
`batch_inference`. One session covers one operation, one task, one graph, one artifact.
Ask only if the request genuinely reads both ways.

---

## `train` — required fields (ask if missing)

| Field | What it must include |
|---|---|
| **`source_scope`** | Where the tables live, in a form the active host accepts (`hosts/<host>/adapter.md`, *Source scope*): for example a database + schema, a dataset, a catalog + schema, a bucket prefix or a directory of files, or a connector name the host knows. On a host that supports them, a fully qualified semantic view (`DB.SCHEMA.VIEW`) contributes exactly its base tables, which must share one schema (`semantic_view.md`). This is where tables are enumerated. Credentials are never requested in chat. |
| **`prediction_entity`** | What receives one prediction — "each account", "each customer". A physical table/key is welcome but not required; preflight resolves it. |
| **`prediction_target`** | The observable outcome (classification/regression) or the positive interaction (link prediction), **stated by the user**. For classification this includes the positive-label condition — "at least one fraud event"; "predict fraud" alone leaves it missing. For link prediction and ranking it includes the **item** being recommended and the **interaction** that counts as positive — "spare-part families the customer ordered"; "recommend to customers" alone leaves both missing. |
| **`time_window`** | Tri-state. `specified` ("next 30 days") → pass. `not_critical` → the task is static (no temporal component: static link prediction, static regression/classification) → pass, do not ask. `missing` → the task is temporal and no window was given → ASK BACK. Decide static vs. temporal with the test below. |

**Static or temporal — decide from what is being predicted, never from the data.** Almost every
table has a date column (`booked_date`, `event_at`); its presence says nothing about the task.

- **Static** (`not_critical`, do not ask):
  - the target is a value the entity row itself carries — "the final price of each listing",
    "the category of each article". The entity is one record; nothing has to happen after it.
  - the user asks for **embeddings**, or for links **learned from observed activity** without a
    forecasting framing — "learn user and product embeddings from past clicks",
    "suggest tags for each article from the article-tag links we already have".
- **Temporal** (window required): the target is something that happens, or is aggregated, over a
  period after a point in time — "will churn", "will have a fraud event", "total spend",
  "what each customer will reorder", and recommenders, which by default predict future
  interactions ("a recommender for each member").

If the task is static, restate `Window: static` at the task gate and do not ask. A time column
may still be used as the graph's time column — that is structure, not a window.

A request that names the dataset ("the e-commerce fraud dataset", a schema name) has supplied
`source_scope` if that name resolves to a location the host accepts. Do not ask the user to restate a
location they have already given in an earlier turn.

## `train` — inferred, then confirmed (never ask)

| Field | How it is obtained |
|---|---|
| **seed tables** | The physical tables behind the entity and the target, found by preflight inside `source_scope` |
| **primary keys, time columns** | Preflight; each shown as `[inferred]` unless the user stated it (`[supplied]`) or a semantic view declares it (`[declared]`). A key that preflight could only fail to disqualify is shown `[inferred, unverified]` and must be confirmed before the table can be an entity or an edge destination |
| **links** | Preflight; shown as `[inferred]`, or `[declared]` when read from a semantic view |
| **`table_mode`** | Default `all-connected` — every table reachable from the seeds. Shown as `[default]` in the graph proposal; the user may switch to `named-table` there. |

User-supplied values always win over inferred ones.
The complete proposal — every table, key, time column, link, and the table mode — is shown and
confirmed **before** any PQL is written.

## `train` — optional (never ask; preserve if given)

| Field | Default |
|---|---|
| **`output_location`** | none — training yields a notebook; when given, the same notebook also predicts into it (SKILL.md Step 2) |
| **artifact format** | one notebook (user may ask for a script) |
| **embeddings** | not included; add only when the user explicitly asks (see below) |
| **binary classification threshold** | `0.5`, shown `[default]` at Node 1d for every binary classification, whether predictions are written or stay local. Without one, `predict` leaves `TARGET_PRED` null; it has no default of its own (`kumoai-reference/references/kumo_finetuning_sdk.md` §9) |
| **metrics** | by task type, see `SKILL.md` |

If the user supplies an `output_location`, carry it verbatim into the task interpretation gate
and the generated artifact — the notebook's predict call writes there. If none is supplied,
predictions stay local and the notebook shows how to add a destination. The destination must be
one the host can write (`hosts/<host>/adapter.md`, *Destinations*). **A table is written with
exactly the name passed** — no suffix, no case change (the host's `runtime.md`,
`kumoai-reference/references/kumo_finetuning_sdk.md` §9.1). A file path follows §9.2. Older SDK builds appended
`_predictions`; the current one does not.

---

## `batch_inference`

Required, resolved and invalid fields for scoring with a saved model are in
`batch_inference.md`. Read it only when the operation is `batch_inference`.

---

## Embeddings

Default to predictions alone. Add embeddings **only when the user explicitly asks for them** —
never ask back for this, and never infer it from the task being recommendation-shaped.

Recognise: "we need shopper and item embeddings", "learn customer and product embeddings from
completed orders", "export the entity and target-item embeddings from that model".

When embeddings are requested, say so explicitly in every reply that restates the request — the
graph proposal and the task interpretation gate — the user must see that embeddings are part of
the output. Carry it into codegen as
`PredictionOutputConfig(artifact_type=...)`; see `kumoai-reference/references/kumo_finetuning_sdk.md` §9.

---

## Canonical good example (`train`)

> "Create a notebook that trains a Kumo model for our fraud use case. Data is in
> `USER_DATA.E_COMMERCE`. Entity: each customer. Outcome: at least one fraud event. Window: next 30 days."

What makes it good:
- `source_scope`: `USER_DATA.E_COMMERCE` ✓
- `prediction_entity`: each customer ✓
- `prediction_target` with positive-label condition: at least one fraud event ✓
- `time_window`: `specified`, 30 days ✓
- No columns, keys or joins described — **correct**; preflight infers them and the user confirms.

## Canonical bad example (`train`)

> "Use my sales data to figure out which customers are risky."

What is missing:
- `source_scope`: "sales data" does not resolve to a location ✗
- `prediction_target`: "risky" is not an observable outcome ✗
- `time_window`: `missing` for what reads as a temporal question ✗

Not missing, and not to be asked for: schema, join keys, output table.

---

## What to do when a required field is missing

Ask for **exactly** the missing required fields, once, in one turn. Do not ask for structure,
table mode, or an output table on a training request. On a `batch_inference` request, do not ask
for the source schema, population, or anchor date (see the *resolved, defaulted, or inferred*
table). Do not invent values. Ask in the user's
vocabulary ("what should be recommended to each customer?"), and do not attach questions
about columns or values for the fields that *are* present — those are confirmed later, as
`[inferred]`. Example:

> "To proceed I need:
> 1. **Where the data lives** — for example a database and schema, or a folder of files.
> 2. **The prediction window** — e.g. 'in the next 30 days'."

Ask only for missing, ambiguous, or contradictory inputs, restate the resolved contract, and never
request secrets in chat.
