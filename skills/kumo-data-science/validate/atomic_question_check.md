# Atomic Question Check

**Goal:** Confirm that the prediction target is a single, well-defined question before investing in graph construction and model training.

A composite or ambiguous target produces a model that optimizes for nothing clearly, or forces the user to re-do work after training.

---

## The rule

A request **passes** the atomic question check when:
1. There is exactly **one prediction target** (one column, one label, one outcome).
2. The target is **operationally defined** — the user's words name a concrete outcome ("at least one fraud event", "a cancelled booking"). Which column or value records it is inferred by preflight and confirmed at the gate, not asked (`accept/intake_contract.md`, *named intent, inferred mapping*).
3. The time horizon is **specified** when the prediction is time-dependent. `time_window` is
   tri-state: `specified` → pass; `not_critical` (a static task — static link prediction,
   static regression/classification, no "in the next N days" semantics) → pass without asking;
   `missing` (temporal question, no window) → ASK BACK for the window only.
   Static vs. temporal follows `accept/intake_contract.md` (*static or temporal*): a date
   column in the data never makes a task temporal.

A request **fails** when any of the following are true:
- Multiple targets are bundled ("predict X and Y")
- The target is vague or subjective ("risky customers", "high-value users") without a concrete operationalization
- The time horizon is ambiguous for a temporal prediction ("will churn soon")
- The target requires data that is not in the source scope (checked by preflight, not by asking)

---

## Pass examples

| Request | Why it passes |
|---|---|
| "Predict which customers will churn — no purchase in the next 60 days — using `churned` boolean." | Single target, time horizon defined, operationally concrete. |
| "Predict the `expected_order_amount` for each customer in the next 30 days." | Single regression target, horizon clear. |
| "Predict whether a transaction (`transaction_id`) is fraudulent (`is_fraud` = 1)." | Single binary label, no ambiguity. |
| "Recommend the top 5 products (from `PRODUCTS.product_id`) most likely to be purchased by each customer in the next 7 days." | Single ranking target, K and horizon defined. |

---

## Fail examples

### Fail 1 — Multiple targets bundled
> "Predict which customers will churn AND which products to recommend to retain them."

**Problem:** Two distinct tasks (binary classification + recommendation). They require different graph structures and different PQL queries. Split into two separate requests.

**Suggested fix:** "Please submit these as two separate requests — one for churn prediction, one for retention recommendations."

---

### Fail 2 — Vague/undefined target
> "Identify our high-value customers."

**Problem:** "High-value" has no operational definition. Is it total spend? Predicted LTV? Number of orders? Without a concrete column or formula, no model can be trained.

**Suggested fix:** "Please define what 'high-value' means in terms of a concrete column or computable metric. For example: 'Predict which customers will have `total_spend > $500` in the next 90 days.'"

---

### Fail 3 — Ambiguous time horizon
> "Predict which customers are about to churn."

**Problem:** "About to" is undefined. 7 days? 30 days? 6 months? The time horizon determines the label construction, training cutoff, and feature window.

**Suggested fix:** "Please specify the prediction horizon. For example: 'Predict which customers will churn in the next 30 days (no purchase within 30 days from today).'"

---

### Fail 4 — Target column not in schema
> "Predict customer satisfaction score." (Preflight finds no satisfaction column or survey table in the source scope.)

**Problem:** The target cannot be computed from the tables in the source scope. Either the data lives elsewhere, or this is the wrong dataset for the task.

**Suggested fix:** "The tables in `<source_scope>` do not contain a satisfaction score or any data that could derive one. Please point me at the schema that holds it, or clarify how the target would be computed."

---

## How to handle a failure

Return a response that:
1. Names the specific failure mode (bundled targets / vague target / ambiguous horizon / missing column).
2. Quotes or paraphrases the problematic part of the request.
3. Gives a concrete example of a fixed version.
4. Does **not** proceed to preflight, the graph gate, or code generation.
