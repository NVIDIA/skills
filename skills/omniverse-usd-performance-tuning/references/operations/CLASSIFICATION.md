---
agent_context: usd-performance-workflow
agent_routes:
  - omniverse-usd-performance-tuning
agent_next:
  - README.md
  - EXECUTION.md
freshness: 2026-05-20
version: "0.1.0"
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Operation Classification Rubric

Every operation's nested `curation` block in `references/operations/operations.json` has a `status` field. **`status` is DERIVED, not hand-authored**: it is computed by the upstream status-derivation rubric (run during skill development) and materialized onto each entry, the same way the index is regenerated from source. The development-time coverage audit rewrites any drifted statuses.

**The upstream status-derivation rubric is the single source of truth for the algorithm** — the derivation precedence (deprecated/specialty overrides → destructive→specialty → read-only→analysis → refs-present→canonical → else documentary), the reference-evidence rule (`wired_into` non-empty OR `pipelines` non-empty), and the clause definitions (C1/C2, S1/S2, A1–A3, D1–D3, X1/X2). This file does NOT restate them; it explains what each tier *means* for agent behavior and the hand-authored override contract.

This rubric is local routing policy only. Usd Optimize operation mechanics
belong to upstream `usd-optimize`; use
[`usd-optimize` upstream handoff](../upstreams/usd-optimize.md) for the central
package and operation-guide resolution rule.

## What each tier means for agent selection

The code computes the label; this is what the label means when the agent picks an op:

- **`canonical`** — reach for it by default; part of the standard 7-phase optimization flow.
- **`specialty`** — reach for it only on an explicit need or named workflow (e.g. proxy creation, restructure orchestration); not a default choice.
- **`analysis`** — a read-only finding/report producer; surface it to inspect, never to mutate the stage.
- **`documentary`** — recommend only when the user explicitly names the op or describes a use case it uniquely fits.
- **`deprecated`** — warn before recommending, and name the replacement.

## Overrides and `rationale`

`deprecated` and `specialty` are the only values `curation.status_override` may take — they are the two statuses the per-op data cannot express, so they are authored, not derived. Each override entry carries an authored `rationale` (forbidden on every other entry):

- `deprecated` — `"deprecated: <replacement>: <one-sentence justification naming the recommended replacement>"`.
- `specialty` (S2) — `"specialty: <one-sentence justification naming the narrow workflow / explicit-need constraint>"`.

Authored shape on an override entry:

```json
"curation": {
  "status": "specialty",
  "status_override": "specialty",
  "rationale": "specialty: legacy standalone welder superseded by meshCleanup; reach for it only when the user explicitly needs the upstream-documented behavior.",
  "wired_into": ["skills/.../workflow.md"]
}
```

Non-override entries carry just the generated `status` and the authored `wired_into` evidence — no `rationale`.

The schema at `scripts/operations.schema.json` describes the `curation` block (the `status` enum, the `status_override` enum, and the `rationale`-only-on-override rule). The coverage audit enforces it: it derives `status` (failing on any mismatch with the materialized value), requires a `rationale` starting with `<status>:` whenever `status_override` is set and forbids `rationale` otherwise, verifies `canonical`-status ops have a non-empty `wired_into`, and verifies each `wired_into` target file actually references the op.
