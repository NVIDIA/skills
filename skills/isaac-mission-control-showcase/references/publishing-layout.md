# Publishing Layout

`isaac-mission-control-showcase` is the only discoverable skill in this package. The
nested `references/` tree is intentional and must not be flattened.

## Only one SKILL.md

Nested references use `README.md`. Promoting one to `SKILL.md` would make it a
separate catalog skill and break the single-workflow contract.

## Two layers

- **Bundled, parent-owned.** `references/`, `shared/`, `scripts/`, `assets/`,
  `config/`. Integration adapters and routing contracts written for this
  showcase. Safe to publish.
- **Runtime-resolved.** The five upstream skills, read and invoked from pinned
  checkouts outside this package. Never copied in, never published here.

A reference directory therefore contains a contract and at most a thin adapter.
It never contains an upstream `SKILL.md`, script, resource, or runbook. Vendoring
one back in would fork upstream logic and is prohibited. Nothing in this package
enforces that automatically, so treat it as a review obligation: a change that
adds an upstream file under `references/` is a defect regardless of what it
fixes.

## Single source of truth

`upstream-versions.lock.json` is the one dependency declaration. There is no
second lock, no vendoring digest, and no copy-oriented sync tool.
