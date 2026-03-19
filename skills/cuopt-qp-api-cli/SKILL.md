---
name: cuopt-qp-api-cli
version: "26.04.00"
description: QP with cuOpt — CLI (e.g. cuopt_cli with QP-capable input). Use when the user is solving QP from the command line.
---

# cuOpt QP — CLI

QP objectives must be **minimization**. For maximization, negate the objective.

This skill is **CLI only** for QP.

## QP via CLI

cuOpt CLI supports QP (quadratic objectives). Use the same `cuopt_cli` tool; input format and options may extend the LP/MILP MPS workflow to allow quadratic terms (see repo docs or `cuopt_cli --help` for QP-specific options).

## Basic usage

```bash
# Solve QP (syntax may match or extend LP/MILP CLI; check --help)
cuopt_cli problem.mps

# With time limit
cuopt_cli problem.mps --time-limit 60
```

Check `cuopt_cli --help` and the repository documentation (e.g. `docs/cuopt/source/cuopt-cli/`) for QP file format and any QP-specific flags.

**Reference:** This skill's [assets/README.md](assets/README.md) — CLI options and repo docs.

## Getting the CLI

CLI is included with the Python package (`cuopt`). Install via pip or conda; then run `cuopt_cli --help` to verify.

## Escalate

If the problem is linear, use LP/MILP CLI. For contribution or build-from-source, see the developer skill.
