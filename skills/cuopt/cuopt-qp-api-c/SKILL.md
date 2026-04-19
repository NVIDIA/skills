---
name: cuopt-qp-api-c
version: "26.06.00"
description: Quadratic Programming (QP) with cuOpt — C API. Use when the user is embedding QP in C/C++.
---

# cuOpt QP — C API

Confirm the objective has squared or cross terms (QP); if purely linear, use LP/MILP. QP must be minimization.

This skill is **C only**.

QP uses the same cuOpt C library as LP/MILP; the API extends to quadratic objectives. Use the same include/lib paths and build pattern as for LP/MILP C (see this skill's assets/README.md); then use the QP-specific creation/solve calls from the cuOpt C headers.

**Reference:** This skill's [assets/README.md](assets/README.md) — build pattern and repo QP C API docs.

## Escalate

If the problem is linear, use LP/MILP. For contribution or build-from-source, see the developer skill.
