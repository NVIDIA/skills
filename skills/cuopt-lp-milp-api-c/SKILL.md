---
name: cuopt-lp-milp-api-c
version: "26.04.00"
description: LP and MILP with cuOpt — C API only. Use when the user is embedding LP/MILP in C/C++.
---

# cuOpt LP/MILP — C API

Confirm problem type and formulation (variables, objective, constraints, variable types) before coding.

This skill is **C only**.

## Quick Reference: C API

```c
#include <cuopt/linear_programming/cuopt_c.h>

// CSR format for constraints
cuopt_int_t row_offsets[] = {0, 2, 4};
cuopt_int_t col_indices[] = {0, 1, 0, 1};
cuopt_float_t values[] = {2.0, 3.0, 4.0, 2.0};
char var_types[] = {CUOPT_CONTINUOUS, CUOPT_INTEGER};

cuOptCreateRangedProblem(
    num_constraints, num_variables, CUOPT_MINIMIZE,
    0.0, objective_coefficients,
    row_offsets, col_indices, values,
    constraint_lower, constraint_upper,
    var_lower, var_upper, var_types,
    &problem
);
cuOptSolve(problem, settings, &solution);
cuOptGetObjectiveValue(solution, &obj_value);
```

## Debugging (MPS / C)

**MPS parsing:** Required sections in order: NAME, ROWS, COLUMNS, RHS, (optional) BOUNDS, ENDATA. Integer markers: `'MARKER'`, `'INTORG'`, `'INTEND'`.

**OOM or slow:** Check problem size (variables, constraints); use sparse matrix; set time limit and gap tolerance.

## Examples

- [examples.md](resources/examples.md) — LP/MILP with build instructions
- [assets/README.md](assets/README.md) — Build commands for all reference code below
- [lp_basic](assets/lp_basic/) — Simple LP: create problem, solve, get solution
- [lp_duals](assets/lp_duals/) — Dual values and reduced costs
- [lp_warmstart](assets/lp_warmstart/) — PDLP warmstart (see README)
- [milp_basic](assets/milp_basic/) — Simple MILP with integer variable
- [milp_production_planning](assets/milp_production_planning/) — Production planning with resource constraints
- [mps_solver](assets/mps_solver/) — Solve from MPS file via `cuOptReadProblem`

For **CLI** (MPS files), use `cuopt_cli` and product docs.

## Escalate

If the problem is quadratic (squared or cross terms in the objective), use QP. For contribution or build-from-source, use product or repo documentation.
