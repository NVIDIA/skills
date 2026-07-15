# cuOpt Numerical Optimization — C API

## Required Headers

```c
#include <cuopt/mathematical_optimization/cuopt_c.h>   // Core API
#include <cuopt/mathematical_optimization/constants.h> // Parameter name macros
```

## API Call Sequence

```
cuOptCreateRangedProblem(...)   // build CSR constraint matrix + variable types
cuOptCreateSolverSettings(...)
cuOptSetFloatParameter(...)     // time_limit, tolerances
cuOptSetIntegerParameter(...)   // log_to_console, method
cuOptSolve(problem, settings, &solution)
cuOptGetObjectiveValue(solution, &obj)
cuOptGetPrimalSolution(solution, values)
cuOptGetDualSolution(...)       // LP/QP only
// cleanup
cuOptDestroyProblem(...)
cuOptDestroySolverSettings(...)
cuOptDestroySolution(...)
```

## Parameter Setting Functions

| Function | Use for |
|----------|---------|
| `cuOptSetFloatParameter` | `time_limit`, tolerances |
| `cuOptSetIntegerParameter` | `log_to_console`, `method`, `presolve` |

**Common mistake:** `cuOptSetIntParameter` does not exist — use `cuOptSetIntegerParameter`.

## LP Example

```c
#include <cuopt/mathematical_optimization/cuopt_c.h>
#include <cuopt/mathematical_optimization/constants.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cuOptOptimizationProblem problem = NULL;
    cuOptSolverSettings settings = NULL;
    cuOptSolution solution = NULL;

    cuopt_int_t num_variables = 2, num_constraints = 2;

    // Constraint matrix in CSR format
    cuopt_int_t row_offsets[] = {0, 2, 4};
    cuopt_int_t col_indices[] = {0, 1, 0, 1};
    cuopt_float_t values[] = {3.0, 4.0, 2.7, 10.1};

    cuopt_float_t obj_coeffs[] = {-0.2, 0.1};
    cuopt_float_t con_lb[] = {-CUOPT_INFINITY, -CUOPT_INFINITY};
    cuopt_float_t con_ub[] = {5.4, 4.9};
    cuopt_float_t var_lb[] = {0.0, 0.0};
    cuopt_float_t var_ub[] = {CUOPT_INFINITY, CUOPT_INFINITY};
    char var_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

    cuopt_int_t status = cuOptCreateRangedProblem(
        num_constraints, num_variables, CUOPT_MINIMIZE, 0.0,
        obj_coeffs, row_offsets, col_indices, values,
        con_lb, con_ub, var_lb, var_ub, var_types, &problem
    );
    if (status != CUOPT_SUCCESS) { return 1; }

    cuOptCreateSolverSettings(&settings);
    cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, 60.0);

    status = cuOptSolve(problem, settings, &solution);

    cuopt_float_t obj;
    cuOptGetObjectiveValue(solution, &obj);
    printf("Objective: %f\n", obj);

    cuopt_float_t* sol = malloc(num_variables * sizeof(cuopt_float_t));
    cuOptGetPrimalSolution(solution, sol);
    printf("x1=%f x2=%f\n", sol[0], sol[1]);
    free(sol);

    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroySolution(&solution);
    return 0;
}
```

For MILP, set `var_types[i] = CUOPT_INTEGER` for integer variables and use `CUOPT_MIP_RELATIVE_GAP` / `CUOPT_MIP_ABSOLUTE_TOLERANCE` settings.

## QP via C API (beta)

QP uses the same library, headers, and build pattern — only the problem-creation call differs (it accepts a quadratic objective). See `cpp/include/cuopt/mathematical_optimization/` for QP-specific creation calls and `docs/cuopt/source/cuopt-c/lp-qp-milp/` for end-to-end QP examples.

**QP rules:** MINIMIZE only (`CUOPT_MINIMIZE`); continuous variables only (`CUOPT_CONTINUOUS`); Q should be PSD.

## Dual Values (LP / QP)

`cuOptGetDualSolution` and `cuOptGetReducedCosts` return duals for LP and QP. They return `NaN` arrays when the model has quadratic constraints. Not available for MILP.

See [assets/c/lp_duals/](../assets/c/lp_duals/) for the call sequence.

## Constants Reference

```c
CUOPT_MINIMIZE / CUOPT_MAXIMIZE
CUOPT_CONTINUOUS / CUOPT_INTEGER
CUOPT_INFINITY / -CUOPT_INFINITY
CUOPT_SUCCESS  // 0

// Float parameters
CUOPT_TIME_LIMIT
CUOPT_ABSOLUTE_PRIMAL_TOLERANCE
CUOPT_MIP_RELATIVE_GAP
CUOPT_MIP_ABSOLUTE_TOLERANCE
CUOPT_MIP_RELATIVE_TOLERANCE

// Integer parameters
CUOPT_LOG_TO_CONSOLE
CUOPT_METHOD        // CUOPT_METHOD_CONCURRENT(0), PDLP(1), DUAL_SIMPLEX(2), BARRIER(3)
CUOPT_PRESOLVE
```

Full list: `cpp/include/cuopt/mathematical_optimization/constants.h`

## Build

See [assets/c/README.md](../assets/c/README.md) for the conda-env include/library/`LD_LIBRARY_PATH` setup and `gcc` build command.

## Reference Models

| Model | Type | Location |
|-------|------|----------|
| Simple LP | LP | [assets/c/lp_basic/](../assets/c/lp_basic/) |
| Dual values | LP | [assets/c/lp_duals/](../assets/c/lp_duals/) |
| PDLP warmstart | LP | [assets/c/lp_warmstart/](../assets/c/lp_warmstart/) |
| Integer variable | MILP | [assets/c/milp_basic/](../assets/c/milp_basic/) |
| Production planning | MILP | [assets/c/milp_production_planning/](../assets/c/milp_production_planning/) |
| MPS file solver | LP/MILP | [assets/c/mps_solver/](../assets/c/mps_solver/) |
