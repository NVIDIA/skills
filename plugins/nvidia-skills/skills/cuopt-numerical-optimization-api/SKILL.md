---
name: cuopt-numerical-optimization-api
version: "26.08.00"
description: LP, MILP, and QP (beta) with cuOpt — Python, C, and CLI. Use when the user is solving LP, MILP, or QP with any cuOpt interface.
license: Apache-2.0
metadata:
  author: NVIDIA cuOpt Team
  tags:
    - cuopt
    - linear-programming
    - milp
    - qp
    - python
    - c-api
    - cli
---


# cuOpt Numerical Optimization API

Model and solve LP, MILP, and QP problems using NVIDIA cuOpt's GPU-accelerated solver.

## Interface Selection

Choose the reference for the user's interface:

| Interface | When to use | Reference |
|-----------|-------------|-----------|
| **Python** | User is writing Python code | [references/python_api.md](references/python_api.md) |
| **C / C++** | User is embedding in a C/C++ application | [references/c_api.md](references/c_api.md) |
| **CLI** | User is solving from MPS files on the command line | [references/cli_api.md](references/cli_api.md) |

If the interface is not yet clear, ask before writing any code.

**Already using a modeling language?** cuOpt also works as a solver backend for third-party
modeling tools — **AMPL, GAMS / GAMSPy, PuLP, JuMP, Pyomo, and CVXPY** — with near-zero code
changes (point the model's solver at cuOpt). CVXPY additionally covers convex QP and, in beta,
QCQP / SOCP. Prefer this when the user already has a model in one of these tools rather than porting
it to the cuOpt API. See
[Third-Party Modeling Languages](https://docs.nvidia.com/cuopt/user-guide/latest/thirdparty_modeling_languages/index.html).

## Choosing LP vs MILP vs QP

**Decide from the objective and variables:**

| If the objective is... | And variables are... | Use |
|---|---|---|
| Linear (sum of `c_i * x_i`) | All continuous | **LP** |
| Linear | Some integer or binary | **MILP** |
| Has squared (`x*x`) or cross (`x*y`) terms | Continuous (integer QP not supported) | **QP** (beta) |

**Prefer LP when the problem allows it.** LP solves faster and has stronger optimality guarantees. Use MILP only when the problem logically requires whole numbers or yes/no decisions. Use QP only when the objective is genuinely quadratic (variance, squared error, kinetic energy).

- **Use LP** when every quantity can meaningfully be fractional: flows, proportions, rates, dollars, hours, tonnes of material, etc.
- **Use MILP** when the problem mentions **counts** of discrete entities, **yes/no** choices, or **either/or** decisions (e.g. open a facility or not, assign a person to a shift, number of trucks).
- **Use QP** when the objective minimizes variance, squared error, or any expression with `x*x` or `x*y` terms (portfolio optimization, least squares, regularized regression).

## Integer vs Continuous from Wording

| Problem wording / concept | Variable type | Examples |
|---------------------------|---------------|----------|
| **Discrete entities (counts)** | **INTEGER** | Workers, cars, trucks, machines, pilots, facilities, units to manufacture |
| **Yes/no or on/off** | **INTEGER** (binary, lb=0 ub=1) | Open a facility, run a machine, assign a person to a shift |
| **Amounts that can be fractional** | **CONTINUOUS** | Tonnes, litres, dollars, hours, kWh, proportion of capacity |
| **Rates or fractions** | **CONTINUOUS** | Utilization, percentage, share of budget |

**Rule of thumb:** "How many *things*" → INTEGER. "How much" → CONTINUOUS.

## QP Rules (all interfaces)

- **MINIMIZE only** — the solver rejects MAXIMIZE for quadratic objectives. To maximize `f(x)`, minimize `-f(x)` and negate the reported objective value.
- **Continuous variables only** — integer QP is not supported.
- **Q should be positive semi-definite** for a convex, well-posed problem.
- **Beta** — API may evolve; treat as production-capable for typical convex QP.

## Dual Values

Duals and reduced costs are available for **LP and QP only**:
- **MILP** — no duals (integer optima are not continuous).
- **Quadratic constraints** — duals unavailable even for LP/QP; all values return `NaN`.
- **PDLP warmstart** — LP only; MILP solves do not accept a PDLP warmstart.

## Common Issues (all interfaces)

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Infeasible | Conflicting constraints | Check constraint logic and bounds |
| Unbounded | Missing bounds | Add variable bounds |
| Slow solve | Large problem | Set time limit; increase gap tolerance |
| QP rejected with MAXIMIZE | QP only supports MINIMIZE | Negate the objective; negate the result |
| QP returns non-optimal | Q not PSD or badly scaled | Check Q is PSD; rescale variables |

## Solver Settings (concepts)

| Setting | Purpose |
|---------|---------|
| `time_limit` | Stop after N seconds |
| `mip_relative_gap` | Stop MILP when within X% of optimal |
| `mip_absolute_tolerance` | Absolute MIP gap stop |
| `log_to_console` | Enable solver logging |

Syntax varies by interface — see the interface reference file.
