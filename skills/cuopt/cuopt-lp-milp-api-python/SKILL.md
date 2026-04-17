---
name: cuopt-lp-milp-api-python
version: "26.06.00"
description: Solve Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) with the Python API. Use when the user asks about optimization with linear constraints, integer variables, scheduling, resource allocation, facility location, or production planning.
---

# cuOpt LP/MILP Skill

Model and solve linear and mixed-integer linear programs using NVIDIA cuOpt's GPU-accelerated solver.

## Before You Start

Use a formulation summary (parameters, constraints, decisions, objective) if available; otherwise ask for decision variables, objective, and constraints. Then confirm **variable types** (see below) and **interface** (Python API recommended).

## Choosing LP vs MILP

**Prefer LP (all continuous variables) when the problem allows it.** LP solves faster and has stronger optimality guarantees. Use **MILP** only when the problem logically requires whole numbers or yes/no decisions.

**Problem types that need extra care:** Multi-period planning and goal programming are easy to misinterpret. Double-check that rates and constraints apply to the right time period or priority level (AGENTS.md: verify understanding before code).

- **Use LP** when every quantity can meaningfully be fractional: flows, proportions, rates, dollars, hours, tonnes of material, etc.
- **Use MILP** when the problem mentions **counts** of discrete entities, **yes/no** choices, or **either/or** decisions (e.g. open a facility or not, assign a person to a shift, number of trucks).

## Integer vs continuous from wording

Choose variable type from what the problem describes.

| Problem wording / concept | Variable type | Examples |
|---------------------------|---------------|----------|
| **Discrete entities (counts)** | **INTEGER** | Workers, cars, trucks, machines, pilots, facilities, units to manufacture (when "units" means whole items), trainees, vehicles |
| **Yes/no or on/off** | **INTEGER** (binary, lb=0 ub=1) | Open a facility, run a machine, produce a product line, assign a person to a shift |
| **Amounts that can be fractional** | **CONTINUOUS** | Tonnes, litres, dollars, hours, kWh, proportion of capacity, flow volume, weight |
| **Rates or fractions** | **CONTINUOUS** | Utilization, percentage, share of budget |
| **Unclear** | Prefer **INTEGER** if the noun is a countable thing (a worker, a car); prefer **CONTINUOUS** if it's a measure (amount of steel, hours worked). If the problem says "whole" or "integer" or "number of", use INTEGER. |

**Rule of thumb:** If the quantity is "how many *things*" (people, vehicles, items, sites), use **INTEGER**. If it's "how much" (mass, volume, money, time) or a rate, use **CONTINUOUS** unless the problem explicitly requires whole numbers.

## Quick Reference: Python API

### LP Example

```python
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

# Create problem
problem = Problem("MyLP")

# Decision variables
x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

# Constraints
problem.addConstraint(2*x + 3*y <= 120, name="resource_a")
problem.addConstraint(4*x + 2*y <= 100, name="resource_b")

# Objective
problem.setObjective(40*x + 30*y, sense=MAXIMIZE)

# Solve
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

# Check status (CRITICAL: use PascalCase!)
if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"Objective: {problem.ObjValue}")
    print(f"x = {x.getValue()}")
    print(f"y = {y.getValue()}")
```

### MILP Example (with integer variables)

```python
from cuopt.linear_programming.problem import Problem, CONTINUOUS, INTEGER, MINIMIZE

problem = Problem("FacilityLocation")

# Binary variable (integer with bounds 0-1)
open_facility = problem.addVariable(lb=0, ub=1, vtype=INTEGER, name="open")

# Continuous variable
production = problem.addVariable(lb=0, vtype=CONTINUOUS, name="production")

# Linking constraint: can only produce if facility is open
problem.addConstraint(production <= 1000 * open_facility, name="link")

# Objective: fixed cost + variable cost
problem.setObjective(500*open_facility + 2*production, sense=MINIMIZE)

# MILP-specific settings
settings = SolverSettings()
settings.set_parameter("time_limit", 120)
settings.set_parameter("mip_relative_gap", 0.01)  # 1% optimality gap

problem.solve(settings)

# Check status
if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(f"Open facility: {open_facility.getValue() > 0.5}")
    print(f"Production: {production.getValue()}")
```

## CRITICAL: Status Checking

**Status values use PascalCase, NOT ALL_CAPS:**

```python
# ✅ CORRECT
if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(problem.ObjValue)

# ❌ WRONG - will silently fail!
if problem.Status.name == "OPTIMAL":  # Never matches!
    print(problem.ObjValue)
```

**LP Status Values:** `Optimal`, `NoTermination`, `NumericalError`, `PrimalInfeasible`, `DualInfeasible`, `IterationLimit`, `TimeLimit`, `PrimalFeasible`

**MILP Status Values:** `Optimal`, `FeasibleFound`, `Infeasible`, `Unbounded`, `TimeLimit`, `NoTermination`

## Common Modeling Patterns

### Binary Selection
```python
# Select exactly k items from n
items = [problem.addVariable(lb=0, ub=1, vtype=INTEGER) for _ in range(n)]
problem.addConstraint(sum(items) == k)
```

### Big-M Linking
```python
# If y=1, then x <= 100; if y=0, x can be anything up to M
M = 10000
problem.addConstraint(x <= 100 + M*(1 - y))
```

### If-then "must also produce"
When the problem says *if we do X then we must also do Y*, enforce both (i) the binary link and (ii) that Y is actually produced:
```python
# y_X <= y_Y (if we do X, we must "do" Y)
problem.addConstraint(y_X <= y_Y)
# Production of Y when Y is chosen: produce at least 1 (or a minimum) when y_Y=1
problem.addConstraint(production_Y >= 1 * y_Y)  # or min_amount * y_Y
```
Otherwise the solver can set y_Y=1 but production_Y=0, satisfying the binary link but not the intent.

### Building large expressions
Chained `+` over many terms can hit recursion limits in the API. Prefer building objectives and constraints with **LinearExpression**:
```python
from cuopt.linear_programming.problem import LinearExpression

# Build as list of (vars, coeffs) instead of v1*c1 + v2*c2 + ...
vars_list = [x, y, z]
coeffs_list = [1.0, 2.0, 3.0]
expr = LinearExpression(vars_list, coeffs_list, constant=0.0)
problem.addConstraint(expr <= 100)
```
See reference models in this skill's `assets/` for examples.

### Piecewise Linear (SOS2)
```python
# Approximate nonlinear function with breakpoints
# Use lambda variables that sum to 1, at most 2 adjacent non-zero
```

## Solver Settings

```python
settings = SolverSettings()

# Time limit
settings.set_parameter("time_limit", 60)

# MILP gap tolerance (stop when within X% of optimal)
settings.set_parameter("mip_relative_gap", 0.01)

# Logging
settings.set_parameter("log_to_console", 1)
```

## Common Issues

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| Status never "OPTIMAL" | Using wrong case | Use `"Optimal"` not `"OPTIMAL"` |
| Integer var has fractional value | Defined as CONTINUOUS | Use `vtype=INTEGER` |
| Infeasible | Conflicting constraints | Check constraint logic |
| Unbounded | Missing bounds | Add variable bounds |
| Slow solve | Large problem | Set time limit, increase gap tolerance |
| Maximum recursion depth | Building big expr with chained `+` | Use `LinearExpression(vars_list, coeffs_list, constant)` |

## Getting Dual Values (LP only)

```python
if problem.Status.name == "Optimal":
    constraint = problem.getConstraint("resource_a")
    shadow_price = constraint.DualValue
    print(f"Shadow price: {shadow_price}")
```

## Reference Models

All reference models live in this skill's **`assets/`** directory. Use them as reference when building new applications; do not edit them in place.

### Minimal / canonical examples (LP & MILP)
| Model | Type | Description |
|-------|------|-------------|
| [lp_basic](assets/lp_basic/) | LP | Minimal LP: variables, constraints, objective, solve |
| [lp_duals](assets/lp_duals/) | LP | Dual values and reduced costs |
| [lp_warmstart](assets/lp_warmstart/) | LP | PDLP warmstart for similar problems |
| [milp_basic](assets/milp_basic/) | MILP | Minimal MIP; includes incumbent callback example |
| [milp_production_planning](assets/milp_production_planning/) | MILP | Production planning with resource constraints |

### Other reference
| Model | Type | Description |
|-------|------|-------------|
| [mps_solver](assets/mps_solver/) | LP/MILP | Solve any problem from standard MPS file format |

**Quick command to list models:** `ls assets/` (from this skill's directory).

## When to Escalate

Use troubleshooting and diagnostic guidance if:
- Infeasible and you can't determine why
- Numerical issues
