# cuOpt Numerical Optimization — Python API

## Quick Reference

```python
from cuopt.linear_programming.problem import Problem, CONTINUOUS, INTEGER, MINIMIZE, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings
```

## LP Example

```python
problem = Problem("MyLP")

x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

problem.addConstraint(2*x + 3*y <= 120, name="resource_a")
problem.addConstraint(4*x + 2*y <= 100, name="resource_b")
problem.setObjective(40*x + 30*y, sense=MAXIMIZE)

settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"Objective: {problem.ObjValue}")
    print(f"x = {x.getValue()}, y = {y.getValue()}")
```

## MILP Example

```python
problem = Problem("FacilityLocation")

open_facility = problem.addVariable(lb=0, ub=1, vtype=INTEGER, name="open")
production = problem.addVariable(lb=0, vtype=CONTINUOUS, name="production")

problem.addConstraint(production <= 1000 * open_facility, name="link")
problem.setObjective(500*open_facility + 2*production, sense=MINIMIZE)

settings = SolverSettings()
settings.set_parameter("time_limit", 120)
settings.set_parameter("mip_relative_gap", 0.01)
problem.solve(settings)

if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(f"Open: {open_facility.getValue() > 0.5}, Production: {production.getValue()}")
```

## QP Example (beta — MINIMIZE only)

```python
problem = Problem("Portfolio")
x1 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_a")
x2 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_b")
x3 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_c")

problem.setObjective(
    0.04*x1*x1 + 0.02*x2*x2 + 0.01*x3*x3
    + 0.02*x1*x2 + 0.01*x1*x3 + 0.016*x2*x3,
    sense=MINIMIZE,
)
problem.addConstraint(x1 + x2 + x3 == 1, name="budget")
problem.addConstraint(0.12*x1 + 0.08*x2 + 0.05*x3 >= 0.08, name="min_return")

problem.solve(SolverSettings())
if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"Variance: {problem.ObjValue}")
```

See [qp_examples.md](qp_examples.md) for least-squares, maximization workaround, and covariance matrix expansion.

## CRITICAL: Status Values Use PascalCase

```python
# ✅ CORRECT
if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(problem.ObjValue)

# ❌ WRONG — silently never matches
if problem.Status.name == "OPTIMAL":
    ...
```

**LP:** `Optimal`, `NoTermination`, `NumericalError`, `PrimalInfeasible`, `DualInfeasible`, `IterationLimit`, `TimeLimit`, `PrimalFeasible`

**MILP:** `Optimal`, `FeasibleFound`, `Infeasible`, `Unbounded`, `TimeLimit`, `NoTermination`

**QP:** same set as LP.

## Solver Settings

```python
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
settings.set_parameter("mip_relative_gap", 0.01)  # MILP: stop within 1% of optimal
settings.set_parameter("log_to_console", 1)
```

## Dual Values (LP / QP)

```python
if problem.Status.name == "Optimal":
    constraint = problem.getConstraint("resource_a")
    print(f"Dual value: {constraint.DualValue}")  # NaN if model has quadratic constraints
```

## Common Modeling Patterns

### Binary Selection
```python
items = [problem.addVariable(lb=0, ub=1, vtype=INTEGER) for _ in range(n)]
problem.addConstraint(sum(items) == k)
```

### Big-M Linking
```python
M = 10000
problem.addConstraint(x <= 100 + M*(1 - y))
```

### If-then "must also produce"
```python
problem.addConstraint(y_X <= y_Y)
problem.addConstraint(production_Y >= 1 * y_Y)
```

### Large Expressions (avoid recursion limit)
```python
from cuopt.linear_programming.problem import LinearExpression

expr = LinearExpression([x, y, z], [1.0, 2.0, 3.0], constant=0.0)
problem.addConstraint(expr <= 100)
```

## Reference Models

| Model | Type | Location |
|-------|------|----------|
| Minimal LP | LP | [assets/python/lp_basic/](../assets/python/lp_basic/) |
| Dual values | LP | [assets/python/lp_duals/](../assets/python/lp_duals/) |
| PDLP warmstart | LP | [assets/python/lp_warmstart/](../assets/python/lp_warmstart/) |
| Integer variables | MILP | [assets/python/milp_basic/](../assets/python/milp_basic/) |
| Production planning | MILP | [assets/python/milp_production_planning/](../assets/python/milp_production_planning/) |
| Portfolio variance | QP | [assets/python/portfolio/](../assets/python/portfolio/) |
| Least squares | QP | [assets/python/least_squares/](../assets/python/least_squares/) |
| Maximization workaround | QP | [assets/python/maximization_workaround/](../assets/python/maximization_workaround/) |
| MPS file solver | LP/MILP | [assets/python/mps_solver/](../assets/python/mps_solver/) |
