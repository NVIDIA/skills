---
name: qp-formulation
version: "26.06.00"
description: Quadratic Programming (QP) — problem form and constraints. Domain concepts; no API or interface. QP is beta.
---

# QP Formulation

Domain concepts for quadratic programming. No API or interface details here. **QP support in cuOpt is currently in beta.**

## What is QP

- **Objective**: Quadratic in the variables (e.g. x², x·y terms). Example: portfolio variance xᵀQx.
- **Constraints**: Linear only. cuOpt does not support quadratic constraints.

## Important domain rule: minimize only

QP objectives must be **minimization**. To maximize a quadratic expression, negate it and minimize; then negate the optimal value.

## Required questions (problem formulation)

Ask these if not already clear:

1. **Objective** — Does it have squared or cross terms (x², x·y)? If purely linear, use LP/MILP instead.
2. **Minimize or maximize?** — If maximize, user must negate objective and minimize.
3. **Convexity** — For minimization, the quadratic form (matrix Q) should be positive semi-definite for well-posed problems.
4. **Constraints** — All linear (no quadratic constraints)?

## Typical use cases

- Portfolio optimization (minimize variance subject to return and budget).
- Least squares (minimize ‖Ax − b‖²).
- Other quadratic objectives with linear constraints.
