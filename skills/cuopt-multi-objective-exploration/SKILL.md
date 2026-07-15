---
name: cuopt-multi-objective-exploration
version: "26.08.00"
description: Trace and interpret the Pareto frontier across competing objectives using repeated single-objective cuOpt solves (weighted-sum and ε-constraint).
license: Apache-2.0
origin: cuopt-skill-evolution
metadata:
  author: NVIDIA cuOpt Team
  tags:
    - multi-objective
    - pareto
    - epsilon-constraint
    - tradeoff
    - workflow
---



# Multi-Objective Exploration


cuOpt optimizes **one** objective per solve. Many real problems have several objectives that pull against each other — cost vs. service level, return vs. risk, makespan vs. overtime, distance vs. vehicle count. A single solve answers "what's optimal *for one particular weighting*," but it hides the tradeoff the user actually needs to see.

This skill turns a sequence of single-objective cuOpt solves into a **Pareto frontier** — the set of solutions where you can't improve one objective without giving up another — and gives the discipline to read it. It adds no solver features; it orchestrates the LP / MILP / QP solves already covered by the formulation and API skills.

## When this applies

Reach for this workflow when the problem has **two or more objectives with no agreed-upon weighting**, signalled by language like:

- "balance X and Y", "trade off", "as cheap as possible *without* hurting service"
- "minimize cost *and* maximize coverage", "I want options, not one answer"
- any objective the user is willing to relax in exchange for another

If there is a single clear objective (everything else is a hard constraint), this skill does not apply — formulate and solve once.

## Core idea — one solve is one point on a curve

A single optimum encodes **one implicit weighting** of the objectives. Change the weighting and the optimum moves. The frontier is the curve traced by all the non-dominated optima.

A solution **A dominates** B when A is at least as good on every objective and strictly better on one. Dominated solutions are never worth choosing. The **Pareto frontier** is exactly the non-dominated set; the user's job is to pick a point on it, and yours is to show them the whole curve plus where the tradeoff is sharpest.

Do not collapse a multi-objective problem to a single weighted number and report its optimum as "the answer" — that silently makes the tradeoff decision *for* the user. Trace the frontier and let them choose.

Objectives and constraints are interchangeable. A requirement currently treated as fixed — a coverage floor, a fairness cap, a budget — is often a latent objective: its level was assumed, not given. Promoting such a constraint to a parametric ε-constraint and sweeping it reveals a tradeoff you'd otherwise hide, so read a single-objective model's hard constraints as candidate objectives, not just limits — but only when the level was an assumption. A genuinely fixed, non-negotiable limit (a hard budget cap, a regulatory minimum) stays a constraint; don't manufacture a tradeoff that isn't there. Express any promoted quantity linearly so it can serve as an ε-constraint (see `cuopt-numerical-optimization-formulation`).

## Step 1 — define the objectives

An informative frontier needs objectives that genuinely conflict: if they don't pull against each other, it collapses to a single point with nothing to trade off. And each objective has to be formulated correctly, since a wrong form, sense, or scale distorts the tradeoff and shifts where the knee falls. Formulate each one with `cuopt-numerical-optimization-formulation` before sweeping.

## Step 2 — build a payoff table (anchor each objective)

Solve each objective **on its own** first. For *k* objectives this is *k* solves. Record, for each, the value of every objective at that optimum:

```text
              f1        f2        f3
min f1   →   f1*       f2(at f1*) f3(at f1*)
min f2   →   ...       f2*        ...
min f3   →   ...       ...        f3*
```

The diagonal (`f1*`, `f2*`, …) is each objective's best achievable value; the off-diagonals give the **range** each objective spans across the others' optima. This table does double duty:

- It sets the **sweep bounds** for the ε-constraint method (the feasible range of each constrained objective).
- It supplies the **scales** for normalization — objectives in dollars, percent, and hours can't be weighted meaningfully until divided by their ranges.

If any single-objective solve is already infeasible, stop and fix the model before sweeping — the frontier doesn't exist yet.

## Step 3 — choose a scalarization

### Weighted sum

Combine the objectives into one and sweep the weights:

```text
minimize  w1·f1(x) + w2·f2(x) + ... ,   for a grid of weight vectors w
```

Cheap and trivial with any solver. Two limitations to respect:

- **It only finds points on the convex hull of the frontier.** Concave (non-convex) regions of the frontier are unreachable no matter how you choose weights, and for MILP the reachable points can be sparse with large gaps. A frontier that looks suspiciously linear or has only a few clustered points is the symptom.
- **Weights are not priorities until the objectives are normalized.** Divide each `f_k` by its payoff-table range first; otherwise the largest-magnitude objective dominates regardless of intent.

### ε-constraint (preferred for a complete frontier)

Keep one objective; move the rest to constraints and sweep their right-hand sides:

```text
minimize  f1(x)
subject to  f2(x) ≤ ε2
            f3(x) ≤ ε3
            (original constraints)
```

Sweep each `ε_k` across the range from the payoff table. Each `(ε2, ε3, …)` combination is a single standard cuOpt solve. This recovers the **full** frontier, including the concave regions weighted-sum cannot reach, which is why it's the default when completeness matters. The cost is more solves (a grid over the constrained objectives) and bookkeeping of the ε values.

ε-constrain *linear* objectives directly. A quadratic objective (e.g. risk `xᵀΣx`) is simplest kept as the objective `f1` while you ε-constrain the linear ones. A **convex** quadratic objective *can* instead be ε-constrained directly: add it as a quadratic constraint `xᵀQx ≤ ε`, which cuOpt supports. Non-convex or equality quadratic constraints are unsupported, and the MILP path stays linear-constraint only.

Spot it in existing code: a hand-coded loop over a target or budget value (a return target, a cost cap) is already the ε-constraint method — name it as such, filter dominated points, and read the swept constraint's dual (LP/QP only).

**Read that dual as the local exchange rate.** Where the frontier is smooth, the dual on a swept ε-constraint is its slope — how much the kept objective `f1` moves per unit of the bound — at no cost beyond the solve already run; at a kink it gives only a one-sided rate. A **zero** dual means the bound is slack: the sweep has run past the frontier's edge. This reading needs LP/QP and a *linear* ε-constraint (MILP optima and problems with quadratic constraints return no duals) — where duals are unavailable, difference adjacent frontier points instead.

**Picking a method:** weighted-sum for a quick convex sketch or when you know the frontier is convex (e.g. a pure-LP/QP tradeoff); ε-constraint when the problem is MILP, when the frontier may be non-convex, or when the user needs a faithful and complete curve.

## Step 4 — sweep, collect, and filter

```text
frontier = []
for each weight vector (or ε vector) in the grid:
    set the combined objective (or ε right-hand sides)
    solve with cuOpt              # reuse the prior solution as a warm start
    if status is Optimal/Feasible:
        record (objective values, solution)
discard dominated and duplicate points
sort the survivors to form the frontier
```

Practical notes:

- **Warm-start LP sweeps.** For an LP frontier, carry the previous solve's PDLP warmstart data into the next to cut solve time. Per cuOpt this is **LP-only**: a MILP solve doesn't take a PDLP warmstart (you can optionally seed a MIP start instead). See `cuopt-numerical-optimization-api` for the calls.
- **Cap each MILP solve.** Set a per-solve time limit on MILP sweeps (see `cuopt-numerical-optimization-api`) — a sweep is many solves, and branch-and-bound can over-spend certifying optimality past a tiny gap, while cuOpt sets no limit by default and won't warn. Report the points as optimal *to the gap you set*, not certified optimal.
- **Filter dominated points.** A correct sweep can still emit dominated points (especially weighted-sum near the hull, or MILP). Drop them; they are not part of the frontier.
- **Resolution is a budget.** Curve fidelity trades against solve count. Start coarse to see the shape, then refine the grid only where the curve bends.
- **Spend the budget where the slope changes (LP/QP).** Because the ε-constraint dual is the frontier's local slope, compare it across solved points: where it barely changes, the curve is nearly straight — interpolate rather than add solves; where it jumps by more than the solve tolerance, the frontier bends between those points — refine there (smaller differences are solver noise, not curvature). This concentrates solves where the curve actually bends instead of spreading them over a uniform grid. On MILP, judge where to refine from the gaps between primal objective values instead.
- **Verify, don't assume.** When you claim one method beats another, measure it — e.g. count the efficient points ε-constraint recovered that weighted-sum missed — rather than asserting it; and flag any solve returning feasible-but-not-`Optimal` so a non-certified point is never read as exact.

## Step 5 — interpret the frontier

- **Report tradeoffs, not single numbers.** A frontier point means nothing in isolation. Quote the exchange rate — "≈ $4k of extra cost per 1% of added coverage in this region" — so the user can judge whether a move is worth it. On an LP/QP frontier this exchange rate is the swept constraint's dual at that point — the local slope of the frontier, accurate to the solve's optimality tolerance (tighten it before relying on a dual); on MILP, estimate it from the gap to the adjacent frontier point.
- **Flag knee points; don't auto-pick them.** The "knee" is where the curve bends most sharply — beyond it you pay a lot for a little. It's often the best-balanced compromise and worth highlighting, but the final choice is the user's preference, not a rule. At the knee the slope is two-sided — the dual just below differs from just above — so quote the exchange rate there as a range, not one number.
- **Treat dominated or gappy output as a diagnostic.** If dominated points survive filtering, or the frontier is implausibly sparse or perfectly linear, suspect the sweep or the model — most often weighted-sum hiding a concave region (switch to ε-constraint) or a normalization mistake.
- **State the weighting/ε you used.** Every reported point is conditional on its scalarization. Make that explicit so a single solve is never mistaken for "the" optimum. On LP/QP, the ε-constraint duals are the *implicit weights* at that point — the effective price the solution puts on each constrained objective, and the weights a weighted-sum solve would need to reproduce that tradeoff. Reporting them makes the accepted tradeoff ratio explicit.

## Interfaces

This skill is solver- and interface-agnostic. The per-solve mechanics — building the objective, adding the ε constraints, passing a warm start, reading status — live in the API skills:

- `cuopt-numerical-optimization-api` — LP, MILP, QP solves (Python, C, CLI).
- `cuopt-routing-api-python` — the same frontier workflow applies to routing tradeoffs (distance vs. vehicles vs. time).
