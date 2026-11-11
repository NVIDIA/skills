# Run:ai Quota Accounting — reference

The specialist knowledge the audit needs. Load this before interpreting the
numbers from `list_project_resources`.

## The four numbers per (project, node-pool) cell
- **deserved** — the project's *guaranteed* share in this pool. The scheduler will
  protect up to this much for the project when it has demand.
- **allocated** — GPUs *currently scheduled* to the project in this pool (this is
  the "used" figure on the quota axis).
- **limit** — the *maximum* the project may reach by borrowing unused capacity
  (deserved + over-quota headroom). `allocated` can sit between `deserved` and
  `limit` when the project is borrowing.
- **overQuotaWeight** — the project's relative weight when the scheduler divides
  *unused* capacity among competing borrowers. Higher weight → larger share of
  the over-quota pool. Not a quota by itself.

Derived:
- `over_quota_used = max(0, allocated − deserved)` — GPUs the project is borrowing
  above its guarantee. **This is the reclaimable amount.**
- `deficit = max(0, deserved − allocated)` — guaranteed GPUs the project is *not*
  currently getting. Combined with pending demand ⇒ starvation.

## Fairness, borrowing, reclaim
- Over-quota borrowing is **legitimate** when capacity is otherwise idle — a
  project using `allocated > deserved` is not "cheating," it is using slack.
- It becomes a **fairness problem** only when another project in the **same pool**
  is **starved** — below its deserved and has pending work it can't place. Then the
  borrower is holding GPUs the starved project is entitled to.
- **Reclaim** targets *over-quota* (unrequested-by-right) capacity only. The
  scheduler reclaims from over-quota borrowers to satisfy in-quota demand.
- **In-quota protection is a priority, not a hard guarantee.** The scheduler
  prioritizes protecting in-quota workloads from reclaim, but under contention it
  is best-effort — do not promise a running in-quota job is untouchable.
- Reclaim ordering across departments has had edge-case bugs historically; when the
  picture is close, present the numbers and flag uncertainty rather than asserting
  a precise reclaim sequence.

## Preemptibility governs borrowing (critical)
- **Non-preemptible workloads are capped at the project's `deserved` quota.** They
  cannot borrow over-quota — a non-preemptible request beyond deserved goes Pending
  with reason *"GPU quota exceeded for non-preemptible workloads."*
- **Only preemptible workloads borrow** into over-quota (unreserved) capacity — and
  the scheduler can reclaim them for in-quota demand.
- Consequence: a *stable* same-pool "starved + over-borrowing" pair is rare. A
  borrower must be preemptible (so it's reclaimable) — so the moment an in-quota
  project needs its deserved, the scheduler reclaims and the starvation resolves.
  Persistent starvation usually means **deserved quota is over-subscribed** (sum of
  deserved on the pool exceeds capacity) or non-preemptible in-quota work fills it.

## STARVED needs the pending *reason*, not just a pending count
`deficit > 0 AND pending > 0` is **not sufficient** — a project can be below deserved
with pending workloads for reasons that are *not* unfair starvation:
- **"…exceeded for non-preemptible workloads"** → self-inflicted: the project asked
  for non-preemptible capacity beyond its own deserved. NOT starvation.
- **Unschedulable spec** (needs more GPUs than any node has, bad node selector, etc.)
  → a workload problem, not a quota-fairness problem.
- **Waiting for capacity / reclaim** (pool full of others' usage, its deserved not
  yet freed) → THIS is genuine starvation.

So STARVED detection must inspect the pending workloads' reason (IDs from
`list_workloads(phase="Pending", projectName=…)`, then `phaseMessage` from
`get_workload_status(workloadId=…)`) and only count the "waiting for deserved
capacity held by others" case. Report the reason category, never just "N pending."

## Two causes of starvation — they need different fixes
A project below its deserved with genuine (waiting-for-capacity) pending is starved,
but *why* determines the fix. Diagnose the cause before recommending anything:

1. **Borrowing** — a same-pool project is OVER_BORROWING (allocated > deserved). It
   holds GPUs the starved project is entitled to. *Fix:* reclaim
   `min(over_quota_used, deficit)` from the borrower. This is the classic fairness
   violation.
2. **Over-subscription** — no one is borrowing, but the pool's `sum(deserved)`
   exceeds its GPU capacity, so not every guarantee can be honored. *Fix:* reduce
   deserved / reassign quota. **Reclaim does not apply** — there is no over-quota
   holder to reclaim from; inventing one gives wrong advice.
3. **Capacity** — no borrower, not over-subscribed: the pool is simply full of
   in-quota usage. A capacity constraint, not a fairness problem.

Why this matters: a well-behaved scheduler *prevents* case 1 (it reclaims
preemptible over-quota for in-quota demand), and non-preemptible work can't borrow at
all — so in practice **over-subscription is often the real cause**. A validated
example: node-pool-2 (32 GPU) with `hpc-fill` deserved 24 (running 16, non-preemptible)
+ `starved-svc` deserved 16 + `quota-hog` deserved 2 → 42 deserved > 32 capacity →
`starved-svc` gets only 12 of its 16, pends the rest. No borrower exists; the fix is
the over-subscription, not a reclaim.

## Fractional GPU / swap caveats (do not misread `allocated`)
- A **fractional** request (e.g. `gpu-fraction: 0.5`) or a **multi-device
  fractional** request (`gpu-fraction-num-devices: N`) still consumes
  **device-count** quota, not memory-fraction quota. A project with quota "2 GPU
  devices" is measured in devices.
- Therefore `allocated` for cells with fractional workloads may not equal raw
  physical GPU cards — it reflects the scheduler's device accounting. Report it as
  the platform reports it; do not "correct" it to card counts.
- With swap enabled, request vs limit can diverge widely (small request, large
  limit); quota is generally checked against the **request/device count**, not the
  swap limit. If a quota question hinges on swap accounting, say the accounting is
  device/request-based and point the admin to confirm the specific setting rather
  than guessing.

## Department roll-up
- `list_department_resources` gives the department's *own* deserved per pool.
- A department is over-allocated when the sum of its projects' `allocated` in a
  pool exceeds the department deserved for that pool — surfaced separately from the
  project-level fairness pair.

## Current-state only
This skill audits the **live cluster state** — what is allocated vs deserved right
now. All classification is a point-in-time read of `list_project_resources`; there
is no trend or time-window analysis here.

## Open items (validate with an SE before publishing)
- OVER_BORROWING flags any over-quota allocation (`over_quota_used > 0`); whether to
  apply a materiality threshold (to ignore sub-GPU noise) is open — confirm with the
  Kai-scheduler team how they define "over-borrowing" and "starved."
- Pending counts come from `get_workloads_summary(orgType="project", orgName=…)`
  without `nodePoolName`: the pool filter matches current placement, so unscheduled
  pending workloads drop out and the count reads 0. There is no per-pool pending
  count; attribute pending demand to the project's under-deserved pools.