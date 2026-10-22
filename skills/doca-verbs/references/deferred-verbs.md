## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows so
the agent does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification, `pkg-config` wiring — defer to
  `doca-setup/SKILL.md` and to the
  install-tree layout in
  doca-public-knowledge-map ## Layout of an installed DOCA package (`doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package`).
  The [`## install`](install.md) anchor in this file holds only the
  per-skill readiness checks; the deep install chain itself lives
  in `doca-setup`.
- **deploy.** Deploying raw-verbs-using applications at scale,
  Kubernetes operator workflows, multi-tenant RDMA isolation —
  out of scope for Phase 1 and reserved for a future platform
  skill. For single-host first-run testing, the right verb in
  this skill is [`## run`](../TASKS.md#run).
- **rollback.** Coordinated rollback across multiple hosts /
  DPUs — out of scope. For a single in-session raw-verbs
  configuration rollback, the right verb in this skill is
  destroying the verbs context (`doca_verbs_context_destroy`
  after every per-object `_destroy`) and re-running
  [`## configure`](../TASKS.md#configure) with corrected parameters.
- **kernel-level driver install / firmware burn.** Out of scope
  for this skill. Route to
  `doca-hardware-safety/SKILL.md`
  for the meta-policy and to
  `doca-setup TASKS.md ## debug` (`doca-setup/TASKS.md#debug`)
  driver layer for the env-side recovery story.
- **Climb back to the matching higher-level library for a use
  case it actually covers.** Not a verb in this skill at all —
  that conversation belongs in
  `doca-rdma TASKS.md ## configure` (`doca-rdma/TASKS.md#configure`)
  (or `doca-eth/SKILL.md` /
  `doca-rmax/SKILL.md`). This skill's job is to
  *recognize* when the climb-back is the right answer, then hand
  the conversation off.

