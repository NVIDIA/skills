## Deferred topic boundaries

This skill scopes itself to the raw-verbs surface inside DOCA
Core. Adjacent topics the agent will get asked but should route
elsewhere:

- **General DOCA RDMA work (Send / Receive / Read / Write / Atomic
  task patterns).** Owned by
  [`doca-rdma`](../../doca-rdma/SKILL.md). This skill exists *only*
  for cases the higher-level library does not cover.
- **General DOCA Ethernet queue work.** Owned by
  [`doca-eth`](../../doca-eth/SKILL.md). This skill's
  `doca_verbs_eth_sq_*` / `doca_verbs_eth_rq_*` surface exists
  *only* for the cases doca-eth does not expose.
- **Timing-precise media / data-over-IP streaming.** Owned by
  [`doca-rmax`](../../doca-rmax/SKILL.md). Most Rivermax cases stay
  there.
- **General libibverbs programming** (raw ibverbs lifecycle, queue
  pair theory, memory-region semantics outside DOCA Core) —
  outside this skill. Route to the upstream RDMA / IB
  documentation; this skill assumes the user already understands
  the abstractions and is asking *how to express them inside the
  DOCA Core model*.
- **DOCA Core context and progress engine internals** — owned by
  [`doca-common`](../../doca-common/SKILL.md). This skill *uses* the
  Core context lifecycle and the PE drive loop; it does not
  redefine them.
- **Cross-library `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the raw-verbs overlay, not the taxonomy itself.
- **Cross-cutting debug ladder** (install / version / build / link
  / runtime / program / driver) — owned by
  [`doca-debug ## debug`](../../../doca-debug/TASKS.md#debug). This
  skill's `## debug` redirects there for layers 1-4 and layer 7;
  layers 5-6 carry the raw-verbs-specific overlay.
- **Cross-library `doca_caps` invocation patterns** — owned by
  [`doca-caps`](../../../tools/doca-caps/SKILL.md). This skill
  references the *raw-verbs capability query family*
  (`doca_verbs_query_device` + `doca_verbs_device_attr_get_*`),
  which is per-library; the *cross-library capability snapshot
  tool* is a separate surface routed there.
