## Pattern overview

Every raw-verbs question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply across every
raw-verbs use case, not just the worked example shown.

| Raw-verbs pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Decide drop-down vs stay-up | The agent first re-checks whether the matching higher-level DOCA library (`doca-rdma/SKILL.md` / `doca-eth/SKILL.md` / `doca-rmax/SKILL.md`) covers the case; only if it explicitly does not does the conversation continue here | [`## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes) higher-level-vs-doca-verbs table + [TASKS.md ## configure](../TASKS.md#configure) step 1 |
| 2. Place the surface against libibverbs | The agent teaches that `doca-verbs` is *not* a synonym for libibverbs even though the object names rhyme; mixing handles across the boundary is unsupported | [`## Safety policy`](../CAPABILITIES.md#safety-policy) no-mixing rule + [TASKS.md ## modify](../TASKS.md#modify) porting block |
| 3. Bring the verbs context up inside DOCA Core | The agent walks `doca_dev` + `doca_pe` + `doca_verbs_context_create` + per-object PD / QP / CQ / SRQ / AH creation in DOCA Core terms (not in libibverbs poll-CQ terms) | [`## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes) object-model section + [TASKS.md ## configure](../TASKS.md#configure) |
| 4. Capability-query the specific verb / opcode / QP feature | The user wanted raw verbs because some specific thing was missing upstairs — the agent must confirm that *specific* thing is supported here on the device, via `doca_verbs_query_device` + the matching `doca_verbs_device_attr_get_*` against the active `doca_devinfo` | [`## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes) cap-query rule + [TASKS.md ## configure](../TASKS.md#configure) |
| 5. Observe what the HW actually did | Three valid surfaces: the DOCA progress engine (preferred default, integrates with the rest of DOCA Core), manual completion-queue handling via `doca_verbs_bridge_poll_cq`, or completion-channel event delivery via `doca_verbs_comp_channel_create` + `doca_verbs_get_cq_event` — the agent picks one explicitly | [`## Observability`](../CAPABILITIES.md#observability) + [TASKS.md ## debug](../TASKS.md#debug) |
| 6. Interpret a `DOCA_ERROR_*` from a raw-verbs call | Map the error to a layer (lifecycle / cap / permission / completion-status); the IO_FAILED case has its own overlay because the answer lives on the CQE, not on the submit return | [`## Error taxonomy`](../CAPABILITIES.md#error-taxonomy) + [TASKS.md ## debug](../TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Smoke-before-scale.** Always start with one QP + one WR + one
  completion before adding any second QP, second connection, or
  second WR opcode. Raw verbs amplifies the cost of a hidden
  configuration bug; a single-shot smoke isolates the cause
  cleanly. The full eval-loop overlay is in
  [TASKS.md ## test](../TASKS.md#test).
- **Discover the version-installed surface, do not assume.** Every
  pattern above gates on `pkg-config --modversion doca-verbs` and
  on the `doca_verbs_query_device` + `doca_verbs_device_attr_get_*`
  capability queries against the active `doca_devinfo`. Quoting a
  verb / opcode / QP option / SRQ feature without a live `doca_verbs_query_device` capability query is the
  most common hallucination failure mode for raw verbs.

