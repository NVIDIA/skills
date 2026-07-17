# DOCA Flow capabilities, version compatibility, errors, observability, safety

**Where to start:** Pick the pattern from the overview below, then drill
into the H2 that owns the substance. For step-by-step workflows that
*use* these capabilities (configure, build, modify, run, test, debug)
see [TASKS.md](TASKS.md). Public-doc URLs and installed-package paths
live in
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Pattern overview

Every Flow-class question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply across every
pipe spec, not just the worked example shown.

| Flow pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Confirm device placement | Confirm the opened function actually owns the steering plane before quoting any feature | [`## Capabilities and modes`](#capabilities-and-modes) |
| 2. Bring up port + representor | Port-init, representor binding, lifecycle order | [`TASKS.md ## configure`](TASKS.md#configure) |
| 3. Express *<match X, do Y>* as a pipe | Match-criteria + action set + pipe-type pick (basic / hairpin / control / ordered) | [`## Capabilities and modes`](#capabilities-and-modes) pipe-type table + [`TASKS.md ## modify`](TASKS.md#modify) |
| 4. Validate the spec before commit | Validation is constructor-time inside `doca_flow_pipe_create` — there is no separate read-only validate API; treat constructor failure as the validate step (details in [`## Safety policy`](#safety-policy)). | [`## Safety policy`](#safety-policy) + [`TASKS.md ## test`](TASKS.md#test) |
| 5. Observe what the HW actually did | Per-entry / per-pipe counters via `doca_flow_resource_query_entry()` | [`## Observability`](#observability) + [`TASKS.md ## debug`](TASKS.md#debug) |
| 6. Interpret a `DOCA_ERROR_*` from a Flow call | Map the error to a layer (env / build / link / runtime / program), then route | [`## Error taxonomy`](#error-taxonomy) + [`TASKS.md ## debug`](TASKS.md#debug) |

Two rules cut across all six: **discover the installed surface** before
quoting anything (`pkg-config --modversion doca-flow` + a `doca_caps`
snapshot — skipping this is the #1 hallucination) and **validate before
commit** (see [`## Safety policy`](#safety-policy)).

## Capabilities and modes

DOCA Flow programs the accelerated steering hardware on a supported
NVIDIA NIC/DPU. Before writing any pipe spec, the agent should know which
configuration and feature set the device is in:

- **Device placement — check this FIRST.** On a BlueField in
  **separated-host / NIC mode** (`INTERNAL_CPU_MODEL = SEPARATED_HOST` in
  `mlxconfig`) the hardware-steering plane belongs to the DPU (Arm) cores,
  not the x86 host — so the **host** function cannot bring up a Flow port at
  all: `doca_flow_port_start` (or the first switch-port
  `doca_flow_pipe_create`) fails at the cap-query stage with the
  [`## Error taxonomy`](#error-taxonomy) signature (`Failed to get hws cap` /
  `dest action ROOT … err -121`). No pipe-spec edit fixes this. Decide
  *where Flow runs* first: the DPU Arm side (native for a separated-host
  BlueField), or — if it must run host-side — change the card's mode
  (`mlxconfig` + reboot, possibly firmware) via the
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) overlay. This
  is step 1 of [`TASKS.md ## configure`](TASKS.md#configure); being
  enumerable in `doca_caps` ≠ having a usable steering plane.
- **Steering mode.** DOCA Flow runs **only** on hardware steering (HWS) —
  there is no software-steering datapath, so HWS is not an optional mode to
  check for: the port must be probed with `dv_flow_en=2` (plus a `hws`
  `mode_args`) or the Flow port will not start. What actually varies is the
  match / action / pipe-type **feature set** (by device + firmware) —
  confirm specific support via capability discovery (below).
- **Pipe types.** Basic match-action, hairpin (RX-to-TX without the host
  CPU), control, and ordered/unordered list pipes. Hairpin and
  ordered-list pipes carry additional per-release constraints.
- **Match kinds.** L2 (dst MAC, VLAN), L3 (IPv4/IPv6 src/dst, protocol),
  L4 (TCP/UDP ports, flags), tunnel headers (VXLAN, GENEVE, GRE — varies
  by firmware), and metadata. Verify the requested kind is in the device's
  capability set first.
- **Action kinds.** Forward to representor (`DOCA_FLOW_FWD_PORT`), to a
  pipe (`DOCA_FLOW_FWD_PIPE`), to a *target* (`DOCA_FLOW_FWD_TARGET` — see
  § *Forward-to-target actions*), RSS (`DOCA_FLOW_FWD_RSS`), drop
  (`DOCA_FLOW_FWD_DROP`), modify (header rewrite, decap, encap), and
  counter. Encap/decap depends on the firmware feature set.
- **Capability discovery at runtime.** Query through the installed
  `doca_caps` tool and the Flow capability-query API rather than guessing
  from docs. When the user has not checked feature support, capability
  discovery in [`TASKS.md ## configure`](TASKS.md#configure) is the
  first move — not guessing a pipe spec.

## Pipe decomposition (one logic step per pipe)

A pipe expresses **one logic step**: its execution order is fixed
*match → actions → fwd*, and pipe configuration (match fields, actions,
forwarding) is **immutable after creation** — only entries are
added/removed/updated at runtime. Chain pipes with `DOCA_FLOW_FWD_PIPE`
when the request implies more than one step.

Decomposition rules:

1. **One pipe cannot OR over different match fields.** Every entry in a
   pipe matches the **same set of fields**. "Match IPv4 src/dst *or* IPv6
   src/dst" is two pipes (one per L3 type) behind a classifier.
2. **One pipe cannot match *after* an action.** Execution order inside a
   pipe is fixed. To act on a packet and then match on the new state (e.g.
   decap, then classify on the inner header), use **two pipes** chained
   with `DOCA_FLOW_FWD_PIPE`.
3. **Group by transform shape, not entry count.** Many entries doing the
   same transform in one pipe is cheaper than several small pipes; mixing
   transform shapes in one pipe is impossible (rule 1).

Worked examples:

| Request | Decomposition |
| --- | --- |
| Forward to port 1/2 by IPv4 src+dst | **1 pipe** — match the IPv4 5-tuple, `DOCA_FLOW_FWD_PORT` to either port |
| Forward by IP (v4 **or** v6) src+dst | **3 pipes** — a root control/classifier pipe branching on `parser_meta.outer_l3_type`, plus one IPv4 and one IPv6 pipe |
| If tunnel: decap, then check inner IP src, send to port 1 | **2 pipes** — a tunnel classifier (matches `tun.type` / `parser_meta.outer_l3_type`, branches {tunnel, no-tunnel} × {inner-v4, inner-v6}) chaining via `DOCA_FLOW_FWD_PIPE` into the IPv4 / IPv6 src-match + `DOCA_FLOW_FWD_PORT` pipe |
| Gateway: forward most flows, drop a few | **2 pipes** — a high-priority drop pipe (explicit drop entries) chained via `fwd_miss` into a default forward pipe |
| Large encap: many flows, few distinct encap headers | **2 pipes** — Pipe 1 (large) matches per-flow criteria and only sets `meta.pkt_meta` to an encap tag (no encap action); Pipe 2 (small, one entry per distinct header) matches `meta.pkt_meta` and does the encap + FWD. Keeps the big pipe's per-entry action set tiny (cheaper HW, faster insertion) |

Root-pipe rule: exactly **one root pipe** per port
(`doca_flow_pipe_cfg_set_is_root(cfg, true)`); LPM, ACL, hash, and
ordered-list pipes **cannot be root** — front them with a basic or control
pipe. For fine-grained priority, use a control pipe
(`doca_flow_pipe_control_add_entry(..., priority)`, 0 highest, 7 lowest);
otherwise priority is implicit (chain via `fwd_miss`, or later-inserted
overlapping entries win). `fwd_miss` only supports `DOCA_FLOW_FWD_PIPE` and
`DOCA_FLOW_FWD_DROP`.

## Choosing between valid designs that trade off

Some well-formed requests have more than one correct pipe-graph, with a
genuine tradeoff and no single best answer — e.g. **one pipe with N
unique encap handles** (lowest latency, but one HW encap instance per
entry) vs **two pipes / one pipe with K shared encap handles** (lowest
HW-resource use, cost ∝ K distinct configs). In autonomous /
non-interactive use, **default to the lower-HW-resource option** (shared
resources — see
[`## Shareable resources`](#shareable-resources-the-scaling-lever)) and
record what was traded away in a top-of-file comment naming the choice
made (memory-optimal K shared handles) and the alternative (latency-
optimal N unique handles, drops a pipe hop, costs HW resources). When
the user is interactive, surface the choice rather than deciding for
them.

## Shareable resources (the scaling lever)

DOCA Flow can allocate a small pool of **shared resource instances** at
port-init time; many pipe entries then reference an instance by id instead
of each carrying its own copy. HW cost scales with the number of
**distinct configurations K**, not the number of **entries N**. Reach for
this first whenever the shape is "N flows, K distinct configs, K ≪ N".

Kinds that can be shared (members of `enum doca_flow_shared_resource_type`
in `doca_flow.h` — verify the set on the install, new kinds appear across
releases):

| Kind | Typical use | Share when |
| --- | --- | --- |
| `DOCA_FLOW_SHARED_RESOURCE_COUNTER` | per-class byte/packet counter | one rolled-up number for many entries (telemetry choice) |
| `DOCA_FLOW_SHARED_RESOURCE_METER` | rate-limit a class | many entries share one rate policy (telemetry/policy choice) |
| `DOCA_FLOW_SHARED_RESOURCE_ENCAP` | tunnel encap (VXLAN / GENEVE / GRE / …) | small set of distinct tunnels, many flows per tunnel |
| `DOCA_FLOW_SHARED_RESOURCE_DECAP` | tunnel decap | small set of distinct decap configs |
| `DOCA_FLOW_SHARED_RESOURCE_RSS` | RSS queue distribution | small set of distinct RSS configs |
| `DOCA_FLOW_SHARED_RESOURCE_IPSEC_SA` | IPsec SA | always — an SA is a registered object many flows reference by id |
| `DOCA_FLOW_SHARED_RESOURCE_PSP` | PSP key / context | always — same shape as IPsec SA |

**SHARED vs NON_SHARED, per action you place on an entry.** Same
config repeats across many entries with distinct configs ≪ entry count
→ **SHARED** (HW scales with K). Every entry needs a unique config →
**NON_SHARED** (HW scales with N); don't allocate one shared slot per
entry. For counters/meters the choice is **telemetry granularity**
(shared = rolled-up, non-shared = per-entry), not performance.

**The 4-step pattern (identical for every kind):** reserve K slots
before `doca_flow_init` (count then type), populate each slot after
init, bind the slot range to the port once at startup, then reference
from each entry by setting the kind-specific `*_type` to
`DOCA_FLOW_RESOURCE_TYPE_SHARED` and the matching `shared_*_id` field.
The verbatim API call sequence — for every kind in the table above —
lives in [`TASKS.md ## shared-resources`](TASKS.md#shared-resources).

**Broken-hybrid — a common scaling failure.** If any line sets a
`*_type = DOCA_FLOW_RESOURCE_TYPE_SHARED` (or a shared id field), the
**init** code MUST also reserve the matching pool (step 1) *and* set
each slot's cfg *and* bind. The pattern is all-or-nothing per kind:
writing only the type marker compiles fine but fails at runtime the
first time an entry references an unallocated slot.

**Sizing — many flows, few distinct tunnels (N flows, K distinct encap
headers, K ≪ N).** Preferred: **one pipe + K shared encap handles** (HW
cost ∝ K, not N). The two-pipe `meta.pkt_meta`-classifier split is an alternative
only when you need a *changeable* shared encap **and** a *changeable* FWD
in one entry (one pipe cannot do both). Attaching a NON_SHARED encap to
the big pipe is the anti-pattern — one HW encap instance per entry (O(N)).
Reference samples: `flow_vxlan_shared_encap/`, `flow_shared_meter/`,
`flow_shared_counter/`. For counter-pool specifics see
[`## Observability`](#observability); for the verbatim call sequence see
[`TASKS.md ## shared-resources`](TASKS.md#shared-resources).

## API surface and name guards

DOCA Flow ships **no** backward-compat alias header and **no** generic
add-entry function. The names in the left column are **not declared** in
`doca_flow*.h` — agents invent them by analogy. Verify against the header
(see [`SKILL.md`](SKILL.md#ground-rule-verify-every-api-name-against-the-installed-header))
and use the canonical name:

| Commonly invented (does not exist) | Canonical name |
| --- | --- |
| `doca_flow_global_init()` | `doca_flow_init(cfg)` (one-shot, takes `doca_flow_cfg *`) |
| `doca_flow_bringup()` | `doca_flow_port_start(port_cfg, &port)` (per-port) |
| `doca_flow_pipe_add_entry()` | pipe-type-specific add-entry (table below) |
| `doca_flow_pipe_rm_entry()` | `doca_flow_pipe_remove_entry()` (full word `remove`) |
| `doca_flow_pipe_update_entry()` | pipe-type-specific update (`*_basic_` / `*_lpm_` / `*_acl_update_entry`) |
| `doca_flow_destroy_all()` | per-resource teardown: `*_destroy()` each pipe → `doca_flow_port_stop` → `doca_flow_destroy` |
| `doca_flow_pipe_destroy_force()` | `doca_flow_pipe_destroy(pipe)` |
| `DOCA_FLOW_NO_WAIT` | `DOCA_FLOW_ENTRY_FLAGS_NO_WAIT` |
| `DOCA_FLOW_WAIT_FOR_BATCH` | `DOCA_FLOW_ENTRY_FLAGS_WAIT_FOR_BATCH` |
| `DOCA_FLOW_ENTRY_FLAGS_NONE` | no such flag — pass `0` for default; only `_NO_WAIT` / `_WAIT_FOR_BATCH` exist |
| `doca_pci.h` / `doca_pci_bdf` (open a NIC by PCI BDF) | no DOCA "pci" header exists — open by BDF with `open_doca_device_with_pci(bdf, NULL, &dev)` (`samples/common.h`), or match a `doca_devinfo` via `doca_devinfo_is_equal_pci_addr()` |

Add-entry is **pipe-type-specific** — there is no generic
`doca_flow_pipe_add_entry()`. An `undefined reference to
doca_flow_pipe_add_entry` at link time means exactly this:

| Pipe type | add-entry function |
| --- | --- |
| BASIC | `doca_flow_pipe_basic_add_entry()` |
| LPM | `doca_flow_pipe_lpm_add_entry()` |
| ACL | `doca_flow_pipe_acl_add_entry()` |
| CONTROL | `doca_flow_pipe_control_add_entry()` |
| HASH | `doca_flow_pipe_hash_add_entry()` |
| ORDERED-LIST | `doca_flow_pipe_ordered_list_add_entry()` |

Removal is the single `doca_flow_pipe_remove_entry()` (all pipe types);
update is pipe-type-specific (`doca_flow_pipe_basic_update_entry()`,
`doca_flow_pipe_lpm_update_entry()`, `doca_flow_pipe_acl_update_entry()`).
Creating two entries with identical match keys is not supported — use the
update call instead.

**Argument lists — copy from the header, never from memory.** The
`*_add_entry()` functions take more than three arguments, are
pipe-type-specific, and DOCA Flow can re-order or add arguments across
releases — so reconstruct nothing from memory. The exact slot-by-slot
shape (the per-type argument counts and the most-missed `action_idx`)
is in [`TASKS.md ## configure`](TASKS.md#configure).

The entry-flag constants carry the full `DOCA_FLOW_ENTRY_FLAGS_`
container
prefix (`_NO_WAIT`, `_WAIT_FOR_BATCH`, `_NONE`); a compiler
`did you mean 'DOCA_FLOW_…'` suggestion is a string-distance match, not the
real symbol — re-grep the prefix. If the compiler points at a name ending
in `_vN` (e.g. `doca_flow_pipe_basic_add_entry_v1`), that suffix is
ABI-versioning machinery for the same function; apply the fix to the
un-suffixed call.

Names that *look* invented but **are** real (verify, then use without
second-guessing): `doca_flow_port_switch_get()`,
`doca_flow_port_pipes_flush()`, `doca_flow_pipe_resize()`,
`doca_flow_entries_process()`.

## Metadata / scratch area (`doca_flow_meta`)

`doca_flow_meta` carries `pkt_meta` (preserved across NIC/FDB domain
crossings and visible in software Rx — use it for app-level tagging) plus a
programmable `u32[]` scratch array. The `u32` slots are application-owned,
but several pipe types consume specific slots internally — do not reuse a
slot a pipe type already claims:

| Pipe type / feature | Scratch slots it uses |
| --- | --- |
| Hash pipe | `u32[3]` bits [31:16] |
| LPM | `u32[0]` (LPM+EM also uses `u32[1]`) |
| ACL | `u32[0]`..`u32[3]` |
| IPsec anti-replay | `u32[0]` bits [7:0], `u32[1]` |
| Meter color | `u32[1]` bits [7:0] |
| NAT64 | `u32[0]`..`u32[2]` |

`u32[0..4]` are the always-available application slots; `u32[5..]` depend
on HW generation and mode (ConnectX-7+ switchdev VNF exposes `u32[5..8]`,
ConnectX-7+ switch exposes `u32[5..7]`, older HW often only `u32[5]`).
Treat `u32[5..]` as opt-in and validate at runtime. Always pair a
`doca_flow_meta` match with an explicit `match_mask` — meta fields are not
subject to implicit match rules.

## Match & action semantics (easy to get wrong)

**Match fields:**

- A field set to all-ones (`0xff…ff`) in the pipe match means
  **CHANGEABLE per entry**, NOT "match any". "Match any" is the field left
  at `0` with no mask (implicit), or an explicit mask of `0`.
- A non-zero, non-all-ones value with no mask is **CONSTANT for the whole
  pipe** — do not also pass a per-entry value for that field.
- `l3_type`, `l4_type_ext`, and `tun.type` **cannot be CHANGEABLE**.
- The `outer` / `inner` / `tun` `type` fields are **selectors only** —
  they do not enforce a protocol match by themselves. To gate on protocol,
  add a `parser_meta.outer_l3_type` / `outer_l4_type` match in an earlier
  classifier pipe.

**Actions:**

- Action execution order inside a pipe is fixed: crypto-decrypt → decap →
  pop → meta → outer → tun → push → encap → crypto-encrypt. You cannot
  reorder these.
- A single pipe **cannot** combine a *changeable* shared encap/decap
  action with a *changeable* FWD — split into two sequential pipes (this is
  the rule behind the P-class "per-flow encap AND per-flow forward in one
  stage" refusal in [`SKILL.md`](SKILL.md#when-to-refuse-push-back-before-writing-code)).
- To set a field to a literal `0`, set `actions.<field> = 0` **and**
  `actions_mask.<field> = 0xff…ff`; otherwise it is treated as "no action".

## Forward-to-target actions (pass-to-kernel and friends)

DOCA Flow exposes a *forward-to-target* action kind
(`DOCA_FLOW_FWD_TARGET`) that is **the only safe forward action for
demos or production filters that run inline on a port carrying live
host traffic** (e.g. a NIC/DPU PF that the host is currently using
for SSH, package mirrors, or telemetry). Picking the wrong forward
action on a live management port can disrupt the host's network
session; getting this right is therefore part of the
[`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) overlay,
not just a doca-flow detail.

**Pattern (three public API symbols):**

| Symbol | Where it comes from | What it does |
|---|---|---|
| `DOCA_FLOW_FWD_TARGET` | `doca_flow.h` (`enum doca_flow_fwd_type`) | Tells the pipe that the forward action is *send-to-target* rather than send-to-port / RSS / drop. |
| `enum doca_flow_target_type` (e.g. `DOCA_FLOW_TARGET_KERNEL`) | `doca_flow.h` | Names which built-in target the action resolves to. `DOCA_FLOW_TARGET_KERNEL` is the *pass-traffic-back-to-the-host-Linux-kernel* target — matched packets are observed by the pipe (counters, …) and **continue up the kernel networking stack on the same port** instead of being diverted away. |
| `doca_flow_get_target(target_type, &target_ptr)` | `doca_flow.h` | Resolves a `doca_flow_target_type` enum value to a `struct doca_flow_target *` that the pipe's `doca_flow_fwd` action carries in its `.target` field. |

**Canonical wiring (concept).** Resolve a kernel target via
`doca_flow_get_target(DOCA_FLOW_TARGET_KERNEL, &kernel_target)`, then
set BOTH `fwd.type` AND `fwd_miss.type` to `DOCA_FLOW_FWD_TARGET` with
`.target = kernel_target` — matched *and* unmatched traffic continue up
the kernel path on the same port. This dual-target wiring is what makes
the filter "inline" and safe on a live management port. The verbatim C
shape lives in
[`TASKS.md ## build`](TASKS.md#build) Track 1, derived from the shipped
sample at `/opt/mellanox/doca/samples/doca_flow/flow_fwd_target/`; the
agent quotes the sample, never invents the field names.

**When to pick this action (binding decision table):**

| The user is building … | On … | Pick |
|---|---|---|
| An *inline filter* that should count / observe matched traffic without diverting it from the host | A live host management port (the user can still SSH into the host while the filter is up) | **`DOCA_FLOW_FWD_TARGET` + `DOCA_FLOW_TARGET_KERNEL`** for both `fwd` AND `fwd_miss` (the demo / DPU-traffic-gate shape) |
| A VNF that *replaces* the kernel data path on a dedicated DOCA-managed port | A NIC/DPU PF or VF that the host does NOT use | `DOCA_FLOW_FWD_PORT` to the egress representor (the classic VNF shape) |
| A connection-tracked NAT / 5-tuple flow | A supported NIC/DPU with HWS + CT enabled | the CT module of `doca-flow` (see `## flow-ct` below); CT wraps the underlying forward action transparently |

**Required safety overlays when the user picks `FWD_TARGET` on a live
management port:**

- [`doca-hardware-safety ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) — even with the pass-to-kernel action, capture per-port counters BEFORE and AFTER, keep an out-of-band recovery path, and revert via `stop_doca_flow_ports()` + `doca_flow_destroy()` if any host-side counter regresses.
- [`AGENTS.md ## The universal verification contract`](../../../AGENTS.md#the-universal-verification-contract) — the green signal for an `FWD_TARGET` inline filter is **sustained counter growth under controlled traffic**, not a single-packet match. A single matched packet can be coincidence on a live port; agents declaring "done" on a one-packet read are violating the contract.
- [`TASKS.md ## test`](TASKS.md#test) — the staged-entry-on-a-single-port pattern still applies; widen to both ports only after the single-port smoke is green.

**Anti-patterns the agent must refuse:**

1. *"Point `fwd_miss.type = DOCA_FLOW_FWD_DROP` to discard unmatched
   traffic — it's simpler."* → **NO.** On a live management port,
   dropping unmatched traffic disconnects the host (SSH dies, monitoring
   breaks). The *miss* path on a live port MUST be `FWD_TARGET → KERNEL`.
2. *"Invent a target kind like `DOCA_FLOW_TARGET_HOST_RAW`."* → **NO.**
   `enum doca_flow_target_type` is fixed by the installed `doca_flow.h`;
   query the header instead of inventing values. If the kind the user
   needs is not in the enum, surface that and route to the
   [DOCA Flow Programming Guide](https://docs.nvidia.com/doca/sdk/doca-flow/index.html).

## flow-ct

DOCA Flow Connection Tracking (CT module of `doca-flow`, header
`doca_flow_ct.h`) is a **companion** that EXTENDS the stateless surface in
[`## Capabilities and modes`](#capabilities-and-modes) with
hardware-accelerated 5-tuple connection tracking, aging timers, and
NAT-aware actions (SNAT / DNAT). It is **not** a separate library — it ships
inside doca-flow.

**Layering rule (non-negotiable).** `doca_flow_ct_init(cfg)` (configured via
`struct doca_flow_ct_cfg`) is a one-time, process-global call that runs AFTER
`doca_flow_init()` but BEFORE any port starts; CT entries can be added only
once the ports and the wrapped pipes are up. CT is **global, not per-port** —
one `doca_flow_ct_init` even across several ports; there is no per-port CT
context. It owns the CT entry table, the aging config, and the CT-aware pipes
wrapping the ports' pipes, and is torn down by `doca_flow_ct_destroy(void)`
before `doca_flow_destroy`. CT **extends** the doca-flow setup, never replaces
it: treating CT as a standalone library, a per-port context, or rebuilding
doca-flow from scratch to add CT is the layering error. If doca-flow is not
initialized yet, route to [`TASKS.md ## configure`](TASKS.md#configure) first.

**Path selection — stateless vs CT vs Linux kernel conntrack.**

| User intent | Right artifact |
| --- | --- |
| Stateless steering only (match-and-forward, no per-connection state) | This skill's stateless surface in [`## Capabilities and modes`](#capabilities-and-modes) alone |
| Hardware-accelerated stateful firewall offload, hardware NAT gateway, per-connection telemetry tied to flow rules, conntrack-aware dataplane actions | the CT module of `doca-flow` (this section) on top of stateless doca-flow |
| Software / kernel-side conntrack is acceptable (low connection rate, host CPU has headroom) | Linux netfilter (`nf_conntrack`, `iptables -m state`, `nft ct`) — different code path, out of scope here. Do NOT use the doca-flow CT module as a wrapper around kernel conntrack |
| Traffic dominated by one-packet flows (CT entries would churn faster than aging can keep up) | This skill's stateless surface alone — CT entries have a non-zero per-flow cost |

**The 5-tuple CT match — the only match the agent should quote
as default.**

| Match field | What it carries |
| --- | --- |
| Source IP (v4 / v6) | The connection's source address — half of the connection identity |
| Destination IP (v4 / v6) | The connection's destination address — the other half |
| Source port | TCP / UDP source port — separates concurrent flows from the same source host |
| Destination port | TCP / UDP destination port — separates concurrent flows to the same destination host |
| Protocol | IP protocol number (TCP=6, UDP=17, …) — the same (IP, port, IP, port) tuple may legitimately exist for different protocols |
| VRF / VNI (overlay only) | Routing-domain identifier (VRF) or overlay network identifier (VNI) for VXLAN / GENEVE / … — required when the same 5-tuple may exist in multiple overlay tenants |

If the user asks for a CT match that is *less* than 5-tuple
(e.g. *"track by source IP only"*), that is almost always a
stateless steering question dressed up as CT — route back to
[`## Capabilities and modes`](#capabilities-and-modes). If they
ask for *more*, confirm via the cap-query before promising it.

**CT-aware actions — state-tracking, NAT, overlay-aware.**

| Action class | How to confirm support |
| --- | --- |
| State-tracking only (new → established → related → closed) | Base CT capability — gated by `doca_flow_ct_cap_is_dev_supported(devinfo)` |
| SNAT (rewrite source address / port; reverse is symmetric) | Same device-support query; NAT direction is requested through `doca_flow_ct_cfg_set_direction` and the CT actions, not a separate cap query |
| DNAT (rewrite destination address / port; reverse is symmetric) | Same device-support query; configured via `doca_flow_ct_cfg_set_direction` and the CT actions |
| SNAT + DNAT combined (full-cone, hairpin, double NAT) | Same device-support query; confirm the combined behavior empirically — there is no per-variant cap symbol |
| Overlay-aware CT (inner 5-tuple over VXLAN / GENEVE / …) | Same device-support query; confirm overlay handling against the shipped CT sample |

**Capability discovery — one device-support query.** CT exposes a single
cap symbol, `doca_flow_ct_cap_is_dev_supported(devinfo)` ("does this device
support CT at all"). There is NO per-axis `doca_flow_ct_cap_*` family — do
not claim separate cap symbols for flow count, aging range, NAT variants, or
overlays. Call it against the active `doca_devinfo` BEFORE proposing CT;
everything below is then sized via the `doca_flow_ct_cfg_*` setters and
verified empirically, not through more cap queries:

| Concern | How it is actually handled |
| --- | --- |
| Max concurrent CT flows | Sized through the CT cfg setters (`doca_flow_ct_cfg_set_*`); oversubscription surfaces as `DOCA_ERROR_FULL` / `_NO_MEMORY` at runtime, so validate against the workload's peak |
| Aging-timer configuration | Set via `doca_flow_ct_cfg_set_aging_query_delay` and the aging plugin ops; an unworkable value surfaces as `DOCA_ERROR_INVALID_VALUE` at init/configure |
| NAT variants (SNAT / DNAT / combined) | Requested through `doca_flow_ct_cfg_set_direction` and the CT actions; an unsupported request surfaces as `_NOT_SUPPORTED` at entry add — there is no per-variant cap query |
| Overlay encapsulations for CT | Confirmed against the shipped CT sample and runtime behavior, not a per-overlay cap symbol |

**Version pairing — CT ships inside doca-flow.** There is no separate
`doca-flow-ct` pkg-config module; `pkg-config --modversion doca-flow` MUST
equal `doca_caps --version`. A skew between the doca-flow library and the
rest of the install is the canonical *"my CT entry returns `_DRIVER` on a
device the cap-query says supports it"* cause — route disagreement to
[`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2
BEFORE any CT-layer diagnosis.

**CT-specific error overlay.** Add to the cross-library taxonomy
in [`## Error taxonomy`](#error-taxonomy):

| Error | CT-specific cause |
| --- | --- |
| `DOCA_ERROR_BAD_STATE` | Layering / lifecycle violation: `doca_flow_ct_init` called after a port was already started (it must run before port start), OR a CT entry add before the ports and wrapped pipes are up, OR calling `doca_flow_ct_destroy` out of order relative to `doca_flow_destroy` |
| `DOCA_ERROR_NOT_SUPPORTED` | NAT variant / overlay / aging range / CT feature is unsupported on this device + firmware combo. Re-run `doca_flow_ct_cap_is_dev_supported(devinfo)` to confirm the device supports CT at all; surface which DOCA version is installed. Do not retry the same spec on the same device |
| `DOCA_ERROR_FULL` (or `_NO_MEMORY`) | CT entry table at capacity. Read the per-CT-entry counters to identify idle / stale entries; either wait for aging to evict them, evict explicitly, or — if the workload genuinely needs more concurrent flows than the device supports — re-run the cap query for the max-concurrent-flows axis and consider whether the workload fits this device at all |
| `DOCA_ERROR_INVALID_VALUE` | Malformed 5-tuple (zero protocol, mismatched IP versions on src / dst), NAT translation that conflicts with an existing entry (two entries cannot map the same translated 5-tuple to two different connections), unsupported overlay configuration, or aging timer outside the cap-advertised range |
| `DOCA_ERROR_IN_USE` | CT entry remove while the entry is still being referenced by in-flight traffic. Quiesce the affected 5-tuple (or wait for the aging timer to evict the entry naturally), then retry. Do NOT force-remove — doing so can corrupt the per-connection state on the wire |

**Safety overlay.** Inherits [`## Safety policy`](#safety-policy) plus three
CT rules:

1. **Aging-table sizing.** Cap-advertised max concurrent flows is the
   ceiling; size to the expected *peak* (not average). If the estimate
   exceeds the ceiling, surface the device-fit gap — do NOT over-commit. The
   aging timer must fit both the cap-advertised range and the granularity.
2. **Do not invent NAT translations to resolve a conflict.**
   `DOCA_ERROR_INVALID_VALUE` on a NAT entry add is almost always a policy
   bug (two NAT rules that should not coexist) — surface it; the policy layer
   fixes it.
3. **This skill does not define firewall policy.** CT *tracks* connections
   and *applies* the requested actions; it does NOT implement policy. On
   *"what rules should I write"*, refuse to invent policy and route to the
   user's networking / security expertise.

For the configure / build / modify / run / test / debug shape
specific to CT, see
[`TASKS.md ## flow-ct`](TASKS.md#flow-ct).

## Version compatibility

For the DOCA version-detection chain, the four-way match rule, NGC
container semantics, and the headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The Flow-specific overlay:

- The `doca_flow_*` symbols available on an install are observable from the
  installed Flow headers (path in
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package)).
  On an `undefined reference` for a `doca_flow_*` symbol, confirm the
  version per [`doca-version`](../../doca-version/TASKS.md#configure),
  verify the symbol exists in the installed headers, then read the Flow
  guide for *that* release.
- `pkg-config --modversion doca-flow` is the build-time anchor for the
  version-matrix lookup in
  [`doca-version`](../../doca-version/TASKS.md#test) step 2.

Read the headers and the matching release notes for symbol
availability; a static symbol table drifts silently and should not be
trusted.

## Error taxonomy

Flow API calls return either `DOCA_SUCCESS` or a `DOCA_ERROR_*` code.
The agent should treat these as a layered taxonomy when deciding the
next move:

| Class | Examples | Typical cause | Right next move |
| --- | --- | --- | --- |
| Configuration error | `DOCA_ERROR_INVALID_VALUE` on pipe creation | Spec contradicts itself or violates schema | Re-validate the spec against the pipe-spec rules (`pipe validation` workflow in TASKS.md). |
| Capability error | `DOCA_ERROR_NOT_SUPPORTED` on pipe creation or entry add | Match kind, action kind, or steering mode is unsupported on this device/firmware | Re-run capability discovery (TASKS.md `## configure`); compare requested capability to the device's actual capability set. Do not retry the same spec on the same device. |
| Resource error | `DOCA_ERROR_NO_MEMORY`, `DOCA_ERROR_FULL` on entry add | Pipe entry budget exhausted or actions-memory pool depleted | Inspect counters and pipe statistics (`## Observability` below); enlarge the pool or evict entries before retrying. |
| Lifecycle error | `DOCA_ERROR_BAD_STATE` on start/stop | Object operated on outside its allowed lifecycle window | Re-read the object's lifecycle in TASKS.md; ensure operations happen in the documented order (port started before pipe created, pipe created before entries added, etc.). |
| Port-config-incomplete error (compiles, dies at start — the #1 first-run failure) | `doca_flow_port_start` aborts with `failed creating port - port ID is mandatory` (`engine_port.c`), `... either doca_dev or doca_dev_rep must be provided` (`engine_port.c`), or `... port representor cannot be created in VNF mode` (`engine_port.c`); or the binary exits earlier with `rte_eth_dev_count_avail`=0 / "need N ports" (`dpdk_utils.c`) | A mandatory `doca_flow_port_cfg` setter is missing, or the DPDK ports were never probed (collecting BDFs from argv ≠ probing them — the probe is what creates the DPDK port from the `doca_dev`, `samples/doca_flow/flow_common.c`) | Set **both** `doca_flow_port_cfg_set_port_id()` and `doca_flow_port_cfg_set_dev()` (a real `doca_dev` in VNF, not `dev_rep`) before `doca_flow_port_start`, and call `doca_dpdk_port_probe(dev, "dv_flow_en=2")` for **every** opened device *before* `rte_eth_dev_count_avail()`. **On count == 0, or count < expected, return non-zero from `main()` immediately — do NOT enter the keep-alive loop on an un-armed bridge** (a long-running `main()` over zero hardware ports stays alive but never forwards a packet, and the watchdog cannot distinguish it from a healthy program). See [`SKILL.md ## Port bring-up`](SKILL.md#port-bring-up-the-gate-lives-in-tasksmd) + [`TASKS.md ## configure`](TASKS.md#configure) step 6 + [`TASKS.md ## run`](TASKS.md#run) step 6. |
| Placement / steering-plane-unavailable error | Port refuses to start — `doca_flow_port_start` (or the first switch-port `doca_flow_pipe_create`) returns `DOCA_ERROR_DRIVER` / a failed start, and the SDK log shows `Failed to query WQE based flow table capabilities` → `Failed to get hws cap` (devx `op_mod=0x37`, `BAD_PARAM_ERR`), or `failed to create dest action ROOT, flag 64, err -121` | The opened function has no usable hardware-steering plane — almost always running **host-side against a BlueField in `SEPARATED_HOST` / NIC mode**, where the steering plane belongs to the DPU Arm. Identical on every card on such a host, in BOTH `vnf` and `switch` modes. | **Do not touch the pipe spec or steering-mode string.** This is the device-placement signature: check `INTERNAL_CPU_MODEL` per [`## Capabilities and modes`](#capabilities-and-modes) device-placement bullet + [`TASKS.md ## configure`](TASKS.md#configure) step 1. Run DOCA Flow on the DPU Arm side, or change the card's mode via [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md). |
| Hardware/firmware error | `DOCA_ERROR_DRIVER` and similar | The kernel driver, firmware, or PCIe path is in a state Flow cannot recover from | Stop. This is not a Flow-spec problem. Capture device state via the platform's diagnostic CLIs and escalate. |

Flow does not invent error codes outside the `DOCA_ERROR_*` family;
**any error in a Flow API trace that is not a `DOCA_ERROR_*` constant is
either a wrapper layer the user added or a bug worth filing** — do not
silently translate it to a guess.

## Observability

Flow exposes distinct observable surfaces, each with its own primitive —
do not fuse them:

- **"Did this entry see traffic?" — per-entry counters.** Attach a
  counter at entry-create time and read it back with
  `doca_flow_resource_query_entry()` (and
  `doca_flow_resource_query_pipe_miss()` for the miss path). This is the
  canonical way to confirm *traffic is matching* an entry. If the counter
  is zero while the user reports traffic should match, the pipe spec is
  most likely wrong — check it before blaming the packet generator.
- **"Is the pipe itself healthy?" — pipe statistics.** Per-pipe
  statistics (entry count, hit count where exposed, errors) describe
  whether the pipe is healthy. Use these before blaming individual
  entries.
- **"What is actually programmed in my pipe right now?"** — for
  programmed-state inspection, route the agent to the
  [`doca-flow-tune`](../../tools/doca-flow-tune/SKILL.md) skill.

Workflow: when investigating "traffic is going to the wrong place", the
canonical order is *per-entry counters first → pipe statistics second*.
For verbose runtime logging during active debugging, see
[`TASKS.md ## debug`](TASKS.md#debug).

### Counters & monitoring — two-step setup

Per-entry counters need BOTH halves of the setup; either half alone
yields no telemetry:

1. **Reserve the HW counter pool at `doca_flow_init` time.**
   `doca_flow_cfg_set_nr_counters(cfg, N)` is **mandatory** for total
   NON_SHARED per-entry counters across every pipe; without it,
   per-entry `NON_SHARED` monitors silently fail or return `ENOSPC` —
   there is no pool to query against. Size `N` to the sum of
   `NON_SHARED` monitor entries across every pipe plus a small (~10%)
   headroom. Shared counters/meters take a different path:
   `doca_flow_cfg_set_nr_shared_resource(cfg, M,
   DOCA_FLOW_SHARED_RESOURCE_COUNTER)` (count then type, mind the
   argument order), `doca_flow_shared_resources_bind(...)` once at
   startup, then reference the id from each entry's monitor through the
   nested `monitor.shared_counter.shared_counter_id` field with
   `monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_SHARED`. The
   verbatim API call sequence lives in
   [`TASKS.md ## build`](TASKS.md#build) step 4 (NON_SHARED per-entry
   pool reservation at `doca_flow_init` time) and
   [`TASKS.md ## shared-resources`](TASKS.md#shared-resources) (shared
   pool bind + reference).
2. **Attach the monitor at entry-add time AND query it at runtime.**
   Setting `monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED`
   on an entry but never calling `doca_flow_resource_query_entry()` /
   `doca_flow_resource_query_pipe_miss()` pays for the HW counter and
   gets no telemetry back out of it.

Performance rules:

- Per-entry counters cost HW resources and a small per-packet cycle hit
  (counter granularity is 1 second). Only enable
  `monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED` on entries
  you will actually query. For very large entry counts (≥10K) where most
  flows share a class, prefer shared counters/meters (HWS only).
- LPM and ACL pipes support **non-shared counters only**, and their
  monitor cannot do meters.

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy). When the two layers disagree, the stricter wins; when either says STOP, the agent stops.

Programming the steering hardware is **not** a free-form operation. Wrong
specs can take traffic offline; wrong actions can drop or misroute
unintended traffic. Three policies follow:

1. **Validate before committing to hardware.** DOCA Flow has no separate
   read-only pipe-validation API; validation happens at constructor time
   inside `doca_flow_pipe_create`, whose `DOCA_ERROR_INVALID_VALUE` /
   `DOCA_ERROR_NOT_SUPPORTED` return IS the validate signal. Treat a
   successful `doca_flow_pipe_create` (optionally backed by the
   staged-entry / dry-run sample pattern) as the validation step
   **before** the entry-add call hits the hardware. The lifecycle is
   *build (create = validate) → start → add entries → read counters*.
   Skipping it is the most common cause of "my pipe takes the link down".
2. **Hairpin pipes must be staged.** A hairpin pipe (RX-to-TX without the
   host CPU) effectively rewires the steering plane, so its ordering is
   stricter: **build** with explicit ingress/egress port ids and an
   explicit match key (implicit-match hairpin specs are forbidden — they
   are silently catch-everything); **validate** against the active hairpin
   rules and reject any spec that would shadow a higher-priority pipe on
   either port; **stage** entries on a single representor and read
   counters under controlled traffic; **commit** the production entries
   only after the staged ones report the expected counters.
3. **Capability check before action change.** Any change to a pipe's
   action set must re-run the capability check from
   `## Capabilities and modes` against the *new* action set. An action
   supported when the pipe was first built can become unsupported if the
   device or firmware was reconfigured between sessions.

The agent's job is to **enforce these orderings in the workflow**, not
just describe them. If the user says "skip the dry-run, just program it",
refuse and explain the cost.
