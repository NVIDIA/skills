# DOCA RDMA workflows

**Where to start:** The verbs run `configure → build → modify → run
→ test → debug`. Skip ahead only when the user is already past a
verb. The `## test` verb is an iterative loop (capability check →
permission check → bidi smoke → completion drain → loop back if the
permission set or task set changed), not a one-shot pass — see the
eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the underlying capability matrix, version
compatibility, error taxonomy, observability surface, and safety
policy that these workflows assume, see
[CAPABILITIES.md](CAPABILITIES.md). For where to find docs, the
installed DOCA layout, or release notes, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through the
steps in order, verifying preconditions before recommending the next
call.

## configure

Goal: bring up a DOCA RDMA context on a host or BlueField and confirm
both sides are in a state where data-movement tasks are meaningful.

Steps the agent should walk the user through:

1. **Confirm the installed DOCA version.** Use the procedure in
   [`doca-setup CAPABILITIES.md ## Version compatibility`](../../doca-setup/CAPABILITIES.md#version-compatibility).
   Quote the version observed (`pkg-config --modversion doca`,
   then `doca_caps --version`); do not assume "latest".
2. **Discover the device capability surface for RDMA.** Run
   `doca_caps --list-devs` ([`doca-caps`](../../tools/doca-caps/SKILL.md))
   to see which devices have RDMA capability, then run the
   per-`doca_devinfo` `doca_rdma_cap_*` queries against the candidate
   device. Record which task types are supported, which transport
   types are supported (IB / RoCE / DC-alpha), and what
   `doca_rdma_cap_get_max_*` returns for the queue / buf-list sizes.
   The capability matrix to compare against lives in
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes).
3. **Decide the connection method.** RDMA CM (callback-driven) vs
   bridge / OOB vs gRPC. The pick decides the next-step code shape;
   the trade-off matrix lives in
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes)
   connection-methods table. Do not pick *for* the user when their
   intent is ambiguous — ask.
4. **Configure the RDMA instance.** Mandatory before
   `doca_ctx_start()`: enable at least one task type
   (`doca_rdma_task_*_set_conf`); set permissions
   (`doca_rdma_set_permissions`); set the mmap permissions
   (`doca_mmap_set_permissions`) to match the task type per the
   matrix in [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy).
   Optional but commonly needed: `doca_rdma_set_transport_type()`,
   `doca_rdma_set_max_num_connections()`, `doca_rdma_set_*` queue
   sizes.
5. **Sanity check before any task submission.** Confirm with the
   user: which side is requesting work (Read/Write originator), which
   side is exporting memory (`doca_mmap_export_rdma()`), which
   transport, which connection method. If any of those are unclear,
   stop and ask — do not invent.

If any step fails with a `DOCA_ERROR_*`, route through the error
taxonomy in [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

Goal: produce a binary that links DOCA RDMA against the user's
installed DOCA, using the canonical cross-library build pattern.

The build pattern for any DOCA C/C++ consumer is **identical** across
libraries — `pkg-config` for include + link flags, meson or CMake as
the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).

**Step 0 — resolve the pkg-config module (do this first; do not assume).**
DOCA RDMA does **not** ship a standalone `doca-rdma.pc` on current
installs. The RDMA library is delivered inside the **umbrella `doca`**
pkg-config module. Resolve it on the target before writing any build file:

1. Make sure pkg-config can see DOCA. If `pkg-config --exists doca`
   fails, export the install's pkgconfig dir first:
   `export PKG_CONFIG_PATH=/opt/mellanox/doca/lib/$(uname -m)-linux-gnu/pkgconfig:$PKG_CONFIG_PATH`
   (the arch segment is e.g. `aarch64-linux-gnu` on BlueField, `x86_64-linux-gnu` on a host).
2. List what actually exists: `pkg-config --list-all | grep -i doca`.
   Use the **most specific RDMA module that exists** if a split install
   exposes one; otherwise use the umbrella **`doca`** (the normal case).
   The public RDMA header is `doca_rdma.h` and the shared object is
   `libdoca_rdma.so` regardless of which `.pc` resolves them.

> **Guardrail — DOCA is the deliverable, not "any RDMA".** If
> `pkg-config doca-rdma` (or any `--libs`/`--cflags` probe) fails, the
> cause is almost always a wrong module name or an unset
> `PKG_CONFIG_PATH` — **fix that**. Do **NOT** silently fall back to
> raw `libibverbs` / `librdmacm` / RDMA-CM and hand-roll a verbs
> program. That technically moves bytes but abandons DOCA entirely,
> which defeats the purpose of this work. A build that does not link
> `libdoca_rdma` is a failed DOCA-RDMA build — troubleshoot the module
> resolution above (and escalate via the error taxonomy) instead of
> bypassing DOCA.

This skill carries only the RDMA-specific overlay:

| Slot | Value for RDMA | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca` (umbrella; verify per Step 0 — there is normally no `doca-rdma.pc`) | The `.pc` that resolves the RDMA cflags/libs on current DOCA host packages |
| Required runtime libs | whatever `pkg-config --libs doca` reports — it pulls in `libdoca_common`, `libdoca_rdma`, and the device-side providers | RDMA depends on Core + the rdma transport providers shipped with DOCA |
| Header check | `doca_rdma.h` resolvable under the include directory `pkg-config --cflags doca` reports (do not hardcode the include path — the install layout can move) | If `--cflags` resolves but `doca_rdma.h` is missing, the install is partial |
| Minimum required DOCA version | Query with `pkg-config --modversion doca`; never hardcode in build files | Cross-version build/runtime mixing breaks per [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) |

For non-C consumers (Rust, Go, Python), the link surface is the same
`*.so` files (cgo `#cgo pkg-config: doca`, Rust `pkg-config` crate on
`doca`, etc.); the FFI wrapper layer is the language-specific binding
and is out of scope for this skill — but the slots above (and the
guardrail) are still the load-bearing inputs the wrapper needs.

## modify

Goal: take a shipped DOCA RDMA sample as the verified starting point
and apply a minimum-diff modification to express the user's intent.

**Non-C languages (Go / Rust / Python): this is still the right verb.**
Do not reimplement RDMA in raw libibverbs to avoid C. Start from the
shipped sample under `/opt/mellanox/doca/samples/doca_rdma/<name>/`,
then expose its entry points (e.g. the sample's send/receive or
write/read driver functions) through a **thin** FFI shim — for Go, one
small `*.c`/`*.h` pair compiled via `#cgo pkg-config: doca` calling the
sample functions, with `main.go` orchestrating. This keeps the verified
DOCA datapath intact and links `libdoca_rdma`; it is a single small
wrapper, not a full re-binding of the DOCA API.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The RDMA-specific overlay is the *modify-from-sample
schema fill* — the five slots the agent must elicit from the user
before recommending any code-level edit:

| Slot | What the agent asks the user | RDMA-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_rdma/`? | Pick the closest in *task direction* (one-sided vs two-sided) and *connection method* (CM vs bridge vs OOB) to the user's intent. Do NOT bridge across both axes — a smaller diff is always safer than a re-architecture |
| 2. Task types added or removed | Which task types from the eleven? | Each added type needs its own `doca_rdma_task_*_set_conf` call before `doca_ctx_start()`, plus its matching mmap-permission flags |
| 3. Permission changes | Which mmap / RDMA permissions change? | Refer to the permission matrix in [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy); over-broad permissions are a silent security regression |
| 4. Connection method | Change CM → bridge or vice versa? | This is a re-architecture, not a tweak. If yes, recommend the user start from the sample that already uses the target method instead of patching one over |
| 5. Transport / connection sizing | Change transport type or `max_num_connections`? | If yes, re-run the capability queries from `## configure` step 2; transport-type support is device-conditional |

The agent emits an *intent description + the five filled slots*; the
*actual* unified diff against the sample source is produced by the
modify-from-sample renderer (deferred to a future round on the
maintainer roadmap). Until the renderer ships, the agent must walk
the user through the diff line-by-line against the sample source
they read on disk, and have the user paste back the result for
validation.

## run

Goal: actually execute the built binary against the user's installed
DOCA on a host or BlueField, including a peer to connect to.

Steps the agent should walk the user through:

1. **Confirm the peer is reachable.** RDMA needs a peer; running the
   binary on one side alone produces a misleading hang. Either both
   sides are on networks that route IB or RoCE to each other, **or**
   both run as local processes on a single host over the same
   `doca_dev` (see *Single-host loopback* below) — the latter is the
   right first-run smoke test when only one BlueField / NIC is on hand.
2. **Run the side that listens first.** Server (RDMA CM) or
   accept-side (bridge / OOB) must be running and at the
   *connection-listening* state before the client / connect-side
   starts. Confirm with the user which side is which.
3. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace` for
   the first run (see [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the lifecycle and connection
   state transitions visible on first failure.
4. **Capture the completion events on the PE.** A run that produces
   no completion events but doesn't error is almost always a missed
   `doca_pe_progress()` call. Confirm the progress engine is being
   driven on both sides.

**Single-host loopback (only one BlueField / NIC available).** A peer
does *not* have to be a second host. Both endpoints can run as two
local processes bound to the same `doca_dev`; the device loops the
traffic internally. The two sides still exchange connection state out
of band, so the shipped send/receive (and read/write) samples
implement a **file-based descriptor handshake**: each side writes its
own `doca_rdma_export()` descriptor (and, for one-sided tasks, its
`doca_mmap_export_rdma()` descriptor) to a local file and reads the
peer's file, then each presses enter to advance from
*connection-listening* to *connected*. On a single box, point the two
processes at a shared filesystem (the requester's `local` path is the
responder's `remote` path and vice-versa), start both, then release
the handshake once both descriptor files exist. Two caveats that bite
agents: (a) the **write/atomic** samples expect a *second* enter on
the responder after the requester reports its task done — a single
enter looks like a hang; (b) **loopback traffic stays internal to the
device, so physical-port and `vport_*` RDMA hw_counters do not
move** — prove success from the completion event and the received
payload (e.g. the responder logging the bytes the requester sent), not
from port counters.

## test

Goal: prove the configured RDMA instance can actually move data
correctly between the two sides on the user's hardware, and that the
permission and capability set was sized right.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the capability set, the permission set, the task set, or the
sizing. The loop terminates when either (a) the user's intended
data-movement pattern flows end-to-end with the expected completions,
or (b) the agent has narrowed the failure cause to a layer outside
RDMA itself (driver / firmware / network) and escalated to the
matching skill.

Iteration shape:

1. **Capability re-check.** Re-run `doca_rdma_cap_task_*_is_supported`
   for every task the user intends to submit, against the active
   `doca_devinfo`. If false → that's the answer; the user's device
   or DOCA version does not support the task. Update the user's
   intent or update the install.
2. **Permission cross-check.** Compare the configured mmap +
   RDMA-permission pair against the matrix in
   [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy).
   Mismatches surface as `DOCA_ERROR_NOT_PERMITTED` on the first
   task submission, not at configure time.
3. **Bidirectional smoke.** Always start with a Send + Receive
   pair, even if the user's intent is one-sided (Read / Write). If
   the bidirectional control path works, the connection +
   two-sided permissions are correct, and any subsequent
   one-sided failure narrows cleanly to the one-sided permission
   set. Skipping this step is the most common reason "Write fails
   and we don't know why".
4. **Completion drain.** Confirm completion events arrive on the
   PE for every submitted task. *Submitted but no completion* is
   the most expensive class of bug to discover late; confirm it on
   the smoke pair before adding bulk submissions.
5. **Negative test.** Once the positive path works, intentionally
   submit one task type the device should NOT support (per step 1)
   and confirm the failure is the expected
   `DOCA_ERROR_NOT_SUPPORTED`. This validates the agent's
   capability-discovery is itself correct.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on a task we expected to work | A `doca_rdma_cap_*` returned `true` for a task type but the runtime rejects it | The agent quoted the *library* capability; the *device* capability per `doca_devinfo` is the real gate. Re-narrow to the device-level query. |
| `DOCA_ERROR_NOT_PERMITTED` on a one-sided task | The local-side smoke worked; the cross-side Read/Write/Atomic fails | The mmap was not exported, or the peer's mmap permissions don't include the matching RDMA flag. Re-check the matrix. |
| Submitted task produces no completion | `doca_task_submit()` returned `DOCA_SUCCESS`; the PE produces nothing | Either the PE is not being progressed, or the peer disconnected silently. Wire the connection-state callbacks. |
| Bulk submit returns `DOCA_ERROR_FULL` | First N submissions succeed, then `FULL` | The queue sizing is below the user's intended in-flight depth. Raise `send_queue_size` and re-run, or drain completions between bursts. |
| Connection callback never fires | RDMA CM connect was called; nothing happens for >30s | Server side is not listening, the network does not route, or `connection_request_timeout` is too short. Check the network first, then the timeout. |

Loop termination: stop iterating once two consecutive iterations of
the same kind don't change anything — that means the cause is below
RDMA. Escalate to [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug)
with the captured layer-1-through-5 evidence.

## debug

Goal: when a DOCA RDMA call returns a `DOCA_ERROR_*` (or the
program doesn't make forward progress), narrow the cause to a
specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending RDMA-specific
fixes. This skill's overlay names the RDMA-specific manifestation
at layers 5 (runtime) and 6 (program):

**Layer 5 (runtime) — RDMA overlay.**

- Walk the connection state machine: was the context started? Was
  the connection request received? Was the connection established?
  Was a task submitted before the connection reached established?
  The wrong order returns `DOCA_ERROR_BAD_STATE`, not a clear
  symptom.
- Confirm the PE is being progressed. *No completion events* is
  almost always a missing `doca_pe_progress()` in the user's main
  loop.
- Confirm both sides agreed on transport type. A mixed-transport
  pair returns `DOCA_ERROR_DRIVER` from the layer below; the fix
  is at configure time, not in the RDMA call.

**Layer 6 (program) — RDMA overlay.**

- Permission matrix: the most common RDMA program-layer bug is an
  mmap-permission / RDMA-permission mismatch surfaced as
  `DOCA_ERROR_NOT_PERMITTED`. Walk the matrix in
  [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy)
  against the user's configured permissions.
- Lifecycle order: configure → start → connect → use → stop →
  destroy. Out-of-order returns `DOCA_ERROR_BAD_STATE`. The most
  common case is destroying an mmap before `doca_ctx_destroy()`.
- Bulk-submit sizing: `DOCA_ERROR_FULL` on bulk submit is a
  sizing-vs-drain mismatch, not a hardware failure.

Once the layer is identified, route to the matching debug verb on
the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); cross-cutting
runtime to [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are commonly
asked in the same conversations. Route them as follows so the agent
does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification, `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the install-tree
  layout in
  [doca-public-knowledge-map ## Layout of an installed DOCA package](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed.
- **deploy.** Deploying RDMA-using applications at scale across
  many hosts / DPUs, Kubernetes operator workflows for RDMA
  workloads, multi-tenant RDMA isolation — out of scope for Phase 1
  and reserved for a future platform skill. For single-host
  first-run testing, the right verb in this skill is `## run`; do
  not invent a "deploy" workflow.
- **rollback.** Coordinated rollback of RDMA-using applications
  across multiple hosts / DPUs — out of scope for Phase 1 and
  reserved for a future platform skill. For a single in-session
  RDMA configuration rollback, the right verb in this skill is
  destroying the context (`doca_ctx_stop` → `doca_ctx_destroy`) and
  re-running `## configure` with the corrected parameters; do not
  invent a "rollback" workflow.
- **kernel-level driver install / firmware burn.** Installing the
  `mlx5_core` driver, burning new ConnectX firmware, or modifying
  `mlxconfig` parameters that need a reset — out of scope for this
  skill. Route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5 (driver), then to the upstream MLNX OFED / firmware
  documentation reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Command appendix

Every command below is **cross-cutting on DOCA RDMA** — it answers a
recurring class of question that comes up in the verbs above. The
agent should treat the *class* as load-bearing; the worked example
is a single instance. Run-as user is the unprivileged user unless
noted. Rows that need elevated privileges call that out explicitly.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env --json`
   for version + devices + libraries + drivers + hugepages in one
   shot; `doca-capability-snapshot` for per-device capability flags;
   `version-matrix.json` for *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the row.
   Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC, headers-win)
   are owned by [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca` | `## configure` step 1; `## build` slot 4 | What is the build-time DOCA version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2). Note: there is normally no `doca-rdma.pc`; use the umbrella `doca` module (verify with `pkg-config --list-all | grep -i doca`) |
| `pkg-config --cflags --libs doca` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs doca` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores (`libdoca_rdma.so`), and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. If this probe fails, fix the module name / `PKG_CONFIG_PATH` per `## build` Step 0 — do NOT fall back to raw libibverbs. |
| `grep -RHn 'DOCA_VERSION_' $(pkg-config --variable=includedir doca)/doca_version.h` | `## configure` step 1 | What macros does this DOCA install expose for compile-time version checks? | A `DOCA_VERSION_MAJOR`, `MINOR`, `PATCH` triple matching the runtime version |
| `doca_caps --list-devs` | `## configure` step 2 | Which devices on this host can be used as a `doca_dev`? | One row per visible device with PCIe address and capability flags |
| `doca_caps --version` | `## configure` step 1; `## test` step 1 | What is the *runtime* DOCA version on this host? | A semver string matching `pkg-config --modversion doca` |
| `ls /opt/mellanox/doca/samples/doca_rdma/` | `## modify` slot 1 | Which RDMA samples ship in this install, and which is the closest starting point? | A list of sample directories named after the task pattern they demonstrate |
| `cat /opt/mellanox/doca/applications/VERSION` | `## configure` step 1; `## debug` layer 1 | What does the install tree itself claim its version is? | A semver string matching the other two version sources |
| `dmesg | tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last RDMA call? | Empty or recent benign messages. Repeated mlx5/IB errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `mlxconfig -d <pcie> q | head -n 40` (sudo) | `## debug` layer 7 | What firmware config does the underlying NIC report? | Stable firmware config; transient values indicate a partial reset |
| `ibv_devinfo` (sudo) | `## configure` step 2; `## debug` layer 7 | What does the underlying `libibverbs` see for this device? | One device row with `state: PORT_ACTIVE` and a sane MTU |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 3 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition and every task submission. Silence after submission = PE not progressed |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the RDMA-specific rows on top.
