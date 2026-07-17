# DOCA Management workflows

**Where to start:** The verbs run `install → configure → build →
modify → run → test → debug → use`. Skip ahead only when the user
is already past a verb. The `## modify` verb is **HIGH STAKES**
because the management plane writes device-level state; the
agent walks the apply-with-rollback workflow there even when the
write looks trivial. The `## test` verb is an iterative loop
(symbol-presence → cap-supported → pre-state capture → write →
re-query → loop back if the diff is unexpected), not a one-shot
pass — see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the underlying object model, version
compatibility, error taxonomy, observability surface, and safety
policy that these workflows assume, see
[CAPABILITIES.md](CAPABILITIES.md). For where to find docs, the
installed DOCA layout, or release notes, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## install

Goal: confirm the user's installed DOCA actually ships
`doca-mgmt` and the host kernel exposes the underlying `fwctl`
path before any management-plane work begins.

This skill does **not** own DOCA installation; that path lives
in [`doca-setup`](../../doca-setup/SKILL.md). The
management-specific preconditions the agent verifies after a
DOCA install:

1. **`doca-mgmt` `.pc` file is present.** `pkg-config
   --modversion doca-mgmt` resolves and reports a semver
   matching `doca_caps --version`. If it does not resolve, the
   installed DOCA package set does not include doca-mgmt; the
   user needs to install the matching package (the exact
   package name is platform-specific and looked up via
   [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package)).
2. **Supporting `.pc` files are present.** doca-mgmt requires
   `doca-common`. Both `pkg-config --modversion` results must
   agree on the same DOCA semver per the four-way match in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
3. **Installed headers expose the symbols.** Check the public
   header set (`doca_mgmt.h`,
   `doca_mgmt_device_caps_general.h`,
   `doca_mgmt_cc_global_status.h`,
   `doca_mgmt_diagnostics_data.h`,
   `doca_mgmt_icm_quota.h`) resolves under the installed DOCA
   infrastructure include tree. If the `.pc` resolves but the
   headers are missing, the install is partial.
4. **`fwctl` is available on the host kernel.** doca-mgmt's
   raw-command path issues `fwctl` ioctls; the host kernel
   must expose the matching character device. The agent does
   not name a specific kernel version from memory; route to
   [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)
   layer 5 (driver) for `fwctl`-side checks. A
   `DOCA_ERROR_OPERATING_SYSTEM` from `doca_mgmt_raw_cmd` on a
   freshly-installed system is almost always this gate.
5. **User privileges.** Most management-plane operations
   require root or an equivalent device-administration
   capability. Confirm the user can either run as root or has
   been granted the necessary `cap_*` capabilities on the
   binary before any write is attempted.

If any precondition fails, **stop and route to
[`doca-setup`](../../doca-setup/SKILL.md)** for the install /
kernel side or to the operator's privilege-management process
for the privilege side; a doca-mgmt-layer diagnosis against a
half-installed system wastes the user's time.

## configure

Goal: bring up the management context (and optionally a
representor context) and reach the state where capability
queries and / or sub-domain operations can run.

Steps the agent should walk the user through:

1. **Confirm the installed DOCA version.** Use the procedure in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   Quote the version observed (`pkg-config --modversion
   doca-mgmt`, then `doca_caps --version`); do not assume
   "latest". Surface that every `doca_mgmt_*` symbol the agent
   recommends is `EXPERIMENTAL` per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
2. **Open the device.** Use the standard `doca_dev` discovery
   path (looked up via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   when the user is unfamiliar with it) to obtain a
   `doca_dev*` for the target BlueField / ConnectX.
3. **Create the management device context.** Call
   `doca_mgmt_dev_ctx_create(dev, &dev_ctx)`. The lifetime of
   `dev_ctx` is bound to the lifetime of `dev`; destroy
   `dev_ctx` first, then close the device.
4. **(Optional) Create the representor context.** If the user
   wants to operate on a VF / function-level surface, obtain a
   `doca_dev_rep*` and call
   `doca_mgmt_dev_rep_ctx_create(dev_ctx, rep, &rep_ctx)`. If
   the VF's kernel-side representor is not exposed but the PCI
   address is known, use
   `doca_mgmt_dev_rep_ctx_create_by_pci_addr(dev_ctx,
   "0000:3a:00.2", &rep_ctx)` with the documented HEX
   `Domain:Bus:Device.Function` format.
5. **Probe sub-domain support.** Before any sub-domain `_set`
   or `_modify`, call the sub-domain's capability gate:
   `doca_mgmt_cap_icm_quota_is_supported(dev_ctx)` for
   icm-quota; `doca_mgmt_cap_diagnostics_data_multi_domain_is_supported(dev_ctx)`
   for the multi-domain diagnostics data; for caps-general and
   cc-global-status, attempt a `_get` against a transient
   handle and treat `DOCA_ERROR_NOT_SUPPORTED` as the
   *not-supported* signal.
6. **Sanity check before any write.** Confirm with the user:
   which device, which representor (if any), which sub-domain,
   which fields. If any of those are unclear, stop and ask —
   do not invent.

If any step fails with a `DOCA_ERROR_*`, route through the
error taxonomy in
[CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

Goal: produce a binary that links DOCA Management against the
user's installed DOCA, using the canonical cross-library build
pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson
or CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the mgmt-specific overlay:

| Slot | Value for doca-mgmt | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-mgmt` | The library's `.pc` file installed by the DOCA host packages |
| Co-required modules | `doca-common` | doca-mgmt depends on Core / Common |
| Header check | The public mgmt header set (`doca_mgmt.h` plus the sub-domain headers) resolvable under the installed DOCA infrastructure include tree (path via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)) | If `pkg-config --cflags doca-mgmt` resolves but the headers are missing, the install is partial |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-mgmt`; never hardcode in build files | The EXPERIMENTAL tag means a version pin from agent memory is wrong by construction |
| Runtime privilege model | The built binary will need root or an equivalent capability set to issue management operations; ensure the deployment plan accounts for this | A binary that compiles but cannot exercise its primary surface in deployment is a productivity sink |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; FFI wrappers are out of scope for this skill.

## modify

Goal: change device-level state through doca-mgmt — toggle a
caps-general field, change a cc-global-status setting, modify
diagnostics-data, set an ICM quota, or issue a raw command — in
a way that preserves the operator's ability to roll back.

**This verb is HIGH STAKES.** The agent NEVER recommends a
doca-mgmt write without walking the apply-with-rollback workflow
below, even when the write looks trivial. The workflow is the
[`doca-hardware-safety` change-application discipline](../../doca-hardware-safety/CAPABILITIES.md#capabilities-and-modes)
applied at the API surface.

Apply-with-rollback workflow (every modify pattern):

1. **Pre-flight inventory.** Run the bundle-wide pre-flight
   inventory per
   [`doca-hardware-safety TASKS.md ## configure`](../../doca-hardware-safety/TASKS.md#configure)
   FIRST. The doca-mgmt-specific addendum is the *sub-domain
   handle pre-state*: create the sub-domain handle, call the
   matching `_get` / `_query` against the target context, and
   record every field the upcoming `_set` will touch.
2. **Confirm capability support.** Re-run the sub-domain's
   capability gate from [`## configure`](#configure) step 5
   against the actual target context. The gate may differ
   between device context and representor context.
3. **Identify the scope.** If the operation is a `raw_cmd`,
   pick the narrowest `enum doca_mgmt_cmd_scope` per the
   ladder in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   If a dedicated sub-domain wrapper exists, prefer it over
   `raw_cmd`.
4. **Name the rollback path.** Before writing, write down (and
   read back to the user) the exact `_set` / `raw_cmd` call
   that would revert to the captured pre-state. If the
   pre-state cannot be reproduced by the same surface, the
   change has *no rollback* and the agent refuses to apply it
   per
   [`doca-hardware-safety ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy).
5. **Maintenance window + OOB precondition.** For any write
   that could affect link state or hardware traffic — most
   notably `cc_global_status` enable/disable, `raw_cmd`
   `CONFIGURATION` or `DEBUG_WRITE_FULL` against a port-
   affecting opcode, and any change applied during live
   traffic — confirm with the operator the maintenance window
   and OOB-access classes per
   [`doca-hardware-safety ## Capabilities and modes`](../../doca-hardware-safety/CAPABILITIES.md#capabilities-and-modes).
6. **Apply on replica first.** Per
   [`doca-hardware-safety ## Capabilities and modes`](../../doca-hardware-safety/CAPABILITIES.md#capabilities-and-modes)
   replica-first rule, apply on a non-prod replica that
   matches the production hardware class first; run the
   post-write re-query gate (step 8 below) on the replica; only
   then schedule the production change.
7. **Apply the write.** Call the sub-domain's `_set` /
   `_modify` / `_set_limit` / `raw_cmd` on the target context.
   Quote `doca_error_get_descr()` verbatim if the call
   returns an error; do not paraphrase.
8. **Post-write re-query.** Immediately re-run the matching
   `_get` / `_query` and diff against the pre-state. The diff
   should be exactly the change the user intended. Any
   unexpected delta is a regression; trigger the rollback path
   from step 4.
9. **Hold the rollback path ready for the duration of the
   change window.** The user does not declare the change "done"
   until either (a) the workload has resumed and the device's
   observability surface is healthy, or (b) the rollback has
   been applied.

The agent emits the *intent description + the apply-with-
rollback workflow filled out for the user's specific call*; the
*actual* unified diff against any sample code the user is
modifying is produced line-by-line via the universal modify-a-
sample workflow in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).

## run

Goal: actually execute the built binary against the user's
installed DOCA on a host with a BlueField / ConnectX device,
with appropriate privileges.

Steps the agent should walk the user through:

1. **Privilege check.** Confirm the binary will run with the
   privileges its management operations require (root or
   equivalent `cap_*` capabilities). A
   `DOCA_ERROR_OPERATING_SYSTEM` from `doca_mgmt_raw_cmd` on a
   non-root invocation is almost always a privilege gate.
2. **Capture the structured log.** Set
   `DOCA_LOG_LEVEL=trace` for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   The doca-mgmt surface is small and quiet; the trace log is
   the cheapest way to see lifecycle transitions.
3. **Run read-only operations first.** Before any write, run
   the binary's read-only path
   (`doca_mgmt_device_caps_general_get`,
   `doca_mgmt_cc_global_status_get`,
   `doca_mgmt_diagnostics_data_query_for_dev`,
   `doca_mgmt_icm_quota_query`, or
   `doca_mgmt_raw_cmd` with `DEBUG_READ_ONLY`) against the
   target device. The output is the baseline.
4. **Confirm the baseline matches the operator's mental
   model.** If the device's reported pre-state diverges from
   what the operator expected, *stop* — the discrepancy is a
   bug to investigate before any write. Writing onto an
   unexpected pre-state compounds the bug.
5. **Only then run writes,** following the apply-with-rollback
   workflow from [`## modify`](#modify).

## test

Goal: prove the configured management surface actually returns
correct data and applies writes the device honors, without
disturbing live state any more than necessary.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the symbol set, the cap-supported result, the pre-state
capture, the write outcome, or the post-write re-query diff. The
loop terminates when either (a) the user's intended management
operation flows end-to-end with the expected device-side state
change, or (b) the agent has narrowed the failure cause to a
layer outside doca-mgmt itself (fwctl, firmware, driver) and
escalated to the matching skill.

Iteration shape:

1. **Symbol-presence check.** Confirm every `doca_mgmt_*`
   symbol the user's code references is exported by the
   installed `libdoca_mgmt.so` and matched in the installed
   headers. A `function not found` at link time is almost
   always a partial install or a wrong-version pairing per
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   layer 2.
2. **Capability-supported check.** Re-run the sub-domain's
   `_is_supported` against the active context. If false →
   that's the answer; the user's device or DOCA version does
   not support the sub-domain.
3. **Pre-state capture.** Run the `_get` / `_query` for every
   field the upcoming write will touch. The result is the
   rollback baseline.
4. **Apply the write on a replica.** Per the
   [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
   replica-first rule, the test environment is the replica;
   never the production device. Apply the write.
5. **Post-write re-query.** Run the `_get` / `_query` again
   and diff against step 3. The diff should match the user's
   intent exactly.
6. **Negative test.** Construct one deliberately invalid
   call (e.g. an ICM-quota limit above `_cap_get_max_limit`,
   or a `_set` on a context whose `_is_supported` returned
   failure) and confirm the API returns
   `DOCA_ERROR_INVALID_VALUE` / `DOCA_ERROR_NOT_SUPPORTED` as
   expected. This validates the agent's capability-discovery
   understanding is itself correct on this DOCA version.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on a sub-domain we expected to ship | The user's code references a sub-domain the installed DOCA does not expose | Re-confirm via the sub-domain's capability gate; the EXPERIMENTAL tag means sub-domains can be added / renamed between releases |
| `DOCA_ERROR_BAD_CONFIG` from `_set` | The handle was not populated with all required fields before `_set` | Re-walk the sub-domain handle's field accessors; every field the surface requires must be set or `_clear`-ed before `_set` |
| `DOCA_ERROR_IN_USE` on a representor `_set` | The representor context is already initialized for this surface | Destroy and re-create the representor context, then retry; some surfaces are not field-mutable in place |
| `DOCA_ERROR_OPERATING_SYSTEM` on `raw_cmd` | The host-side `fwctl` ioctl path is not reachable | Confirm the host kernel exposes the `fwctl` interface; route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) layer 5 |
| `DOCA_ERROR_IO_FAILED` on `raw_cmd` | The firmware rejected the command | Re-confirm the four-way version match; re-confirm the opcode against the vendor docs; if both check out, the device state is the gate — route to [`doca-hardware-safety TASKS.md ## debug`](../../doca-hardware-safety/TASKS.md#debug) |
| Post-write re-query diff doesn't match intent | The `_set` returned success; the `_get` shows unexpected state | The write landed but with side effects the agent did not anticipate; rollback per [`## modify`](#modify) step 4, then re-investigate |

Loop termination: stop iterating once two consecutive
iterations of the same kind don't change anything — that means
the cause is below doca-mgmt. Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured layer-1-through-5 evidence.

## debug

Goal: when a DOCA Management call returns a `DOCA_ERROR_*`
(or a write lands but the device state diverges from the
operator's intent), narrow the cause to a specific layer and
act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending
doca-mgmt-specific fixes. This skill's overlay names the
management-specific manifestation at layers 5 (runtime), 6
(program), and at the hardware-state-investigation cross-link:

**Layer 5 (runtime) — doca-mgmt overlay.**

- Confirm the management contexts were created in the right
  order: device context first, then representor context (the
  representor context's `_create` requires an existing
  `dev_ctx`). Out-of-order returns `DOCA_ERROR_INVALID_VALUE`.
- Confirm the user-side privilege model. A
  `DOCA_ERROR_OPERATING_SYSTEM` is almost always either
  missing root or a missing `cap_*` set; check the binary's
  privileges before deeper investigation.
- Confirm the `fwctl` ioctl path is reachable. The kernel-side
  check for `fwctl` is owned by
  [`doca-setup`](../../doca-setup/SKILL.md); route there.

**Layer 6 (program) — doca-mgmt overlay.**

- Sub-domain field discipline: every `_set` requires its
  fields to be populated via the matching `_set_<field>`
  accessor before the apply call. `DOCA_ERROR_BAD_CONFIG` is
  the signal for an unset field.
- Handle re-use: a sub-domain handle is *transient*; it can
  be re-used across multiple operations via `_clear`, but
  cannot be shared across threads without external
  synchronization.
- PCI address format for `_create_by_pci_addr`: the documented
  format is HEX `Domain:Bus:Device.Function`
  (e.g. `"0000:3a:00.2"`). A malformed string returns
  `DOCA_ERROR_INVALID_VALUE`.

**Hardware-state investigation cross-link.** When a write
returned success but the device's reported state diverges from
the intended value, OR when `raw_cmd` returns
`DOCA_ERROR_IO_FAILED` after the four-way version match and
opcode check both pass, the cause is at the
hardware/firmware layer — route to
[`doca-hardware-safety TASKS.md ## debug`](../../doca-hardware-safety/TASKS.md#debug)
for the change-application incident discipline. doca-mgmt's
debug overlay does NOT include the firmware-state recovery
ladder; that's the meta-policy's job.

Once the layer is identified, route to the matching debug verb
on the matching skill: install / build / link / driver to
[`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug);
device-state recovery to
[`doca-hardware-safety TASKS.md ## debug`](../../doca-hardware-safety/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## use

Goal: integrate a working doca-mgmt component into a fleet-
management agent, an orchestration plugin, or a device-
administration tool — and operate it against many devices over
time without surprise.

The integration shape this skill teaches:

1. **Per-application init order.** Open the device → create
   `doca_mgmt_dev_ctx` → optionally create
   `doca_mgmt_dev_rep_ctx` for representor-targeted operations
   → run the sub-domain capability gates → cache the result for
   the session. The cached cap snapshot is the baseline every
   future operation in the same session compares against.
2. **Per-application teardown order.** Destroy representor
   contexts before the device context; destroy sub-domain
   handles before the context they were used against; close
   the device last. Out-of-order destroy is the canonical
   leak.
3. **Bulk operation discipline.** When the fleet agent walks
   many devices, each device gets its own
   `doca_mgmt_dev_ctx`; do NOT share a context across devices.
   The library is not documented as cross-device-safe and the
   agent does not assume it.
4. **Per-release re-verification.** Because the API surface
   is EXPERIMENTAL, every DOCA upgrade requires re-running
   [`## test`](#test) end-to-end against the new install. A
   sub-domain that worked on DOCA *X* may be reshaped on DOCA
   *Y*; the agent does not assume the fleet tool survives a
   version bump without re-testing.
5. **Operational handoff.** Production deployment uses the
   bundle's hardware-safety meta-policy
   ([`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md))
   for every write. The fleet tool's *change-control runbook*
   names the maintenance window, the OOB access class, the
   replica-first stage, the rollback path, and the
   observability gate per the meta-policy.
6. **`raw_cmd` operator discipline.** The raw-command path is
   the escape valve for vendor-documented opcodes without a
   dedicated wrapper. The agent treats every `raw_cmd` call
   site as a *change to the device's firmware-control
   surface*: opcode + scope + payload reviewed; pre-state
   captured; rollback documented; the operation runs inside
   the same change-control discipline as `mlxconfig` writes.

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows
so the agent does not invent guidance:

- **install (of DOCA itself).** Installing DOCA, choosing
  packages, post-install verification, `pkg-config` wiring —
  defer to [`doca-setup`](../../doca-setup/SKILL.md) and to
  the install-tree layout in
  [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill's `## install` verb assumes DOCA is already
  installed and only checks the mgmt-specific preconditions.
- **deploy.** Deploying fleet-management agents at scale across
  many hosts and BlueFields, Kubernetes operator workflows,
  multi-tenant isolation — out of scope and reserved for a
  future platform skill. For single-host first-run testing,
  the right verb is `## run`.
- **rollback (fleet-wide).** Coordinated rollback of
  doca-mgmt-applied changes across many devices is fleet-tool
  operator workflow, not a doca-mgmt API concern; the API
  surface for per-device rollback is the same `_set` / `_modify`
  / `raw_cmd` used for the original write, called with the
  pre-state captured per [`## modify`](#modify). The
  cross-cutting [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
  meta-policy owns the discipline.
- **mlxconfig direct operation.** Some configuration spaces are
  reachable both via doca-mgmt and via `mlxconfig`; the latter
  is outside this skill's surface and routes through
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
  for the change-application discipline.
- **Firmware burn / BFB reflash.** Out of scope. Route to
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
  for the meta-policy and to the public firmware-tooling
  documentation reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Command appendix

Every command below is **cross-cutting on DOCA Management** —
it answers a recurring class of question that comes up in the
verbs above. The agent should treat the *class* as load-bearing;
the worked example is a single instance.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent probes for the matching structured helper FIRST
(`doca-env --json` for version + devices + libraries +
drivers; `doca-capability-snapshot` for per-device capability
flags; `version-matrix.json` for *"available since"* lookups).
If the probe succeeds, the structured tool's output is the
authoritative answer. If the probe fails, fall back to the
manual command in the row.

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-mgmt` | [`## install`](#install) step 1; [`## configure`](#configure) step 1 | What is the build-time DOCA Management version? | A semver matching `doca_caps --version`. Disagreement = partial install; route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| `pkg-config --modversion doca-common doca-mgmt` | [`## install`](#install) step 2 | Do both `.pc` files agree on the same DOCA semver? | A single semver repeated twice. Any disagreement is the partial-install pattern |
| `pkg-config --cflags --libs doca-mgmt` | [`## build`](#build) | What include + link flags does the linker need? | Includes resolve under whichever include directory `pkg-config --cflags` reports on this install (do not hardcode the path); libs include `-ldoca_mgmt -ldoca_common` |
| `doca_caps --list-devs` | [`## configure`](#configure) step 2 | Which devices on this host can be used as a `doca_dev` for management operations? | One row per visible device with PCIe address and capability flags |
| `doca_caps --version` | [`## install`](#install) step 1 | What is the *runtime* DOCA version on this host? | A semver matching `pkg-config --modversion doca-mgmt` |
| `cat /opt/mellanox/doca/applications/VERSION` | [`## install`](#install) step 1; [`## debug`](#debug) layer 1 | What does the install tree itself claim its version is? | A semver matching the other version sources |
| `id -u` and `getcap <binary>` | [`## run`](#run) step 1 | Does the user / binary have the privileges management operations require? | `0` for root invocation, or a capability set that includes the device-administration `cap_*` the operation requires |
| `ls /dev/fwctl*` | [`## install`](#install) step 4 | Is the host kernel exposing the `fwctl` character device doca-mgmt's raw command path uses? | One or more `/dev/fwctl*` character devices visible to root |
| `DOCA_LOG_LEVEL=trace ./<binary>` | [`## run`](#run) step 2 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition and every management-plane call |
| `dmesg | tail -n 40` (sudo) | [`## debug`](#debug) layer 7 | What did the kernel / driver / firmware log around the last mgmt call? | Empty or recent benign messages. Repeated firmware / `fwctl` errors → route to [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) for the device-state investigation |
| `mlxconfig -d <pcie> q | head -n 40` (sudo) | [`## debug`](#debug) layer 7 | What firmware-stored config does the NIC / DPU report? | Stable firmware config. Mismatched values vs the pre-state captured in [`## modify`](#modify) indicate the change had side effects beyond the user's intent — trigger the rollback path |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the mgmt-specific rows on top.
