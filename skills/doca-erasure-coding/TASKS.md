# DOCA Erasure Coding workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop (cap check →
permission cross-check → known-vector recover smoke → update
smoke → small-bulk → full-bulk → loop back if a task-type or
sizing assumption changed), not a one-shot pass — see the
eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the underlying capability surface, the
three task types (create / recover / update), the
matrix-type + N + K + block-size configuration surface, per-task
capability-query rules, error taxonomy, observability, and
safety / path-selection policy, see
[CAPABILITIES.md](CAPABILITIES.md). For the cross-library DOCA
patterns layered under everything below (the universal lifecycle,
the cross-library `DOCA_ERROR_*` taxonomy, the
modify-a-shipped-sample workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: bring up a DOCA Erasure Coding context on a host or
BlueField and confirm the device's accelerator supports the task
type(s), block size, N + K layout, and matrix type the user
actually intends to use.

Steps the agent should walk the user through:

1. **Confirm the installed DOCA version.** Use the procedure in
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
   Quote the version observed (`pkg-config --modversion
   doca-erasure-coding`, then `doca_caps --version`); do not
   assume "latest". The four-way match rule lives in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility);
   if the observed sources disagree, route there before any EC
   diagnosis.
2. **Discover the device capability surface for Erasure Coding.**
   Run `doca_caps --list-devs` (per
   [`doca-caps`](../../tools/doca-caps/SKILL.md)) to see which
   devices are visible, then run the per-`doca_devinfo`
   `doca_ec_cap_*` queries against the candidate device. Record
   at minimum:
   `doca_ec_cap_task_create_is_supported(devinfo)`,
   `doca_ec_cap_task_recover_is_supported(devinfo)`,
   `doca_ec_cap_task_update_is_supported(devinfo)`,
   `doca_ec_cap_task_galois_mul_is_supported(devinfo)` (the 4th
   public task — bundle previously omitted this row),
   `doca_ec_cap_get_max_block_size(devinfo)`,
   `doca_ec_cap_get_max_buf_list_len(devinfo, &max_buf_list_len)`, and a
   per-variant `doca_ec_matrix_create()` / `_create_from_raw()`
   attempt against the active device to enumerate matrix
   types the device advertises. The capability surface to
   compare against lives in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
3. **Confirm Erasure Coding is the right primitive for this
   workload at all.** Per the path-selection bullets in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   doca-erasure-coding is the right answer ONLY for
   storage-domain resilience workloads (RAID-6-style block
   layouts, distributed file system parity, object-storage
   erasure-coded buckets). For a network-FEC problem, a
   small-cluster replication problem, or a non-Reed-Solomon
   coding scheme, doca-erasure-coding is the wrong primitive —
   route the user back to the correct primitive (replication
   for small clusters; CPU library for non-RS schemes; DOCA
   networking libraries for wire-side problems) instead of
   bringing up a `doca_ec` context.
4. **Pick the task type(s) to enable.** Create
   (`doca_ec_task_create`) for first-time encoding of N data → K
   redundancy; recover (`doca_ec_task_recover`) for
   reconstruction workers; update (`doca_ec_task_update`) for
   in-place editors that touch one source block at a time. **A
   user who describes editing one source block in an existing
   N + K layout should be steered to update, not to a fresh
   create** — re-encoding all N data blocks just to refresh K
   parity is linear-in-N work the update task is designed to
   avoid. The task-type table lives in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
5. **Pick the coding parameters: matrix type, N, K, block size.**
   Matrix type must be in the
   variant set the device advertises via successful
   `doca_ec_matrix_create()` / `_create_from_raw()` constructor
   (the public header does NOT ship a `doca_ec_cap_get_matrix_*`
   family — do not
   assume Vandermonde is universally available). N + K must
   satisfy `N + K ≤ doca_ec_cap_get_max_buf_list_len(devinfo, &max_buf_list_len)`.
   Block size must satisfy
   `block_size ≤ doca_ec_cap_get_max_block_size(devinfo)`; ALL
   blocks in a single task share this block size. The K choice
   reflects the durability target (the layout tolerates any K
   simultaneous block losses); the agent should ask the user
   for the durability target rather than picking K silently.
6. **Configure the EC instance.** Mandatory before
   `doca_ctx_start()`: enable at least one task type
   (`doca_ec_task_create_set_conf` / `_recover_set_conf` /
   `_update_set_conf`, each with its success and error
   completion callbacks plus its max-num-tasks budget); commit
   the four coding parameters (matrix type, N, K, block size)
   per the device cap query in step 2; set source mmap
   permissions
   (`doca_mmap_set_permissions` to include
   `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY`) on EVERY source buffer
   (N data buffers for create / update; up to N + K surviving
   buffers for recover); set destination mmap permissions
   (`DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`) on EVERY destination
   buffer (K redundancy buffers for create; up to K recovered
   buffers for recover; K refreshed parity buffers for update);
   confirm every buffer length equals the configured block
   size. Per the matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
7. **Sanity check before any task submission.** Confirm with
   the user: which task type(s), matrix type, N, K, block size,
   and the buffer count on each side. Run a known-vector
   recover smoke (encode a tiny N + K layout from a fixed
   input, drop one block, recover, bit-compare against the
   original) before any real storage data flows. If any step
   fails with a `DOCA_ERROR_*`, route through the error
   taxonomy in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   before retrying.

For the canonical DOCA universal lifecycle that underlies steps
4-7, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill adds the EC overlay; do not re-explain the lifecycle
here.

## build

Goal: produce a binary that links DOCA Erasure Coding against
the user's installed DOCA, using the canonical cross-library
build pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson
or CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the EC-specific overlay:

| Slot | Value for Erasure Coding | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-erasure-coding` | The library's `.pc` file installed by the DOCA host packages. Wrong module name = `pkg-config: Package 'doca-erasure-coding' was not found` |
| Required runtime libs | `libdoca-common`, `libdoca-erasure-coding`, plus whatever `pkg-config --libs doca-erasure-coding` resolves transitively | EC depends on Core; the link line should not pull in unrelated DOCA libraries |
| Header check | The public header that `pkg-config --cflags` for this artifact resolves to actually exists on disk at the path pkg-config reports (do not hardcode the include path) | If `pkg-config --cflags doca-erasure-coding` resolves but the include is missing, the install is partial — route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-erasure-coding`; never hardcode in build files | Cross-version build/runtime mixing breaks per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; the FFI wrapper layer is the language-specific
binding and is out of scope for this skill — but the four slots
above are still the load-bearing inputs the wrapper needs.

## modify

Goal: take a shipped DOCA EC sample as the verified starting
point and apply a **minimum-diff modification** to express the
user's intent.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The EC-specific overlay is the *modify-from-sample
schema fill* — the slots the agent must elicit from the user
before recommending any code-level edit:

| Slot | What the agent asks the user | EC-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_erasure_coding/`? | Pick the closest in *task direction* (create vs recover vs update) to the user's intent. Do NOT bridge across directions — a smaller diff is always safer than a re-architecture. A user editing one source block at a time should start from an update sample, not from a create sample with the encode ripped out |
| 2. Task type(s) added or removed | Which of the three task types — create, recover, update? | Each enabled task needs its own `doca_ec_task_*_set_conf` call before `doca_ctx_start()`, plus its matching cap-query in [`## configure`](#configure) step 2. Removing an unused task type from a sample shrinks the configuration surface and is recommended for single-purpose pipelines |
| 3. Matrix-type change | Switching the Reed-Solomon matrix variant? | Re-attempt `doca_ec_matrix_create()` (or `_from_raw()`) against the active `doca_dev` for the new variant; constructor failure = "not supported on this device." The public header does NOT ship a `doca_ec_cap_get_matrix_*` family. Matrix-type drift between configure and submit returns `DOCA_ERROR_NOT_SUPPORTED` |
| 4. N + K layout change | Changing the number of data or redundancy blocks? | Confirm `N + K ≤ doca_ec_cap_get_max_buf_list_len(devinfo, &max_buf_list_len)`. Changing K changes the durability target (tolerates K simultaneous losses); the agent should surface that trade-off rather than picking K silently |
| 5. Block-size change | Changing the per-block size? | Confirm `block_size ≤ doca_ec_cap_get_max_block_size(devinfo)`. ALL N + K buffers in a single task share this block size; mismatched block sizes within a task return `DOCA_ERROR_INVALID_VALUE` at submit. Resize every buffer, not just one |
| 6. Permissions on the extra buffers | Are the source mmaps RO and destination mmaps RW on EVERY block (not just the first)? | EC tasks have multiple source / destination buffers (N + K total per layout); a missing permission flag on ANY one of them surfaces as `DOCA_ERROR_NOT_PERMITTED`. Over-broad permissions are a silent security regression |
| 7. Build manifest | Keep the sample's existing `meson.build` (which already wires `pkg-config doca-erasure-coding`)? | Yes. Do not switch to a hand-rolled Makefile for *"simplicity"* — it removes the version-check rail |

The agent emits an *intent description + the filled slots*; the
*actual* unified diff against the sample source is produced by
the modify-from-sample renderer (deferred to a future round, per
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify)).
Until the renderer ships, the agent must walk the user through
the diff line-by-line against the sample source they read on
disk, and have the user paste back the result for validation.

## run

Goal: actually execute the built binary against the user's
installed DOCA on a host or BlueField, with a real input.

Steps the agent should walk the user through:

1. **Confirm the device is reachable.** EC runs on a single
   side (no peer); the only env-side requirement is that the
   `doca_dev` the binary opens corresponds to a device whose
   accelerator the user expects. Mismatched `doca_dev`
   selection (opening a NIC without the requested EC task on
   its accelerator) returns `DOCA_ERROR_NOT_SUPPORTED` at task
   submit, not at open. Re-quote the output of `doca_caps
   --list-devs` ([`doca-caps`](../../tools/doca-caps/SKILL.md))
   and confirm the device the binary opens is the same one the
   cap-query ran against.
2. **Run the known-vector recover smoke first.** A binary that
   encodes a tiny N + K layout from a fixed input, simulates
   losing one block, runs recover, and bit-compares the
   recovered block against the original — that is the cheapest
   correctness signal. Do not bulk-encode storage data before
   this passes. If the user's workflow includes update, run an
   update smoke (encode → update one block → re-encode the new
   state → compare parity byte-for-byte) as the second smoke.
3. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace`
   for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the lifecycle and
   task-submit transitions visible on first failure.
4. **Capture the completion events on the PE.** A run that
   produces no completion events but doesn't error is almost
   always a missed `doca_pe_progress()` call. Confirm the
   progress engine is being driven on the main thread.

## test

> **Performance harness routing.** For *throughput / latency
> measurement* on the configured EC context (or for cross-library
> comparison against the other DOCA crypto primitives), route the
> user to [`doca-bench TASKS.md ## test`](../../tools/doca-bench/TASKS.md#test)
> — `doca-bench` is the cross-library performance harness with
> documented warm-up / steady-state / outlier semantics, and it
> explicitly supports Erasure-Coding. The iteration loop below stays
> the *correctness* harness; `doca-bench` is the *performance* harness.

Goal: prove the configured EC context can actually produce
correct parity (create), correctly reconstructed blocks
(recover), and correctly refreshed parity (update), at the
user's intended throughput, on the user's hardware, and that
the task selection, matrix type, N + K, block size, and
permission set were right.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the task type, the matrix type, N + K, the block size,
the permission set, or the path-selection assumption. The loop
terminates when either (a) the user's intended workload
produces correct output end-to-end with acceptable throughput,
or (b) the agent has narrowed the failure cause to a layer
outside DOCA Erasure Coding itself (driver / firmware / device)
and escalated to the matching skill.

Iteration shape:

1. **Capability re-check.** Re-run
   `doca_ec_cap_task_create_is_supported(devinfo)`,
   `_task_recover_is_supported(devinfo)`,
   `_task_update_is_supported(devinfo)`,
   `doca_ec_cap_get_max_block_size(devinfo)`,
   `doca_ec_cap_get_max_buf_list_len(devinfo)`, and the
   `doca_ec_matrix_create()` constructor success per variant
   against the active
   `doca_devinfo`. If any required task type or matrix type
   returns false / unexpected → that's the answer; the user's
   device or DOCA version does not support the requested config.
   Update the intent or update the install.
2. **Permission cross-check.** Compare the configured source +
   destination mmap permissions against the matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   on EVERY one of the N + K buffers (not just the first).
   Mismatches surface as `DOCA_ERROR_NOT_PERMITTED` on the
   first task submission, not at configure time.
3. **Known-vector recover smoke.** Encode a small fixed N + K
   layout (e.g. N = 4, K = 2, block_size = 4 KiB), drop one
   block, run recover, and bit-compare the recovered block
   against the original. If the recovered block differs, the
   configuration is wrong — do not proceed to bulk input. The
   most common bugs at this step are wrong matrix type (asked
   for Vandermonde, configured Cauchy or vice versa), wrong
   N or K, and source / destination buffer swap.
4. **Update smoke (if update is in scope).** Encode a small
   fixed N + K layout, update one data block (changing its
   contents), then re-encode the modified data via create from
   scratch and compare the two parity sets byte-for-byte. They
   must match — if they don't, the update task is configured
   wrong (most commonly: wrong matrix type, wrong index
   identification of the changed block, or stale source
   buffer wiring).
5. **Completion drain.** Confirm completion events arrive on
   the PE for every submitted task. *Submitted but no
   completion* is the most expensive class of bug to discover
   late; confirm it on the known-vector smoke before bulk
   submissions.
6. **Small-bulk test.** Submit a small series of EC tasks at
   block sizes approaching `_get_max_block_size` and N + K
   layouts approaching `_get_max_num_blocks` but staying under
   both ceilings; verify recover round-trips (encode →
   intentionally drop K blocks → recover → bit-compare) and
   update consistency (update → re-encode → compare parity).
   Throughput numbers come from this step; correctness comes
   from steps 3 and 4.
7. **Full-bulk run.** Once the small-bulk passes, scale up to
   the user's intended dataset size and submission rate. Watch
   for `DOCA_ERROR_AGAIN` (drain the PE before retrying) and
   for `DOCA_ERROR_INVALID_VALUE` (a block-size or N + K
   boundary was crossed — re-narrow to the cap-query axis).
8. **Negative test.** Once the positive path works,
   intentionally request a task type the device should NOT
   support (per step 1), a matrix type outside the
   variant set advertised via `doca_ec_matrix_create()`
   constructor success, a block size larger than
   `_get_max_block_size`, an N + K layout larger than
   `_get_max_num_blocks`, or a recover task with more than K
   missing blocks; confirm the failure is the expected
   `DOCA_ERROR_NOT_SUPPORTED` / `DOCA_ERROR_INVALID_VALUE`.
   This validates the agent's capability-discovery is itself
   correct.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on a task we expected to work | The doc page lists the task but the cap query returns false | The agent quoted the *library* surface; the *device* capability per `doca_devinfo` is the real gate. Re-narrow to the device-level query for the specific task type. |
| `DOCA_ERROR_NOT_SUPPORTED` on the configured matrix type | Configure accepts the matrix-type enum but submit (or `set_conf`) returns NOT_SUPPORTED | The matrix variant is not supported by the active device's accelerator. Re-attempt `doca_ec_matrix_create()` (or `_from_raw()`) per variant the user wants and re-pick from those that succeed; the public header does NOT ship a `doca_ec_cap_get_matrix_*` family. |
| `DOCA_ERROR_INVALID_VALUE` on first submit | Block size larger than `_get_max_block_size`, OR N + K larger than `_get_max_num_blocks`, OR mismatched block sizes between source / destination buffers in a single task | Re-size the buffer using the cap-query output (block size) or re-narrow the layout (N + K). Confirm every block in the task is the same size. The error is sizing-vs-cap mismatch, not corruption. |
| `DOCA_ERROR_INVALID_VALUE` on a recover task | Recover request lists more than K missing blocks | The layout is unrecoverable. Stop; do not invent a workaround. Restore from another replica (if the system layers replication on top of EC) or accept the data loss. |
| Recover smoke produces wrong byte content | Configuration accepted but the recovered block does not match the original | Wrong matrix type, wrong N + K, or source / destination buffer swap. Re-check the matrix type in the `_set_conf` call, re-check the N and K against the device cap, and re-check the source / destination wiring before any other diagnosis. |
| Update smoke produces parity that disagrees with re-encode | Update task accepted, but the refreshed parity does not match the parity a fresh create would produce | Wrong matrix type, wrong index identification of the changed source block, or stale source buffer wiring. Walk the update inputs (which source block changed, what was its old content, what is its new content) before any other diagnosis. |
| Submitted task produces no completion | `doca_task_submit()` returned `DOCA_SUCCESS`; the PE produces nothing | The PE is not being progressed. Add a `doca_pe_progress()` call in the main loop. |
| Bulk submit returns `DOCA_ERROR_AGAIN` | First M submissions succeed, then `AGAIN` | The task queue is full. Drain completions between bursts via `doca_pe_progress()`, or raise the configured queue depth at configure time. |
| User asks "I changed one block, should I run create again?" | The user's instinct is to re-encode N data → K parity from scratch | Route to `doca_ec_task_update`; re-encoding is wasteful when one source block changed. This is a workflow-design correction, not a bug. |

Loop termination: stop iterating once two consecutive iterations
of the same kind don't change anything — that means the cause is
below DOCA Erasure Coding. Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured cap-query snapshot + recover smoke diff as
evidence.

## debug

Goal: when a DOCA EC call returns a `DOCA_ERROR_*` (or the
program produces no completion event), narrow the cause to a
specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending EC-specific
fixes. This skill's overlay names the EC-specific manifestation
at layers 5 (runtime) and 6 (program):

**Layer 5 (runtime) — EC overlay.**

- Walk the lifecycle: was the context started? Were the
  task(s) enabled before start (`doca_ec_task_*_set_conf`
  before `doca_ctx_start()`)? Submitting before the requested
  task is enabled returns `DOCA_ERROR_BAD_STATE`, not a clear
  symptom.
- Confirm the PE is being progressed. *No completion events*
  is almost always a missing `doca_pe_progress()` in the
  user's main loop.
- Confirm EVERY source and destination mmap is still alive at
  submit time (EC has N + K buffers per task, not just two).
  Destroying any of them before `doca_ctx_destroy` is a
  use-after-free that surfaces as `DOCA_ERROR_BAD_STATE` from
  subsequent calls.

**Layer 6 (program) — EC overlay.**

- Block-sizing matrix: the most common EC program-layer bug is
  a per-buffer length that doesn't match the configured block
  size — surfaces as `DOCA_ERROR_INVALID_VALUE` at submit.
  Confirm every source AND destination buffer length equals
  the configured block size, and that the block size itself is
  ≤ `doca_ec_cap_get_max_block_size(devinfo)`. The matching
  N + K error is `N + K > doca_ec_cap_get_max_buf_list_len(devinfo, &max_buf_list_len)`
  at configure / submit; the fix there is to shrink the layout
  or pick a device with a higher cap.
- Matrix-type drift: a `_set_conf` call that quotes a matrix
  variant `doca_ec_matrix_create()` fails to construct on the
  active device for
  returns `DOCA_ERROR_NOT_SUPPORTED`. Re-enumerate the
  supported matrix types against the active `doca_devinfo`; do
  not assume from prior installs.
- Task-type drift: an `_allocate_init` call that asks for a
  task type the cap query returns false for returns
  `DOCA_ERROR_NOT_SUPPORTED`. Re-run the matching
  `doca_ec_cap_task_*_is_supported` against the active
  `doca_devinfo`; do not assume from prior installs.
- Recover-out-of-budget: a recover task that lists more than K
  missing blocks returns `DOCA_ERROR_INVALID_VALUE` — the
  layout is unrecoverable. The fix is upstream (more
  redundancy at encode time, or replication on top of EC), not
  a tweak inside the EC call.
- Update-vs-create mis-routing: a user re-encoding via
  `doca_ec_task_create` on every single-block change is not
  errored — it just wastes accelerator cycles. The "fix" is a
  workflow correction (route to `doca_ec_task_update`), not a
  `DOCA_ERROR_*` interpretation.
- Recover smoke produces wrong bytes: the recovered block is
  the wrong length → block-size mismatch or N + K mismatch.
  The recovered block is the right length but content differs
  → wrong matrix type, or source / destination buffer swap, or
  the fixture used for the smoke is itself corrupt. Walk the
  matrix-type + N + K + buffer wiring before any other
  diagnosis.

Once the layer is identified, route to the matching debug verb
on the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); version
to [`doca-version ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows
so the agent does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification, `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the
  install-tree layout in
  [doca-public-knowledge-map ## Layout of an installed DOCA package](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed.
- **deploy.** Deploying EC-using applications at scale (storage
  daemons running create / recover / update on many nodes,
  rebuild workers, Kubernetes operator workflows) — out of
  scope for Phase 1 and reserved for a future platform skill.
  For single-host first-run testing, the right verb in this
  skill is `## run`; do not invent a "deploy" workflow.
- **rollback.** Coordinated rollback of EC-using applications
  across many hosts — out of scope. For a single in-session
  EC configuration rollback, the right verb in this skill is
  destroying the context (`doca_ctx_stop` →
  `doca_ctx_destroy`) and re-running [`## configure`](#configure)
  with corrected parameters.
- **storage-stack design.** How to layer replication on top of
  EC, how to schedule rebuilds, how to choose N + K per
  durability target across many nodes, how to rebalance after
  node failure — outside this skill. EC is one primitive
  inside the user's storage stack; the stack design lives
  upstream of this skill.
- **kernel-level driver install / firmware burn.** EC depends
  on the underlying ConnectX firmware and BlueField BFB; if
  the debug ladder lands on a driver-layer issue
  (`DOCA_ERROR_DRIVER` from an EC submit, repeated accelerator
  errors in `dmesg`), the fix is via `mlxconfig` /
  `mlxfwreset` / re-imaging the BFB, all of which belong to
  [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
  layer 5.

## Command appendix

Every command below is **cross-cutting on DOCA Erasure Coding**
— it answers a recurring class of question that comes up in the
verbs above. The agent should treat the *class* as load-bearing;
the worked example is a single instance. Run-as user is the
unprivileged user unless noted. Sudo is called out per row.

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
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-erasure-coding` | `## configure` step 1; `## build` slot 4 | What is the build-time DOCA EC version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-erasure-coding` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `doca_caps --list-devs` | `## configure` step 2; `## run` step 1 | Which devices on this host can be used as a `doca_dev` for EC? | One row per visible device with PCIe address and capability flags; the agent must still run `doca_ec_cap_*` per-device to confirm per-task and matrix-type support |
| `doca_caps --version` | `## configure` step 1; `## test` step 1 | What is the *runtime* DOCA version on this host? | A semver string matching `pkg-config --modversion doca-erasure-coding` |
| `ls /opt/mellanox/doca/samples/doca_erasure_coding/` | `## modify` slot 1 | Which EC samples ship in this install, and which is the closest starting point? | A list of sample directories named after the task direction they demonstrate (create / recover / update) |
| `cat /opt/mellanox/doca/applications/VERSION` | `## configure` step 1; `## debug` layer 1 | What does the install tree itself claim its version is? | A semver string matching the other two version sources |
| `dmesg | tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last EC call? | Empty or recent benign messages. Repeated mlx5 / accelerator errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 3 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition and every task submission. Silence after submission = PE not progressed |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the EC-specific rows on top.
