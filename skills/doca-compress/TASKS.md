# DOCA Compress workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop (cap check →
permission cross-check → round-trip smoke → small-bulk → full-bulk
→ loop back if a task-type or buffer-sizing assumption changed),
not a one-shot pass — see the eval-loop overlay in `## test`
below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the underlying capability surface, the
compress-vs-decompress task split (including decompression-only
as a standalone shape), per-task capability-query rules, error
taxonomy, observability, and safety / path-selection policy, see
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

Goal: bring up a DOCA Compress context on a host or BlueField and
confirm the device's accelerator supports the task type(s) the
user actually intends to use, with buffer sizing that respects
the per-task max.

Steps the agent should walk the user through:

1. **Confirm the installed DOCA version.** Use the procedure in
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
   Quote the version observed (`pkg-config --modversion
   doca-compress`, then `doca_caps --version`); do not assume
   "latest". The four-way match rule lives in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility);
   if the observed sources disagree, route there before any
   Compress diagnosis.
2. **Discover the device capability surface for Compress.** Run
   `doca_caps --list-devs` (per
   [`doca-caps`](../../tools/doca-caps/SKILL.md)) to see which
   devices are visible, then run the per-`doca_devinfo`
   `doca_compress_cap_*` queries against the candidate device.
   Record at minimum:
   `doca_compress_cap_task_compress_deflate_is_supported(devinfo)`,
   `doca_compress_cap_task_decompress_deflate_is_supported(devinfo)`,
   and the matching
   `doca_compress_cap_task_compress_deflate_get_max_buf_size(devinfo)`
   /
   `doca_compress_cap_task_decompress_deflate_get_max_buf_size(devinfo)`
   for each task the user is enabling. The capability surface to
   compare against lives in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
3. **Decide whether to offload at all.** Per the size-threshold
   path-selection bullets in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   doca-compress is the right answer only when the input is bulk
   (rule of thumb: ≥ a few KiB), repeated, or already pinned in
   `doca_mmap` memory. For a tiny one-shot input (e.g. a
   64-byte buffer), recommend CPU `zlib` / `zstd` — do not invent
   a doca-compress use case the user did not ask for. For
   non-DEFLATE algorithms, doca-compress is not the answer (the
   public task names commit it to DEFLATE).
4. **Pick the task type(s) to enable.** Compress
   (`doca_compress_task_compress_deflate`) for outbound encoding;
   decompress (`doca_compress_task_decompress_deflate`) for
   inbound decoding; both for a round-trip flow. **A
   decompress-only consumer is a fully valid shape** — do not
   auto-enable the compress task on a user who only described an
   inbound decoding need. The trade-off table lives in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
5. **Configure the Compress instance.** Mandatory before
   `doca_ctx_start()`: enable at least one task type
   (`doca_compress_task_compress_deflate_set_conf` and/or
   `doca_compress_task_decompress_deflate_set_conf`, each with
   its success and error completion callbacks plus its
   max-num-tasks budget); set source mmap permissions
   (`doca_mmap_set_permissions` to include
   `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY`); set destination mmap
   permissions (`DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`); size the
   source buffer to ≤ the matching `_get_max_buf_size` and the
   destination buffer to the worst-case output size (a compress
   destination may need to be slightly larger than the source for
   poorly-compressible inputs; a decompress destination may need
   to be substantially larger than the compressed source). Per
   the matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
6. **Sanity check before any task submission.** Confirm with the
   user: which task type(s), source-buffer size, destination-buffer
   size, and whether the intended source size is within the
   per-task `_get_max_buf_size`. Run a round-trip smoke (compress
   one short input and decompress the output, or — for a
   decompress-only consumer — decompress one small DEFLATE blob
   produced by `zlib`) and verify the result matches the original
   byte-for-byte before any user data flows. If any step fails
   with a `DOCA_ERROR_*`, route through the error taxonomy in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   before retrying.

For the canonical DOCA universal lifecycle that underlies steps
4-6, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill adds the Compress overlay; do not re-explain the
lifecycle here.

## build

Goal: produce a binary that links DOCA Compress against the
user's installed DOCA, using the canonical cross-library build
pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson
or CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the Compress-specific overlay:

| Slot | Value for Compress | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-compress` | The library's `.pc` file installed by the DOCA host packages. Wrong module name = `pkg-config: Package 'doca-compress' was not found` |
| Required runtime libs | `libdoca-common`, `libdoca-compress`, plus whatever `pkg-config --libs doca-compress` resolves transitively | Compress depends on Core; the link line should not pull in unrelated DOCA libraries |
| Header check | The public header that `pkg-config --cflags` for this artifact resolves to actually exists on disk at the path pkg-config reports (do not hardcode the include path) | If `pkg-config --cflags doca-compress` resolves but the include is missing, the install is partial — route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-compress`; never hardcode in build files | Cross-version build/runtime mixing breaks per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; the FFI wrapper layer is the language-specific
binding and is out of scope for this skill — but the four slots
above are still the load-bearing inputs the wrapper needs.

## modify

Goal: take a shipped DOCA Compress sample (or the File
Compression reference application) as the verified starting point
and apply a **minimum-diff modification** to express the user's
intent.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The Compress-specific overlay is the
*modify-from-sample schema fill* — the slots the agent must
elicit from the user before recommending any code-level edit:

| Slot | What the agent asks the user | Compress-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_compress/`? Or the File Compression reference application under `/opt/mellanox/doca/applications/`? | Pick the closest in *direction* (compress vs decompress vs both) to the user's intent. Do NOT bridge across both directions — a smaller diff is always safer than a re-architecture; a decompress-only consumer should start from a decompress sample, not from a compress sample with the encoding ripped out |
| 2. Task type(s) added or removed | Which task type(s) from the two? | Each enabled task needs its own `doca_compress_task_*_set_conf` call before `doca_ctx_start()`, plus its matching cap-query in [`## configure`](#configure) step 2. A decompress-only consumer should *remove* the compress-task setup the sample wires up, not leave it dangling |
| 3. Buffer-size changes | Source or destination buffer size changing? | Source must be ≤ the matching `_get_max_buf_size` for the task type; destination must be sized for the worst-case output (compress: input that does not compress well can produce slightly more bytes than the input; decompress: output can be substantially larger than the compressed input). Over-broad permissions are a silent security regression |
| 4. Round-trip vs one-way pipeline | Is the user running a round-trip (compress → decompress in the same process) or a one-way pipeline (compress here, decompress elsewhere; or decompress only)? | Round-trip enables both tasks on the same context and round-trip-validates locally; one-way pipelines enable only the matching task and validate against a CPU reference (zlib for the missing side) |
| 5. Build manifest | Keep the sample's existing `meson.build` (which already wires `pkg-config doca-compress`)? | Yes. Do not switch to a hand-rolled Makefile for *"simplicity"* — it removes the version-check rail |

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

1. **Confirm the device is reachable.** Compress runs on a
   single side (no peer); the only env-side requirement is that
   the `doca_dev` the binary opens corresponds to a device whose
   accelerator the user expects. Mismatched `doca_dev` selection
   (opening a NIC without the requested Compress task on its
   accelerator) returns `DOCA_ERROR_NOT_SUPPORTED` at task
   submit, not at open. Re-quote the output of `doca_caps
   --list-devs` ([`doca-caps`](../../tools/doca-caps/SKILL.md))
   and confirm the device the binary opens is the same one the
   cap-query ran against.
2. **Run the round-trip (or decompress-known-fixture) smoke
   first.** A binary that compresses one short input and
   decompresses the output, comparing to the original — or, for
   a decompress-only consumer, decompresses one small DEFLATE
   fixture produced by `zlib` and compares to the original — is
   the cheapest correctness signal. Do not bulk-encode or
   bulk-decode before this passes.
3. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace`
   for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the lifecycle and task-submit
   transitions visible on first failure.
4. **Capture the completion events on the PE.** A run that
   produces no completion events but doesn't error is almost
   always a missed `doca_pe_progress()` call. Confirm the
   progress engine is being driven on the main thread.

## test

Goal: prove the configured Compress context can actually produce
correct encoded / decoded output at the user's intended
throughput, on the user's hardware, and that the task selection,
buffer sizing, and permission set were right.

> **Performance harness routing.** For *throughput / latency
> measurement* on the configured Compress context (or for cross-library
> comparison against the other DOCA crypto primitives), route the
> user to [`doca-bench TASKS.md ## test`](../../tools/doca-bench/TASKS.md#test)
> — `doca-bench` is the cross-library performance harness with
> documented warm-up / steady-state / outlier semantics, and it
> explicitly supports Compress. The iteration loop below stays the
> *correctness* harness; `doca-bench` is the *performance* harness.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the task type, the buffer sizing, the permission set, or
the path-selection assumption. The loop terminates when either
(a) the user's intended workload produces correct output
end-to-end with acceptable throughput, or (b) the agent has
narrowed the failure cause to a layer outside DOCA Compress
itself (driver / firmware / device) and escalated to the
matching skill.

Iteration shape:

1. **Capability re-check.** Re-run
   `doca_compress_cap_task_compress_deflate_is_supported(devinfo)`,
   `_decompress_deflate_is_supported(devinfo)`, and the matching
   `_get_max_buf_size` queries against the active `doca_devinfo`.
   If any required task type returns false / unexpected → that's
   the answer; the user's device or DOCA version does not
   support the requested config. Update the intent or update the
   install.
2. **Permission cross-check.** Compare the configured source +
   destination mmap permissions against the matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
   Mismatches surface as `DOCA_ERROR_NOT_PERMITTED` on the first
   task submission, not at configure time.
3. **Round-trip smoke (or decompress-known-fixture smoke).**
   Compress one short fixed input and decompress the output,
   comparing to the original byte-for-byte; OR, for a
   decompress-only consumer, decompress one small DEFLATE blob
   produced by a published CPU encoder (`zlib`) and compare to
   the original byte-for-byte. If the comparison fails, the
   configuration is wrong — do not proceed to bulk input.
4. **Completion drain.** Confirm completion events arrive on the
   PE for every submitted task. *Submitted but no completion* is
   the most expensive class of bug to discover late; confirm it
   on the round-trip smoke before bulk submissions.
5. **Small-bulk test.** Submit a small series of tasks at sizes
   approaching the per-task `_get_max_buf_size` but staying
   under it; verify the outputs decompress back to the inputs
   (round-trip) or match a CPU reference. Throughput numbers
   come from this step; correctness comes from step 3.
6. **Full-bulk run.** Once the small-bulk passes, scale up to
   the user's intended size and submission rate. Watch for
   `DOCA_ERROR_AGAIN` (drain the PE before retrying) and for
   `DOCA_ERROR_INVALID_VALUE` (a per-task source-size or
   destination-size boundary was crossed — re-narrow to the
   cap-query axis or to the worst-case-output-sizing axis).
7. **Negative test.** Once the positive path works, intentionally
   request a task type the device should NOT support (per step
   1), or submit a source buffer larger than `_get_max_buf_size`
   for the configured task, and confirm the failure is the
   expected `DOCA_ERROR_NOT_SUPPORTED` / `DOCA_ERROR_INVALID_VALUE`.
   This validates the agent's capability-discovery is itself
   correct.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on a task we expected to work | The doc page lists the task but the cap query returns false | The agent quoted the *library* surface; the *device* capability per `doca_devinfo` is the real gate. Re-narrow to the device-level query for the specific task type. |
| `DOCA_ERROR_INVALID_VALUE` on first submit | Source buffer is larger than `_get_max_buf_size` for the task, OR destination buffer is too small for the worst-case output | Re-size the buffer using the cap-query output (source) or against a worst-case calculation (destination). The error is sizing-vs-cap mismatch, not corruption. |
| Round-trip smoke produces wrong output | Configuration accepted but the decompressed output does not match the original | Wrong task enabled (asked for decompress, configured compress), wrong buffer wired (source / destination swapped), or the smoke fixture is corrupt. Re-check the task type in the `_alloc_init` call and the source / destination buffer wiring before any other diagnosis. |
| Submitted task produces no completion | `doca_task_submit()` returned `DOCA_SUCCESS`; the PE produces nothing | The PE is not being progressed. Add a `doca_pe_progress()` call in the main loop. |
| Bulk submit returns `DOCA_ERROR_AGAIN` | First N submissions succeed, then `AGAIN` | The task queue is full. Drain completions between bursts via `doca_pe_progress()`, or raise the configured queue depth at configure time. |
| Smoke passes; small-bulk is slower than CPU `zlib` end-to-end | The per-call DMA-to-accelerator round-trip dominates per-input time at this size | Re-walk the size-threshold path-selection rule in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes). At small input sizes, CPU compression is the right answer; the cap query reports max input but not the min-input floor. |

Loop termination: stop iterating once two consecutive iterations
of the same kind don't change anything — that means the cause is
below DOCA Compress. Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured cap-query snapshot + round-trip diff as
evidence.

## debug

Goal: when a DOCA Compress call returns a `DOCA_ERROR_*` (or the
program produces no completion event), narrow the cause to a
specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending Compress-specific
fixes. This skill's overlay names the Compress-specific
manifestation at layers 5 (runtime) and 6 (program):

**Layer 5 (runtime) — Compress overlay.**

- Walk the lifecycle: was the context started? Were the
  task(s) enabled before start (`doca_compress_task_*_set_conf`
  before `doca_ctx_start()`)? Submitting before the requested
  task is enabled returns `DOCA_ERROR_BAD_STATE`, not a clear
  symptom.
- Confirm the PE is being progressed. *No completion events* is
  almost always a missing `doca_pe_progress()` in the user's
  main loop.
- Confirm both the source mmap and the destination mmap are
  still alive at submit time. Destroying either before
  `doca_ctx_destroy` is a use-after-free that surfaces as
  `DOCA_ERROR_BAD_STATE` from subsequent calls.

**Layer 6 (program) — Compress overlay.**

- Buffer-sizing matrix: the most common Compress program-layer
  bugs are (a) a source buffer larger than the matching
  `_get_max_buf_size` for the enabled task — surfaces as
  `DOCA_ERROR_INVALID_VALUE` at submit; fragment at the
  application layer — and (b) a destination buffer too small for
  the worst-case output. For compress, worst-case output can be
  slightly larger than the input on incompressible data; for
  decompress, the decompressed output can be substantially
  larger than the compressed input.
- Task-type drift: an `_alloc_init` call that asks for a task
  type the cap query returns false for returns
  `DOCA_ERROR_NOT_SUPPORTED`. Re-run the matching
  `doca_compress_cap_task_*_is_supported` against the active
  `doca_devinfo`; do not assume from prior installs.
- Source / destination swap: a configuration that wires the
  encoded buffer as the source of a *compress* task (or the raw
  buffer as the source of a *decompress* task) will produce a
  completion that decodes back to garbage on round-trip. Walk
  the buffer wiring against the task direction in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  before any other diagnosis.
- Round-trip mismatch: if the decompressed output is the wrong
  length, the task type is likely wrong (asked for compress,
  configured decompress, or vice versa). If the length is right
  but the bytes don't match the original, the source / destination
  wiring is wrong or the fixture used for the smoke is itself
  corrupt.

Once the layer is identified, route to the matching debug verb
on the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); version
to [`doca-version ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

**5-phase universal debug-loop instantiation (Compress).** Layer
identification above is phase 1 of the
[universal debug-loop contract](../../doca-debug/CAPABILITIES.md#universal-debug-loop-contract).
The agent MUST walk the remaining four phases on every Compress
debug answer before declaring done:

1. **Layer identification** — above (capability / lifecycle /
   data-path).
2. **Triple capture (READ-ONLY).** Capture (a) capability map:
   `doca_compress_cap_task_compress_deflate_is_supported(devinfo)`
   + `doca_compress_cap_*` max buffer / algorithm support on the
   actual devinfo, (b) submitted-task vs completed-task counters
   from `doca_pe_progress` callback log, (c) DOCA log at
   `DOCA_LOG_LEVEL=DEBUG` for the offending task with the
   request / response buffer addresses. The triple is the
   rollback target.
3. **Single-variable mutation SMALLER than the original
   change.** Examples: shrink the input buffer to a single
   compressible payload (not the production batch); switch
   `doca_mmap` to a fresh pinned region (not the production
   pool); reduce parallelism to one outstanding task (not the
   pipeline depth). Larger mutations void the experiment.
4. **Re-capture and compare.** Re-run the triple; the
   request/response counter diff IS the evidence.
5. **Exit with named green signal OR escalate.** Green = one
   round-trip (compress → decompress) returns bytes-equal at
   the source. If two consecutive iterations don't change
   anything, escalate via the layer route table above with the
   captured triple.

## rollback

Compress contexts are stateful (started context + registered
mmap regions + in-flight tasks on the progress engine) and the
agent's failure mode is to leave in-flight tasks dangling on a
context that is being torn down, returning
`DOCA_ERROR_BAD_STATE` on the next program run. The
[universal verification contract](../../doca-setup/CAPABILITIES.md#universal-verification-contract)
step 1 (preconditions) requires *"the rollback path is
documented"* on every change-recommending answer; this is the
Compress instantiation.

**Snapshot before mutate.** Before any change-recommending
Compress answer, capture (a) the started-context registration
map (mmap region IDs + task-type conf flags from `## configure`
step 3), (b) the outstanding-task count from
`doca_pe_progress`, and (c) the input/output buffer ownership
list. The triple IS the rollback target.

1. **Drain outstanding tasks FIRST.** Walk
   `doca_pe_progress` until the outstanding-task counter is
   zero. Do NOT submit new tasks after rollback intent is
   declared. If the drain stalls (counter not decrementing
   within the bounded debug-loop window), fire the
   [deploy-loop bridge](../../doca-setup/CAPABILITIES.md#deploy-loop-bridge--step-5-not-green-is-the-debug-loop-trigger)
   on the stalled-drain symptom before continuing the rollback.
2. **`doca_ctx_stop` on the Compress context.** Returns
   `DOCA_ERROR_BAD_STATE` if step 1 was skipped — that is
   diagnostic, not a retry trigger; re-walk step 1 with a
   higher-resolution drain log.
3. **Unregister mmap regions in reverse-register order.**
   `doca_mmap_destroy` on every region created with
   `doca_mmap_create_*`; the underlying host buffers may be
   freed after this step.
4. **Destroy the Compress context.** `doca_compress_destroy`.
   The underlying `doca_dev` remains valid and must not be
   torn down by this step.
5. **Re-verify with the shipped round-trip smoke.** Re-run
   the round-trip from [`## test`](#test) step 1 against a
   fresh context to confirm the device + driver path is
   intact post-rollback. If the smoke does not return
   bytes-equal, the rollback corrupted device state —
   surface as a residual gap, do NOT retry.
6. **Document the rollback verb in the verification contract
   preconditions block.** The step 1 line for a Compress add
   reads: *"the rollback path is the five-step reversal in
   [`## rollback`](#rollback); the agent has captured the
   started-context registration map and outstanding-task
   count."* Without that line, the contract is incomplete
   and the agent is NOT eligible to declare done.

The rollback is bounded — on the second non-green re-verify at
step 5, the agent MUST surface the unresolved residual gap
instead of recommending another Compress retry.

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
- **deploy.** Deploying Compress-using applications at scale
  (compress-before-write storage workers across many hosts,
  decompress-on-read network receivers, Kubernetes operator
  workflows) — out of scope for Phase 1 and reserved for a
  future platform skill. For single-host first-run testing, the
  right verb in this skill is `## run`; do not invent a "deploy"
  workflow.
- **rollback.** Coordinated rollback of Compress-using
  applications across many hosts — out of scope. For a single
  in-session Compress configuration rollback, the right verb in
  this skill is destroying the context (`doca_ctx_stop` →
  `doca_ctx_destroy`) and re-running [`## configure`](#configure)
  with corrected parameters.
- **kernel-level driver install / firmware burn.** Compress
  depends on the underlying ConnectX firmware and BlueField BFB;
  if the debug ladder lands on a driver-layer issue
  (`DOCA_ERROR_DRIVER` from a Compress submit, repeated
  accelerator errors in `dmesg`), the fix is via `mlxconfig` /
  `mlxfwreset` / re-imaging the BFB, all of which belong to
  [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer
  5.

## Command appendix

Every command below is **cross-cutting on DOCA Compress** — it
answers a recurring class of question that comes up in the verbs
above. The agent should treat the *class* as load-bearing; the
worked example is a single instance. Run-as user is the
unprivileged user unless noted. Rows that need elevated privileges call that out explicitly.

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
| `pkg-config --modversion doca-compress` | `## configure` step 1; `## build` slot 4 | What is the build-time DOCA Compress version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-compress` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `doca_caps --list-devs` | `## configure` step 2; `## run` step 1 | Which devices on this host can be used as a `doca_dev` for Compress? | One row per visible device with PCIe address and capability flags; the agent must still run `doca_compress_cap_*` per-device to confirm per-task support |
| `doca_caps --version` | `## configure` step 1; `## test` step 1 | What is the *runtime* DOCA version on this host? | A semver string matching `pkg-config --modversion doca-compress` |
| `ls /opt/mellanox/doca/samples/doca_compress/` | `## modify` slot 1 | Which Compress samples ship in this install, and which is the closest starting point? | A list of sample directories named after the task direction they demonstrate (compress, decompress, or both) |
| `ls /opt/mellanox/doca/applications/` (then locate the File Compression reference app) | `## modify` slot 1 | Is the File Compression reference application present as a fuller worked example? | A directory whose contents include a DEFLATE-based file-compression flow |
| `cat /opt/mellanox/doca/applications/VERSION` | `## configure` step 1; `## debug` layer 1 | What does the install tree itself claim its version is? | A semver string matching the other two version sources |
| `printf 'abcabcabcabcabcabc' \| python3 -c 'import sys, zlib; sys.stdout.buffer.write(zlib.compress(sys.stdin.buffer.read()))' \| xxd` | `## test` step 3 | What is the CPU-reference DEFLATE encoding for the round-trip smoke fixture? | A short DEFLATE-encoded byte sequence the agent can compare against the doca-compress compress output, or feed into doca-compress decompress for a decompress-only smoke |
| `dmesg \| tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last Compress call? | Empty or recent benign messages. Repeated mlx5 / accelerator errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 3 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition and every task submission. Silence after submission = PE not progressed |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the Compress-specific rows on top.
