# DOCA SHA workflows

**Where to start:** The verbs run `configure → build → modify → run
→ test → debug`. Skip ahead only when the user is already past a
verb. The `## test` verb is an iterative loop (cap check →
known-vector smoke → small-bulk → full-bulk → loop back if the
algorithm or buffer sizing changed), not a one-shot pass — see the
eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the underlying capability surface,
algorithm enums, one-shot-vs-partial task split, capability-query
rules, error taxonomy, observability, and safety / path-selection
policy, see [CAPABILITIES.md](CAPABILITIES.md). For the cross-library
DOCA patterns layered under everything below (the universal
lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, the
modify-a-shipped-sample workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through the
steps in order, verifying preconditions before recommending the
next call.

## configure

Goal: bring up a DOCA SHA context on a host or BlueField and
confirm the device's accelerator supports the algorithm and buffer
sizes the user actually intends to use.

Steps the agent should walk the user through:

1. **Confirm the installed DOCA version.** Use the procedure in
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
   Quote the version observed (`pkg-config --modversion doca-sha`,
   then `doca_caps --version`); do not assume "latest".
2. **Discover the device capability surface for SHA.** Run
   `doca_caps --list-devs` to see which devices have SHA capability,
   then run the per-`doca_devinfo` `doca_sha_cap_*` queries against
   the candidate device. Record at minimum:
   `doca_sha_cap_task_hash_get_supported(devinfo, algorithm)` and
   `doca_sha_cap_task_partial_hash_get_supported(devinfo, algorithm)`
   for each algorithm the user is considering — these two folded
   calls cover both task-support and algorithm-support; there is
   no separate `_is_supported` or `doca_sha_cap_is_algorithm_supported`
   cap query in the public header.
   Also record `doca_sha_cap_get_min_dst_buf_size(devinfo, algorithm)`
   and `doca_sha_cap_get_max_src_buf_size(devinfo)`. The capability
   surface to compare against lives in
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes).
3. **Decide whether to offload at all.** Per the path-selection
   bullets in [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy),
   doca-sha is the right answer only when the input is bulk,
   repeated, or already pinned in `doca_mmap` memory. For a tiny
   one-shot hash, recommend CPU — do not invent a doca-sha use
   case the user did not ask for.
4. **Pick the task type.** One-shot (`doca_sha_task_hash`) if the
   input fits in `max_src_buf_size` and is available in full at
   submit time; partial (`doca_sha_task_partial_hash`) if the input
   is larger or arrives in chunks. The pick decides the next-step
   code shape; the trade-off table lives in
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes).
   Do not pick *for* the user when their intent is ambiguous — ask.
5. **Configure the SHA instance.** Mandatory before
   `doca_ctx_start()`: enable at least one task type
   (`doca_sha_task_hash_set_conf` and/or
   `doca_sha_task_partial_hash_set_conf`); set source mmap
   permissions (`doca_mmap_set_permissions` to include
   `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY`); set destination mmap
   permissions (`DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`); size the
   destination buffer to at least
   `doca_sha_cap_get_min_dst_buf_size(devinfo, algorithm)`. Per the
   matrix in [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy).
6. **Sanity check before any task submission.** Confirm with the
   user: which algorithm, which task type, source-buffer size, and
   destination-buffer size. Hash a known input (a published NIST
   SHA test vector) and verify the digest matches before any user
   data flows. If any step fails with a `DOCA_ERROR_*`, route
   through the error taxonomy in
   [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy)
   before retrying.

For the canonical DOCA universal lifecycle that underlies steps 4-6,
see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill adds the SHA overlay; do not re-explain the lifecycle
here.

## build

Goal: produce a binary that links DOCA SHA against the user's
installed DOCA, using the canonical cross-library build pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson or
CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the SHA-specific overlay:

| Slot | Value for SHA | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-sha` | The library's `.pc` file installed by the DOCA host packages |
| Required runtime libs | `libdoca-common`, `libdoca-sha`, plus whatever `pkg-config --libs doca-sha` resolves to | SHA depends on Core; the link line should not pull in unrelated DOCA libraries |
| Header check | The public header that `pkg-config --cflags` for this artifact resolves to actually exists on disk at the path pkg-config reports (do not hardcode the include path) | If `pkg-config --cflags doca-sha` resolves but the include is missing, the install is partial |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-sha`; never hardcode in build files | Cross-version build/runtime mixing breaks per [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; the FFI wrapper layer is the language-specific
binding and is out of scope for this skill — but the four slots
above are still the load-bearing inputs the wrapper needs.

## modify

Goal: take a shipped DOCA SHA sample as the verified starting
point and apply a minimum-diff modification to express the user's
intent.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The SHA-specific overlay is the *modify-from-sample
schema fill* — the five slots the agent must elicit from the user
before recommending any code-level edit:

| Slot | What the agent asks the user | SHA-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_sha/`? Or the File Integrity reference application under `/opt/mellanox/doca/applications/`? | Pick the closest in *task direction* (one-shot vs partial / incremental) to the user's intent. Do NOT bridge across both axes — a smaller diff is always safer than a re-architecture |
| 2. Task type added or removed | Which task type from the two? | Each added type needs its own `doca_sha_task_*_set_conf` call before `doca_ctx_start()`, plus its matching cap-query in [`## configure`](#configure) step 2 |
| 3. Algorithm change | Switching from SHA-256 to SHA-1 or SHA-512? | Re-run `doca_sha_cap_task_hash_get_supported(devinfo, new_algorithm)` (and `_task_partial_hash_get_supported(devinfo, new_algorithm)` if using partial-hash), and `doca_sha_cap_get_min_dst_buf_size(devinfo, new_algorithm)`; the digest size and minimum destination buffer change per algorithm |
| 4. Buffer-size changes | Source or destination buffer size changing? | Source must be ≤ `doca_sha_cap_get_max_src_buf_size(devinfo)`; destination must be ≥ `doca_sha_cap_get_min_dst_buf_size(devinfo, algorithm)`; over-broad permissions are a silent security regression |
| 5. Streaming model change | Adding chunk-by-chunk streaming on top of an existing one-shot sample? | This is a re-architecture, not a tweak. If yes, recommend the user start from the partial-hash sample instead of patching a one-shot sample over |

The agent emits an *intent description + the five filled slots*;
the *actual* unified diff against the sample source is produced by
the modify-from-sample renderer (deferred to a future round). Until
the renderer ships, the agent must walk the user through the diff
line-by-line against the sample source they read on disk, and have
the user paste back the result for validation.

## run

Goal: actually execute the built binary against the user's
installed DOCA on a host or BlueField, with a real input.

Steps the agent should walk the user through:

1. **Confirm the device is reachable.** SHA runs on a single side
   (no peer); the only env-side requirement is that the `doca_dev`
   the binary opens corresponds to a device whose accelerator the
   user expects. Mismatched `doca_dev` selection (opening a NIC
   without SHA accelerator support) returns
   `DOCA_ERROR_NOT_SUPPORTED` at task submit, not at open.
2. **Run the known-vector smoke first.** A binary that hashes one
   short input with a published digest and compares the output is
   the cheapest correctness signal. Do not bulk-hash before this
   passes.
3. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace` for
   the first run (see [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the lifecycle and task-submit
   transitions visible on first failure.
4. **Capture the completion events on the PE.** A run that produces
   no completion events but doesn't error is almost always a missed
   `doca_pe_progress()` call. Confirm the progress engine is being
   driven on the main thread.

## test

Goal: prove the configured SHA context can actually produce correct
digests at the user's intended throughput, on the user's hardware,
and that the algorithm and buffer sizing were right.

> **Performance harness routing.** For *throughput / latency
> measurement* on the configured SHA context (or for cross-library
> comparison against the other DOCA crypto primitives), route the
> user to [`doca-bench TASKS.md ## test`](../../tools/doca-bench/TASKS.md#test)
> — `doca-bench` is the cross-library performance harness with
> documented warm-up / steady-state / outlier semantics, and it
> explicitly supports SHA. The iteration loop below stays the
> *correctness* harness; `doca-bench` is the *performance* harness.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the algorithm, the buffer sizing, the task type, or the
permission set. The loop terminates when either (a) the user's
intended hash workload produces correct digests end-to-end with
acceptable throughput, or (b) the agent has narrowed the failure
cause to a layer outside DOCA SHA itself (driver / firmware /
device) and escalated to the matching skill.

Iteration shape:

1. **Capability re-check.** Re-run
   `doca_sha_cap_task_hash_get_supported(devinfo, algorithm)`
   (and `_task_partial_hash_get_supported(devinfo, algorithm)`),
   `doca_sha_cap_get_min_dst_buf_size`, and
   `doca_sha_cap_get_max_src_buf_size` against
   the active `doca_devinfo`. If any return false / unexpected →
   that's the answer; the user's device or DOCA version does not
   support the requested config. Update the intent or update the
   install.
2. **Permission cross-check.** Compare the configured source +
   destination mmap permissions against the matrix in
   [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy).
   Mismatches surface as `DOCA_ERROR_NOT_PERMITTED` on the first
   task submission, not at configure time.
3. **Known-vector smoke.** Hash one short fixed input (e.g. the
   empty string, "abc", or the million-`a` NIST SHA test vector)
   and compare the accelerator's digest against the published
   digest byte-for-byte. If the digests differ, the configuration
   is wrong — do not proceed to bulk input.
4. **Completion drain.** Confirm completion events arrive on the
   PE for every submitted task. *Submitted but no completion* is
   the most expensive class of bug to discover late; confirm it on
   the known-vector smoke before bulk submissions.
5. **Bulk one-shot test.** If the user intends one-shot hashing,
   submit a series of one-shot tasks (small inputs first, then
   sizes approaching `max_src_buf_size`) and verify the digests
   against a CPU SHA reference (OpenSSL or similar). Throughput
   numbers come from this step; correctness comes from step 3.
6. **Partial-hash test (if used).** If the user intends partial
   / incremental hashing, submit a sequence of intermediate
   chunks followed by the finalize call; verify the final digest
   against the equivalent one-shot digest (or against a CPU
   reference). Order matters here: marking the task final on the
   first buffer (before any intermediate submit) is **undefined
   behavior**, not a defined error — if a single buffer is all you
   have, use a non-partial `doca_sha_task_hash` instead. That is
   the most common partial-hash bug.
7. **Negative test.** Once the positive path works, intentionally
   request an algorithm the device should NOT support (per step 1)
   and confirm the failure is the expected
   `DOCA_ERROR_NOT_SUPPORTED`. This validates the agent's
   capability-discovery is itself correct.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on an algorithm we expected to work | The doc page lists the algorithm but the cap query returns false | The agent quoted the *library* surface; the *device* capability per `doca_devinfo` is the real gate. Re-narrow to the device-level query. |
| `DOCA_ERROR_INVALID_VALUE` on first submit | Destination buffer is smaller than `_get_min_dst_buf_size`, OR source buffer is larger than `_get_max_src_buf_size` | Re-size the buffer using the cap-query output. The error is sizing-vs-cap mismatch, not corruption. |
| Known-vector smoke produces a wrong digest | Configuration accepted but output mismatches the published vector | Algorithm mis-selection (asked for SHA-256, configured SHA-1) or endianness assumption in the comparison. Re-check the algorithm enum in the `_alloc_init` call before any other diagnosis. |
| Submitted task produces no completion | `doca_task_submit()` returned `DOCA_SUCCESS`; the PE produces nothing | The PE is not being progressed. Add a `doca_pe_progress()` call in the main loop. |
| Partial-hash misbehaves when finalized on the first buffer | The task was marked final before any intermediate chunk was submitted | This is **undefined behavior**, not a defined `DOCA_ERROR_*` — if a single buffer is all you have, use a non-partial `doca_sha_task_hash` instead; otherwise submit (and complete) at least one intermediate chunk before marking the task final. |
| Bulk submit returns `DOCA_ERROR_AGAIN` | First N submissions succeed, then `AGAIN` | The task queue is full. Drain completions between bursts via `doca_pe_progress()`, or raise the configured queue depth at configure time. |

Loop termination: stop iterating once two consecutive iterations of
the same kind do not change anything — that means the cause is
below DOCA SHA. Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured cap-query snapshot + known-vector diff as
evidence.

## debug

Goal: when a DOCA SHA call returns a `DOCA_ERROR_*` (or the program
produces no completion event), narrow the cause to a specific layer
and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending SHA-specific
fixes. This skill's overlay names the SHA-specific manifestation at
layers 5 (runtime) and 6 (program):

**Layer 5 (runtime) — SHA overlay.**

- Walk the lifecycle: was the context started? Was the task
  enabled before start (`doca_sha_task_*_set_conf` before
  `doca_ctx_start()`)? Submitting before the task is enabled
  returns `DOCA_ERROR_BAD_STATE`, not a clear symptom.
- Confirm the PE is being progressed. *No completion events* is
  almost always a missing `doca_pe_progress()` in the user's main
  loop.
- Confirm both the source mmap and the destination mmap are still
  alive at submit time. Destroying either before `doca_ctx_destroy`
  is a use-after-free that surfaces as `DOCA_ERROR_BAD_STATE` from
  subsequent calls.

**Layer 6 (program) — SHA overlay.**

- Buffer-sizing matrix: the most common SHA program-layer bug is a
  destination buffer smaller than
  `doca_sha_cap_get_min_dst_buf_size(devinfo, algorithm)` —
  surfaces as `DOCA_ERROR_INVALID_VALUE` at submit. The matching
  source-buffer error is a source larger than
  `doca_sha_cap_get_max_src_buf_size(devinfo)`; the fix there is
  to switch to `doca_sha_task_partial_hash`.
- Algorithm enum drift: an `_alloc_init` call that quotes a
  `DOCA_SHA_ALGORITHM_*` enum the cap query returns false for
  returns `DOCA_ERROR_NOT_SUPPORTED`. Re-run the cap query against
  the active `doca_devinfo`; do not assume from prior installs.
- Partial-hash ordering: marking a `doca_sha_task_partial_hash`
  task final on the first buffer (before any intermediate submit)
  is **undefined behavior**, not a defined error — use a
  non-partial `doca_sha_task_hash` instead when a single buffer is
  all you have. Walk the user's submit sequence chunk-by-chunk;
  at least one intermediate chunk must be submitted before the
  task is marked final.
- Known-vector mismatch: if the digest is the wrong length, the
  algorithm enum is wrong. If the digest is the right length but
  the bytes don't match the published vector, the input buffer
  contents differ from what the user thinks (off-by-one length, a
  stray trailing newline, or an endian assumption on the
  comparison).

Once the layer is identified, route to the matching debug verb on
the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); version to
[`doca-version ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows so
the agent does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification, `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the install-tree
  layout in
  [doca-public-knowledge-map ## Layout of an installed DOCA package](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed.
- **deploy.** Deploying SHA-using applications at scale (file
  integrity workers across many hosts, content-addressed storage
  daemons, Kubernetes operator workflows) — out of scope for Phase
  1 and reserved for a future platform skill. For single-host
  first-run testing, the right verb in this skill is `## run`; do
  not invent a "deploy" workflow.
- **rollback.** Coordinated rollback of SHA-using applications
  across many hosts — out of scope. For a single in-session SHA
  configuration rollback, the right verb in this skill is
  destroying the context (`doca_ctx_stop` → `doca_ctx_destroy`)
  and re-running [`## configure`](#configure) with corrected
  parameters.
- **kernel-level driver install / firmware burn.** SHA depends on
  the underlying ConnectX firmware and BlueField BFB; if the debug
  ladder lands on a driver-layer issue, the fix is via `mlxconfig`
  / `mlxfwreset` / re-imaging the BFB, all of which belong to
  [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer 5.

## Command appendix

Every command below is **cross-cutting on DOCA SHA** — it answers a
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
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-sha` | `## configure` step 1; `## build` slot 4 | What is the build-time DOCA SHA version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-sha` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `doca_caps --list-devs` | `## configure` step 2 | Which devices on this host can be used as a `doca_dev` for SHA? | One row per visible device with PCIe address and capability flags; the agent must still run `doca_sha_cap_*` per-device to confirm algorithm support |
| `doca_caps --version` | `## configure` step 1; `## test` step 1 | What is the *runtime* DOCA version on this host? | A semver string matching `pkg-config --modversion doca-sha` |
| `ls /opt/mellanox/doca/samples/doca_sha/` | `## modify` slot 1 | Which SHA samples ship in this install, and which is the closest starting point? | A list of sample directories named after the task pattern they demonstrate |
| `ls /opt/mellanox/doca/applications/` (then locate the File Integrity reference app) | `## modify` slot 1 | Is the File Integrity reference application present as a fuller worked example? | A directory whose contents include a SHA-based file-integrity flow |
| `cat /opt/mellanox/doca/applications/VERSION` | `## configure` step 1; `## debug` layer 1 | What does the install tree itself claim its version is? | A semver string matching the other two version sources |
| `openssl dgst -sha256 <(printf 'abc')` | `## test` step 3 | What is the CPU-reference digest for the known-vector smoke? | The well-known SHA-256 digest of `"abc"`; compare byte-for-byte against the doca-sha output |
| `dmesg | tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last SHA call? | Empty or recent benign messages. Repeated mlx5 / accelerator errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 3 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition and every task submission. Silence after submission = PE not progressed |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the SHA-specific rows on top.
