# DOCA DMA workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop (capability
check → permission check → small-buffer smoke → bulk → loop back
if a cap or permission boundary changed), not a one-shot pass —
see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the DMA capability surface, the
single-task type, the capability-query rule, the path-selection
rule against RDMA / Comch / CPU memcpy, the error taxonomy,
observability, and safety policy, see
[CAPABILITIES.md](CAPABILITIES.md). For where to find docs, the
installed DOCA layout, or release notes, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: bring up a DOCA DMA context on a host or BlueField and
confirm the device, the memcpy task type, and the source /
destination mmap permissions are all in a state where a memcpy
task is meaningful.

Steps the agent should walk the user through:

1. **Confirm the installed DOCA version.** Use the procedure in
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
   Quote the version observed (`pkg-config --modversion doca-dma`,
   then `doca_caps --version`); do not assume "latest". The
   four-way match rule lives in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility);
   if the observed sources disagree, route there before any DMA
   diagnosis.
2. **Discover the device capability surface for DMA.** Run
   `doca_caps --list-devs` (per
   [`doca-caps`](../../tools/doca-caps/SKILL.md)) to see which
   devices are present, then run the per-`doca_devinfo`
   `doca_dma_cap_task_memcpy_*` queries against the candidate
   device:
   `_is_supported`, `_get_max_buf_size`, `_get_max_buf_list_len`.
   Quote the queried values back to the user; do not assume from
   prior installs. The capability matrix to compare against
   lives in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
3. **Confirm DMA is the right library for this copy.** Walk the
   path-selection rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   if the data has to traverse the network, route to
   [`doca-rdma`](../doca-rdma/SKILL.md); if it is producer /
   consumer messaging, route to
   [`doca-comch`](../doca-comch/SKILL.md); if it is tiny and
   one-shot, recommend a CPU `memcpy`. Picking DMA *for* the
   user when the path-selection rule rules it out is a wrong
   answer regardless of how cleanly the rest of the configure
   step goes.
4. **Configure the `doca_dma` instance.** Mandatory before
   `doca_ctx_start()`: enable the memcpy task via
   `doca_dma_task_memcpy_set_conf` (passing the success and
   error completion callbacks plus the max-num-tasks budget).
   Set permissions on the source and destination
   `doca_mmap` regions (`doca_mmap_set_permissions` →
   `doca_mmap_start`) per the matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
   For cross-peer copies, export the source mmap via the
   matching `doca_mmap_export_*` and import it on the peer side.
5. **Sanity check before any task submission.** Confirm with
   the user: which side owns the source mmap, which side owns
   the destination, what length they intend to memcpy, and
   whether that length is within
   `doca_dma_cap_task_memcpy_get_max_buf_size`. If any of those
   are unclear, stop and ask — do not invent.

If any step fails with a `DOCA_ERROR_*`, route through the error
taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

Goal: produce a binary that links DOCA DMA against the user's
installed DOCA, using the canonical cross-library build pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson
or CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the DMA-specific overlay:

| Slot | Value for DMA | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-dma` | The library's `.pc` file installed by the DOCA host packages. Wrong module name = `pkg-config: Package 'doca-dma' was not found` |
| Required runtime libs | `libdoca-common`, `libdoca-dma`, plus whatever `pkg-config --libs doca-dma` resolves transitively | DMA depends on Core; the resolver pulls in the right transitive set |
| Header check | the artifact's public header resolvable under whichever include directory `pkg-config --cflags` reports (do not hardcode the include path — the install layout can move) | If `pkg-config --cflags doca-dma` resolves but the include is missing, the install is partial — route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-dma`; never hardcode in build files | Cross-version build/runtime mixing breaks per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; the FFI wrapper layer is the language-specific
binding and is out of scope for this skill — but the four slots
above are still the load-bearing inputs the wrapper needs.

## modify

Goal: take a shipped DOCA DMA sample (or the DMA Copy reference
application) as the verified starting point and apply a
**minimum-diff modification** to express the user's intent.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The DMA-specific overlay is the *modify-from-sample
schema fill* — the slots the agent must elicit from the user
before recommending any code-level edit:

| Slot | What the agent asks the user | DMA-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_dma/`, or the *DMA Copy* reference application? | Pick the closest in *direction* (intra-host vs host ↔ DPU) and *buffer shape* (single buffer vs scatter-gather list) to the user's intent. A smaller diff is always safer than a re-architecture |
| 2. Buffer length and shape | Single buffer, or scatter-gather list? What total length? | Re-validate against `_get_max_buf_size` and `_get_max_buf_list_len` (per [`## configure`](#configure) step 2); a sample sized for one length will not just work at 8x the length |
| 3. Permission changes | Are source / destination mmap permissions the same as the sample, or do they need to flip (e.g. the user is now the importing peer rather than the exporting peer)? | Refer to the matrix in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy); a wrong flag is a silent first-submit failure |
| 4. Cross-peer vs local | Is the copy intra-host, or does it cross a host ↔ DPU boundary? | Cross-peer requires the `doca_mmap_export_*` step on the source side; intra-host does not. Switching one to the other is a sample swap, not an in-place tweak |
| 5. Build manifest | Keep the sample's existing `meson.build` (which already wires `pkg-config doca-dma`)? | Yes. Do not switch to a hand-rolled Makefile for *"simplicity"* — it removes the version-check rail |

The agent emits an *intent description + the filled slots*; the
*actual* unified diff against the sample source is produced by
the modify-from-sample renderer (deferred to a future round, per
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify)).
Until the renderer ships, the agent must walk the user through
the diff line-by-line against the sample source they read on
disk, and have the user paste back the result for validation.

## run

Goal: actually execute the built binary against the user's
installed DOCA on a host or BlueField, with the source +
destination mmaps materialized and the progress engine driven.

Steps the agent should walk the user through:

1. **Confirm the active `doca_dev` and the active
   `doca_devinfo`.** A binary that links cleanly but never
   produces a memcpy completion is most often opening the wrong
   device. Re-quote the output of `doca_caps --list-devs`
   ([`doca-caps`](../../tools/doca-caps/SKILL.md)) and confirm
   the device the binary opens is the same one the cap-query
   ran against.
2. **For cross-peer copies, run the side that exports first.**
   The exporting peer must have its `doca_mmap_export_*` complete
   before the importing peer can use the export blob as a source
   `doca_buf`. Order matters; mismatched startup is a silent
   `DOCA_ERROR_NOT_PERMITTED` on the first submit.
3. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace`
   for the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the lifecycle and per-task
   submission visible on first failure.
4. **Drive the progress engine.** A run that produces no
   completion events but doesn't error is almost always a
   missed `doca_pe_progress()` call in the user's main loop.
   Confirm the PE is being driven on the side that submits
   tasks.

## test

Goal: prove the configured DMA instance can actually move data
correctly between the source and destination mmaps on the user's
hardware, and that the permission and capability set was sized
right for the user's intended workload.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the capability set, the permission set, the buffer size,
or the buffer-list length. The loop terminates when either (a)
the user's intended copy length flows end-to-end with the
expected per-task completions, or (b) the agent has narrowed the
failure cause to a layer outside DMA itself (driver / firmware /
mmap-export bookkeeping) and escalated to the matching skill.

Iteration shape:

1. **Capability re-check.** Re-run
   `doca_dma_cap_task_memcpy_is_supported`,
   `_get_max_buf_size`, and (if scatter-gather is in use)
   `_get_max_buf_list_len` against the active `doca_devinfo`.
   If `_is_supported` is false → that's the answer; the user's
   device or DOCA version does not support the memcpy task.
   Update the user's intent or update the install.
2. **Permission cross-check.** Compare the configured source /
   destination mmap permission flags against the matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
   For cross-peer copies, confirm the source side exported via
   `doca_mmap_export_*` and the destination side imported the
   blob. Mismatches surface as `DOCA_ERROR_NOT_PERMITTED` on the
   first task submission, not at configure time.
3. **Small-buffer smoke.** Always start with a single small
   memcpy (a few KiB) before bulk. If the small smoke works, the
   permission set and the lifecycle are correct; any subsequent
   oversize-buffer failure narrows cleanly to the cap-query /
   size axis. Skipping this step is the most common reason
   *"bulk DMA fails and we don't know why"*.
4. **Completion drain.** Confirm a per-task completion event
   arrives on the PE for every submitted memcpy. *Submitted but
   no completion* is the most expensive class of bug to discover
   late; confirm it on the smoke pair before adding bulk
   submissions.
5. **Bulk run.** Once the smoke is green, scale up to the user's
   intended length and submission rate. Watch for
   `DOCA_ERROR_AGAIN` (drain the PE before retrying) and for
   `DOCA_ERROR_INVALID_VALUE` (a length boundary was crossed —
   re-narrow to the cap-query axis).

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on a memcpy submit | The cap query at configure time was assumed; the device-level reality differs | Re-run `doca_dma_cap_task_memcpy_is_supported` against the *active* `doca_devinfo`, not a sibling device. The wrong-device case is the most common cause |
| Small-buffer smoke passes; bulk submit returns `INVALID_VALUE` | The bulk length crossed `_get_max_buf_size` (single buffer) or `_get_max_buf_list_len` (scatter-gather) | Fragment at the application layer; the cap is device-bound and not raisable from software |
| Submitted task produces no completion | `doca_task_submit()` returned `DOCA_SUCCESS`; the PE produces nothing | The PE is not being progressed, or the context is not in `RUNNING`. Wire the lifecycle trace per [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug) |
| Cross-peer submit returns `NOT_PERMITTED` after smoke worked locally | The local smoke used a local source mmap; the cross-peer case requires the source to be exported via `doca_mmap_export_*` | Re-walk the cross-peer rule in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) |
| `DOCA_ERROR_AGAIN` on bulk submit | First N submissions succeed, then `AGAIN` | The task queue is full. Drain completions on the PE before re-submitting; or raise `max-num-tasks` at configure time |

Loop termination: stop iterating once two consecutive iterations
of the same kind don't change anything — that means the cause is
below DMA. Escalate to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug) with the
captured layer-1-through-5 evidence.

## debug

Goal: when a DOCA DMA call returns a `DOCA_ERROR_*` (or the
program doesn't make forward progress), narrow the cause to a
specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug). Walk
through it in order — install → version → build → link →
runtime → program → driver — *before* recommending DMA-specific
fixes. This skill's overlay names the DMA-specific manifestation
at layers 5 (runtime) and 6 (program):

**Layer 5 (runtime) — DMA overlay.**

- Walk the lifecycle: was the context started? Were the source
  and destination mmaps started before the first submit? Was a
  task submitted before the context reached `RUNNING`? The wrong
  order returns `DOCA_ERROR_BAD_STATE`, not a clear symptom.
- Confirm the PE is being progressed on the submitting side. *No
  completion events* is almost always a missing
  `doca_pe_progress()` in the user's main loop.
- Confirm `_AGAIN` is not being treated as a bug. The DMA queue
  fills; the contract is that the program drains completions
  before re-submitting, not that the queue is infinite.

**Layer 6 (program) — DMA overlay.**

- Permission matrix: the most common DMA program-layer bug is a
  source / destination mmap permission flag missing or the
  cross-peer export step skipped, surfaced as
  `DOCA_ERROR_NOT_PERMITTED`. Walk the matrix in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  against the user's configured permissions.
- Buffer-size sizing: `DOCA_ERROR_INVALID_VALUE` on submit is a
  cap-vs-payload mismatch, not a hardware failure. Re-read
  `_get_max_buf_size` / `_get_max_buf_list_len` against the
  active `doca_devinfo` and fragment at the application layer.
- Lifecycle order: configure → start → submit → progress → stop
  → destroy. Out-of-order returns `DOCA_ERROR_BAD_STATE`. The
  most common case is destroying a source or destination mmap
  before `doca_ctx_destroy()`.

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
- **deploy.** Deploying DMA-using applications at scale across
  many hosts / DPUs, Kubernetes operator workflows for
  DMA-accelerated workloads, multi-tenant DMA isolation — out of
  scope for Phase 1 and reserved for a future platform skill.
- **rollback.** Coordinated rollback across multiple hosts /
  DPUs — out of scope for Phase 1. For a single in-session DMA
  configuration rollback, the right answer in this skill is
  destroying the context (`doca_ctx_stop` → `doca_ctx_destroy`)
  and re-running [`## configure`](#configure) with the corrected
  parameters; do not invent a "rollback" workflow.
- **kernel-level driver install / firmware burn.** If the debug
  ladder lands on a driver-layer issue
  (`DOCA_ERROR_DRIVER` from a memcpy submit, repeated mlx5
  errors in `dmesg`), the fix is via the env-side skill:
  [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer
  5, then upstream MLNX OFED / firmware documentation reachable
  through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Command appendix

Every command below is **cross-cutting on DOCA DMA** — it answers
a recurring class of question that comes up in the verbs above.
The agent should treat the *class* as load-bearing; the worked
example is a single instance. Run-as user is the unprivileged
user unless noted. Sudo is called out per row.

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
| `pkg-config --modversion doca-dma` | `## configure` step 1; `## build` slot 4 | What is the build-time DOCA DMA version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-dma` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `doca_caps --list-devs` | `## configure` step 2; `## run` step 1 | Which devices on this host can be used as a `doca_dev` for DMA? | One row per visible device with PCIe address and capability flags |
| `doca_caps --version` | `## configure` step 1; `## test` step 1 | What is the *runtime* DOCA version on this host? | A semver string matching `pkg-config --modversion doca-dma` |
| `ls /opt/mellanox/doca/samples/doca_dma/` | `## modify` slot 1 | Which DMA samples ship in this install, and which is the closest starting point? | A list of sample directories named after the buffer / direction pattern they demonstrate |
| `cat /opt/mellanox/doca/applications/VERSION` | `## configure` step 1; `## debug` layer 1 | What does the install tree itself claim its version is? | A semver string matching the other two version sources |
| `dmesg | tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last DMA call? | Empty or recent benign messages. Repeated mlx5 / DMA-engine errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 3 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition and every task submission. Silence after submission = PE not progressed |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the DMA-specific rows on top.
