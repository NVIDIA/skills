# DOCA Bench Extension — Tasks

**Where to start:** The verbs that carry real workflow
content for `doca-bench-extension` are `## configure`,
`## build`, `## run`, `## test`, and `## debug`. `## modify`
is substantive for this tool because the operator IS
modifying the shipped reference extension or authoring one
from scratch. `## install` and `## use` carry routing stubs
and a tightly-scoped agent-side workflow.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent
through the task verbs every artifact in this bundle exposes.

## install

The bench-extension framework itself is **shipped pre-built**
with the DOCA install — the reference
`doca_bench_cuda_impl` shared library and the in-tree
`/opt/mellanox/doca/tools/bench_extension/` source tree (which an
operator copies forward into a custom extension) ship with
DOCA when the bench / GPU components are present in the
install profile.

Routing for nearby "install" questions:

- *"`doca-bench` is not on my system."* → that is a parent
  tool install question. Route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  and [`## no-install`](../../doca-setup/TASKS.md#no-install).
- *"The reference `doca_bench_cuda_impl` library is
  missing."* → same routing — the install profile that
  excluded the bench GPU component will have excluded the
  reference extension too.
- *"I need to install the CUDA toolkit for my GPU-side
  extension."* → not a DOCA install question. Route to
  the public CUDA Toolkit documentation on
  `docs.nvidia.com`.
- *"I want to install a newer version of the extension
  headers without upgrading DOCA."* → the bench extension
  surface is versioned WITH DOCA per
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility);
  the `DOCA_EXPERIMENTAL` rebuild rule says there is no
  independent upgrade path. Route the cross-release
  decision through
  [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).

## configure

`configure` for the bench-extension framework is *"commit
to the three-axis decision (extension is genuinely needed
× the reference's API surface fits × the DOCA toolchain
preconditions hold) AND validate the parent-tool
relationship BEFORE any extension build work"*. Skipping
any step is the canonical failure mode.

Steps the agent should walk the user through, in order:

1. **Confirm DOCA is installed and `doca-bench` is healthy.**
   Run [`doca-setup ## test`](../../doca-setup/TASKS.md#test);
   then confirm the parent `doca-bench` is reachable per
   [`doca-bench TASKS.md ## test`](../doca-bench/TASKS.md#test).
   The extension has no value without a working parent.
2. **Re-validate the extension-is-needed decision.** Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   (Extension-as-exit-ramp): walk
   [`doca-bench TASKS.md ## configure`](../doca-bench/TASKS.md#configure)
   FIRST. Confirm no built-in workload mode covers the
   class. The agent must NOT skip this — extensions add
   real cost (toolchain, version churn, experimental
   surface).
3. **Locate the shipped reference extension.** Confirm
   the operator's install contains
   `/opt/mellanox/doca/tools/bench_extension/doca_bench_cuda/`
   (the reference exemplar source tree). The agent must
   ask the operator to confirm the path on their install
   rather than asserting one.
4. **Read the reference exemplar's API surface.** Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   (API surface shape): the shipped `doca_bench_cuda.h`
   on the operator's install is the schema by example.
   Walk the operator through the `DOCA_EXPERIMENTAL`
   entry-point family, the per-workload settings structs,
   the accounting struct, and the lifetime contract
   (the `stop_flag`).
5. **Decide if a NEW extension is needed or if the
   shipped reference suffices.** If the workload is
   already covered by the reference's nop / eth-recv /
   eth-send / eth-bidir kernels (i.e. a GPUNetIO RX / TX
   class workload), the operator may just use the
   reference; no custom extension is needed. If the
   workload is genuinely outside the reference, the
   operator authors a NEW extension by COPYING the
   reference subtree and adapting it.
6. **Commit to the toolchain preconditions.** When the
   extension is GPU-side: the CUDA toolkit version
   compatible with the running DOCA / driver, the GPU
   architecture flags, the GPUNetIO preconditions per
   [`doca-gpunetio`](../../libs/doca-gpunetio/SKILL.md).
   When the extension is non-GPU: the relevant
   primitive's preconditions.
7. **Capture the version stack.** Per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility):
   DOCA version, BlueField / ConnectX generation,
   firmware version, BlueField mode, CUDA toolkit version
   (when applicable), the parent `doca-bench`'s version,
   the shipped reference extension's version (matches the
   DOCA `version`). Partial captures break the
   reproducibility leg of the safety policy.
8. **Sanity check before any build.** Confirm with the
   user: is the workload genuinely outside `doca-bench`'s
   built-in modes? Is the reference exemplar reachable?
   Are the toolchain preconditions met? Is OOB / reset
   access available (when GPU-side)? If any answer is
   unclear, stop and ask.

Do not invent build flags, entry-point function names, or
runtime discovery paths beyond what the shipped reference
demonstrates and what the public DOCA Bench documentation
on `docs.nvidia.com` describes.

## build

`build` for the bench-extension framework is *"build a
versioned shared library that matches the running DOCA's
`soversion`, adapted from the shipped reference exemplar's
`meson.build` and source layout"*. The skill walks the
shape, not a verbatim recipe.

Steps the agent should walk the user through, in order:

1. **Copy the reference exemplar subtree to the
   operator's working tree.** The reference is
   `/opt/mellanox/doca/tools/bench_extension/doca_bench_cuda/` on
   the DOCA install. The operator copies the subtree,
   not edits it in place — the shipped reference must
   stay pristine for cross-check during
   [`## debug`](#debug).
2. **Adapt the `meson.build`.** Per the shipped
   `/opt/mellanox/doca/tools/bench_extension/meson.build` shape:
   - the shared library MUST carry
     `version : doca_version` and
     `soversion : doca_so_version` so the parent loader
     accepts it — this is the
     [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
     rule at build time;
   - the compile args (the reference uses
     `tool_cpp_args = [base_cpp_args, gpu_compile_flags]`)
     reflect what the parent's build system expects;
   - GPU extensions inherit `gpu_dependencies`;
   - `install : true` places the library where the parent
     loader can find it at runtime.
   The agent must NOT invent build flags; the reference
   `meson.build` is the source of truth.
3. **Adapt the source files.** Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   (API surface shape): keep the
   `DOCA_EXPERIMENTAL`-marked entry-point shape, adapt
   the per-workload settings structs / accounting struct
   / lifetime contract for the new workload. The
   reference's `*_init`, `*_device_query`,
   `*_device_synchronize` shape is the lifecycle
   blueprint; new workloads layer their own
   `*_start_<workload>_kernel` entry points.
4. **Build with the matching DOCA / CUDA toolchain.**
   Confirm the CUDA toolkit version (when GPU-side) is
   compatible with the running DOCA driver — mismatches
   surface in
   [`## debug`](#debug) layers 2 and 6.
5. **Confirm the shared library version stamp.** Run
   `readelf -d` or `objdump -p` on the built library to
   confirm `SONAME` carries the expected `soversion`.
6. **STOP here.** Do NOT point the parent `doca-bench` at
   the extension until the smoke-before-bulk loop in
   [`## test`](#test) has passed.

When recording the build for downstream consumers, write
down: the DOCA version, the CUDA toolkit version, the
extension's `meson.build` adaptations relative to the
reference, the build command line, the resulting
`SONAME` / `soversion`, and the install path.

## modify

`modify` for the bench-extension framework is substantive
— extensions are by definition source artifacts the
operator authors or adapts.

**Do not modify the shipped reference exemplar
(`/opt/mellanox/doca/tools/bench_extension/doca_bench_cuda/`) in
place.** It is the pristine baseline the agent
cross-references in [`## debug`](#debug) and on every DOCA
upgrade per the
[`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
`DOCA_EXPERIMENTAL` rebuild rule.

What the agent *does* modify is the operator's COPY of
the reference, per [`## build`](#build) steps 1-3.
Recurring rules for the modification:

- **Keep the API surface shape from the shipped
  reference.** The `DOCA_EXPERIMENTAL` entry-point
  family, per-workload settings struct, accounting
  struct, and lifetime contract (`stop_flag`) are the
  template; the operator adapts the *content* of each,
  not the *shape*.
- **Maintain the `version` / `soversion` stamps.** Per
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility):
  every operator-built extension MUST carry the running
  DOCA's `version` and `soversion` so the parent loader
  accepts it.
- **Re-smoke after every modification.** Per
  [`## test`](#test): every change to entry-point
  signature, settings struct, kernel body, or build flags
  re-opens the smoke. The agent must NOT recommend
  scaling up after a modification without re-running the
  smoke.
- **Cross-link to the right neighbouring skill for the
  workload domain.** GPUNetIO workloads route to
  [`doca-gpunetio`](../../libs/doca-gpunetio/SKILL.md);
  Flow workloads route to
  [`doca-flow`](../../libs/doca-flow/SKILL.md); Comch
  workloads route to
  [`doca-comch`](../../libs/doca-comch/SKILL.md). The
  extension's BODY uses the underlying primitive's API;
  this skill does not duplicate that semantics.

Routing for nearby "modify" questions:

- *"Patch `doca-bench` itself to add a new built-in mode."*
  → out of scope; that is contributor work, not
  external-consumer work. Use the extension framework
  exactly because it lets the operator add workloads
  WITHOUT modifying the parent.
- *"Modify the shipped reference in place."* → no. The
  shipped reference is the cross-check baseline.

## run

The start → smoke → measure flow. The full invocation
surface lives in the public DOCA Bench documentation on
`docs.nvidia.com`; this section names the *shape* of the
flow.

1. **Confirm preconditions.** Per
   [`## configure`](#configure) steps 1-7 and
   [`## build`](#build) steps 1-5. The build product
   exists, has the expected `SONAME`, and is on a path
   the parent loader can find.
2. **Run a no-op invocation first.** Per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   (smoke-before-bulk, twice): point the parent
   `doca-bench` at the extension with the cheapest
   possible workload — the reference exemplar's
   `doca_bench_cuda_start_nop_kernel` is the canonical
   example, and any custom extension SHOULD expose an
   equivalent no-op entry. This proves the parent loaded
   the library, found the registration handshake, and can
   call into it.
3. **Confirm the no-op invocation completed cleanly.**
   Parent logs report the extension loaded; the no-op
   entry returned `DOCA_SUCCESS`; the parent's
   accounting consumed the extension's stats struct.
4. **Run ONE minimal workload invocation.** The smallest
   non-no-op kernel the extension exposes — for the GPU
   reference exemplar, a short-duration eth-recv or
   eth-send kernel with a small `max_rx_pkts` /
   `timeout_ns`. The agent must confirm the `stop_flag`
   discipline before scaling.
5. **Scale up to the operator's full workload.** ONLY
   after steps 2-4 are clean. The agent captures the
   full output per [`## test`](#test) (capture step).

When recording the run for downstream consumers, write
down: the DOCA version, the host, the BlueField /
ConnectX generation, the firmware version, the BlueField
mode, the CUDA toolkit version (when applicable), the
parent `doca-bench`'s version, the extension's
`SONAME` / `soversion`, the parent's full invocation
command, and the extension's per-workload settings struct
contents. Partial captures break the downstream debug
ladder.

## test

The bench-extension `## test` is **the canonical
smoke-before-bulk loop for the framework**. *"Test"* in
this skill means *"prove ONE no-op invocation, then ONE
minimal workload invocation, completes end-to-end before
scaling up to the operator's full workload"*, not
*"unit-test the framework"*.

**`## test` is an iterative loop, not a one-shot pass.**
Every mutation — an extension source edit, a build flag
change, a `meson.build` edit, a DOCA upgrade, a CUDA
toolkit upgrade — re-opens BOTH smokes (the no-op smoke
AND the minimal-workload smoke).

The smoke-before-bulk shape:

1. **Run the no-op smoke.** Per [`## run`](#run) step 2.
2. **Confirm the parent loaded the extension and called
   the no-op entry cleanly.** Parent logs show the load
   event; the no-op entry returned `DOCA_SUCCESS`. If
   either fails, walk [`## debug`](#debug) layers 2-4.
3. **Run the minimal-workload smoke.** Per
   [`## run`](#run) step 4.
4. **Confirm the minimal workload completed cleanly and
   the `stop_flag` discipline holds.** The kernel
   terminated when the parent set the stop flag; the
   accounting struct's `jobs_processed` /
   `bytes_processed` are non-zero and consistent.
5. **Scale up to the operator's full workload** per
   [`## run`](#run) step 5.
6. **Repeat the scaled run multiple times to establish
   variance** when the operator is measuring (not just
   validating loading). A single-iteration result is
   not defensible.
7. **Capture the full version stack alongside every
   reported number** per [`## configure`](#configure)
   step 7.

Eval-loop overlay (rows apply to every extension
invocation, not just one):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → ## debug | The parent cannot find / load the library; walk the load-failed layer, then re-build with corrected `soversion` / install path and re-run step 1 | [`## debug`](#debug) layers 2-3 |
| 2 → ## debug | The parent loaded the library but cannot register the entry points; walk the registration-mismatch layer | [`## debug`](#debug) layer 4 |
| 3 → ## debug | The minimal workload kernel fails or hangs; walk the runtime-call layer and confirm the `stop_flag` discipline | [`## debug`](#debug) layer 5 |
| 5 → step 6 | Per-iteration variance is high on the scaled measurement; the printed average is misleading | [`doca-bench TASKS.md ## debug`](../doca-bench/TASKS.md#debug) (parent's variance handling) |
| any → DOCA upgrade → 1 | Per the `DOCA_EXPERIMENTAL` rebuild rule, every DOCA upgrade re-opens the smoke; the prior smoke is stale | [`## configure`](#configure) step 7 + [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |
| any → extension source edit → 1 | After ANY edit, re-run BOTH smokes; the prior smoke is stale | [`## modify`](#modify) |
| any → CUDA toolkit upgrade → 1 | After a CUDA toolkit upgrade, re-run BOTH smokes; the prior smoke is stale | [`## debug`](#debug) layer 6 |

The agent's rule: every state-changing action on the
extension source, the build flags, the DOCA version, or
the toolchain re-opens BOTH smokes. Saving a stale smoke
is exactly the failure mode this loop is here to prevent.

This skill does **not** ship a "test fixture" pass / fail
oracle for a custom extension. The expected output is
workload-specific; pinning one would mislead operators on
a different workload.

## debug

When the user reports a build failure, a load failure, a
registration mismatch, a runtime call failure, or an
unexpected measurement, walk the
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order:

1. **Built-in-mode-would-have-sufficed.** First, confirm
   the operator actually needs an extension. If a built-in
   `doca-bench` mode covers the workload, route back to
   [`doca-bench TASKS.md ## configure`](../doca-bench/TASKS.md#configure)
   and drop the extension entirely.
2. **Build-failed.** Quote the compiler / linker error
   verbatim. Common causes: wrong DOCA headers (route to
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)),
   wrong CUDA toolkit version (route to the public CUDA
   docs), missing GPU build flags (cross-reference the
   shipped reference's `meson.build`).
3. **Load-failed.** Run `ldd` on the built library, check
   `LD_LIBRARY_PATH`, check `SONAME` matches the running
   DOCA `so_version`. Route the dynamic-linker side to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md).
4. **Registration-mismatch.** Cross-check the
   extension's exported symbols (`nm -D` on the library)
   against the shipped reference's
   `DOCA_EXPERIMENTAL`-marked surface. Confirm against
   the public DOCA Bench documentation on
   `docs.nvidia.com` (via
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)).
5. **Runtime-call-failed.** Quote the parent's error
   verbatim. Check the per-workload settings struct
   contents; the most common GPU-side failure is the
   `stop_flag` never being set, leading to a hung kernel
   that requires OOB reset.
6. **Version.** Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end. Common extension-specific symptoms: the
   extension was built against one DOCA release's headers
   and is loaded by another release's `doca-bench`; the
   CUDA toolkit version doesn't match the driver; the
   firmware doesn't expose what the extension assumes.
7. **Cross-cutting.** Hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
   for env-side layers (driver, firmware, CUDA driver,
   PCIe, dynamic linker).

In every case: **capture the parent's logs, the
extension's logs, `ldd` output, `nm -D` exported symbols,
and the version stack verbatim BEFORE retrying.**
Paraphrasing the error is the canonical lost-fidelity
failure for this skill.

## use

`## use` is the agent-side workflow for *consuming* a
captured `doca-bench-extension` run as evidence.

1. **Read the parent `doca-bench`'s output first.** The
   parent is the canonical sink for the extension's
   measurement; the extension's own logging is a
   debugging aid, not the result.
2. **Read the version stack alongside the result.** A
   result without the version stack (DOCA, firmware,
   CUDA, parent `doca-bench`, extension `SONAME`) is not
   actionable; route to [`## debug`](#debug) layer 6 if
   any leg is missing.
3. **Cross-check `jobs_processed` / `bytes_processed`
   (or the equivalent accounting fields for the custom
   extension).** Non-zero counts are necessary but not
   sufficient; the operator must confirm against the
   workload's domain expectations
   (GPUNetIO / Flow / Comch).
4. **Compare ONLY against extension runs with the
   matching version stack.** Comparing across DOCA
   releases (per the `DOCA_EXPERIMENTAL` rebuild rule)
   is NOT apples-to-apples; the agent says so explicitly.
5. **Route to [`doca-bench TASKS.md ## use`](../doca-bench/TASKS.md)**
   for the parent's standard `## use` workflow; the
   extension result feeds into the parent's reporting
   pipeline, not a separate one.

## Deferred task verbs

The verbs below are not `doca-bench-extension` work and
should be routed out before the agent does any of them
under this skill's name.

- **install DOCA** ⇒
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  and [`## no-install`](../../doca-setup/TASKS.md#no-install).
- **pick a built-in `doca-bench` workload mode** ⇒
  [`doca-bench`](../doca-bench/SKILL.md). Always the
  first answer to try before any extension work.
- **CUDA toolkit installation** ⇒ NVIDIA's public CUDA
  Toolkit documentation on `docs.nvidia.com`. The
  bench-extension framework does not duplicate it.
- **DOCA GPUNetIO programming semantics** ⇒
  [`doca-gpunetio`](../../libs/doca-gpunetio/SKILL.md).
  The extension's kernel BODY uses GPUNetIO; this skill
  describes the wrapper, not the underlying API.
- **DOCA Flow programming semantics** ⇒
  [`doca-flow`](../../libs/doca-flow/SKILL.md) when the
  extension drives a Flow workload.
- **firmware / driver upgrade** ⇒
  [`doca-version`](../../doca-version/SKILL.md) +
  [`doca-setup`](../../doca-setup/SKILL.md). The
  bench-extension framework has no opinion beyond the
  rebuild-on-DOCA-upgrade rule.
- **contributor work on the in-tree extensions** ⇒ out
  of scope; this skill is for external operators
  authoring custom extensions, not contributors patching
  the shipped ones.

## Command appendix

`doca-bench-extension`-specific invocation classes the
verbs above reach for. Every row is a CLASS — the agent
must not invent build flags, runtime discovery paths, or
extension API entry-point names beyond what the shipped
reference exemplar demonstrates and what the public DOCA
Bench documentation on `docs.nvidia.com` describes.

**Infra-aware preamble (every row below).** Per the
bundle's detect → prefer → fall back → report contract
documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST
   (`doca-env --json`; `doca-capability-snapshot`;
   `version-matrix.json`).
2. If the probe succeeds, the structured tool's output is
   the authoritative answer.
3. If the probe fails, fall back to the manual command in
   the row.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Purpose (class) | Invocation (shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Discover the documented bench-extension surface | Inspect the shipped `/opt/mellanox/doca/tools/bench_extension/` source tree on the user's install + the public DOCA Bench documentation on `docs.nvidia.com` (via [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)) | [`## configure`](#configure) steps 3-4 + [`## debug`](#debug) layer 4 | The reference exemplar is reachable on the user's install; the public guide documents the runtime discovery mechanism the parent uses. |
| Confirm DOCA `version` / `so_version` for the build | `pkg-config --modversion doca-common` or equivalent DOCA version probe on the build host | [`## configure`](#configure) step 7 + [`## debug`](#debug) layer 6 | The reported version matches the parent `doca-bench`'s linked DOCA version per [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2. |
| Build the extension shared library | A `meson` configure + compile invocation adapted from the shipped `/opt/mellanox/doca/tools/bench_extension/meson.build`, with `version : doca_version` and `soversion : doca_so_version` preserved | [`## build`](#build) steps 2-4 | Exits 0; produces a `.so` with the expected `SONAME`; install path is on the parent loader's search path. |
| Confirm the built library's `SONAME` | `readelf -d <library>.so` or `objdump -p <library>.so` | [`## build`](#build) step 5 + [`## debug`](#debug) layer 3 | `SONAME` carries the matching `so_version`; `ldd` resolves all dependencies. |
| Confirm the extension exports the expected entry points | `nm -D <library>.so` cross-referenced against the `DOCA_EXPERIMENTAL`-marked surface in the shipped `doca_bench_cuda.h` | [`## debug`](#debug) layer 4 | The exported symbols match the surface shape (and the *spelling* matches what the parent expects per the public DOCA Bench documentation). |
| Run the no-op smoke through the parent | `doca-bench` invocation that points at the extension and invokes its no-op kernel (the reference's `doca_bench_cuda_start_nop_kernel` family is the example; the exact CLI surface the parent exposes lives in the parent's documentation) | [`## run`](#run) step 2 + [`## test`](#test) step 1 | Exit 0; parent logs show the extension loaded; the no-op entry returned `DOCA_SUCCESS`. |
| Run the minimal-workload smoke through the parent | `doca-bench` invocation that points at the extension and invokes the smallest non-no-op workload kernel | [`## run`](#run) step 4 + [`## test`](#test) step 3 | Exit 0; parent's accounting consumed the extension's stats struct; `jobs_processed > 0`; `stop_flag` discipline held (kernel terminated when the parent set the flag). |
| Save a session snapshot for debug | Capture (a) the parent's invocation + logs, (b) the extension's logs, (c) `ldd` and `nm -D` output for the library, (d) the version stack (DOCA, firmware, CUDA, parent `doca-bench`, extension `SONAME`) | [`## test`](#test) capture step + [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) | The saved bundle is consumed by the cross-cutting debug ladder. |

Three cross-cutting rules for this appendix:

- **Never invent a build flag, runtime discovery path, or
  `DOCA_EXPERIMENTAL` entry-point name beyond what the
  shipped reference exemplar demonstrates.** The shipped
  `/opt/mellanox/doca/tools/bench_extension/` source tree on the
  user's install plus the public DOCA Bench documentation
  are the joint contract; prose-derived names are the
  most common hallucination failure for this skill.
- **No-op smoke before any non-no-op invocation.** A
  non-no-op invocation that has not been preceded by a
  clean no-op smoke is not defensible; the agent re-runs
  the smoke per [`## test`](#test) before issuing
  anything else.
- **Cross-link instead of duplicate.** Cross-cutting
  commands (`pkg-config`, `dmesg`, `ldd`, `nm`, `readelf`)
  live in
  [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md);
  parent-tool commands (the `doca-bench` CLI itself) live
  in
  [`doca-bench TASKS.md ## Command appendix`](../doca-bench/TASKS.md);
  GPUNetIO-specific commands live in
  [`doca-gpunetio`](../../libs/doca-gpunetio/SKILL.md);
  this appendix names only bench-extension-specific
  invocations on top.

## Cross-cutting

A few rules that apply across every verb in this file:

- The **public DOCA Bench documentation on
  `docs.nvidia.com`** + the shipped
  `/opt/mellanox/doca/tools/bench_extension/` source tree + the
  parent `doca-bench`'s `--help` are the joint source of
  truth.
- The **shipped reference exemplar is the schema by
  example.** Quote the reference's
  `DOCA_EXPERIMENTAL`-marked surface; do not paraphrase
  entry-point names from prose memory.
- **No-op smoke before bulk.** Per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  the rule is non-negotiable; loading custom code into a
  benchmark that touches the device requires graduated
  validation.
- **`DOCA_EXPERIMENTAL` rebuild on DOCA upgrade.** Every
  DOCA upgrade re-opens the smoke; a custom extension
  built against one DOCA release is NOT guaranteed to
  load against another.
- **Stop-flag discipline.** Every long-running extension
  kernel respects a parent-set stop signal; the agent
  refuses to recommend a kernel design without bounded
  termination.
- This skill **assumes a healthy DOCA install** (or the
  public NGC DOCA container), a healthy parent
  [`doca-bench`](../doca-bench/SKILL.md), and the
  relevant toolchain for the extension's domain (CUDA
  toolkit when GPU-side). If any of those is in doubt,
  route to the appropriate skill before doing any
  extension work here.
