# DOCA SHA Offload Engine — Tasks

**Where to start:** The verbs that carry real workflow
content are `## configure`, `## build`, `## run`,
`## test`, `## debug`, and `## use`. The other verbs
(`install`, `modify`) carry preconditions and a
refuses-to-patch routing stub.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent
through the task verbs every artifact in this bundle
exposes (`install / configure / build / modify / run /
test / debug / use`), then defers task verbs that do not
belong here.

## install

Goal: confirm the host has every precondition the engine's
build + load + run sequence needs **before** any
engine-specific work begins.

This skill does **not** own DOCA installation; that path
lives in [`doca-setup`](../../doca-setup/SKILL.md). The
`doca-sha-offload-engine`-specific preconditions:

1. **`doca-sha.pc` is present.** Run
   `pkg-config --modversion doca-sha`. If the `.pc` does
   not resolve, the installed DOCA does not include
   doca-sha; route to
   [`../../libs/doca-sha/SKILL.md`](../../libs/doca-sha/SKILL.md)
   and [`doca-setup`](../../doca-setup/SKILL.md).
2. **OpenSSL ≥ 1.1.1 is present** per the shipped
   `readme.md`. Verify via `openssl version`. OpenSSL
   3.x is supported via the legacy ENGINE code path
   per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
3. **`libssl-dev` (or distribution equivalent) is
   installed** for the build host per the shipped
   `test_cmdline_mode/readme.md`. On Ubuntu:
   `sudo apt install libssl-dev`. On RHEL:
   `sudo dnf install openssl-devel`. The exact package
   name is distribution-specific; the agent does not
   invent it.
4. **DOCA SHA device is visible to DOCA.**
   `doca_caps --list-devs` (per
   [`../doca-caps/TASKS.md#run`](../doca-caps/TASKS.md#run))
   reports a device whose PCIe address the operator
   plans to bind via `set_pci_addr`. The default PCIe
   address in the shipped source is `03:00.0`; the
   operator must confirm this matches their hardware
   before relying on it.
5. **The application embedding the engine has access to
   OpenSSL's ENGINE API.** For programmatic embedding,
   the app must link against `libcrypto`; for CLI
   embedding the `openssl` binary on the host is enough.

If any precondition fails, stop and route; engine-layer
diagnosis against a missing OpenSSL, a missing
`libssl-dev`, or an absent doca-sha install wastes time.

## configure

Goal: pick the PCIe address of the DOCA SHA device, the
load pattern (CLI vs programmatic), and the OpenSSL
algorithm coverage the deployment needs.

Steps the agent walks the user through, in order:

1. **Confirm the engine is the right surface.** Walk
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   engine-vs-library selection table. If the user is
   writing a new pipeline, route to
   [`../../libs/doca-sha/SKILL.md`](../../libs/doca-sha/SKILL.md);
   if the user needs SHA-224 or another non-covered
   algorithm, the engine is the wrong answer.
2. **Identify the DOCA SHA PCIe address.** Run
   `doca_caps --list-devs` and pick the device the
   workload should bind. If the host's DOCA SHA device
   is at `03:00.0`, the engine's build-time default
   works without override; otherwise the operator must
   either re-build with the alternative PCIe address
   (per the shipped `test_cmdline_mode/readme.md` edit-
   the-`meson.build` workflow) OR pass the PCIe address
   at load time via the `set_pci_addr` ctrl-cmd.
3. **Decide CLI vs programmatic load.** For an `openssl
   dgst` / `openssl speed` driven pipeline, CLI is the
   right answer (no application change). For an
   application calling OpenSSL programmatically, the
   verified `ENGINE_load_dynamic` / `ENGINE_by_id` /
   `ENGINE_ctrl_cmd_string` / `ENGINE_init` /
   `ENGINE_set_default_digests` block from the shipped
   `readme.md` is the minimum source change.
4. **Decide whether to use `-engine_impl`.** Per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability),
   `-engine_impl` disables OpenSSL's software fallback.
   In **production** the operator may want fallback
   enabled (the engine fails open to software SHA when
   the device is unavailable); in **verification** the
   operator wants `-engine_impl` (so a missed offload
   surfaces as an error, not a silent software hash).
5. **For `openssl speed` perf characterization, decide
   the `-async_jobs` and `-multi` values.** Per the
   shipped readmes, the engine's perf surface depends on
   the OpenSSL async pipeline; the agent surfaces this
   to the user before they ask for a *"quick benchmark"*
   that would not exercise the engine's actual scaling
   shape.
6. **Confirm the build inputs.** `PKG_CONFIG_PATH`
   includes the install's `pkgconfig` directory; OpenSSL
   headers are visible to the `meson` build.

## build

The engine is **not pre-built** under `/opt/mellanox/doca/`
on all install profiles; the user typically builds it
from source under `doca/tools/sha_offload_engine/`
against the installed DOCA. The build pattern (per the
shipped `test_cmdline_mode/readme.md`):

1. **(Optional) edit the build-time PCIe address.** If
   the user's DOCA SHA device is not at `03:00.0`, edit
   `doca/tools/sha_offload_engine/meson.build` to
   change the `-DDOCA_ENGINE_PCI_ADDR="03:00.0"` macro
   per the verified `test_cmdline_mode/readme.md`. The
   alternative is to leave the macro alone and pass the
   PCIe address at load time via `set_pci_addr`.
2. **Set the right `PKG_CONFIG_PATH`** so `pkg-config`
   can find `doca-sha.pc`, `doca-common.pc`, and the
   OpenSSL `.pc` files. The DOCA `pkgconfig` directory
   is documented in
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
3. **Build under the DOCA top-level meson tree.** Per
   `test_cmdline_mode/readme.md`: `cd ${DOCA_DIR}`;
   `meson setup build`; `cd build`; `ninja`. The engine's
   `.so` lands under the build tree at a path the
   shipped readme documents.
4. **Confirm the `.so` was produced.** `ls
   <build-dir>/.../libdoca_sha_offload_engine.so` (or
   equivalent for the user's install layout).
5. **Confirm transitive dependencies.** Run
   `ldd <engine.so>` and verify the OpenSSL and DOCA
   shared objects all resolve. Missing transitive
   dependencies are the most common cause of layer-1
   load errors per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).

Routing for nearby "build" questions:

- *"I want to build a PROVIDER instead of an ENGINE for
  OpenSSL 3.x."* → out of scope; the shipped artifact is
  an ENGINE, not a PROVIDER. Route to the OpenSSL
  upstream PROVIDERS migration guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- *"I want to author my own ENGINE wrapping a different
  doca library."* → out of scope; route to
  [`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).

## modify

**Do not patch the shipped engine source tree.** The
shipped `engine/doca_sha_offload_engine.c` and
`lib/doca_sha_offload_lib.{c,h}` are the verified worked
example for this engine class; modifying them puts the
user in contributor-to-DOCA territory.

What the agent *does* modify is the **load + invocation
environment** — the PCIe address (either via the
build-time macro or the run-time ctrl-cmd), the
`-async_jobs` / `-multi` knobs on `openssl speed`, the
inclusion of `-engine_impl`, the OpenSSL `engines`
configuration in `openssl.cnf` for system-wide loads.
Treat *"modify the invocation, not the engine source"*
as the operating mode.

Routing for nearby "modify" questions:

- *"The engine should support SHA-224 / SHA-384 /
  SHA-3 / HMAC-SHA."* → out of scope; that would be a
  source-level extension. The user can either (a) use
  doca-sha directly for those algorithms if the library
  supports them per
  [`../../libs/doca-sha/CAPABILITIES.md`](../../libs/doca-sha/CAPABILITIES.md),
  (b) use OpenSSL's software path for the non-covered
  algorithms, or (c) route to a different bundle that
  ships a customized engine.
- *"I want to enable the engine's debug logging."* → the
  shipped source supports a `PRINT_DEBUG_INFO`
  compile-time macro and a `debug_level` ctrl-cmd; the
  user has to re-build with the macro to surface them.
- *"I need the engine to do incremental hashing across
  many `EVP_DigestUpdate` calls."* → the verified
  engine is one-shot per the shipped `readme.md`. For
  true incremental hashing, route to
  [`../../libs/doca-sha/`](../../libs/doca-sha/SKILL.md)
  and the partial-hash task there.

## run

The load + verify flow — every session goes through it.
The detailed OpenSSL surface lives in the upstream
OpenSSL documentation; this section names the *shape* of
the flow, not the verbatim recipe.

> **Do-not-invent guard (openssl flag spellings).** Real
> downstream agents have hallucinated an `-engine-options`
> flag for `openssl dgst` — **it does not exist** in
> upstream OpenSSL. The bundle's canonical pattern is two
> commands: first
> `openssl engine dynamic -pre NO_VCHECK:1 -pre SO_PATH:<path> -pre LOAD -vvv -t -c`
> to probe the engine, then `openssl dgst -sha256 -engine <path> -engine_impl`
> to run the digest through it. The bundle's exact spelling
> is `-engine_impl` with an **underscore** — not `-engine-impl`
> with a hyphen; some OpenSSL builds may accept both, but the
> bundle's verbatim form is the underscore. The engine ID is
> `doca_sha_offload_engine`; the install path of the `.so` is
> `${DOCA_DIR}/tools/doca_sha_offload_engine/` (per
> `install_dir: 'tools/doca_sha_offload_engine'` in
> `tools/sha_offload_engine/meson.build` and the shipped
> `readme.md`; NOT under `${DOCA_DIR}/infrastructure/`).

1. **Confirm the `.so` and the environment.** Per
   [`## install`](#install), [`## configure`](#configure),
   and [`## build`](#build).
2. **Probe the engine load.** Run the verified
   `openssl engine dynamic -pre NO_VCHECK:1 -pre
   SO_PATH:<path-to-libdoca_sha_offload_engine.so> -pre
   LOAD -vvv -t -c` invocation per the shipped
   `readme.md`. On success the output prints the engine
   name, the `[SHA1, SHA256, SHA512]` available digests,
   and the `set_pci_addr` ctrl-cmd. If the output is
   missing any of those, route to
   [`## debug`](#debug) layer 1 or 2.
3. **Run the SHA-224 negative test first.** Per the
   shipped `test_cmdline_mode/readme.md`, run
   `openssl dgst -sha224 -engine <path> -engine_impl`
   on a known input. The expected output is an error
   (`Error setting digest`); the error PROVES OpenSSL
   actually consulted the engine. If the call succeeds
   anyway, the engine is NOT being consulted — route to
   [`## debug`](#debug) layer 4.
4. **Run a positive test.** Run `openssl dgst -sha256
   -engine <path> -engine_impl` on a known input.
   Expected: a correct SHA-256 hash and a stderr line
   matching `engine "doca_sha_offload_engine" set.` per
   the verified `readme.md` examples.
5. **For programmatic deployment, drop the verified
   load block into the application** per the shipped
   `readme.md`:
   ```
   ENGINE *e;
   const char *doca_engine_path = "${DOCA_DIR}/tools/doca_sha_offload_engine/libdoca_sha_offload_engine.so";
   ENGINE_load_dynamic();
   e = ENGINE_by_id(doca_engine_path);
   ENGINE_ctrl_cmd_string(e, "set_pci_addr", doca_engine_pci_addr, 0);
   ENGINE_init(e);
   ENGINE_set_default_digests(e);
   ```
   Pair it with the verified unload sequence
   (`ENGINE_unregister_digests` →
   `ENGINE_finish` → `ENGINE_free`). Quote these
   verbatim from the shipped readme; do not invent
   alternative function names.
6. **Plan the perf-comparison run** only after the
   verification is green. Per
   [`## test`](#test) the agent walks an `openssl speed`
   comparison with and without the engine.

When recording the run for downstream consumers, write
down: `pkg-config --modversion doca-sha`,
`openssl version`, host OS / kernel / NUMA, DOCA SHA
device PCIe address, the engine load output (`openssl
engine dynamic -vvv -t -c`), the SHA-224 negative test
result, the positive test result, and the
`-engine_impl` flag's presence / absence.

## test

The engine's `## test` verb covers TWO loops: (a)
verification — *did the engine actually run* — and (b)
performance characterization — *for what message-size
mix is the engine a win*.

### Verification loop

**The SHA-224 negative test is mandatory.** Without it,
the operator cannot tell whether DOCA SHA ran or
whether OpenSSL silently used its software path. Per
[`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
prove-offload pattern:

1. Run `openssl dgst -sha224 -engine <path>
   -engine_impl <some-input>`. Expected: error
   (`Error setting digest` per the shipped readme). If
   the call succeeds with a hash, OpenSSL is silently
   using software SHA — the engine is not being
   consulted. Route to [`## debug`](#debug) layer 4.
2. Run `openssl dgst -sha256 -engine <path>
   -engine_impl <some-input>`. Expected: a correct
   SHA-256 hash and a stderr line
   `engine "doca_sha_offload_engine" set.`. If this
   fails, route to [`## debug`](#debug) layer 2 or 3.
3. Repeat (2) for `-sha1` and `-sha512`. All three
   should succeed.
4. Only after all four steps pass is the engine
   verified.

### Performance-characterization loop

Per the shipped `test_cmdline_mode/readme.md` and
`readme.md`, `openssl speed` is the documented surface.
The eval-loop overlay:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `openssl speed -evp sha1 -bytes 10000 -elapsed` without the engine completes; the same command with `--engine` and no `-async_jobs` is slower or about equal | Round-trip cost dominates for small messages without async batching | Add `-async_jobs 256` and re-run; the engine's async surface is what changes the picture. |
| Engine with `-async_jobs 256` wins comfortably; engine without `-async_jobs` does not | Workload's deployment must enable async to benefit | The agent's recommendation is "deploy with async enabled, or move to the doca-sha library directly for finer-grained control of the parallelism shape". |
| Engine wins at 100 KB but not at 10 KB | The crossover point is between those sizes for this platform | Capture the message-size mix the production workload actually uses; if it is dominantly below the crossover, the engine is not a win for that pipeline. |
| Engine wins on synchronous `openssl dgst` for a single large message | The single-call path is doca-sha's natural shape | Quote the result alongside the message size; do NOT extrapolate to a workload with many small messages. |
| Same invocation produces different numbers across two hosts at the same DOCA version | DOCA SHA device, firmware, or NUMA delta below DOCA | Capture the (DOCA + OpenSSL + device + firmware + NUMA) tuple on both hosts; route through [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test). |
| Same invocation produces different numbers on the same host across DOCA versions | Regression signal | Cross-link both baselines, name the changed fields, route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug). |

The agent's rule: every change to the OpenSSL invocation
re-opens the loop. Re-running with `-multi 8` without
re-walking the `-async_jobs` knob first is exactly the
failure mode this loop replaces.

**Baseline-capture rule.** When the goal is a baseline,
the captured artifact must include `pkg-config
--modversion doca-sha`, `openssl version`, the device
PCIe address, the `openssl speed` command lines (engine
on and engine off), the `-async_jobs` and `-multi`
values, and the host's CPU SHA-NI status (for the CPU
baseline).

## debug

When the engine fails to load, register, or engage,
walk the
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order:

1. **Load-layer.** `openssl engine dynamic -pre LOAD
   -vvv` fails or the verbose output does not list
   the engine. Confirm the `SO_PATH` is the verified
   `${DOCA_DIR}/tools/doca_sha_offload_engine/libdoca_sha_offload_engine.so`;
   run `ldd <engine.so>` to surface missing transitive
   dependencies; confirm OpenSSL on the host matches
   what the engine was built against.
2. **Register-layer.** Engine loaded but the
   `[SHA1, SHA256, SHA512]` digests are not listed, or
   `set_pci_addr` is rejected, or
   `ENGINE_set_default_digests` returns 0. Confirm the
   PCIe address is a valid DOCA SHA device via
   `doca_caps --list-devs`; route to
   [`../doca-caps/TASKS.md#run`](../doca-caps/TASKS.md#run).
3. **Runtime-layer.** Engine registered, but a digest
   call fails. Confirm the message size against
   `doca_sha_cap_get_max_src_buf_size` per
   [`../../libs/doca-sha/CAPABILITIES.md`](../../libs/doca-sha/CAPABILITIES.md);
   raise `DOCA_LOG_LEVEL` to surface the engine's
   internal logging.
4. **Engine-not-engaged.** The `openssl dgst` call
   completed and produced a hash, but the SHA-224
   negative test from [`## test`](#test) shows OpenSSL
   silently used the software path. Common causes:
   the engine was loaded but `ENGINE_set_default_digests`
   was not called; the application was started before
   the engine was registered; the OpenSSL `engines`
   config does not actually include the engine. Re-walk
   [`## run`](#run) steps 2 and 3.
5. **Version.** Cross-cutting partial-install /
   mixed-version. Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end; confirm OpenSSL and the engine were
   built against compatible toolchains.
6. **Cross-cutting.** Cause is below DOCA. Hand off to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   and
   [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug).

In every case: **quote what OpenSSL and the engine
reported.** Do not paraphrase the load output; do not
collapse the SHA-224 test into "the engine works".

## use

Goal: turn a verified engine + perf characterization
into an integration decision — *"do I deploy this engine
or rewrite to doca-sha?"*.

The decision shape this skill teaches:

1. **Confirm verification.** The SHA-224 negative test
   from [`## test`](#test) is the gate; an unverified
   engine deployment is a deployment that silently
   falls back to software SHA on a moment's notice.
2. **Quote the win alongside the message-size mix.**
   *"The engine wins at 100 KB with `-async_jobs 256`;
   at 4 KB the CPU wins"* is a defensible deployment
   recommendation; *"the engine is faster"* is not.
3. **For deployments where the workload's message-size
   mix is below the crossover, the right answer is NOT
   the engine.** It may be doca-sha directly with a
   custom batching strategy, or the host's SHA-NI path,
   or a redesign that batches more aggressively before
   hashing.
4. **For deployments where the application code is
   fixed and cannot adopt doca-sha directly,** the
   engine is the right answer even when the win is
   small — the alternative is no offload at all.
5. **Plan the deprecation path on OpenSSL 3.x.** The
   ENGINE API is deprecated in OpenSSL 3.x in favor of
   PROVIDERS; the engine still works via the legacy
   code path per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
   but a deployment that wants to be future-proof
   should plan a doca-sha-direct rewrite (or wait for a
   PROVIDER artifact if one ships in a future DOCA
   release).
6. **Per-release re-verification.** Every DOCA upgrade
   AND every OpenSSL upgrade requires re-running the
   SHA-224 verification AND the `openssl speed`
   characterization. The agent does not assume a
   known-good deployment survives version bumps without
   re-testing.

## Deferred task verbs

The verbs below are not `doca-sha-offload-engine` work
and should be routed out:

- **install DOCA** ⇒ [`doca-setup TASKS.md`](../../doca-setup/TASKS.md).
- **author a new SHA pipeline against doca-sha
  directly** ⇒
  [`../../libs/doca-sha/SKILL.md`](../../libs/doca-sha/SKILL.md).
- **offload algorithms the engine does not cover
  (SHA-224, SHA-3, MD5, HMAC-SHA)** ⇒
  [`../../libs/doca-sha/CAPABILITIES.md`](../../libs/doca-sha/CAPABILITIES.md)
  for the library's coverage; for genuine non-coverage,
  route to OpenSSL's software path or to a different
  bundle.
- **OpenSSL PROVIDER authoring (OpenSSL 3.x
  long-term)** ⇒ out of scope; route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the OpenSSL upstream PROVIDERS migration docs on
  `openssl.org`.
- **hardware-touching changes underneath** (firmware
  burn on the DOCA SHA device, BFB reflash) ⇒
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md).
