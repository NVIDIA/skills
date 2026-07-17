# DOCA SHA Offload Engine — Capabilities

**Where to start:** This file is loaded by [`SKILL.md`](SKILL.md).
It documents *what the engine actually offloads vs falls
back*, *when the engine is the right surface vs the
underlying doca-sha library*, *which OpenSSL + DOCA
combinations it requires*, *the layered error and
observability surfaces*, *the message-size window in
which the engine is a perf win*, and *the safety
overlay*. The pattern overview names the recurring
engine-class questions; pick the pattern first, then
drill into the H2 that owns the substance. For the
*how*, jump to [TASKS.md](TASKS.md).

## Pattern overview

Every `doca-sha-offload-engine`-class question this
skill teaches resolves into one of six patterns.

| Engine pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Pick engine vs doca-sha directly | Decide *before* writing any code (or before wiring the engine into the deployment) whether the OpenSSL ENGINE wrapper is the right answer or whether the application should call doca-sha directly. The engine is right for existing pipelines wanting offload with no code change; doca-sha is right for new pipelines or for fine-grained control. | [`## Capabilities and modes`](#capabilities-and-modes) engine-vs-library selection table |
| 2. Configure the PCIe address | The engine defaults to PCIe address `03:00.0` per the shipped source; on systems where DOCA SHA lives at a different PCIe address, the operator must either edit the build-time `DOCA_ENGINE_PCI_ADDR` macro or set the run-time `set_pci_addr` ctrl-cmd. | [`## Capabilities and modes`](#capabilities-and-modes) PCIe-address configuration + [TASKS.md ## configure](TASKS.md#configure) |
| 3. Load and register the engine | The engine is loaded through OpenSSL's standard ENGINE API — either via `openssl engine dynamic -pre LOAD ...` for CLI usage or via `ENGINE_load_dynamic` / `ENGINE_by_id` / `ENGINE_ctrl_cmd_string` / `ENGINE_init` / `ENGINE_set_default_digests` for programmatic usage. The verified load pattern is in the shipped `readme.md`. | [`## Capabilities and modes`](#capabilities-and-modes) engine-load pattern + [TASKS.md ## run](TASKS.md#run) |
| 4. Prove offload actually engaged | A successful `openssl dgst` does NOT prove the engine ran — OpenSSL falls back to software SHA when the engine is missing or fails to register. The *"SHA-224 negative test"* + the `-engine_impl` flag are the verified proofs that the engine was actually called. | [`## Observability`](#observability) prove-offload pattern + [TASKS.md ## test](TASKS.md#test) |
| 5. Characterize the message-size win | The engine offload adds round-trip cost to each digest; for small messages CPU SHA wins, for large messages (especially with `-async_jobs`) the engine wins. The crossover point is workload- and platform-specific. | [`## Capabilities and modes`](#capabilities-and-modes) message-size-window rule + [TASKS.md ## test](TASKS.md#test) |
| 6. Diagnose a load / digest failure | Walk the layered error taxonomy — load-layer (engine `.so` not found, OpenSSL version mismatch) / register-layer (engine loaded but digests not registered, PCIe address rejected) / runtime-layer (engine engaged but a specific digest call failed) / version / cross-cutting (driver, firmware, NUMA). | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern
above:

- **A successful invocation does not prove offload.**
  Without the *"SHA-224 negative test"* or the
  `-engine_impl` flag (which disables OpenSSL's software
  fallback when the engine does not implement a digest),
  the operator cannot tell whether DOCA SHA ran or
  whether OpenSSL silently used its built-in SHA. The
  *prove-offload* pattern is the load-bearing primitive
  of this skill — the verified pattern lives in
  [`## Observability`](#observability) and the worked
  example lives in the shipped `readme.md`.
- **The engine offloads ONLY one-shot SHA-1 / SHA-256 /
  SHA-512.** Per the verified `engine/doca_sha_offload_engine.c`
  source, the registered digests are `doca_sha_digest_sha1`,
  `doca_sha_digest_sha256`, `doca_sha_digest_sha512` and
  the engine is built on the EVP_Digest one-shot
  interface. Anything else — SHA-224, MD5, SHA-3,
  HMAC-SHA, true incremental SHA via repeated
  `EVP_DigestUpdate` — is either silently handled by
  OpenSSL's software path (without `-engine_impl`) or
  fails (with `-engine_impl`). The shipped `readme.md`
  documents this surface explicitly.

## Capabilities and modes

The DOCA SHA Offload Engine is a **dynamic OpenSSL
ENGINE** shipped as `libdoca_sha_offload_engine.so`,
built from `doca/tools/sha_offload_engine/` against
DOCA's `doca-sha` library and OpenSSL ≥ 1.1.1. There is
no daemon, no CLI of its own, and no programmatic API
specific to the engine; the surface is *the OpenSSL
ENGINE API*.

### Algorithm-coverage matrix

Per the verified source (`engine/doca_sha_offload_engine.c`):

| OpenSSL EVP digest | What the engine does |
| --- | --- |
| `EVP_sha1()` | Offloads to DOCA SHA (one-shot, via the EVP_Digest interface). |
| `EVP_sha256()` | Offloads to DOCA SHA (one-shot). |
| `EVP_sha512()` | Offloads to DOCA SHA (one-shot). |
| `EVP_sha224()`, `EVP_sha384()`, `EVP_sha3_*`, `EVP_md5()` | NOT offloaded. Without `-engine_impl`, OpenSSL falls back to its built-in software SHA. With `-engine_impl`, the digest call fails (this is the basis of the "prove offload actually engaged" pattern). |
| Streaming / chained `EVP_DigestInit_ex` → `EVP_DigestUpdate` → `EVP_DigestFinal_ex` | Per the shipped `readme.md`, the engine is built on the one-shot `EVP_Digest` API. Chained-update workflows that OpenSSL's library code paths convert to one-shot may still be offloaded; workflows that genuinely require incremental hashing are outside the engine's surface — route to [`../../libs/doca-sha/CAPABILITIES.md`](../../libs/doca-sha/CAPABILITIES.md) and the partial-hash task there. |

### Engine vs library: selection rule

| User situation | Right surface | Why |
| --- | --- | --- |
| Existing OpenSSL-based pipeline doing SHA-1 / SHA-256 / SHA-512; wants offload with no code change | **This engine.** | Drop in via the `openssl engine dynamic` load pattern; the only application change is the `ENGINE_load_dynamic` block from the shipped `readme.md`, or even less if the user can rely on an `openssl.cnf` engines section. |
| New SHA-using pipeline being authored from scratch | [`../../libs/doca-sha/`](../../libs/doca-sha/SKILL.md). | The library exposes both one-shot and partial-hash tasks, the full cap-query surface, and the DOCA Core context lifecycle. The engine is a deliberate subset; the library is the full surface. |
| Existing OpenSSL pipeline needing offload for an algorithm the engine does NOT cover (SHA-224, SHA-3, HMAC-SHA) | Neither this engine, nor doca-sha directly for that algorithm. | DOCA SHA's library surface is documented per [`../../libs/doca-sha/CAPABILITIES.md`](../../libs/doca-sha/CAPABILITIES.md); the engine restricts to one-shot SHA-1 / 256 / 512. The agent surfaces the gap honestly. |
| Existing pipeline that requires TRUE incremental hashing (gigabytes-into-one-digest with chunks too large for one DOCA SHA task) | [`../../libs/doca-sha/`](../../libs/doca-sha/SKILL.md) directly, using the partial-hash task. | The engine's one-shot path requires the full message fit in a single DOCA SHA call; the partial-hash task in the library exists for exactly this case. |

The selection rule is the load-bearing piece. The agent
must surface the engine-vs-library choice to the user
before recommending the engine.

### PCIe-address configuration

Per the verified source, the engine has TWO ways to pick
the PCIe address of the DOCA SHA device it binds to:

1. **Build-time default.** The shipped `meson.build`
   sets `-DDOCA_ENGINE_PCI_ADDR="03:00.0"`; the source
   uses `default_engine_pci_addr = "03:00.0"`. The
   `test_cmdline_mode/readme.md` documents the edit-the-
   `meson.build` workflow for systems where DOCA SHA
   lives at a different PCIe address.
2. **Run-time override via the `set_pci_addr` ctrl-cmd.**
   The engine registers a ctrl-cmd named `set_pci_addr`
   (verified in `engine/doca_sha_offload_engine.c`'s
   `engine_cmd_defns`) that takes a STRING argument and
   overrides the build-time default at engine init time.
   The CLI invocation pattern is `openssl engine
   dynamic -pre SO_PATH:... -pre LOAD -pre "set_pci_addr:<bdf>"`;
   the programmatic equivalent is
   `ENGINE_ctrl_cmd_string(e, "set_pci_addr", "<bdf>", 0)`
   per the shipped `readme.md`.

The agent's rule: surface BOTH knobs to the user. On
stock installs `03:00.0` may or may not be the right
device; the operator must check (`doca_caps --list-devs`
or platform-specific PCIe inspection per
[`doca-setup`](../../doca-setup/SKILL.md)).

### Engine-load pattern (the verified surface)

Two load patterns, both verified in the shipped
`readme.md`:

1. **CLI:**
   ```
   openssl engine dynamic \
       -pre NO_VCHECK:1 \
       -pre SO_PATH:${DOCA_DIR}/tools/doca_sha_offload_engine/libdoca_sha_offload_engine.so \
       -pre LOAD \
       -vvv -t -c
   ```
   On success the loader prints the engine name
   (`doca_sha_offload_engine`), the available digests
   (`[SHA1, SHA256, SHA512]`), and the `set_pci_addr`
   ctrl-cmd descriptor.
2. **Programmatic:** per the shipped readme,
   ```
   ENGINE *e;
   const char *doca_engine_path = "${DOCA_DIR}/tools/doca_sha_offload_engine/libdoca_sha_offload_engine.so";
   ENGINE_load_dynamic();
   e = ENGINE_by_id(doca_engine_path);
   ENGINE_ctrl_cmd_string(e, "set_pci_addr", doca_engine_pci_addr, 0);
   ENGINE_init(e);
   ENGINE_set_default_digests(e);
   ```
   Unload mirrors it: `ENGINE_unregister_digests(e)` →
   `ENGINE_finish(e)` → `ENGINE_free(e)`.

The agent quotes the shipped path verbatim
(`${DOCA_DIR}/tools/doca_sha_offload_engine/libdoca_sha_offload_engine.so`)
rather than inventing one.

### Message-size window: when offload is a perf win

The engine offload adds a round-trip cost (the engine
hands the message to DOCA SHA, waits for completion,
returns to the OpenSSL caller). For small messages the
round-trip cost dominates the SHA compute itself; for
large messages it amortizes. Three sub-classes:

| Message size class | Typical engine vs CPU result | Tuning that matters |
| --- | --- | --- |
| Small (e.g. ~10 KB) | CPU SHA may match or beat the engine without `-async_jobs`; `-async_jobs` enables asynchronous engine batching and changes the picture per the shipped `readme.md` `openssl speed` examples. | `-async_jobs 256` (or workload-appropriate value); `-multi <N>` for multi-threaded scaling. |
| Medium (e.g. ~100 KB) | Engine + `-async_jobs` typically wins. | `-async_jobs`. |
| Large (e.g. ≥1 MB) | Engine wins comfortably, even synchronous. | The shipped examples in the `readme.md` and `test_cmdline_mode/readme.md` use 10 000 and 100 000 byte messages with various `-async_jobs` values. |

The exact crossover point depends on the DOCA SHA device,
the CPU's SHA-instruction-set support (SHA-NI), the
OpenSSL version, and the workload's concurrency. The
agent does NOT quote a crossover from memory; the
`openssl speed` comparison pattern in
[`TASKS.md ## test`](TASKS.md#test) is the way to capture
it on the user's actual hardware.

## Version compatibility

For the canonical DOCA version-detection chain, the
four-way match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The `doca-sha-offload-engine`-specific overlay** is:

- **The engine builds against `doca-sha`** per the
  shipped `meson.build`. The version of the installed
  `doca-sha` (`pkg-config --modversion doca-sha`) is the
  authoritative DOCA-side pin. The four-way match across
  `doca-common`, `doca-sha`, and the rest of the install
  must hold per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
- **OpenSSL ≥ 1.1.1** per the shipped `readme.md`. The
  engine's shipped tests cover OpenSSL 1.1.1f on Ubuntu
  20.04 and OpenSSL 3.0.2 on Ubuntu 22.04. OpenSSL 3.x
  deprecates the ENGINE API in favor of PROVIDERS, but
  still supports ENGINEs via the legacy code path
  (per the OpenSSL upstream documentation on
  `openssl.org/docs/manmaster/man7/migration_guide.html`).
  The engine will continue to load on OpenSSL 3.x; the
  load pattern is unchanged.
- **The ENGINE-vs-PROVIDER question.** This skill does
  not ship a PROVIDER version of the offload (the
  shipped artifact is an ENGINE, not a PROVIDER). On a
  pure OpenSSL 3.x deployment where the operator has
  moved to PROVIDERS exclusively, the engine still works
  (legacy code path) but is not the long-term answer; the
  agent surfaces that as a forward-looking concern
  without inventing a PROVIDER artifact that does not
  exist.
- **`libssl-dev` is required** at build time per the
  shipped `test_cmdline_mode/readme.md`. The exact
  package name is distribution-specific; on Ubuntu it is
  `libssl-dev`, on RHEL it is `openssl-devel`. The agent
  routes the package-name lookup through
  [`doca-setup`](../../doca-setup/SKILL.md) rather than
  quoting from memory.
- **No version literal from memory** for DOCA. The
  agent quotes versions observed from the user's host.

## Error taxonomy

The engine's error surface combines the OpenSSL load /
register path with the DOCA SHA runtime path. The layers,
in escalating order:

1. **Load-layer.** `openssl engine dynamic -pre LOAD
   ...` (or `ENGINE_load_dynamic` /
   `ENGINE_by_id`) fails to find or open the `.so`.
   Causes: wrong `SO_PATH`, missing OpenSSL headers /
   ABI mismatch (the engine was built against one
   OpenSSL version and is being loaded by another),
   missing transitive dependencies (the engine's `.so`
   pulls in `libdoca_sha.so` and other DOCA shared
   objects). Routing: confirm the path is the verified
   `${DOCA_DIR}/tools/doca_sha_offload_engine/libdoca_sha_offload_engine.so`;
   run `ldd` on the engine `.so` to surface missing
   dependencies; confirm the OpenSSL on the host
   matches the OpenSSL the engine was built against.
2. **Register-layer.** Engine loaded but the digests
   are not registered, OR the `set_pci_addr` ctrl-cmd
   was rejected, OR `ENGINE_set_default_digests`
   returns failure. Causes: the configured PCIe
   address is not a valid DOCA SHA device; the device
   exists but DOCA cannot bind it (driver / firmware
   precondition not met). Routing: verify the device
   via `doca_caps --list-devs`; route to
   [`doca-caps TASKS.md ## run`](../doca-caps/TASKS.md#run)
   for the capability snapshot.
3. **Runtime-layer.** Engine registered; a specific
   digest call returns an error. Common sub-causes:
   message size exceeds the DOCA SHA cap-reported
   maximum source buffer size (per
   [`../../libs/doca-sha/CAPABILITIES.md#capabilities-and-modes`](../../libs/doca-sha/CAPABILITIES.md#capabilities-and-modes)
   `doca_sha_cap_get_max_src_buf_size`); destination
   buffer too small; transient device errors. The
   engine's source logs at `DOCA_LOG_*` per the
   verified `DOCA_LOG_REGISTER(DOCA_SHA_OFFLOAD_ENGINE)`.
4. **Engine-not-engaged (silent fallback).** The
   `openssl dgst` call completed and produced a correct
   hash, but DOCA SHA never ran — OpenSSL fell back to
   software SHA because the engine was not loaded, the
   engine was loaded but the digest was not in
   `[SHA1, SHA256, SHA512]`, or the engine was loaded
   but `ENGINE_set_default_digests` was not called.
   This is the *most* common silent failure mode for
   this skill and is the reason the `-engine_impl` /
   SHA-224 verification pattern exists per
   [`## Observability`](#observability).
5. **Version.** Cross-cutting partial-install /
   mixed-version per
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   OpenSSL 1.1.1 vs 3.x; DOCA versions across host vs
   container.
6. **Cross-cutting.** Cause is below DOCA — mlx5
   driver, firmware on the SHA-capable device,
   hugepages, NUMA. Route to
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   and
   [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug).

For the cross-library `DOCA_ERROR_*` taxonomy, see
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).

## Observability

The engine's observability surface combines OpenSSL-side
visibility (the `openssl engine dynamic` loader's verbose
output) with DOCA-side logging. Specifically:

- **`openssl engine dynamic -vvv -t -c` output.** On
  successful load the verified output per the shipped
  `readme.md` prints the engine name, the available
  digests (`[SHA1, SHA256, SHA512]`), the `[available]`
  marker, and the `set_pci_addr` ctrl-cmd descriptor.
  This is the agent's first observability signal —
  *"did the engine load and register the right
  digests?"*.
- **The *"prove offload actually engaged"* pattern.**
  This is the load-bearing observability primitive of
  this skill. Two verified techniques from the shipped
  `readme.md` and `test_cmdline_mode/readme.md`:
  - **`-engine_impl` for an algorithm the engine
    supports.** Run `openssl dgst -sha256 -engine
    <path> -engine_impl`; `-engine_impl` disables
    OpenSSL's software fallback, so a successful
    output proves the engine actually computed the
    hash.
  - **The SHA-224 negative test.** Run `openssl dgst
    -sha224 -engine <path> -engine_impl`; SHA-224 is
    NOT in the engine's covered set, and the
    `-engine_impl` flag prevents OpenSSL from falling
    back to software SHA. The expected output is an
    error (per the verified `test_cmdline_mode/readme.md`:
    `Error setting digest`); the error PROVES OpenSSL
    actually consulted the engine and the engine did
    not implement SHA-224. Without this test, the
    operator cannot distinguish *"the engine works"*
    from *"OpenSSL is silently using software SHA"*.
- **`openssl speed` for performance characterization.**
  Per the shipped readmes, the documented invocations
  use `openssl speed -evp <alg> -bytes <N> -elapsed`
  with and without `--engine <path>`. The agent runs
  both, with and without `-async_jobs`, to characterize
  the message-size window per
  [`## Capabilities and modes`](#capabilities-and-modes).
- **DOCA log levels.** `DOCA_LOG_LEVEL` applies per
  [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
  The engine's source includes
  `DOCA_LOG_REGISTER(DOCA_SHA_OFFLOAD_ENGINE)` and
  uses `DOCA_LOG_ERR` / `DOCA_LOG_DBG` for its own
  diagnostics.
- **`ldd <engine.so>` for the load-layer check.** Useful
  when the engine fails to load and the operator needs
  to confirm transitive dependencies are present.

For env-side observability (PCIe scans, firmware
introspection) see
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).
For doca-sha-library-side observability see
[`../../libs/doca-sha/CAPABILITIES.md#observability`](../../libs/doca-sha/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

The engine binds to a specific PCIe device (the DOCA
SHA accelerator) and runs as a shared object loaded by
the application's OpenSSL. The artifact-specific
overlay:

- **Verify offload before quoting any perf claim.** The
  prove-offload pattern in
  [`## Observability`](#observability) is the
  load-bearing safety primitive of this skill. Quoting a
  *"DOCA SHA offload win"* without first running the
  SHA-224 negative test or the `-engine_impl` flag is
  the canonical *"we measured the wrong thing"* failure
  mode — the operator may be benchmarking OpenSSL's
  software SHA against itself.
- **PCIe-address binding is a credential.** The
  `set_pci_addr` ctrl points the engine at a specific
  device; on multi-tenant hosts where multiple
  accelerators exist, the operator must confirm the
  chosen PCIe address belongs to the workload's
  intended device. Routing through
  [`doca-caps`](../doca-caps/SKILL.md) and
  [`doca-setup`](../../doca-setup/SKILL.md) is the
  agent's first move.
- **OpenSSL 3.x deprecation warnings are not blockers
  but they are signals.** Loading an ENGINE on OpenSSL
  3.x may log deprecation warnings per OpenSSL's
  upstream guidance. The engine still functions; the
  agent surfaces the deprecation as a forward-looking
  concern (PROVIDERS are the OpenSSL 3.x answer) but
  does not invent a PROVIDER artifact that the bundle
  does not ship.
- **Do not invent OpenSSL flags or engine ctrl-cmds.**
  The verified surface is `-pre NO_VCHECK:1`, `-pre
  SO_PATH:...`, `-pre LOAD`, `-pre "set_pci_addr:..."`,
  `-vvv -t -c`, `-engine <path>`, `-engine_impl`,
  `-async_jobs <N>`, `-multi <N>`. Anything beyond
  those is either application-side OpenSSL surface
  (route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the OpenSSL upstream docs on `openssl.org`) or a
  hallucination.
- **Hardware-safety meta-policy applies to host-side
  changes.** Any host-side change the engine needs to
  cope with — firmware burn on the SHA-capable device,
  BlueField BFB reflash, host kernel boot parameter
  changes — runs through the cross-cutting meta-policy
  in
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md),
  not through this engine's load-pattern.

## Public-source pointer

The canonical public sources for this engine are:

- The **DOCA SHA** programming guide page on
  `docs.nvidia.com/doca/sdk/`, reached via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- The **OpenSSL ENGINE** upstream documentation on
  `openssl.org/docs/man1.1.1/man1/openssl-engine.html`
  and the `EVP_Digest` man page at
  `openssl.org/docs/manmaster/man3/EVP_Digest.html`
  (both cited in the shipped `readme.md`).
- The shipped source tree at
  `doca/tools/sha_offload_engine/` on the user's
  install (or in the public DOCA SDK download).

Do not invent OpenSSL flags, engine ctrl-cmd names,
DOCA SHA symbols, or message-size crossover literals
beyond what those sources document.
