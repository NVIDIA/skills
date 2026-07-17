# DOCA version handling — capabilities, version compatibility, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(detection sources / four-way match / NGC / per-library overlay /
errors / safety) and read that section end-to-end. The tables in
each section are the load-bearing content; the prose is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern, jump
to [TASKS.md](TASKS.md). For the JSON schemas that helper tools
emit (so the agent can prefer the structured one-shot over the
manual chain), see
[`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md).

## Pattern overview

Every version-handling question this skill teaches resolves into
one of FIVE patterns. The patterns are CLASSES — they apply across
every DOCA release, every library, and every host kind.

| Version pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Detect the installed version | Read every source-of-truth and confirm they agree (four-way match) | [`## Capabilities and modes`](#capabilities-and-modes) source-of-truth table + [TASKS.md ## configure](TASKS.md#configure) |
| 2. Validate consistency | Cross-check sources; flag drift; explain partial-install | [`## Version compatibility`](#version-compatibility) four-way match rule + [TASKS.md ## test](TASKS.md#test) |
| 3. Look up capability availability | Compare a required minimum DOCA version against the installed version | [`## Observability`](#observability) version-matrix lookup + [TASKS.md ## test](TASKS.md#test) |
| 4. Diagnose a version-related error | Map symptom (pkg-config missing / mismatch / wrong API / BFB drift) to root cause | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |
| 5. Author the per-library overlay | When adding a new library skill, write its `## Version compatibility` against THIS skill's template | [`## Safety policy`](#safety-policy) per-library overlay pattern + [TASKS.md ## modify](TASKS.md#modify) |

Two cross-cutting rules that apply to *every* pattern above:

- **Never invent a version, never quote "latest".** Always derive
  the version from one of the sources in
  [`## Capabilities and modes`](#capabilities-and-modes); never
  trust agent memory and never copy a version string from a
  public docs URL without confirming it against the user's
  installed sources.
- **The headers win over the docs.** The C headers under the
  install tree's include directory (resolved via
  `pkg-config --variable=includedir doca-common`) are the
  *authoritative* statement of which symbols exist on this release. A public docs
  page that mentions a symbol absent from the headers is wrong
  for *this* install — the docs describe a release; the headers
  *are* the release.

## Capabilities and modes

The **canonical source-of-truth table** for DOCA version
detection. Every version question the agent answers must derive
its version string from one of these sources; if two sources
disagree, the install is partial and the answer routes to
[`## Error taxonomy`](#error-taxonomy) before any other diagnosis
continues.

| Source | What it tells you | When to read |
| --- | --- | --- |
| `pkg-config --modversion doca-common` | The *build-time* DOCA version your application will link against. The `doca-common` module ships with every DOCA install and is depended on by every other library, so it is the single most reliable build-time source. | Always read first. The agent's first version-detection step on every host. |
| `pkg-config --modversion doca-<library>` | Same as above but for a specific library. Useful when the agent is reasoning about *one* library and wants to confirm the per-library `.pc` agrees with `doca-common`. | When the user's question is library-scoped (Flow, RDMA, …) and the agent has already read `doca-common`. |
| `cat /opt/mellanox/doca/applications/VERSION` | The *install-tree* DOCA version. A flat text file written by the install scripts. This is the **second mandatory leg of the four-way match** — its presence and value are independent evidence even when `pkg-config` works correctly. Its ABSENCE (`No such file or directory`) on a host where `pkg-config` works is a *partial-install signal* (the `doca-applications` host package was not installed) — see [`## Version compatibility`](#version-compatibility) partial-install table. | Always read second (after `pkg-config`). Also serves as a fallback when `pkg-config` is missing or `PKG_CONFIG_PATH` is unconfigured. |
| `doca_caps --version` | The *runtime* DOCA version. Reads the same version metadata the loaded `*.so` libraries report at runtime. The single most reliable runtime source. | After `pkg-config` — these two together establish whether build-time and runtime agree (the most common drift surface). |
| `bfver` (BlueField Arm console; can also run from the host against a standalone BFB file) and `cat /etc/mlnx-release` (on the BlueField Arm side) | The *BFB-image* DOCA version on the BlueField side of a host ↔ DPU pair. `bfver` is the documented primary probe in the NVIDIA BlueField Platform Software Troubleshooting Guide; `cat /etc/mlnx-release` shows the full BFB image version from inside the BlueField. Only relevant on hosts where a BlueField is present. | On BlueField hosts. The fourth leg of the four-way match. Do NOT substitute `mlxprivhost` (configures privileged-host mode, not BFB version) or `bfb-info` (not a real NVIDIA-documented tool); both are common hallucinations. |
| `flint -d <bdf> q` (host-side, MFT tools, runs against any 15b3:* device — BlueField NIC functions AND ConnectX) | The *NIC firmware* version (the `FW Version:` and `FW Release Date:` lines of the output). This is a DIFFERENT anchor from the BFB-image DOCA version above: `flint q` reads the firmware *image* the silicon is currently running; `bfver` reads the BFB *userland*. Use when the BlueField Arm console is not reachable (e.g. NIC personality, BFB SSH access not configured) AND when reasoning about NIC FW vs host DOCA skew. Alternative names for the same surface: `mlxfwmanager --query` (NVIDIA-supported, same info) and `mst status -v` (less detail; only OK as a fallback). | On any 15b3:* host, including hosts without an SSH-reachable BlueField Arm side. Apply [`## Version compatibility`](#version-compatibility) NIC-FW skew rule against the captured `FW Version:` / `FW Release Date:` lines. Cross-reference the host DOCA's required FW level via the *Supported NICs and Firmware* table in the [DOCA Release Notes](https://docs.nvidia.com/doca/sdk/doca-release-notes/index.html). |
| Header path `$(pkg-config --variable=includedir doca-common)/doca_version.h` | The *compile-time* DOCA version constants (`DOCA_VERSION_MAJOR / MINOR / PATCH`). Read by C programs at compile time. | When the user is reasoning about *what their program will see at compile time*; otherwise, `pkg-config` is the same information. |

**Discovery shortcut.** When the host has a structured-tools
helper installed (per
[`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md#doca-env---json-schema)),
`doca-env --json` returns all five sources in one JSON object with
a `version.consistent` boolean that pre-computes the four-way
match. Prefer this when present; fall back to the chain above
when not.

## Version compatibility

The **four-way match rule** is the central constraint DOCA
version handling exists to enforce:

> All of (a) `pkg-config --modversion doca-common`, (b) `cat /opt/mellanox/doca/applications/VERSION`, (c) `doca_caps --version`, and (on BlueField hosts) (d) the BFB-image version MUST match within a release. Any disagreement means the install is partial; the fix is to reinstall consistently, NOT a code change.

**Absent source vs. disagreeing source — distinct partial-install
shapes.** A source returning `No such file or directory` or
`command not found` is NOT the same failure as two sources
*returning different* values. Treat absence as a *partial-install
signal* (one or more DOCA host-packages was not installed
alongside `doca-common`), record which sources are absent in the
captured triple, and continue with the remaining sources to pin
down which package to add — do NOT route an absent-source case to
a full `doca-all` reinstall when a targeted, granular `apt install`
is the actual fix. The canonical partial-install pattern on a host
with `doca-common` installed is:

| What the four sources say                                      | What it means                                          | Fix (DOCA 3.3+ vocabulary)                                       |
| ---                                                            | ---                                                    | ---                                                              |
| (a) returns *X*; (b) and (c) are *absent*                      | Only the `doca-common` host package is installed       | `apt install doca-samples doca-caps` (samples ships `applications/VERSION` too on 3.3+; `doca-caps` ships the `doca_caps` binary; add other per-tool packages — `doca-bench`, `doca-pcc-counters`, `doca-flow-tune`, `doca-comm-channel-admin` — as the workload requires) |
| (a) returns *X*; (b) returns *X*; (c) is *absent*              | The per-tool package(s) that ship the binary you need (e.g. `doca-caps`) were not installed | `apt install doca-caps` (or the specific per-tool package the workload needs) |
| (a) returns *X*; (c) returns *X*; (b) is *absent*              | The `doca-samples` package (which on DOCA 3.3+ ships `/opt/mellanox/doca/applications/VERSION` as well) was not installed | `apt install doca-samples` |
| (a) returns *X*; (b) returns *X*; (c) returns *Y* (*X ≠ Y*)    | Build-time and runtime are from *different* releases   | Reinstall consistently — this IS the case that requires `doca-all`|

**Confirm package vocabulary against the host BEFORE prescribing.**
DOCA package naming has been refactored across releases — on DOCA 3.3
the legacy `doca-applications` and `doca-tools` meta-packages no
longer exist (`applications/VERSION` ships inside `doca-samples`; the
tools are granular per-binary: `doca-caps`, `doca-bench`,
`doca-pcc-counters`, `doca-flow-tune`, `doca-comm-channel-admin`,
…). Before quoting an `apt install` line at the
user, the agent **MUST** run `apt-cache policy <pkg>` (or
`apt-cache search ^doca-`) on the target host to confirm the package
name exists in that host's configured DOCA repo. A prescribed
package name from this table that returns `Unable to locate package`
is itself a finding to report — not silently ignored.

**Binaries are not on `$PATH` by default.** Per-tool binaries
installed by packages like `doca-caps` land under
`/opt/mellanox/doca/tools/` (e.g. `/opt/mellanox/doca/tools/doca_caps`),
NOT under `/usr/bin/`. `command -v doca_caps` and a bare
`doca_caps --version` will both return "not found" on a freshly-
installed DOCA 3.3 host even when the package IS installed. Probe
with the absolute path (`/opt/mellanox/doca/tools/doca_caps --version`)
or extend `PATH` (`export PATH=/opt/mellanox/doca/tools:$PATH`) before
declaring the binary absent. Cross-check with
`dpkg -L doca-caps | grep doca_caps` to confirm where the package
actually placed the binary on this DOCA release.

The cross-version mixing trap (last row) is the single most
common cause of *"the program built but does nothing on the wire"*
reports for first-time DOCA users, which is why this rule sits
at the top of the version skill.

## Apt-repo and OS-matrix preconditions

Before quoting *any* `apt install` / `apt upgrade` line at the user —
whether to fix a partial install, jump from one DOCA release to the
next, or align host to BFB — the agent **MUST** clear two preconditions
that are independent of the four-source chain itself. Skipping either
turns a "should work" recommendation into a recoverable-but-painful
failure on the user's box.

1. **Apt repository pinning + `apt-cache policy <pkg>` precheck.**
   `latest` channels and stale local repo files are the two ways an
   apparently-correct `apt install` line resolves to the wrong release
   on a user's host. The agent must:
   - Prefer the explicit DOCA-release-pinned repo URL form
     (`https://linux.mellanox.com/public/repo/doca/<X.Y.Z>/<distro>/<arch>/`)
     over the `latest` alias for any controlled upgrade — `latest`
     is for fresh installs, not for landing on a known target release.
   - Recommend the user disable / remove stale earlier-release repo
     files (e.g. a leftover DOCA 2.x `doca.sources` while installing 3.x);
     a host with both pinned can resolve to either.
   - **Run `apt-cache policy <pkg>` for every package about to be
     installed BEFORE the install line is quoted.** If `Candidate:`
     is `(none)` or `Installed: (none)` with no `Candidate:`, the
     package name does not exist in any configured repo on this host
     (see the run-4 lesson: legacy `doca-applications` / `doca-tools`
     names are absent on DOCA 3.3+). That is a finding to surface,
     not a line to ignore.

2. **Host OS support-matrix gate.** Every DOCA release publishes a
   *Supported Operating Systems* table; a host whose OS family is
   *listed* but whose minor / point release is outside the supported
   sub-range (the documented case: Ubuntu 24.04.x where `x ≤ 3` per
   DOCA 3.3's matrix, on a host running 24.04.4) is a stop-and-ask,
   NOT silent acceptance. The packages may install cleanly and the
   userspace may even pass the four-source coherence check — and
   then a downstream kernel-module / DKMS / OFED step quietly fails
   in a way that takes hours to triangulate to "you are off-matrix."
   The agent's required move:
   - Read the target release's *Supported Operating Systems* table
     from `docs.nvidia.com/doca/sdk/<release>-release-notes/...`
     (route via [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)).
   - Compare against the user's actual `lsb_release -a` / `uname -r`.
   - If the OS family is unsupported → refuse (this is the
     verification-contract `## step 1 — preconditions` failure mode).
   - If the OS family is supported but the point release is outside
     the documented sub-range → SURFACE the gap to the user verbatim
     ("your Ubuntu 24.04.4 is outside the DOCA 3.3 supported
     sub-range of 24.04.x for x ≤ 3 per the release notes"), name
     the documented sub-range, and ask the user to either downgrade
     the point release into the supported window or to accept the
     off-matrix risk explicitly before proceeding.

These two preconditions sit ABOVE the four-source chain, not inside
it: they protect against failures that the four-source chain cannot
detect because they occur before any DOCA package gets to claim a
version at all.

**Authoritative upstream source for compatibility windows.**
NVIDIA's own statement of which release pairings are *intended* to
work — quarterly GA cadence, October LTS designation (3-year
support, 7-update LTS train), the semver `X.Y.Z` scheme, the three
compatibility types (source / binary / behavioral), and the two
compatibility directions (backward / forward) — is the
[DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html).
Cite this URL whenever the user asks *"is my LTS still supported"*,
*"what does the version string mean"*, or any host ↔ DPU
compatibility question. This skill detects *what is installed*;
the Compatibility Policy describes *which installs NVIDIA intends
to work together*.

**NGC container semantics.** When the user reached an install via
the public NGC DOCA container (per
[`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install)
Path 0), the four-way match is *of the container*: the headers,
`*.so`, samples, and `doca_caps` are all built and shipped together
at the container tag the user pulled. Mixing artifacts built
inside the container with a `*.so` from a different DOCA install
on the host is the same partial-install trap as case (a) ≠ (c) on
a non-container host.

**Cross-version `*.so` loading is not supported.** A program built
against version *X* must run against runtime version *X*. The
`doca_<library>_cap_*` query family is the right way to ask *"is
this capability supported"* without resorting to header probes or
build-time guards.

## Error taxonomy

Version-related errors the agent should recognize and disambiguate
before continuing to a library- or program-level diagnosis. For
the cross-library `DOCA_ERROR_*` taxonomy itself, see
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *version-level* upstream causes that bubble
up *as* those errors.

| Symptom | Most-likely version cause | First action |
| --- | --- | --- |
| `pkg-config: Package 'doca-common' was not found` | `PKG_CONFIG_PATH` is not configured OR the install is missing the `doca-common` package | Locate the live `doca-common.pc` first: `find /opt/mellanox/doca -name doca-common.pc -print -quit` (commonly `/opt/mellanox/doca/lib/<arch>-linux-gnu/pkgconfig/` on DOCA 3.3+, OR `/opt/mellanox/doca/infrastructure/lib/pkgconfig/` on legacy / split-profile installs). If found, fix `PKG_CONFIG_PATH` (see [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)). If not found, reinstall via the appropriate DOCA profile package. |
| `pkg-config --modversion doca-common` returns *X*; `doca_caps --version` returns `command not found` (NOT a version disagreement; the binary is ABSENT) | Either the per-tool package that ships the binary (e.g. `doca-caps`) was not installed, OR the package IS installed but `/opt/mellanox/doca/tools/` is not on `$PATH` (DOCA 3.3+ ships these binaries off-PATH by default). | First disambiguate: `dpkg -l doca-caps` and `ls /opt/mellanox/doca/tools/doca_caps`. If `dpkg -l` says absent → `apt install doca-caps` (NOT the legacy `doca-tools` meta, which no longer exists on 3.3+; verify with `apt-cache policy doca-caps` first). If the binary IS present at `/opt/mellanox/doca/tools/doca_caps` → it's a `$PATH` issue, not a missing-install; either invoke by absolute path or extend `PATH`. See `## Version compatibility` partial-install table. |
| `pkg-config --modversion doca-common` returns *X*; `cat /opt/mellanox/doca/applications/VERSION` errors with `No such file or directory` | Partial install: the package that ships `applications/VERSION` was not installed. On DOCA 3.3+ this file is shipped by `doca-samples` (not by a separate `doca-applications` package, which no longer exists on 3.3+); on older DOCA releases it shipped via `doca-applications`. | `apt install doca-samples` on DOCA 3.3+ (confirm with `apt-cache policy doca-samples` first); `apt install doca-applications` on older releases that still ship that package. |
| `pkg-config --modversion doca-common` returns *X*; `ls /opt/mellanox/doca/samples/` errors with `No such file or directory` | Partial install: the `doca-samples` package was not installed alongside `doca-common`. The bundle's modify-from-sample first-app workflow CANNOT apply on this host — the agent must say so explicitly (see [AGENTS.md `## Ground rules` rule 5](../../AGENTS.md#ground-rules-every-agent-must-follow)) rather than scaffold a sample from memory. | `apt install doca-samples` OR pivot to the NGC DOCA container via [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install) Path 0. |
| `pkg-config --modversion doca-common` returns *X*; `doca_caps --version` returns *Y* (*X ≠ Y*) | Partial install: build-time and runtime are from different DOCA releases | Reinstall consistently. Do NOT work around in code. See [TASKS.md ## debug](TASKS.md#debug) ladder step 2. |
| Program compiles with `DOCA_VERSION_MAJOR = X`; same program returns `DOCA_ERROR_NOT_SUPPORTED` from a call that the docs say is available since *X* | The headers are from version *X*; the runtime `*.so` is from version *Y* < *X* | Same partial-install diagnosis as above; the header path is *not* what the runtime is. |
| BFB image version differs from host package version by more than one minor release | Host ↔ DPU compatibility window may not cover this pair | Cite the [DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html); if the user is outside the supported window, the answer is to bring the BFB and host into agreement, not to patch around the mismatch. |
| NIC firmware version (per `flint -d <bdf> q` `FW Version:` line) major.minor predates the host DOCA's published required-FW level, OR the `FW Release Date:` predates the host DOCA's release by more than 12 months | NIC FW is out-of-window for this DOCA. NIC FW is a DIFFERENT anchor from BFB-image DOCA version — `flint q` reads the FW *image* shipped to the silicon, `bfver` reads the BFB DOCA *userland* — and skew on either independently disqualifies the pairing. | Route to [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify) for a firmware burn (NOT a host-package reinstall). Cross-check the required FW level against the *Supported NICs and Firmware* table in the [DOCA Release Notes](https://docs.nvidia.com/doca/sdk/doca-release-notes/index.html) before burning. |
| Two or more BlueField devices on the same host (e.g. BF2 + BF3, or BF3 ×2 from different procurement waves) with DIFFERENT firmware levels | The four-way match is *per-device*: each BlueField/NIC enumerated by `lspci -d 15b3:` is its own anchor and must independently sit inside the host DOCA's compatibility window. BF2 and BF3 FW levels are independent silicon and need not match each other; do not "average" them or pick one. | Run `flint -d <bdf> q` (and `bfver` when the BlueField Arm console is reachable) once per visible 15b3:* device; apply the FW-skew rule above to each independently; surface a per-device verdict to the user rather than a single global verdict. |
| `doca-flow.pc` exists; `pkg-config --modversion doca-flow` works; `doca_caps --version` is silent or errors | DOCA runtime is not on `LD_LIBRARY_PATH` OR is from a different install | Verify with `ldconfig -p | grep doca`; route to [`doca-setup ## debug`](../doca-setup/TASKS.md#debug) layer 3. |
| Public docs say capability *C* exists; on the user's host, `doca_<library>_cap_*` returns false | The user's installed version pre-dates the capability | Look up the minimum version in the version-matrix (see [TASKS.md ## test](TASKS.md#test)); if installed < min, the answer is to upgrade or to use a different approach. |
| User pastes a URL like `docs.nvidia.com/doca/sdk/.../archive/v2.5.0/...` | Version-pinned doc URL; describes an old release | Tell the user the URL is version-pinned and fetch the current-release equivalent via [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md). |

## Observability

The version handling observability surface is the set of commands
that *read* version state, and the structured-tools fields that
*report* it in one shot. There is no DOCA "version counter" — the
visibility comes from probing the sources in
[`## Capabilities and modes`](#capabilities-and-modes).

Three primary signals the agent should reach for:

1. **The four-way match status.** Either `version.consistent`
   from `doca-env --json` (preferred) or the agent computing it
   itself from the manual chain. The single most informative
   one-line answer to *"is my install consistent"*.
2. **The version-matrix lookup result.** Either a row in
   `version-matrix.json` (preferred) or the manual fallback of
   fetching the per-library docs page and extracting the
   *"available since"* prose (per
   [`doca-structured-tools-contract ## version-matrix.json schema`](../doca-structured-tools-contract/SKILL.md#schemas)).
   **Current shipping state (PR2):** the schema is shipped; the
   populated `version-matrix.json` data file does NOT yet ship in
   the public bundle, so in practice the agent always falls back
   to fetching the *"available since"* prose from the per-library
   docs via [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
   That fallback is correct (the answer comes from NVIDIA's
   freshest source) but slower than a local lookup; an offline
   version-data file together with an offline verifier is on the
   maintainer roadmap. The agent never treats the missing data file
   as a "no answer" — it walks the manual fallback.
3. **The capability-query result.** The per-library
   `doca_<library>_cap_*` API answers *"is this supported on this
   device + this version"* at runtime. The version-matrix is the
   *promise*; the capability query is the *reality*. When they
   disagree, the capability query wins.

For the env-side observability primitives (`LD_LIBRARY_PATH`,
`PKG_CONFIG_PATH`, `ldconfig -p`) see
[`doca-setup CAPABILITIES.md ## Observability`](../doca-setup/CAPABILITIES.md#observability).
For the cross-cutting debug-time observability (`--sdk-log-level`,
the `doca-<lib>-trace` flavor, `DOCA_LOG_LEVEL`) see
[`doca-debug CAPABILITIES.md ## Observability`](../doca-debug/CAPABILITIES.md#observability).

## Safety policy

Version handling's safety surface is **anti-hallucination**.
The single most common bundle failure mode without this skill is
the agent quoting a version from memory or from a public-docs URL
without confirming it against the user's installed sources. The
rules below exist to prevent that.

- **Never quote "latest".** "Latest" is not a version. The user's
  installed version is the version. If the user actually does not
  know what they have installed, route to
  [TASKS.md ## configure](TASKS.md#configure) before answering any
  other question.
- **Never copy a version from a URL.** A URL like
  `docs.nvidia.com/doca/sdk/.../v3.3/...` describes what was
  current when the page was published, not what the user has.
- **Never assume the four-way match.** Always verify the user's
  sources agree before answering a *"is X supported"* question.
  The cost of asking the user to run two commands is much smaller
  than the cost of telling them a feature exists when their
  install pre-dates it.
- **Never recommend a workaround for a partial install.** When the
  four-way match fails, the *only* safe answer is to reinstall
  consistently. Pinning `LD_LIBRARY_PATH` to a different `*.so`,
  manually copying a header, or any other workaround perpetuates
  the bug and makes the next failure harder to diagnose.

**The per-library overlay pattern.** Every library / service / tool
skill in the bundle has a `## Version compatibility` section in its
own `CAPABILITIES.md`. To stop those sections from drifting from
this skill, the bundle convention is:

> A library / service / tool skill's `## Version compatibility` section
> MUST be 3-5 lines that (a) cross-link to this skill for the
> four-way match rule + detection chain + NGC semantics, and (b)
> add at most one library-specific overlay rule (e.g. *"the DOCA
> Comch library was renamed from DOCA Comm Channel in DOCA 2.5;
> the `pkg-config` module name is `doca-comch` on 2.5+"*). It MUST
> NOT redefine the four-way match rule or restate the detection
> chain.

The mechanical enforcer is the lint warning in NVIDIA's internal
skill-conformance CI that flags repeated `/opt/mellanox/doca`
references in a library skill's CAPABILITIES.md; if you find
yourself accumulating warnings on a new skill's `## Version
compatibility`, you are duplicating instead of cross-linking.

## Deferred topic boundaries

This skill scopes itself to DOCA version handling. Adjacent topics
the agent will get asked but should route elsewhere:

- **Installing DOCA, choosing packages, post-install verification.**
  Owned by [`doca-setup`](../doca-setup/SKILL.md). This skill
  assumes something is installed somewhere; the version question
  is *what was installed and is it consistent*.
- **Per-library capability availability** (which symbols exist in
  Flow at version *X*). The version-matrix lookup procedure in
  [TASKS.md ## test](TASKS.md#test) is generic; the per-library
  *interpretation* belongs in the matching library skill's
  `## Capabilities and modes`.
- **Cross-library `DOCA_ERROR_*` taxonomy.** Owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill's `## Error taxonomy` is the *version-level* upstream
  causes that bubble up *as* those errors.
- **General debug ladder** (install / version / build / link /
  runtime / program / driver). Owned by
  [`doca-debug ## debug`](../doca-debug/TASKS.md#debug). This
  skill owns layer 2 (*version mismatch*); the other layers
  redirect.
- **Routing to public docs and the on-disk install layout.**
  Owned by [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
  This skill cites the Compatibility Policy URL once via that
  map; it does not duplicate the routing.
