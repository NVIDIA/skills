# DOCA version handling workflows

**Where to start:** The verbs run `configure → build → modify → run
→ test → debug`. Skip ahead only when the user is already past a
verb. The `## test` verb is the four-way match + version-matrix
lookup loop (read sources → cross-check → look up min-version per
capability → loop back if drift detected), not a one-shot pass —
see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the underlying detection sources, the
four-way match rule, NGC semantics, error taxonomy, observability,
and per-library overlay pattern, see
[CAPABILITIES.md](CAPABILITIES.md). For the JSON schemas of the
structured-tools helpers the agent prefers when present, see
[`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through the
steps in order, verifying preconditions before recommending the
next call.

## configure

Goal: detect the DOCA version installed on this host and confirm
the four-way match (so every subsequent answer in the same session
can quote the verified version).

Steps the agent should walk the user through:

1. **Probe for the structured helper first.** Run `command -v
   doca-env` (per
   [`doca-structured-tools-contract ## The agent behavior contract`](../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract)).
   If present, run `doca-env --json` and read `version.consistent`
   — one line, four-way match pre-computed. Report *"using
   structured `doca-env`"* per the contract's report rule.
2. **If structured helper absent, walk the manual chain.** In
   order: `pkg-config --modversion doca-common`, `cat
   /opt/mellanox/doca/applications/VERSION`, `doca_caps --version`.
   On BlueField hosts also: `bfver` (BlueField Arm console, or on the host against a standalone BFB file) and `cat /etc/mlnx-release` (BlueField Arm side). Do NOT substitute `mlxprivhost` (privileged-host configuration, not BFB version) or `bfb-info` (not a real NVIDIA-documented tool); both are common hallucinations.
   Quote the version observed from each source; never paraphrase.
   Report *"falling back to manual chain"* per the contract.
3. **Compare the four sources.** All four must agree. If any
   disagree, the install is partial; route to [`## debug`](#debug)
   ladder step 2 before continuing.
4. **Identify the host kind.** Host (x86 / Arm with BlueField as
   SmartNIC), BlueField (Arm running the BFB image directly), or
   inside-NGC-container. The host kind decides which
   version-related questions are even meaningful (e.g. *"BFB
   version"* is meaningless inside a non-BlueField NGC container).
5. **Record the version + host_kind for the session.** Every
   subsequent answer in the same conversation quotes this verified
   version. Do not re-detect on every question; re-detect only
   when the user reports an install change.

If any step fails with a `pkg-config: Package ... was not found`
or similar, route through [`## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

Goal: confirm the build-time DOCA version the user's project will
link against matches the runtime DOCA version on the build host.

The build pattern for any DOCA C/C++ consumer uses `pkg-config` for
include + link flags — fully documented in
[`doca-programming-guide TASKS.md ## build`](../doca-programming-guide/TASKS.md#build).
This skill carries only the *version-side* overlay:

| Slot | Value | Why it matters |
| --- | --- | --- |
| Build-time version source | `pkg-config --modversion doca-common` (cross-library) + `pkg-config --modversion doca-<library>` (per library being linked) | The agent quotes the version observed; the build manifest does NOT pin a version (Linux `pkg-config` does that already). Hardcoding a version in the build file is anti-pattern: it creates the partial-install trap on the next reinstall. |
| Compile-time version macros | `DOCA_VERSION_MAJOR / MINOR / PATCH` from `doca_version.h` | The user's program reads these at compile time if it wants conditional compilation across DOCA releases. Cite this when the user is doing version-gated `#ifdef` work. |
| Build-time vs runtime match guarantee | Build host == deploy host case is trivial. Cross-host: the agent MUST surface that the user is building on a host that will deploy elsewhere, and that the four-way match has to be re-verified on the deploy host. | A program built on a 3.3.0 host and deployed to a 3.2.0 host will fail at runtime in confusing ways; this is the multi-host case of the partial-install trap. |

For non-C consumers (Rust, Go, Python), the build-time version
visibility goes through the language's own FFI generator (e.g.
`bindgen` against `doca_version.h`); the four-way match rule still
applies — the wrapper consumes a `*.so` that has its own runtime
version.

## modify

Goal: update a version-related setting in a build manifest, an
install procedure, or a per-library skill's `## Version
compatibility` overlay.

Three concrete shapes the modify workflow handles:

| Modify intent | What changes | Pre-modify checklist |
| --- | --- | --- |
| 1. Update a build-manifest `pkg-config` minimum | The minimum DOCA version the project requires (e.g. `dependency('doca-common', version: '>=2.6.0')` in `meson.build`) | Confirm via the version-matrix that 2.6.0 is in fact when the user's needed capability landed; over-pinning excludes users whose install is older but supports the capability via a different path |
| 2. Update a per-library skill's `## Version compatibility` overlay | The 3-5 line redirect + the one library-specific overlay rule | Per the overlay pattern in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy), the overlay MUST cross-link this skill and MUST add at most one library-specific rule; if you find yourself writing more, the content belongs here instead |
| 3. Bump the version-matrix when a new DOCA release ships | Append rows to `version-matrix.json` for newly-available capabilities, set `max_doca_version` on rows the new release removes | The schema is locked in [`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md#schemas); follow it. The source-quote field MUST be derived from a public docs URL, never agent memory |

Each modify intent produces a *minimum diff*; never expand scope
to "while we're here". A version-only modify that also re-flows
prose in adjacent sections is a lint regression in disguise.

## run

Goal: confirm at runtime that the DOCA version the program is
actually loading matches the version it expects.

Three runtime checks the agent recommends in order:

1. **`doca_caps --version`.** The runtime DOCA version visible to
   the program at execution time. Must match the build-time
   version from [`## build`](#build).
2. **`ldconfig -p | grep doca`.** Which DOCA `*.so` files the
   runtime linker can find. If multiple DOCA installs exist on the
   host, this is where the cross-version trap shows up — the
   runtime resolves to whichever install is first on
   `LD_LIBRARY_PATH`, regardless of which one `pkg-config` found
   at build time.
3. **Per-library cap query at runtime.** Once the program is up,
   call the relevant `doca_<library>_cap_*` query for the
   capabilities the program depends on. The version-matrix told
   the program *what should be there*; the cap query tells it
   *what is actually there*. When they disagree, the cap query
   wins.

## test

Goal: prove the installed DOCA is consistent (four-way match) and
that the capabilities the user's program depends on are actually
available on this version + device.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the source set being checked, the library being looked up,
or the device being capability-queried. The loop terminates when
either (a) the four-way match holds AND every required capability
reports `is_supported == true`, or (b) the agent has narrowed the
failure to a layer outside version handling (driver / firmware /
hardware) and escalated to the matching skill.

Iteration shape:

1. **Four-way match check.** Read all four sources (per
   [`## configure`](#configure) step 2). Pass = all four agree.
   Fail = route to [`## debug`](#debug) ladder step 2.
2. **Version-matrix lookup.** For each capability the user's
   program / question depends on: look up
   `version-matrix.json` (per
   [`doca-structured-tools-contract ## version-matrix.json schema`](../doca-structured-tools-contract/SKILL.md#schemas))
   if present; otherwise fetch the matching per-library docs
   page via [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)
   and extract the *"available since"* prose. Compare the
   minimum DOCA version against the installed version.
3. **Per-library capability query.** For each capability the
   version-matrix says is in this release: run the matching
   `doca_<library>_cap_*` query against the active
   `doca_devinfo`. If false, the device does not support the
   capability even though the version does — narrow to a
   device-level answer (route to the matching library skill).
4. **Cross-host check (when applicable).** If the user builds on
   host A and deploys on host B, re-run steps 1-3 on host B.
   The four-way match is a per-host property; cross-host
   deployment requires both hosts to pass.
5. **Loop back if any check changed the version picture.** A
   reinstall, a `LD_LIBRARY_PATH` change, or a BFB re-image
   changes the version state; re-run from step 1 after any of
   those.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| Four-way match was passing yesterday, fails today | One of the sources is now reading a different version | Something on disk changed: a partial upgrade, a new `LD_LIBRARY_PATH`, a different `PKG_CONFIG_PATH` in the shell. Re-narrow to *which* source moved. |
| Version-matrix says capability X is available; cap query returns false | The release supports the capability; this device does not | Device-level constraint; the answer is at the device layer, not the version layer. Route to the matching library skill's `## configure`. |
| Cap query returns true; runtime call still returns `DOCA_ERROR_NOT_SUPPORTED` | The capability claim is fine at startup; something else changed | Runtime state changed mid-program (different context, different device, different permissions). Route to the matching library skill's `## debug`. |
| Version-matrix lookup gave a definitive answer; the user reports a different result | Either the version-matrix is stale, OR the user's installed version is not what the agent thought | Re-run step 1 (four-way match) first; a stale version-matrix is rare, a forgotten reinstall is common. |
| Cross-host: host A passes; host B fails | The deploy host has a different DOCA install | This is the multi-host case of partial-install. The fix is to bring both hosts to the same release. |

Loop termination: stop iterating once two consecutive iterations
do not change the picture — the cause is below version handling.
Escalate to [`doca-debug ## debug`](../doca-debug/TASKS.md#debug)
with the captured version state as evidence.

## debug

Goal: when a session's symptom narrows to a version-handling
cause, diagnose it through the layered ladder below before any
other layer is considered.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver. This skill OWNS *layer 2 (version)*;
the other layers redirect.

**Layer 2 (version) — full diagnosis ladder.**

1. **Read all four sources.** Per [`## configure`](#configure)
   step 2. Capture them in the session for reference.
2. **Identify the disagreement.** Most common patterns:
   - `pkg-config doca-common` ≠ `cat applications/VERSION` →
     install scripts and `*.pc` files are from different
     releases. Re-install consistently via
     [`doca-setup`](../doca-setup/SKILL.md).
   - `pkg-config doca-common` ≠ `doca_caps --version` → build-time
     and runtime are from different releases. Check
     `LD_LIBRARY_PATH` and `ldconfig -p | grep doca`; the
     runtime is resolving to a different install.
   - `pkg-config doca-common` ≠ `pkg-config doca-<library>` →
     individual library `.pc` file is from a different release.
     The user installed `doca-<library>` from a different
     package than `doca-common`; reinstall via `doca-all`.
   - BFB version differs from host package version → cross-host
     compatibility-window question. Cite the [DOCA Compatibility
     Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html);
     bring host and DPU into agreement.
3. **Verify the fix.** After the reinstall, re-run [`## test`](#test)
   step 1. The four-way match must hold before any other layer
   is re-considered.
4. **Document the version state in the session.** Subsequent
   answers in the same conversation quote the verified version;
   the user does not have to re-detect.

For the other ladder layers — install, build, link, runtime,
program, driver — route to the matching skill (`doca-setup`,
`doca-programming-guide`, `doca-debug`).

## apt-source consistency

The single most expensive class of partial-install failure on Debian /
Ubuntu hosts is *apt-source mismatch*: the host's
`/etc/apt/sources.list.d/doca.list` (or equivalent) is pointed at one
DOCA release channel (frequently `latest`, or a specific point release
like `3.5`), while the host's *already-installed* `doca-*` packages
are pinned to a different release (e.g. `3.1.0105`). The symptom set
is wide and confusing:

- "I rolled back the BlueField to `3.1.0105` for a bug repro, but on
  the host `apt upgrade` re-installed `doca-tools` from `latest` and
  now nothing on host/BF agrees."
- "I ran the documented install line from the DOCA Installation Guide
  for release X, but on this host the package candidate is from
  release Y."
- "Two co-tenants installed DOCA on the same host from different
  channels and now `doca-common` is from one release and
  `doca-tools` is from another."
- "Everything looked fine yesterday; today `apt update && apt
  upgrade` quietly pulled a newer DOCA point release and broke my
  build."

This anchor is the **agent's precheck** for any host that has DOCA
installed AND is about to receive any subsequent `apt install` /
`apt upgrade` invocation. It runs BEFORE
[`doca-setup ## configure`](../doca-setup/TASKS.md#configure) and is
referenced from the cross-cutting precondition step there.

1. **Capture the configured DOCA apt source(s).** On Debian / Ubuntu:
   `grep -R 'doca' /etc/apt/sources.list /etc/apt/sources.list.d/
   2>/dev/null`. The agent reads back to the user (a) the source
   file path(s) DOCA is configured from, and (b) the **release
   channel string** the source resolves to. Three shapes the agent
   should recognize and treat equivalently when reasoning about
   "which release this host will pull from":
    - **Network URL form** — typically
      `https://linux.mellanox.com/public/repo/doca/<channel>/<distro>/<arch>/`,
      where `<channel>` is either `latest` (rolling) or a pinned
      `X.Y.Z` (e.g. `3.3.0/`). This is the canonical form the
      DOCA Installation Guide documents for fresh installs.
    - **Local file-repo form** — `file:/usr/share/doca-host-<full-build-id>/repo ./`,
      dropped on disk by the `doca-host` meta-package on Ubuntu /
      Debian hosts that installed via the offline `.deb` bundle.
      Observed in the wild on real DOCA 3.3 installs (the build-id
      embeds the release: e.g. `doca-host-3.3.0-088910-26.01-ubuntu2204`
      is DOCA 3.3.0 host build 088910, OFED 26.01, on Ubuntu 22.04).
      The release channel is the `X.Y.Z` inside the build-id; it
      is **already pinned** by the bundle and cannot drift across
      `apt upgrade` unless `doca-host` itself is upgraded.
    - **RHEL / OEL equivalent** — `/etc/yum.repos.d/doca*.repo`
      with a `baseurl=` line; the same three-way distinction
      (network URL with channel, or pinned URL with `X.Y.Z`, or
      local file-repo dropped by `doca-host`) applies.

2. **Capture the installed DOCA package version(s).** Run
   `dpkg -l | grep -E '^ii\s+doca-' | awk '{print $2, $3}'` (Ubuntu)
   or `rpm -qa | grep '^doca-' | sort` (RHEL). The agent reads back
   to the user the installed versions of the canonical anchors
   `doca-common`, `doca-tools`, `doca-host`, `doca-samples` (any
   that are installed).

3. **Cross-check the two captures.** The release channel in step 1
   MUST be the **same** release as the installed versions in step 2.
   The mismatches that are *load-bearing failure modes*:
    - Step 1 source channel = `latest`, step 2 installed = a pinned
      `X.Y.Z` — the next `apt upgrade` will silently move the host
      to whatever `latest` resolves to at that moment, which may not
      be the release the operator's BlueField / co-tenant assumes.
    - Step 1 source channel = `X.Y.Z`, step 2 installed = a
      *different* `A.B.C` — the host was installed from one channel,
      the source list points at another. `apt-cache policy
      doca-common` will report a "Candidate" version that *also*
      differs from "Installed", and any subsequent `apt install` of
      a new DOCA-coupled package will pull from the channel, not
      from the installed release.
    - Step 1 returns no DOCA source AND step 2 reports DOCA
      installed — the host was installed once and then the apt
      source was removed; future `apt upgrade` cannot keep DOCA in
      sync, and the agent surfaces that the host is on a frozen
      release.

4. **Action depending on which mismatch.** The agent surfaces the
   mismatch to the user AND surfaces the documented apt-source-pin
   recommendation BEFORE any further `apt install` / `apt upgrade`
   runs:
    - Prefer pinning the apt source to the explicit `X.Y.Z` URL
      form documented in the DOCA Installation Guide (reached
      through
      [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points))
      for any controlled upgrade — `latest` channels are for fresh
      installs, NOT for landing on a known target release.
    - When the source channel and installed version genuinely
      diverge, the safe path is: (a) pick the release the operator
      WANTS this host on, (b) reconfigure the apt source to that
      explicit `X.Y.Z` channel, (c) `apt update && apt-cache policy
      doca-common` to confirm "Candidate" matches the target, (d)
      THEN re-install the DOCA package set to converge the host on
      that release (per the documented Installation Guide procedure
      — the agent does not invent the package list).

5. **Tie back to the four-source detection chain.** Once the apt
   source is reconciled with the installed release, re-run the
   four-source chain in [`## configure`](#configure) step 1 — the
   match is only meaningful AFTER the apt source is no longer
   silently disagreeing with what is installed.

6. **For BFB / BlueField side.** The same class of failure exists on
   the BlueField side when the operator pushed a different BFB
   (e.g. rolled the BF back to `3.1.0105`) but did NOT
   simultaneously reconfigure the host's apt source to match. This
   is exactly the
   [`doca-bare-metal-deployment ## bluefield-state-classifier`](../doca-bare-metal-deployment/TASKS.md#bluefield-state-classifier)
   state `host-bf-version-mismatch`; the apt-source precheck above
   is the *host-side cause* of that state.

The precheck is intentionally simple to run (two grep / dpkg
captures plus a comparison) precisely because the agent should run
it on EVERY install-shaped session. Skipping it is the most common
way an otherwise-clean version-handling session ends with the user
on a release they did not ask for.

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows so
the agent does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification — defer to
  [`doca-setup`](../doca-setup/SKILL.md). This skill assumes
  something is installed; version handling is *what was installed
  and is it consistent*.
- **deploy.** Deploying DOCA-using applications at scale across
  multiple hosts / DPUs, Kubernetes operator workflows — out of
  scope for Phase 1. The multi-host *version match* is handled in
  this skill's [`## test`](#test) cross-host check; the *deploy
  mechanics* are not.
- **rollback.** Coordinated rollback of a DOCA release across
  multiple hosts / DPUs — out of scope for Phase 1 and reserved
  for a future platform skill. For single-host version rollback,
  the right verb is reinstalling the previous release via
  `doca-setup`; this skill does not own the rollback procedure.
- **firmware burn / BFB re-image.** Burning new ConnectX firmware
  or re-imaging a BlueField with a different BFB — out of scope.
  This skill DETECTS the BFB version as one of the four match
  legs; it does not change the BFB.

## Command appendix

Every command below is **cross-cutting on DOCA version handling**
— it answers a recurring class of question that comes up in the
verbs above. The agent should treat the *class* as load-bearing;
the worked example is a single instance. Run-as user is the
unprivileged user unless noted. Rows that need elevated privileges call that out explicitly.

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| Read `version.consistent` from `doca-env --json` if present per [`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md#schemas); else walk the manual chain below | `## configure` step 1 | What is the installed DOCA version and is it consistent? | Single semver matching across all sources (structured) OR four matching semver strings from the manual chain |
| `pkg-config --modversion doca-common` | `## configure` step 2; `## build` slot 1 | What is the build-time DOCA version? | A semver string matching `doca_caps --version` |
| `cat /opt/mellanox/doca/applications/VERSION` | `## configure` step 2 | What does the install tree itself claim? | A semver string matching the other sources |
| `doca_caps --version` | `## configure` step 2; `## run` step 1 | What is the runtime DOCA version? | A semver string matching `pkg-config --modversion doca-common` |
| `bfver` (BlueField Arm console, or on the host against a standalone BFB file) and `cat /etc/mlnx-release` (BlueField Arm side) | `## configure` step 2 (BlueField only) | What is the BFB-side DOCA version? | A semver string from `bfver` and a full BFB image string from `/etc/mlnx-release`, both matching the host package version. Do NOT use `mlxprivhost` or `bfb-info` for this — see [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes). |
| `grep -RHn 'DOCA_VERSION_' $(pkg-config --variable=includedir doca-common)/doca_version.h` | `## configure` step 2; `## build` slot 2 | What macros does this install expose for compile-time version checks? | A `DOCA_VERSION_MAJOR / MINOR / PATCH` triple matching the runtime version |
| `ldconfig -p | grep doca` | `## run` step 2; `## debug` layer 2 | Which DOCA `*.so` files does the runtime linker see? | One install's set of `*.so` files; multiple installs visible = the runtime might resolve to the wrong one |
| `ls "$(dirname "$(find /opt/mellanox/doca -name doca-common.pc -print -quit)")"` (commonly `/opt/mellanox/doca/lib/<arch>-linux-gnu/pkgconfig/` on DOCA 3.3+, OR `/opt/mellanox/doca/infrastructure/lib/pkgconfig/` on legacy / split-profile installs) | `## debug` layer 2 | Which `*.pc` files does this install ship? | One `*.pc` per installed library; `doca-common.pc` MUST be present |
| Look up capability in `version-matrix.json` if present per [`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md#schemas); else fetch the matching per-library docs page via [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md) and extract the *"available since"* prose | `## test` step 2 | Is capability X available on the user's installed version? | Min-version row (structured) OR quoted prose (manual) — both saying *"available since DOCA Y.Z.W"* |
