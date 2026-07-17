# DOCA debug workflows

**Where to start:** [`## debug`](#debug) is the canonical layered
ladder every per-library `## debug` redirects to. Reach for
[`## configure`](#configure) first only if the env needs verbosity
turned up; otherwise the order is `## test` (capture a reproducible
state) → `## debug` (walk the ladder).

Read this file when the loader sent you here from [SKILL.md](SKILL.md). For the underlying debug surface (the layered debug model, version-availability of debug tools, the cross-library error taxonomy, the observability primitives, the debug-side safety constraints), see [CAPABILITIES.md](CAPABILITIES.md). For the env-class debug ladder (install / build prerequisites), see [`doca-setup ## debug`](../doca-setup/TASKS.md#debug). For the program-class debug ladder (lifecycle order, `DOCA_ERROR_*` interpretation), see [`doca-programming-guide ## debug`](../doca-programming-guide/TASKS.md#debug). For where to find official documentation or the on-disk install layout, route through [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).

Each verb below describes the **shape of the workflow**, not a copy-paste recipe. The agent's job is to walk the user through the steps in order, verifying preconditions before recommending the next call.

This skill scopes itself to **cross-cutting debug** — the ladder every library shares and every per-library `## debug` redirects to. Three of the seven lint-required task verbs (`## build`, `## modify`, `## run`) describe their own substance in [`doca-programming-guide`](../doca-programming-guide/SKILL.md); the anchors here exist for lint compliance and route there. The verbs this skill *owns* are `## configure` (debug-env setup), `## test` (capture a reproducible state), and `## debug` (the canonical layered ladder, including the *Where to ask for help* escalation).

## configure

> **Routing summary.** This anchor configures the **environment for a debug session**: turn the verbosity surface up, link against the trace flavor, capture a baseline snapshot. For env preparation in the *non-debug* case (DOCA install verification, `pkg-config`, hugepages, devlink), see [`doca-setup ## configure`](../doca-setup/TASKS.md#configure). This is debug-specific configuration, not first-time install.

Goal: prepare the user's environment so that the debug session has *visible signal*. Default DOCA settings are tuned for production quietness; debug sessions need them turned up.

Steps the agent should walk the user through:

1. **Capture the baseline first** (read-only). Before changing anything, record what the user has now: `doca_caps --version` and `doca_caps --list-devs` (if available — see [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) for when `doca_caps` is present), `pkg-config --modversion doca-common`, `cat /opt/mellanox/doca/applications/VERSION`, and the user's current `DOCA_LOG_LEVEL` if any. Save the output. The baseline is what every later observation is compared against; without it the agent cannot tell what changed.

2. **Raise the SDK log level.** For a *first run* of any new code path, default to `--sdk-log-level 70` (TRACE). For an ongoing debug session where TRACE is too noisy to read, drop to `50` (DEBUG). The log-level mechanics are documented in [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability). The agent's rule: never debug with the production default of `30` (INFO) unless the user explicitly asks to reproduce a production observation that happens at `30`.

3. **Switch to the trace build flavor for the affected library.** Linking against `pkg-config doca-<library>-trace` instead of `pkg-config doca-<library>` selects the trace `.so`. The build-side mechanics live in [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build); the env-side mechanics (where the trace `*.so` is on disk, how to point `LD_LIBRARY_PATH` at it without rebuilding) live in [`doca-setup CAPABILITIES.md ## Capabilities and modes`](../doca-setup/CAPABILITIES.md#capabilities-and-modes). After the switch, the same code path will emit additional log lines at `DEBUG` / `TRACE` level that the release flavor does not emit no matter how high the log level.

4. **Decide whether to attach a debugger now or later.** `gdb` and `valgrind` change the program's wall-clock behavior (gdb pauses; valgrind slows by 10x+) and are invasive. Capture the read-only picture (steps 1–3) first; reach for the debugger when the read-only picture has not produced enough evidence — see [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy) item 3.

5. **Document what you changed.** Before declaring the debug session over, list every env / build mutation made during it (log level, trace flavor, kernel-module reload, hugepage change). The user must be able to revert. Mutating without a paper trail leaves the system in an unknown state.

The anti-pattern to refuse: silently raising the log level in the user's persistent shell profile. Debug-env changes are *session-scoped* unless the user explicitly asks to make them permanent.

## build

> **Anchor exists for lint compliance.** Substantive build content for cross-cutting debug (build flags that aid debugging — `-g`, `-O0`, `-DDOCA_ALLOW_EXPERIMENTAL_API`, the `doca-<library>-trace` `pkg-config` module) lives at [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build). Read that section directly. The trace flavor specifically is documented in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes).

Debug-specific build overlay (a CLASS of build mutations, not a recipe
— each row applies to every DOCA library, not just one):

| Debug-time build change | Why | Where the mechanics live |
| --- | --- | --- |
| `-g -O0` | Symbols + no inlining so `gdb` / `valgrind` show real call sites | [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build) |
| Swap `pkg-config doca-<library>` → `pkg-config doca-<library>-trace` | Selects the trace `.so`; emits TRACE/DEBUG lines the release `.so` never emits | [`## configure`](#configure) step 3 + [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability) |
| Keep `-DDOCA_ALLOW_EXPERIMENTAL_API` (when applicable) | Some debug-relevant APIs live behind the experimental gate; dropping the flag in a debug build re-introduces the build failure the user is debugging | [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build) |
| Use `pkg-config --libs` output verbatim | The single most common link-failure class (layer 4) is a hand-typed `-l` line that omits one of DOCA Flow's 5 split `.so`s | [`## debug`](#debug) layer 4 |
| Rebuild against the *currently installed* DOCA, not a cached one | Mixed-version `*.so` is the second most common cross-cutting bug; ensure the build sees the *runtime* DOCA install | [`## debug`](#debug) layer 2 (version coherence) |

The agent's rule for the *build* verb in a debug context: every row
above is a class. Library-specific build flags (Flow's pipe-trace
build option, RDMA-specific build defines) live in the matching
library skill; this overlay names only the cross-cutting changes.

## modify

> **Anchor exists for lint compliance.** Substantive *modify* content (the universal "derive a custom first app from a shipped sample" workflow, the canonical sample-edit pattern) lives at [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify). The debug-specific overlay — *adding diagnostics to a sample without rewriting it* — is captured under that section as a special case (instrument the sample's lifecycle calls, log the `doca_error_t` from each, capture state at the moment of failure). The agent should walk the user through `doca-programming-guide ## modify` and emphasize the diagnostics overlay rather than re-deriving the workflow here.

Debug-specific modify overlay (a CLASS of sample mutations — the rows
apply when the user is using a shipped sample as the carrier for a
debug session, not when authoring a new app):

| Diagnostic modification | Why | Where the mechanics live |
| --- | --- | --- |
| Log `doca_error_t` from every lifecycle call | The layer-classifier ([`## debug`](#debug) layer 5) needs the *exact* `doca_error_get_descr()` text; samples often silently propagate the error and lose it | [`## debug`](#debug) layer 5 + [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy) |
| Print `cfg → init → start` call order before each | `DOCA_ERROR_BAD_STATE` is almost always lifecycle out of order; logging the order is the cheapest way to confirm or refute the hypothesis | [`## debug`](#debug) layer 6 + [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes) |
| Insert read-only capability dumps before commit | Confirms hardware/firmware actually exposes what the program is about to ask for (`doca_caps` cross-check); turns layer 5 / 7 ambiguity into layer 7 evidence | [doca-caps](../tools/doca-caps/SKILL.md) + [`## debug`](#debug) layer 5 |
| Reduce scope to one unit of damage (one rep, one QP, one channel, one device) | Per [`## test`](#test) step 1: a symptom that disappears at scope=1 is a contention bug, not per-unit | [`## test`](#test) step 1 |
| Add a hash / sequence number to per-packet / per-event logs | Timing-dependent symptoms need ordering evidence the unprefixed log does not preserve | [`## test`](#test) step 4 |

The agent's rule for the *modify* verb in a debug context: every row
above instruments the sample without rewriting it. Re-deriving the
sample from prose is the [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify)
*modify-from-sample schema*'s job; this overlay is purely diagnostic,
not generative.

## run

> **Anchor exists for lint compliance.** Substantive *run* content (how to run a built DOCA program: env vars, command-line conventions, the program-side observability the program is expected to emit) lives at [`doca-programming-guide ## run`](../doca-programming-guide/TASKS.md#run). The debug-specific runtime overlay — *capture stdout, stderr, and the system's view of the program in parallel; reproduce the symptom on the smallest possible scope first* — is the *test* and *debug* verbs below.

Debug-specific run overlay (a CLASS, not a recipe — the rows apply to
every DOCA program, not just one library):

| Debug-time run modification | Why | Where the mechanics live |
| --- | --- | --- |
| `--sdk-log-level 70` (TRACE) on first run | Default INFO hides the lifecycle calls and pipe-construction details a debug session needs | [`## configure`](#configure) step 2 + [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability) |
| Link against `doca-<library>-trace` (trace flavor) | Trace `.so` emits TRACE/DEBUG lines the release `.so` does not, regardless of log level | [`## configure`](#configure) step 3 |
| Run on the smallest unit of damage first | A symptom that disappears at scope=1 is a contention bug, not a per-unit bug | [`## test`](#test) step 1 |
| Restart between attempts | DOCA programs accumulate state (handles, pools, registered loggers); reuse hides repro flakiness | [`## test`](#test) step 2 |
| Capture stdout / stderr / system / DOCA view in parallel | One-channel capture is incomplete; the agent must not theorize from a partial picture | [`## test`](#test) step 3 |
| Time-stamp every captured line | Timing-dependent symptoms require ordering evidence the unprefixed log does not preserve | [`## test`](#test) step 4 |

The agent's rule for the *run* verb in a debug context: every row
above is a class — Flow-specific overlays of these rows live in
[`doca-flow ## run`](../libs/doca-flow/TASKS.md#run); the cross-cutting
shape lives here and nothing in this overlay names a specific library
feature.

## test

Goal: produce a **reproducible capture** of the symptom. A debug session that cannot reproduce its symptom on demand is guessing; a session that can reproduces it on the smallest scope, captures the full state at that moment, and then can iterate.

Steps the agent should walk the user through:

1. **Reduce the scope to the smallest unit of damage.** Every DOCA library has a *unit of damage* — a representor for Flow, a queue-pair for RDMA, a channel for Comch, a service config for DMS. The library skill names it. For debug, run the failing program against *one* of those units before trying to reproduce against many. A symptom that disappears at the smallest scope is a contention / scaling problem; a symptom that survives is a per-unit problem.

2. **Reproduce the symptom on a fresh process.** Many DOCA programs accumulate state across runs (handles, registered loggers, library-internal pools). A symptom that appears only on the *Nth* run after a long-running session is a different bug than one that appears on the first run. For debug, restart the process between attempts so each reproduction is from a clean state.

3. **Capture the full triple at the moment of failure.** When the symptom reproduces, capture all three of:
   - **Program output** — `stdout` and `stderr` of the failing program, including the `--sdk-log-level 70` trace.
   - **System view** — `dmesg | tail -200`, `journalctl --since "5 min ago"`, the matching service's log directory if relevant.
   - **DOCA view** — `doca_caps --list-devs` (capability snapshot at the time of failure), `pkg-config --modversion doca-<library>` (re-confirms the version).

   These three views together are *the artifact* the debug session is operating on. If any one of the three is missing, the picture is incomplete and the next debug step is to capture the missing view, not to form a hypothesis from the partial picture.

4. **Note timing and ordering.** When the failure is timing-dependent (intermittent, only-on-cold-start, only-after-long-uptime), the sequence and the wall-clock spacing of operations matters. Prefer capturing logs with timestamps (`--log-format` if available; otherwise `awk '{print strftime("%H:%M:%S"), $0}'`) and noting which operations happened immediately before the failure.

5. **Bisect when the symptom is recent.** If the symptom appeared after a known change (DOCA upgrade, BFB reflash, kernel update, code change), bisect: revert the change, confirm the symptom disappears, re-apply, confirm it returns. *"It started after we did X"* is a hypothesis until X is bisected.

The anti-pattern to refuse: declaring a fix without first reproducing the symptom on demand and then re-running with the fix in place. A debug session whose final state is *"I changed three things and the symptom went away"* has not learned which of the three was the fix and is not done.

## debug

The canonical layered debug ladder for **any DOCA symptom**. Every per-library `## debug` redirects to this ladder for cross-cutting steps and then layers its library-specific overlay on top.

**The ladder is an iterative loop, not a one-shot walk.** The agent
walks the 7 layers bottom-up, captures the picture at each layer, and
loops back when the picture changes the hypothesis. Treating the
ladder as a one-shot sequence ("if it's not layer 1, it's layer 2,
…") misses the most common case: a fix at layer 3 (build) unmasks a
layer 5 (runtime) symptom that was hidden before; the agent must
restart from layer 1 to confirm the new picture, not assume the
earlier layers are still clean.

The loop shape:

```
   .--> 1. Identify lowest layer the symptom is consistent with (1–7)
   |
   |    2. Capture the read-only picture at that layer (## test step 3 triple)
   |
   |    3. Read the picture: hypothesis or escalation?
   |
   |    4. If hypothesis: apply ONE change; back to (1) and re-capture
   |       If symptom unchanged: layer was wrong; back to (1) at next layer
   |       If symptom changed: hypothesis correct; back to (1) to confirm new state
   |       If ladder exhausted at layer 7: escalate to Developer Forum
   '----- (single-trip walks are the failure mode this loop replaces)
```

The agent's rule: walk the layers in order, top to bottom (which is bottom-of-stack first). Skipping a layer because *"it can't be that"* is the most common debugging mistake. *Most* of the time, *most* of the symptoms in the bundle's audience are install / version / build / link problems wearing the costume of a runtime error.

**Layer 1 — Install.** Before any other debug step, confirm DOCA is actually installed and the install is complete.

- `dpkg -l | grep -i doca` (Debian/Ubuntu) or `rpm -qa | grep -i doca` (RHEL/Rocky). The expected packages depend on the install profile; refer to [`doca-setup CAPABILITIES.md ## Capabilities and modes`](../doca-setup/CAPABILITIES.md#capabilities-and-modes).
- `ls /opt/mellanox/doca/` — the install root. If empty or missing, this is layer 1; route to [`doca-setup ## debug`](../doca-setup/TASKS.md#debug) layers 1–2 and stop here.

**Layer 2 — Version coherence.** Once install is present, confirm all four version strings agree.

- `pkg-config --modversion doca-common` (build-time view).
- `cat /opt/mellanox/doca/applications/VERSION` (install-tree view).
- `doca_caps --version` (runtime view) — see [`doca-caps`](../tools/doca-caps/SKILL.md) for when this is available.
- BFB version on the BlueField, if applicable, via the BFB's own version file.

If any disagree, the install is partial / mixed — route to [`doca-setup ## debug`](../doca-setup/TASKS.md#debug) layer 3 and stop here. Cross-version `*.so` loading is not supported; symptoms above this layer will be misleading.

**Layer 3 — Build.** Once version is coherent, confirm the build can find DOCA.

- `pkg-config --list-all | grep -i doca` — should show the libraries the user expects to link against.
- The full failing compile command, with the exact error string. Most build failures fall into one of three buckets: `pkg-config` cannot find module (env), header not found (env), missing experimental-API flag (program — see the `-DDOCA_ALLOW_EXPERIMENTAL_API` requirement documented in [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build)).
- For build env failures, route to [`doca-setup ## debug`](../doca-setup/TASKS.md#debug) layer 4. For program-side build failures (e.g. wrong source layout, missing experimental-API flag), route to [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build).

**Layer 4 — Link.** Once the build compiles, confirm it links.

- The exact `undefined reference to ...` error, if any. The most common case for first-app developers is **DOCA Flow split into 5 separate `*.so`s on recent versions** (`-ldoca_flow -ldoca_flow_ct -ldoca_flow_info_comp -ldoca_flow_tune_server -ldoca_flow_definitions`). Use `pkg-config --libs doca-<library>` to get the full link line; do not try to construct it by hand.
- `ldd /path/to/built/binary` — confirms which `*.so` files actually got linked at runtime. A binary that links cleanly but `ldd` shows missing or wrong-version `*.so` entries is a runtime path problem (`LD_LIBRARY_PATH`) not a link problem.
- For link failures, the fix is almost always *"use the `pkg-config --libs` output verbatim"*. Inventing the link line by hand is the failure mode this skill exists to prevent.

**Layer 5 — Runtime.** Once the binary runs, you reach DOCA's own runtime surface.

- Re-confirm with the read-only triple from `## test` step 3 (program output / system view / DOCA view).
- Check the `doca_error_t` returned from every DOCA call in the failure path. For each non-`DOCA_SUCCESS`, log `doca_error_get_descr(err)` verbatim. The cross-library taxonomy in [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy) tells the agent which family routes to which layer (program / lifecycle / capability / driver-below).
- If the error is `DOCA_ERROR_BAD_STATE`, the symptom is layer 6 (program). If `DOCA_ERROR_DRIVER`, it's layer 7 (driver-below). If `DOCA_ERROR_NOT_SUPPORTED`, capture `doca_caps` output and confirm the capability is actually expected on this hardware/firmware/mode.

**Layer 6 — Program.** Once runtime errors are characterized, debug the program logic.

- Walk the call sequence against the universal lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes) (`cfg-create → cfg-set-* → init → start → use → stop → destroy`). `DOCA_ERROR_BAD_STATE` is almost always lifecycle order out of sequence.
- Confirm the validate-before-commit rule (see [`doca-programming-guide CAPABILITIES.md ## Safety policy`](../doca-programming-guide/CAPABILITIES.md#safety-policy) item 1): every library that programs hardware exposes a *validate* call separate from the *commit* call. Skipping validate produces runtime symptoms that look like hardware bugs.
- For library-specific program-side debug (Flow pipe match-spec issues, RDMA QP-state transitions, Comch handshake failures), route to the matching library skill's `## debug` (e.g. [`doca-flow ## debug`](../libs/doca-flow/TASKS.md#debug)).

**Layer 7 — Driver / firmware.** When `DOCA_ERROR_DRIVER` returns, or when symptoms are visible in `dmesg` / `mlxconfig` but not in the DOCA program's own logs, the layer below DOCA is the suspect.

- `dmesg | tail -200` — kernel-side device events, `mlx5_core` driver messages.
- `mlxconfig -d <pci_addr> q` — firmware capability snapshot. Compare against the user's expected configuration.
- `devlink dev show` — device-level visibility.
- This skill *cannot* fix driver / firmware problems. Route to [`doca-setup ## debug`](../doca-setup/TASKS.md#debug) layer 5 for env-side reset procedures, and to the public Developer Forum (below) for escalation.

**Where to ask for help (when the ladder runs out).** The DOCA Developer Forum at <https://forums.developer.nvidia.com/c/infrastructure/doca/370> is the right escalation channel. Before posting:

- Capture the read-only triple from `## test` step 3 in the post.
- Quote your `pkg-config --modversion doca-common` and `doca_caps --version`.
- State which layer of this ladder you reached and what you tried at each lower layer.
- Respect the public-sources contract from [`AGENTS.md`](../../AGENTS.md) ground rule #1: no internal NVIDIA hostnames, no internal package mirrors, no internal build numbers in the public post. Customer-public artifacts only.
- See also the Developer Forum row in [`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md).

The anti-pattern to refuse: posting on the forum *first* (before walking the ladder). The forum is for symptoms the public, layered process cannot resolve, not for symptoms the agent has not yet investigated.

**Commercial-support scope (what this bundle does not cover).** The public DOCA Developer Forum is the bundle's *technical* escalation channel and the only escalation channel the bundle is authorized to name. Commercial support contracts, response-time SLAs, escalation paths to NVIDIA engineering, and license-tier questions are out of scope for this bundle and are an [`AGENTS.md ## Non-goals`](../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely) class — the right shape of agent response is *"this bundle does not represent commercial support; the right next step is to engage NVIDIA sales"*, not silence and not improvisation. The Developer Forum is the right answer for *technical* questions an experienced engineer cannot resolve from the public docs; it is not a substitute for a procurement conversation.

## Command appendix

The cross-cutting debug commands the verbs above reach for, grouped by
layer so the agent picks the right family without searching prose.
Library-specific debug tools (Flow's `doca-flow-tune`,
RDMA QP-state dumps, …) overlay in the matching
library skill; this appendix lists only the cross-cutting ones.

| Layer | Command | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Install | `dpkg -l \| grep -i doca` / `rpm -qa \| grep -i doca` | [`## debug`](#debug) layer 1 | Lists the expected package set for the install profile. |
| Install | `ls /opt/mellanox/doca/` | [`## debug`](#debug) layer 1 | Returns a populated tree (samples / lib / include / applications). |
| Version | `pkg-config --modversion doca-common` | [`## debug`](#debug) layer 2 | Matches `doca_caps --version` and `cat /opt/mellanox/doca/applications/VERSION`. |
| Version | `doca_caps --version` | [`## debug`](#debug) layer 2 | Matches `pkg-config --modversion`. |
| Version | `cat /opt/mellanox/doca/applications/VERSION` | [`## debug`](#debug) layer 2 | Matches the other two. Disagreement = partial install. |
| Build | `pkg-config --list-all \| grep -i doca` | [`## debug`](#debug) layer 3 | Lists the libraries expected for the install profile. |
| Build | `pkg-config --cflags doca-<library>` | [`## debug`](#debug) layer 3 | Returns valid `-I` flags rooted at the install include dir. |
| Link | `pkg-config --libs doca-<library>` | [`## debug`](#debug) layer 4 | Returns the canonical `-l` list. Hand-typed `-l` lines are the anti-pattern. |
| Link | `ldd /path/to/binary` | [`## debug`](#debug) layer 4 | All `*.so` entries resolved; no "not found" lines. |
| Runtime | `--sdk-log-level 70` on first run | [`## configure`](#configure) step 2 / [`## run`](#run) overlay | TRACE output appears in stderr; failure path produces named lifecycle calls. |
| Runtime | `doca_caps --list-devs` | [`## debug`](#debug) layer 5 + [doca-caps](../tools/doca-caps/SKILL.md) | Lists every device DOCA can see; capabilities present per device. |
| Runtime | `doca_error_get_descr(<rc>)` (called from program) | [`## debug`](#debug) layer 5 | Returns the canonical description; quote it verbatim, do not paraphrase. |
| Capture | `dmesg \| tail -200` | [`## test`](#test) step 3 | Kernel-side device events; `mlx5_core` messages around the failure window. |
| Capture | `journalctl --since "5 min ago"` | [`## test`](#test) step 3 | Service-level logs for the failure window. |
| Driver / FW | `mlxconfig -d <pcie> q` | [`## debug`](#debug) layer 7 | Firmware capability snapshot matches the user's expected configuration. |
| Driver / FW | `devlink dev show` | [`## debug`](#debug) layer 7 | Devices visible at the kernel-driver layer. |

Three cross-cutting rules for the appendix:

- **Never paraphrase `doca_error_get_descr()`.** The text is the
  contract; paraphrasing loses the disambiguation cues the layer
  classifier ([`## debug`](#debug) layer 5) needs.
- **Never invent a `-l` link line.** `pkg-config --libs doca-<library>`
  is the source of truth, especially for libraries (DOCA Flow on recent
  versions) that ship as multiple `*.so` files.
- **Cross-link instead of duplicate.** Most rows above are also in the
  [`doca-setup`](../doca-setup/SKILL.md) or
  [`doca-programming-guide`](../doca-programming-guide/SKILL.md)
  command appendices for the same reason: they are cross-cutting. This
  appendix names them with their *debug-context* purpose; the other
  appendices name them with their setup-context / program-context
  purpose. Same commands, different framing.

## Deferred task verbs

This skill cross-cuts every library; it does not own library-specific debug, env-class debug-prerequisites, or program-class lifecycle interpretation. The boundaries:

- **Library-internal debug overlays** (Flow pipe trace, RDMA queue-pair stats, Comch channel statistics, GPUNetIO CUDA-side debugging) — owned by the matching library skill's `## debug`. This skill provides the cross-cutting ladder; library skills overlay their library-specific debug on top.
- **Env-class debug prerequisites** (install verification, `pkg-config` setup, hugepages, devlink visibility, kernel-module presence) — owned by [`doca-setup ## debug`](../doca-setup/TASKS.md#debug). Layers 1–4 of this skill's ladder cross-link there for the env-side details; the ladder summary stays here for the cross-cutting flow.
- **Program-class error interpretation** (the `DOCA_ERROR_*` taxonomy, `doca_error_get_descr()` use, lifecycle-order debugging) — owned by [`doca-programming-guide ## debug`](../doca-programming-guide/TASKS.md#debug) and [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy). Layers 5–6 of this skill's ladder cross-link there for the taxonomy; the ladder shape stays here.
- **Performance debugging** (latency tail analysis, throughput regression, jitter characterization) — DOCA performance profiling is a separate problem class that needs its own tooling (`doca-bench`, library-specific perf counters). This skill's ladder is *correctness-debug*; performance-debug is deferred to a future round, with `doca-bench` and per-library inspector tools as the entry points.
- **Production incident response** (SRE-style root-cause analysis under time pressure, on-call escalation) — this skill ladder is for development-time and bug-investigation debug; incident response has its own discipline (timeline reconstruction, blast-radius assessment, comms) that is out of scope here.
