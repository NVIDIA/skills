# DOCA Arg Parser workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already
past a verb. The `## test` verb is an iterative loop (presence /
register / `--help` listing / smoke-parse / JSON-config), not a
one-shot pass — see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the param-registration model, the
parameter-type set, the standard DOCA CLI surface, the
`--json <path>` (`-j <path>`) rule (note: the real flag is
`--json`, NOT `--json-config` — do not invent the longer name),
the error taxonomy, observability,
and the safety / path-selection policy, see
[CAPABILITIES.md](CAPABILITIES.md). For the cross-library DOCA
patterns layered under everything below (the canonical
`pkg-config` + meson build pattern, the universal
modify-a-shipped-sample workflow, the cross-library
`DOCA_ERROR_*` taxonomy), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through
the steps in order, verifying preconditions before recommending
the next call.

## configure

Goal: confirm doca-argp is installed and usable, pick the
shipped-sample `*_main.c` that already wires the Arg Parser, and
plan the register-before-start lifecycle for any new params
before any code change.

Steps the agent should walk the user through:

1. **Confirm the installed DOCA version and that doca-argp is
   present.** Run `pkg-config --exists doca-argp` first
   (presence); if it succeeds, follow with
   `pkg-config --modversion doca-argp` (version). Quote the
   version observed; do not assume "latest". The four-way
   match rule lives in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility);
   if the observed sources disagree, route there before any
   Arg-Parser diagnosis. If `--exists` fails, the user's
   install profile does not include doca-argp — route to
   [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure).
2. **Locate the canonical reference: any sample's
   `*_main.c`.** doca-argp ships no dedicated sample tree;
   the reference is the `*_main.c` file in every sample
   under `/opt/mellanox/doca/samples/<library>/<sample>/`.
   Pick one whose other-library context already matches the
   user's intent (e.g.
   `doca_dma/dma_local_copy/dma_local_copy_main.c` when the
   user is modifying a DMA sample's CLI, or
   `applications/dma_copy/` for the reference *application* —
   `samples/doca_dma/dma_copy/` does not exist as a sample
   directory in the public DOCA install; the public samples
   are `dma_local_copy`, `dma_copy_dpu`, `dma_copy_host`) so
   the surrounding code is familiar.
3. **Sketch the lifecycle on paper before writing code.** Per
   the lifecycle table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   `doca_argp_init` is the FIRST `doca_argp_*` call; every
   `doca_argp_register_param` must run BEFORE
   `doca_argp_start`; `doca_argp_start` runs ONCE;
   `doca_argp_destroy` runs on every exit path including
   error paths. Walking this on paper before editing
   `<sample>_main.c` is the cheapest way to avoid the
   `DOCA_ERROR_BAD_STATE` first-app failure mode.
4. **Plan the new params (if any) by type.** For each
   user-added flag, decide its parameter type per the
   parameter-type table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   (string, int, bool flag, JSON config file). The type
   choice drives both argv validation and JSON-config
   validation; getting it wrong surfaces later as
   `DOCA_ERROR_INVALID_VALUE`.
5. **Confirm the standard surface is preserved.** The sample's
   existing `*_main.c` already registers the standard DOCA
   CLI flags (per the standard-surface table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)).
   The user's modification adds rows; it does not replace
   them. The agent must surface this rule before any code
   change.

If any step fails with a `DOCA_ERROR_*`, route through the
error taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

Goal: compile a doca-argp-using consumer against the user's
installed DOCA, with `pkg-config` as the source of truth for
include + link flags.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson
or CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the Arg-Parser-specific overlay:

| Slot | Value for Arg Parser | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-argp` | The library's `.pc` file installed by the DOCA host packages. Wrong module name = `pkg-config: Package 'doca-argp' was not found`. Presence is profile-dependent; confirm with `pkg-config --exists doca-argp` per [`## configure`](#configure) step 1 |
| Include flags | `pkg-config --cflags doca-argp` | Resolves to `doca_argp.h` under $(pkg-config --variable=includedir doca-common) |
| Link flags | `pkg-config --libs doca-argp` | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator) |
| Header check | the artifact's public header resolvable under whichever include directory `pkg-config --cflags` reports (do not hardcode the include path — the install layout can move) | If `pkg-config --cflags doca-argp` resolves but the include is missing, the install is partial — route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| Companion libraries | Always paired with the user's actual DOCA library (`doca-dma`, `doca-comch`, `doca-rdma`, …) — doca-argp alone does no DOCA work | Building doca-argp into a binary that calls no other `doca_*` symbol is a path-selection mistake; re-read the *Do not use doca-argp when …* column in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) |

For non-C consumers (Rust, Go, Python), the wrapper consumes
`libdoca_argp.so` through FFI; the build-time version
visibility goes through the language's own FFI generator (e.g.
`bindgen` against `doca_argp.h`). The lifecycle and
parameter-type rules still apply — the wrapper consumes a
`*.so` that has its own runtime version per
[`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).

## modify

Goal: take a shipped DOCA sample's `*_main.c` as the verified
starting point and apply a **minimum-diff modification** to add
/ remove / rename a CLI flag without breaking the standard
surface.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The Arg-Parser-specific overlay is the
*modify-from-sample schema fill* — the slots the agent must
elicit from the user before recommending any code-level edit:

| Slot | What the agent asks the user | Arg-Parser-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which `*_main.c` under `/opt/mellanox/doca/samples/<library>/<sample>/`? | Pick the closest in *library context* (a DMA modification starts from a DMA sample's `*_main.c`); the Arg Parser usage is structurally identical across libraries, but the surrounding setup code matters |
| 2. Long + short name | What long name (e.g. `--my-flag`) and (optional) short name? | Long name MUST NOT collide with the standard surface in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) (`--device`, `--representor`, `--rep-list`, `--json`, `--sdk-log-level`); colliding silently overrides the standard flag in `--help` and breaks cross-sample muscle memory |
| 3. Parameter type | String, int, bool flag, or JSON config file (per the parameter-type table in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes))? | The choice determines both argv validation and JSON-config validation; declaring an `int` and then passing `0x40` surfaces as `INVALID_VALUE` per [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy) |
| 4. Value-callback target | Which user-config struct field does the parsed value write into? | The callback runs DURING `doca_argp_start`; do not call back into `doca_argp_*` from inside it (that's the most common `BAD_STATE` first-app failure) |
| 5. Description for `--help` | What one-line description should `--help` show for the new flag? | The description is what the operator reads; vague descriptions surface as user-confusion bug reports, not as parser bugs |
| 6. JSON-config impact | Does the user want operators to be able to set this from `--json <path>` as well? | Every registered param is automatically settable via the JSON file using the long name as the key — the user does not have to register it twice, but does need to communicate the key to operators |
| 7. Build manifest | Keep the sample's existing `meson.build` (which already wires `pkg-config doca-argp` via the parent sample build) | Yes. Do not switch to a hand-rolled Makefile for *"simplicity"* — it removes the version-check rail and breaks the standard build pattern |

The agent's anti-pattern alert: a *"swap doca-argp for getopt
because it'll be simpler"* refactor is **always** a regression
on a shipped sample — it breaks the standard CLI surface, drops
the `--json` integration silently, and removes the
error-taxonomy ladder in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
Per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
the agent must refuse this refactor and explain why before
producing any diff.

## run

Goal: execute the built binary against the user's installed
DOCA and confirm both the new flag and the standard surface
parse correctly.

Steps the agent should walk the user through:

1. **First run: `./<binary> --help`.** Confirm the operator-side
   listing shows the standard surface (`--device`,
   `--representor`, `--rep-list`, `--json`,
   `--sdk-log-level`) AND the user's new flag with the
   description from
   [`## modify`](#modify) slot 5. If the new flag is missing,
   registration never happened (likely a missing
   `doca_argp_register_param` call); if the standard flags are
   missing, `doca_argp_init` was skipped or replaced. Per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability),
   `--help` is the cheapest first signal.
2. **Second run: argv smoke.** Invoke
   `./<binary> --device <PCI> --my-flag <VALUE>` (substituting
   the user's actual PCIe address and value). Confirm the
   binary parses successfully and reaches its first
   non-Arg-Parser code path. A parse failure at this stage is
   one of the `DOCA_ERROR_*` rows in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
3. **Third run: JSON-config smoke.** Author a small
   `./my-config.json` with the user's new flag as a key plus
   one or two standard-surface keys (e.g. `"device":
   "0000:03:00.0"`), then invoke
   `./<binary> --json ./my-config.json`. Confirm the
   binary parses the same set of values it parsed in step 2.
   This confirms the JSON-config path covers the new flag,
   which is the load-bearing reason to reuse doca-argp.
4. **Capture the structured log on first failure.** Set
   `DOCA_LOG_LEVEL=trace` (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability))
   when any of the above three runs fails; the trace shows
   the per-param parse calls and surfaces the
   `--json` file read.

For the runtime version + `LD_LIBRARY_PATH` cross-checks that
underlie *"the program built but does nothing"*, see
[`doca-version TASKS.md ## run`](../../doca-version/TASKS.md#run).

## test

Goal: prove a doca-argp-using consumer is correct end-to-end —
the new flag works, the standard surface still works, and the
JSON-config path covers both — before claiming the *"wire
doca-argp into this sample"* journey is done.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the registration shape, the parameter type, the standard
surface coverage, or the JSON-config coverage. The loop
terminates when the user reports a `--help` smoke + an argv
smoke + a JSON-config smoke all pass AND the standard-surface
flags still behave the same as in a sibling unmodified sample.

Iteration shape:

1. **`--help` listing.** Confirm every registered param
   (standard surface + user-added) appears with its
   description. Missing rows mean registration never happened;
   extra rows mean a stale registration was not removed.
2. **Argv smoke for the new flag.** Pass the new flag with a
   valid value; confirm the value-callback fires and writes
   into the user-config struct as expected. Add a one-line
   log inside the callback per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
   signal 2 if the callback's effect is hard to observe.
3. **Argv smoke for the standard surface.** Re-run the
   sample with the standard flags only (`--device <PCI>` and
   any sibling-sample-mandatory flag) and confirm the
   sample's pre-modification behavior is intact. If the
   standard flags broke after the user's diff, the
   modification overrode a default registration.
4. **JSON-config smoke.** Drive the same configuration from
   `--json ./my-config.json`; confirm argv-equivalent
   behavior. An unknown JSON key surfaces as
   `DOCA_ERROR_NOT_SUPPORTED` (pedantic parsing) per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   — that's a JSON-file typo or a stale config, not a
   library bug.
5. **Cross-version run** (if the user has multiple installs):
   re-run steps 1-4 on each install; quote the version per
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)
   in the report.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `--help` does not show the new flag | The new flag's `doca_argp_register_param` either never ran or ran AFTER `doca_argp_start` | Re-walk the lifecycle table in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) and move the register call before `_start` |
| `--help` shows the new flag but argv smoke returns `DOCA_ERROR_INVALID_VALUE` | The registered parameter type does not match the value the user passed | Re-check the parameter-type table in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes); fix on the registration side OR the operator side, not both |
| Argv smoke passes; JSON-config smoke returns `NOT_SUPPORTED` | The JSON key is not the long name of any registered param (typo, or a renamed flag the JSON file did not catch up with) | Diff the registered long names against the JSON keys; rename whichever side is wrong |
| JSON-config smoke returns `IO_FAILED` | The path is wrong, the file is unreadable, or the JSON is malformed | Resolve at the OS layer (`ls -l <path>`; `cat <path> \| jq .`) before any code change |
| Standard surface broke after the modification | The user's diff overrode a default `doca_argp_init` registration (e.g. re-registered `--device` with a different callback) | Remove the redundant registration; the standard surface is owned by `doca_argp_init` and should not be re-registered |
| Same code passes on host A, fails on host B | Different DOCA version or different install profile (doca-argp not installed on host B) | Re-run [`## configure`](#configure) step 1 (presence + version) + [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test) four-way match on host B |

Loop termination: stop iterating once two consecutive iterations
of the same kind do not change the picture — the cause is
likely outside the Arg Parser layer. Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured `--help`, argv-smoke, and JSON-config-smoke
evidence.

## debug

Goal: when a `doca_argp_*` call returns a `DOCA_ERROR_*` (or
the program's CLI silently misbehaves), narrow the cause to a
specific layer before recommending any code change.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending
Arg-Parser-specific fixes. This skill's overlay names the
Arg-Parser-specific manifestation at layer 6 (program); the
library does not own runtime / driver layers because it has no
hardware accelerator behind it.

**Layer 6 (program) — Arg Parser overlay.**

- **Lifecycle order is the first hypothesis.** Per the
  lifecycle table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
  the most common first-app failure is registering a param
  after `doca_argp_start` (returns `BAD_STATE`), or calling
  `_start` twice in the same process. Walk the user's
  `*_main.c` against the lifecycle before looking elsewhere.
- **Type mismatch is the second hypothesis.** A flag that
  registers cleanly but rejects the operator's value with
  `DOCA_ERROR_INVALID_VALUE` is a type mismatch between the
  registered parameter type (per the parameter-type table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes))
  and the value the operator wrote. The fix is on the
  declaration side OR the operator side, not a retry.
- **Unknown JSON key is the third hypothesis.**
  `DOCA_ERROR_NOT_SUPPORTED` from `doca_argp_start` when
  `--json <path>` is in play (pedantic parsing) means the JSON
  file references a long name no registered param uses. Diff the
  registered long names against the JSON keys; one side is
  stale.
- **Unreadable JSON file is the fourth hypothesis.**
  `DOCA_ERROR_IO_FAILED` is an OS-layer problem (missing
  file, wrong permission, malformed JSON). Resolve via
  `ls -l <path>` and `cat <path> | jq .` before any code
  change.
- **Standard surface override is the fifth hypothesis.** If
  the user's modification broke a standard flag's behavior,
  the diff re-registered a name `doca_argp_init` already
  owns; remove the redundant registration.

Once the layer is identified, route to the matching debug verb
on the matching skill: install / build / link to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug);
version to
[`doca-version ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns (in the *downstream* code
the parsed values feed into) to
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
- **deploy.** Deploying doca-argp-using applications at scale
  (sample CI, distributing JSON-config files to operators) —
  out of scope for Phase 1.
- **manage / monitor.** Per-operator CLI usage analytics or
  config-file change auditing — outside doca-argp's surface;
  the library parses, it does not telemeter.
- **shell-completion generation.** Out of scope; the Arg Parser
  exposes `--help` but does not emit bash / zsh completion
  scripts. Operators wanting completion can layer a
  language-native completion generator on top.

## Command appendix

Every command below is **cross-cutting on DOCA Arg Parser** — it
answers a recurring class of question that comes up in the
verbs above. The agent should treat the *class* as
load-bearing; the worked example is a single instance. Run-as
user is the unprivileged user unless noted. Sudo is called out
per row.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env --json`
   for version + libraries + sample paths in one shot;
   `version-matrix.json` for *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the
   row. Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --exists doca-argp && echo ok` | `## configure` step 1 | Is the Arg Parser installed in this DOCA install profile at all? | `ok`. A non-zero exit = doca-argp not present; route to [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure) |
| `pkg-config --modversion doca-argp` | `## configure` step 1; `## build` slot 4 | What is the build-time Arg Parser version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-argp` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `ls /opt/mellanox/doca/samples/*/*_main.c \| head` | `## configure` step 2; `## modify` slot 1 | Which sample `*_main.c` files ship in this install, and which is the closest starting point for a CLI modification? | A list of `*_main.c` paths, one per shipped sample. Pick the closest in library context |
| `./<binary> --help` | `## run` step 1; `## test` step 1 | Does the built binary's `--help` listing show the standard DOCA CLI surface plus every newly registered param? | A help listing with `--device`, `--representor`, `--rep-list`, `--json`, `--sdk-log-level`, plus every user-added flag with its description |
| `cat <path> \| jq .` | `## run` step 3; `## debug` layer 6 | Is the `--json <path>` file valid JSON before we blame doca-argp for `IO_FAILED`? | A pretty-printed JSON tree. A `jq` parse error = malformed JSON, fix the file before any code change |
| `DOCA_LOG_LEVEL=trace ./<binary> --json <path>` | `## run` step 4; `## debug` layer 6 | What did the structured DOCA logger emit during the Arg Parser parse? | A trace-level line on every per-param parse call and on the JSON-config file read. Silence on a flag = the flag was never registered |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the Arg-Parser-specific rows on top.
