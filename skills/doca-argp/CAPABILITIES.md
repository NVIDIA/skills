# DOCA Arg Parser capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(param model / parameter types / standard CLI surface / JSON
config / version / errors / safety) and read that section
end-to-end. The tables in each section are the load-bearing
content; the prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern
(the verbs `configure / build / modify / run / test / debug`),
jump to [TASKS.md](TASKS.md). For the canonical DOCA
version-handling rules that this skill layers an Arg-Parser
overlay on top of, see
[`doca-version`](../../doca-version/SKILL.md).

## Pattern overview

Every Arg Parser question this skill teaches resolves into one
of FIVE patterns. The patterns are CLASSES — they apply across
every DOCA release and every sample, not just the worked
examples shown.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Reuse, do not rewrite | The user is modifying a shipped DOCA sample's CLI (adding / removing / renaming a flag) and is tempted to swap the Arg Parser for `getopt` / `argparse` / hand-rolled parsing | [`## Capabilities and modes`](#capabilities-and-modes) reuse rule + [TASKS.md ## modify](TASKS.md#modify) |
| 2. Register before start | New params are registered against the Arg Parser instance BEFORE `doca_argp_start` parses argv; registering after parse is the most common first-app failure | [`## Capabilities and modes`](#capabilities-and-modes) lifecycle table + [TASKS.md ## configure](TASKS.md#configure) |
| 3. Pick the parameter type | Choose from the small public set (string, int, bool flag, JSON config file); the type drives both argv validation and JSON-config validation | [`## Capabilities and modes`](#capabilities-and-modes) parameter-type table + [TASKS.md ## modify](TASKS.md#modify) |
| 4. Inherit the standard CLI surface | Keep `--device`, `--representor`, `--rep-list`, `--json` (`-j`), `--sdk-log-level` working the same way they do in the sibling samples; users learn one CLI for all of DOCA | [`## Capabilities and modes`](#capabilities-and-modes) standard-surface table + [TASKS.md ## modify](TASKS.md#modify) |
| 5. Diagnose an Arg Parser error | Map symptom (`BAD_STATE`, `INVALID_VALUE`, `NOT_SUPPORTED`, `NOT_FOUND`, `IO_FAILED`) to root cause without leaving the Arg Parser layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **The shipped sample's `*_main.c` is the canonical reference,
  every time.** doca-argp does not ship its own dedicated
  sample tree; instead, every DOCA sample's `*_main.c` (under
  `/opt/mellanox/doca/samples/<library>/<sample>/`) is a
  working, version-matched Arg Parser usage. Quote the
  installed file, not a remembered API shape.
- **The standard CLI surface is the user's mental model,
  not a stylistic preference.** When a user invokes any DOCA
  sample, they expect `--device`, `--representor` (or
  `--rep-list`), `--json` (`-j`), and `--sdk-log-level` to
  behave the same way as in the sibling samples. Adding a flag
  on top of that surface is fine; replacing the surface breaks
  the cross-sample mental model and is a regression.

## Capabilities and modes

DOCA Arg Parser is a **small, foundational CPU-side library**.
It has no hardware accelerator behind it, no DOCA Core context,
and no progress engine — it owns *parsing*, not *execution*. On
top of that small surface, it layers a param-registration
lifecycle, a small parameter-type set, the shared DOCA standard
flag surface, and the JSON-config file integration.

**The Arg Parser lifecycle.** Every consumer follows the same
register-before-start order. The agent must not invent steps;
the public surface is closed.

| Step | Call | What it does | Order constraint |
| --- | --- | --- | --- |
| Init | `doca_argp_init(<program-name>, <user-config>)` | Creates the per-process Arg Parser instance; allocates the default standard-flag set | Must be the FIRST `doca_argp_*` call in the program |
| Register | `doca_argp_param_create` → `doca_argp_param_set_*` (short name, long name, value-callback, description, type) → `doca_argp_register_param` | Adds one `doca_argp_param` to the instance for each app-specific flag | Must happen AFTER `doca_argp_init` and BEFORE `doca_argp_start`; registration after start is silently ignored or returns `BAD_STATE` |
| Start (parse) | `doca_argp_start(argc, argv)` | Walks argv (and any `--json <path>` / `-j <path>` file), calls each matching param's value-callback with the parsed value, fails fast on the first invalid value | Runs ONCE per process; a second `_start` returns `BAD_STATE` |
| Destroy | `doca_argp_destroy()` | Frees the per-process instance and every registered `doca_argp_param` | Should pair with `_init` on every exit path, including error paths |

**Parameter types.** The full public `enum doca_argp_type` the Arg
Parser exposes. Picking the right one is what makes argv
validation and JSON-config validation Just Work. The enum has
**six** real values (`DOCA_ARGP_TYPE_STRING`,
`DOCA_ARGP_TYPE_INT`, `DOCA_ARGP_TYPE_BOOLEAN`,
`DOCA_ARGP_TYPE_DEVICE`, `DOCA_ARGP_TYPE_DEVICE_REP`,
`DOCA_ARGP_TYPE_DOUBLE`) — do not silently omit the latter three.

| Type (enum) | Argv shape | JSON shape | Typical use |
| --- | --- | --- | --- |
| `DOCA_ARGP_TYPE_STRING` | `--my-flag VALUE` | `"my-flag": "VALUE"` | File paths, free-form strings |
| `DOCA_ARGP_TYPE_INT` | `--my-flag 1234` | `"my-flag": 1234` | Queue depth, message size cap, max-num-tasks |
| `DOCA_ARGP_TYPE_BOOLEAN` | `--my-flag` (presence sets true) | `"my-flag": true` | Enable a code path that is off by default |
| `DOCA_ARGP_TYPE_DEVICE` | `--my-flag <PCI>` | `"my-flag": "<PCI>"` | A PCIe-address-bearing flag whose value the Arg Parser opens as a `doca_dev` for you (e.g. `--device 0000:03:00.0`) — preferred over hand-rolling the device open via `DOCA_ARGP_TYPE_STRING` |
| `DOCA_ARGP_TYPE_DEVICE_REP` | `--my-flag <rep>` | `"my-flag": "<rep>"` | A representor-name-bearing flag whose value the Arg Parser opens as a `doca_dev_rep` for you (e.g. `--representor pf0vf0`) — DPU-side counterpart to `_TYPE_DEVICE` |
| `DOCA_ARGP_TYPE_DOUBLE` | `--my-flag 12.5` | `"my-flag": 12.5` | Floating-point knobs (e.g. timeouts in seconds, fractional caps) |
| JSON config file *(special, not a type)* | `--json /path/to/file.json` (or `-j`) | n/a (this IS the JSON file) | Drive any of the above from a file instead of expanding argv. The real flag is `--json` / `-j`, NOT `--json-config` — do not invent the longer name. |

**The standard DOCA CLI surface.** Every shipped sample already
registers these (via `doca_argp_init` and a small set of
default params) so the user learns ONE CLI for all of DOCA.
Modifications must preserve this surface.

| Flag | Value type | Meaning |
| --- | --- | --- |
| `--device <PCI>` | String (PCIe address, e.g. `0000:03:00.0`) | Which `doca_dev` the program opens |
| `--representor <name>` | String (representor name, e.g. `pf0vf0`) | Which `doca_dev_rep` the DPU side opens (Comch and similar libraries) |
| `--rep-list` | Bool flag | Print the visible representors and exit (DPU-side discovery) |
| `--json <path>` / `-j <path>` | String (path to a JSON file) | Read parameter values from a JSON file instead of expanding argv. The flag is registered as `--json` (`-j`) by `doca_argp.cpp::register_param` (the internal static helper; the public entry point is `doca_argp_register_param`); the bundle previously called this `--json-config`, which is not the real flag. |
| `--sdk-log-level <level>` | String (one of the DOCA Log level names) | Set the SDK-side DOCA Log threshold for this run |

**JSON-config integration.** The `--json <path>` flag (also
short-form `-j <path>`; the long name is `--json`, NOT
`--json-config`) is not optional cosmetic detail — shipped
samples with non-trivial configurations expect operators to
drive them from a file (the command line becomes unreadable
past ~5 flags). The JSON keys match the registered long names;
the JSON value types match the registered parameter types; an
unknown JSON key fails with `DOCA_ERROR_NOT_SUPPORTED` (pedantic parsing) per
[`## Error taxonomy`](#error-taxonomy).

**Path selection — when doca-argp is the right answer.** This
is a small library, but its scope is sharp. Walk this rule
before recommending the Arg Parser path.

| Use doca-argp when … | Do not use doca-argp when … |
| --- | --- |
| Modifying a shipped DOCA sample's CLI (add / remove / rename a flag) — reusing the Arg Parser keeps the sample's CLI consistent with every sibling sample | The program never calls a `doca_*` symbol — there is no DOCA-side interaction to begin with, so a language-native CLI parser is the right answer (and avoids dragging in `libdoca-argp` for no benefit) |
| Building a new DOCA-using app that wants operators to learn the standard `--device` / `--representor` / `--json` (`-j`) / `--sdk-log-level` CLI conventions instead of inventing a fresh one | The user genuinely needs a POSIX getopt feature the Arg Parser does not cover (e.g. optional-argument optional-value forms) — and is willing to accept the cost of breaking sample-CLI consistency for that specific need; even then, prefer layering doca-argp + the extra parser, not replacing it |
| Driving a complex configuration from a JSON file (`--json <path>` / `-j <path>`) — the file format and key validation come for free | The CLI surface is interactive (REPL, prompt loop) — that is not what an arg parser is for; reach for a TUI / REPL library instead |

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The Arg-Parser-specific overlay** is:

- **Presence is the first check; version is the second.**
  doca-argp is a small, foundational library — agents tend to
  assume it is always present. It is not always installed in
  every package profile. Use `pkg-config --exists doca-argp`
  for the presence check FIRST, then `pkg-config --modversion
  doca-argp` for the version. Per the cross-cutting rule in
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the build-time `pkg-config` value is the authority for *"is
  this library available for me to link against?"*.
- **`doca-argp.pc` and `doca-common.pc` must both match
  `doca_caps --version`** at the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  A common partial-install pattern after a DOCA upgrade is
  that `doca-argp.pc` lingers from the previous release; route
  to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before any Arg-Parser-layer diagnosis.
- **Headers in $(pkg-config --variable=includedir doca-common)
  win over public docs.** Per the headers-win-over-docs rule
  in [`doca-version`](../../doca-version/SKILL.md), if a
  public Arg Parser doc page mentions a `doca_argp_*` symbol
  that is not in the installed `doca_argp.h`, the headers
  describe what *this* install can call; the docs describe
  what *some* release shipped. The agent must quote the
  headers, not the docs URL, when the two disagree.

## Error taxonomy

Arg-Parser-specific overlays on the cross-library
`DOCA_ERROR_*` taxonomy. The cross-library taxonomy itself
lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *Arg Parser surface* meaning that the
agent must disambiguate before falling back to the
cross-library response.

| Error | Arg Parser context where it shows up | Arg-Parser-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | `doca_argp_register_param` after `doca_argp_start`; a second `doca_argp_start` in the same process; any `doca_argp_*` call after `doca_argp_destroy` | Lifecycle violation. Walk the lifecycle table in [`## Capabilities and modes`](#capabilities-and-modes); the most common case is the program registering an extra param inside the value-callback of another param (i.e. during `_start`, not before). |
| `DOCA_ERROR_INVALID_VALUE` | `doca_argp_start` when an argv value (or JSON-config value) does not match the registered param's declared type | Type mismatch. Re-check the registered type vs. the value the user is passing (e.g. param declared as `int` but operator wrote `--my-flag 0x40`). The fix is on the declaration side OR the operator side, not a retry. |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_argp_start` when a JSON-config key is not the long name of any registered param (pedantic parsing rejects the unknown field — see `doca_argp.cpp::json_keys_validation`) | The JSON file references a flag the program never registered. Either the key is a typo, or the program is older than the JSON config it is being fed. Diff the registered long names against the JSON keys. |
| `DOCA_ERROR_IO_FAILED` | `doca_argp_start` when `--json <path>` (or `-j <path>`) points at a file the process cannot open or read | File-system failure (missing path, wrong permission, JSON syntax error). Resolve at the OS layer (`ls -l <path>`; `cat <path> \| jq .`) before any code change. |
| `DOCA_ERROR_NOT_SUPPORTED` (operator-side) | `doca_argp_start` against argv containing an unknown long / short name (an unsupported program flag, per the `doca_argp_start` doc in `doca_argp.h`) | The operator passed a flag the program never registered. The fix is on the operator side; the program may surface its own `--help` listing all registered params via the per-param descriptions. |

The agent's rule: **never recommend a retry loop on a
`doca_argp_*` `DOCA_ERROR_*`**. Every row above is an
authoring or operator mistake, not a transient state — the fix
is to correct the call site, the registration, the JSON key, or
the file path, not to retry.

## Observability

The Arg Parser's observability surface is small but
load-bearing: the program's own `--help` output (synthesized
from the registered param descriptions) is what the operator
reads, and DOCA Log is what the program emits at parse time.

Three primary signals the agent should reach for:

1. **The `--help` output.** Every Arg Parser instance
   auto-generates a `--help` listing from `doca_argp_init`'s
   default standard flags plus every registered param's
   description. The agent should ask the user to run
   `./<binary> --help` before any other debug step — *if the
   new flag is not listed, registration never happened*; *if
   the standard flags are not listed, `doca_argp_init` was
   skipped*.
2. **The parsed-value callback fires.** Each registered
   `doca_argp_param`'s value-callback runs once per matched
   flag (argv or JSON-config). A param that registers cleanly
   but whose callback never fires means the operator never
   passed the flag — *not* that the library dropped it. Add a
   one-line log inside the callback before assuming a parser
   bug.
3. **DOCA Log lines from the Arg Parser layer.** Per the
   cross-cutting observability primitives
   (`DOCA_LOG_LEVEL=trace`, `--sdk-log-level`, the trace build
   flavor) in
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability),
   the Arg Parser routes its own messages through DOCA Log; a
   trace-level run shows the per-param parse calls and the
   JSON-config file read.

For the install-tree observability (logger names, package
layout) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

The Arg Parser's safety surface is **lower-stakes than other
DOCA libraries** — it runs entirely in the user's process,
holds no kernel-side resources, requires no privileges, and
opens no devices. There is no permission matrix to walk, no
representor visibility check, no PCIe address reachability
gate. The agent must not invent one.

The load-bearing safety rule is structural, not access-control
shaped: **when modifying a shipped DOCA sample's CLI, reuse the
Arg Parser; do not replace it with `getopt` / `argparse` /
hand-rolled parsing**. The reasons this is a safety rule, not
a stylistic preference:

- **Cross-sample CLI consistency is a user contract.**
  Operators of DOCA samples carry mental models across
  libraries (`--device`, `--representor`, `--json` / `-j`,
  `--sdk-log-level` behave the same in every sample). A
  modified sample that silently breaks that contract becomes a
  trap for operators and a debugging time-sink that looks like
  a library bug.
- **`--json <path>` (`-j <path>`) is shared infrastructure** (the
  long name is `--json`, NOT `--json-config`). Sample
  CI, sample documentation, and operator runbooks all assume
  the JSON-config path works. A hand-rolled parser drops the
  JSON path silently; `--help` still looks right; the failure
  surface is *"my JSON file is ignored"*, which the operator
  has no obvious place to file against.
- **Lifecycle violations are caught only when doca-argp owns
  parsing.** The `BAD_STATE` / `INVALID_VALUE` / `NOT_SUPPORTED` /
  `NOT_FOUND` / `IO_FAILED` ladder in
  [`## Error taxonomy`](#error-taxonomy) only fires when the
  program goes through `doca_argp_*` — a hand-rolled parser
  re-creates this ladder by accident, badly.

The narrow exception (path-selection rule in
[`## Capabilities and modes`](#capabilities-and-modes)): when
the program never calls a `doca_*` symbol at all, doca-argp is
not the right tool — a language-native parser is — and pulling
in `libdoca-argp` adds a build-time dependency for no agent or
user benefit. The agent must distinguish *"DOCA-using app
modifying its CLI"* (reuse mandatory) from *"non-DOCA CLI
tool"* (language-native parser is correct).

## Deferred topic boundaries

This skill scopes itself to the DOCA Arg Parser library.
Adjacent topics the agent will get asked but should route
elsewhere:

- **Generic CLI-parser feature requests** (variadic flags,
  subcommand routing, shell-completion generation) — outside
  this skill. The Arg Parser exposes a deliberately small
  public surface; route the user to a language-native parser
  layered on top if a feature is genuinely required.
- **DOCA Core context, progress engine, and `doca_mmap`
  internals** — owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  Arg Parser is upstream of the Core context (it parses the
  CLI that selects which device the Core context will open);
  it does not redefine the Core context.
- **DOCA Log internals** (registry names, per-source level
  filters) — owned by the public DOCA Log guide reachable
  through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  Arg Parser registers `--sdk-log-level` as a default flag but
  does not own the log layer it configures.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the Arg Parser overlay, not the taxonomy
  itself.
- **Cross-cutting debug ladder** (install / version / build /
  link / runtime / program / driver) — owned by
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug).
  This skill's `## debug` overlays the program layer.
