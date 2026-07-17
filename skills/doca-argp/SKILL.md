---
license: Apache-2.0
name: doca-argp
description: >
  Use this skill for hands-on DOCA Arg Parser CLI work on a
  shipped sample or new DOCA-using app — adding / removing /
  renaming flags; wiring `doca_argp_init` → register params →
  `doca_argp_start` → `doca_argp_destroy` in order; picking a
  parameter type from the full public enum
  (`DOCA_ARGP_TYPE_STRING`, `_INT`, `_BOOLEAN`, `_DEVICE`,
  `_DEVICE_REP`, `_DOUBLE` — six values, not three);
  preserving the standard `--device` / `--representor` /
  `--json` (`-j`; real flag is `--json`, NOT `--json-config`) /
  `--sdk-log-level` surface; or debugging
  `DOCA_ERROR_BAD_STATE` / `INVALID_VALUE` / `NOT_SUPPORTED` / `NOT_FOUND` /
  `IO_FAILED` from `doca_argp_*`. Trigger on implicit
  phrasings: "add a custom flag to a DOCA sample", "should I
  use getopt here", "BAD_STATE registering a new param", "my
  JSON config key is rejected", or "my sample's --json is
  ignored". Refuse and route elsewhere for variadic-flag /
  subcommand / shell-completion features, DOCA Core context,
  or DOCA Log internals.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU or
  ConnectX NIC attached. Reads the user's local install via
  `pkg-config doca-argp` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA Arg Parser

**Where to start:** This skill assumes DOCA is already installed
and the user is doing **hands-on CLI work** on a DOCA sample or
new DOCA-using app. Open [`TASKS.md`](TASKS.md) if the user wants
to *do* something (configure / build / modify / run / test /
debug); open [`CAPABILITIES.md`](CAPABILITIES.md) when the
question is *what can the Arg Parser express* on this version. If
the user has not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
about to rewrite a sample's CLI with `getopt` / `argparse` /
custom parsing instead of reusing the Arg Parser, read the
load-bearing rule in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
before any code change.

## Example questions this skill answers well

The CLASSES of Arg Parser questions this skill is built to
answer, each with one worked example. The agent should treat the
*class* as the load-bearing piece — the worked example is a
single instance.

- **"How do I add a new flag to a DOCA sample without breaking
  the standard CLI?"** — worked example: *"add `--my-flag` to
  `/opt/mellanox/doca/samples/doca_dma/dma_local_copy/` so the
  sample still accepts `--device <PCI>` and `--sdk-log-level
  <level>` the same way it did before"*. Answered by the
  reuse-the-Arg-Parser rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the register-before-start workflow in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **"Why does `doca_argp_param_set_*` return `BAD_STATE` on my
  second call?"** — worked example: *"registering a new param
  after `doca_argp_start` has already parsed argv"*. Answered by
  the lifecycle order in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the error-taxonomy row in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  for `DOCA_ERROR_BAD_STATE`.
- **"Can I drive a sample from a JSON file instead of a long
  command line?"** — worked example: *"point a sample at
  `./my-config.json` so the operator does not have to type out
  ten flags every time"*. Answered by the `--json <path>`
  integration in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the JSON-config-file workflow in
  [`TASKS.md ## modify`](TASKS.md#modify) and
  [`TASKS.md ## run`](TASKS.md#run).
- **"My `--my-flag X` value is rejected as `INVALID_VALUE` — why?"** —
  worked example: *"declared the param as `int` but passed
  `--my-flag 0x40`"*. Answered by the parameter-type table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the type-mismatch row in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
- **"Is `doca-argp` even on my installed DOCA?"** — worked
  example: *"a colleague's sample mentions doca-argp but I want
  to confirm before I depend on it"*. Answered by the presence
  + version-detection rule in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md).
- **"Should I use doca-argp here, or is this case actually
  outside its scope?"** — worked example: *"writing a host-side
  CLI tool that never calls a `doca_*` symbol"*. Answered by the
  path-selection rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  *Use doca-argp when … / Do not use doca-argp when …* bullets.

## Audience

This skill serves **external developers building or modifying
DOCA-using applications** — i.e., users whose code already calls
`doca_*` (directly in C/C++, or through FFI/bindings from
another language) and who need the standard DOCA CLI surface so
operators of the resulting binary do not have to relearn how to
invoke each sample. It is *not* for NVIDIA developers
contributing to the Arg Parser library itself.

**Language scope.** DOCA Arg Parser ships as a C library with
`pkg-config` module name `doca-argp`. The shipped samples are
written in C. C and C++ consumers are the canonical case; the
worked examples in `TASKS.md` assume that path. Other-language
consumers (Rust, Go, Python, …) consume the same `*.so` through
FFI or language-specific bindings; the skill's contribution in
that case is to keep the lifecycle, parameter-type, JSON-config,
standard-flag-surface, and error-taxonomy guidance
language-neutral, and to route the agent to the public C ABI as
the authoritative surface that any wrapper will eventually call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA Arg Parser
work, in any language. Concretely:

- Adding, removing, or renaming a CLI flag on a shipped DOCA
  sample or on a new app that wants to share the standard DOCA
  CLI surface (`--device <PCI>`, `--representor <name>`,
  `--rep-list`, `--json <path>`, `--sdk-log-level
  <level>`).
- Wiring `doca_argp_init` / `doca_argp_start` /
  `doca_argp_destroy` into a `main()`, including the
  register-before-start lifecycle and the cleanup-on-exit
  contract.
- Registering a `doca_argp_param` (short name, long name, value
  callback, description for `--help`) with a parameter type
  drawn from the small public set (string, int, bool flag, JSON
  config file).
- Reading complex configurations from a JSON file via the shared
  `--json <path>` flag instead of expanding the command
  line.
- Confirming the build- and runtime-side Arg Parser version on
  the user's install (`pkg-config --exists doca-argp`,
  `pkg-config --modversion doca-argp`) before depending on it.
- Debugging a `DOCA_ERROR_*` returned from a `doca_argp_*` call
  (lifecycle vs. type-mismatch vs. unknown JSON key vs.
  unreadable file).
- Designing or extending non-C bindings (Rust, Go, Python, …)
  that wrap the Arg Parser C ABI — for the lifecycle,
  parameter-type, JSON-config, and standard-flag rules the
  wrapper must honor.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, or non-Arg-Parser library questions. For those,
use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
Arg-Parser-specific material lives in two companion files:

- `CAPABILITIES.md` — what the Arg Parser can express on this
  version: the param-registration model, the small set of
  public parameter types, the standard DOCA CLI surface every
  sample shares, the `--json <path>` file integration,
  the register-before-start lifecycle, the Arg Parser error
  taxonomy (mapped onto the cross-library `DOCA_ERROR_*` set),
  the observability surface (the `--help` output and the
  DOCA Log channel), and the safety / path-selection policy
  (when reusing doca-argp is mandatory; when a language-native
  parser is the right answer).
- `TASKS.md` — step-by-step workflows for the six in-scope Arg
  Parser verbs: `configure`, `build`, `modify`, `run`, `test`,
  `debug`. Plus a `Deferred task verbs` block that points
  out-of-scope questions at the right next skill.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location. It does not cover installing
DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA Arg Parser usage code, in any language.**
  The verified Arg Parser usage is the `*_main.c` file in every
  shipped DOCA sample at
  `/opt/mellanox/doca/samples/<library>/<sample>/`. The agent's
  job is to route the user to that file and prescribe a
  minimum-diff modification on it via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the Arg-Parser-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, `Cargo.toml`, …) parked inside the skill.
  The agent constructs the build manifest *in the user's
  project directory* against the user's installed DOCA, where
  `pkg-config --modversion doca-argp` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree,
  even one labeled "reference", is misleading: users will read
  it as buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope.
2. **For the param-registration model, parameter types, the
   standard DOCA CLI surface, the `--json <path>` rule,
   the register-before-start lifecycle, error taxonomy,
   observability, and the path-selection / safety policy, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "Arg-Parser-specific
guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The Arg
  Parser URL is
  `https://docs.nvidia.com/doca/sdk/DOCA-Arg-Parser/index.html`;
  the canonical on-disk usage example is any sample's
  `*_main.c` under `/opt/mellanox/doca/samples/`.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, and the *I have no install yet* path
  with the public NGC DOCA container. This skill assumes its
  preconditions are satisfied.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  the Arg-Parser-specific presence-check overlay.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer
  / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library:
  the canonical `pkg-config` + meson build pattern, the
  universal modify-a-shipped-sample first-app workflow, the
  universal lifecycle, the cross-library `DOCA_ERROR_*`
  taxonomy, and the program-side debug order. This skill layers
  Arg-Parser specifics on top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Arg-Parser-specific debug (lifecycle
  violations, type-mismatch on a registered param, unknown JSON
  key) overlays on top of that ladder.
