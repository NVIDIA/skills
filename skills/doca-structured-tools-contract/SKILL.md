---
license: Apache-2.0
name: doca-structured-tools-contract
description: >
  Use this skill whenever another DOCA skill says "prefer the
  structured tool per doca-structured-tools-contract", or when the
  user wants a one-shot answer that consolidates info multiple
  manual commands would produce — DOCA env / version / devices /
  capabilities / validate / host vs DPU state. Trigger even when
  the user does not explicitly mention "structured tool" or
  "doca-env --json" — typical implicit phrasings include "is there
  one command that tells me everything about my DOCA install",
  "what version is X capability available since", "every PF/VF/SF
  visible on this BlueField with PCIe address", "will this pipe
  pass validate before commit", "diff host vs DPU state", or "why
  does the agent give a one-line answer on host A and five commands
  on host B". Refuse and route elsewhere for general DOCA
  orientation, specific library API how-to, or install-from-scratch
  guidance — those belong to the per-library skill,
  doca-public-knowledge-map, or doca-setup.
metadata:
  kind: knowledge
compatibility: >
  No DOCA install required to read this skill (it is an overlay
  loaded against any DOCA artifact skill); the validation steps
  within DO require a live DOCA install at /opt/mellanox/doca.
---

# DOCA structured-tools contract

**Where to start:** Reach for this skill whenever a workflow in
another skill says *"prefer the structured tool per
`doca-structured-tools-contract`"*. Read
[`## The agent behavior contract`](#the-agent-behavior-contract)
first; then drill into the matching schema in [`## Schemas`](#schemas).
If the host has the structured tool, prefer its output. If it does
not, fall back to the manual command chain in the same schema
section. **Always report which path was taken** so the user can fix
the gap (or so a future bundle update can detect that the structured
path was never tried).

## Example questions this skill answers well

These are the question SHAPES this skill is designed to route, with
one worked example each. The agent should treat the *shape* as the
load-bearing piece — the worked example is a single instance.

- **"Is there a single command that tells me everything about my
  DOCA install?"** — worked example: *"version, devices, libraries
  installed, sample paths, drivers, hugepages — all at once"*.
  Answered by the `doca-env --json` schema in
  [`## Schemas`](#schemas) and the four-step fallback chain in the
  same section.
- **"How do I look up which DOCA version a capability requires
  without fetching a doc page?"** — worked example: *"is symmetric
  RSS hash mode in DOCA 2.6.0"*. Answered by the
  `version-matrix.json` schema in [`## Schemas`](#schemas), with
  the manual fallback being to fetch the matching per-library doc
  page via [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
- **"What does my hardware actually look like to DOCA — every PCIe
  function, every representor?"** — worked example: *"every PF / VF
  / SF visible on this BlueField, with PCIe address and link
  state"*. Answered by the `collect-host-state` / `collect-dpu-state`
  schema in [`## Schemas`](#schemas), with the manual fallback being
  `devlink dev show` + `lspci | grep Mellanox` + `ip -j link`.
- **"How do I know my Flow / RDMA / DMS / etc. spec is valid before
  I program hardware with it?"** — worked example: *"will this Flow
  pipe pass validate before commit"*. Answered by the
  `validate-before-commit` schema in [`## Schemas`](#schemas), with
  the manual fallback being the library's own constructor-time
  validation surface (e.g. for DOCA Flow, treat the
  `DOCA_ERROR_INVALID_VALUE` / `DOCA_ERROR_NOT_SUPPORTED` return
  from `doca_flow_pipe_create` as the validate signal — the
  public Flow header does not ship a separate `doca_flow_pipe_validate`
  call at the bundle-aligned release, and the agent must not
  invent one).
- **"My agent is acting differently on different hosts — is one of
  them missing the helpers?"** — worked example: *"the agent on host
  A gave a one-line answer; the agent on host B walked five manual
  commands"*. Answered by the report-which-path rule in
  [`## The agent behavior contract`](#the-agent-behavior-contract) —
  if the agent on host A did not say "structured tool used", the user
  should ask explicitly.

If the question fits a different shape (how to write code, how to
debug a crash, how to set up an env), route to the matching skill
instead — see [`AGENTS.md`](../../AGENTS.md) for the routing table.

## When to load this skill

Load this skill whenever another skill's workflow tells the agent to
*prefer the structured tool*, OR whenever the user's question implies
they want a single one-shot answer that consolidates information
multiple manual commands would otherwise produce.

Concretely:

- A library / service / tool skill's Command appendix references
  this skill in its first column.
- The user asks "is there one command that tells me X about my DOCA
  install" (env / devices / version / capabilities / hardware
  topology).
- The user asks "how do I know X is valid before I commit" for
  any DOCA library that has a validate-before-commit call.
- The agent has computed the manual fallback answer and wants to
  *also* surface the equivalent structured-tool one-liner so the
  user can adopt it next time.

Do **not** load this skill for general DOCA orientation, for
specific library API questions, or for install-from-scratch
guidance. For those, use the matching library skill +
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)
+ [`doca-setup`](../doca-setup/SKILL.md).

## Ground rules for any agent using this skill

1. **Detect first; never assume the tool is present.** Each schema
   below names the *probe command* that decides whether the
   structured tool is installed on this host. Run the probe before
   reading the schema's output as authoritative.
2. **Prefer structured when present; fall back to manual when not.**
   When the probe succeeds, the structured tool's JSON output is
   the source of truth. When the probe fails, walk the manual
   command chain in the same schema section and synthesize the
   equivalent answer.
3. **Report which path you took.** Always tell the user at the
   start of the answer: *"using structured `doca-env --json`"* OR
   *"falling back to manual chain (no structured helpers
   installed)"*. The user adopting the helpers is a downstream
   improvement; the user knowing whether they were used is the
   *now*.
4. **Schemas are locked here; per-skill overlays are NOT.** A
   library / service / tool skill MAY add a per-skill row to its
   own Command appendix that *uses* a schema; it MUST NOT redefine
   the schema. If a schema needs to grow, the change happens here
   first and every Command appendix that consumes it inherits the
   change automatically.
5. **Never invent a JSON field that is not in the schema.** The
   structured tool's output is exactly the shape this contract
   says it is. If the user pastes JSON that contains a field not in
   the schema, treat the extra field as advisory and quote the
   official schema as the boundary.
6. **Schemas describe contracts, not implementations.** The
   executables that satisfy these contracts are deferred to a
   subsequent PR on the maintainer roadmap. This skill exists so
   every other skill in the bundle can be *infra-aware* before the
   executables ship.

## The agent behavior contract

The contract is a four-step loop the agent runs every time a skill's
Command appendix references this contract:

1. **Detect.** Run the probe command listed in the schema section
   for the relevant tool. Examples: `command -v doca-env`,
   `test -f /opt/mellanox/doca/share/version-matrix.json`,
   `command -v doca-capability-snapshot`. Probes are read-only and
   safe to run on any host.
2. **Prefer.** If the probe succeeds, invoke the structured tool
   and parse its JSON per the schema in [`## Schemas`](#schemas).
   The output is the authoritative answer for this turn. Do NOT
   also run the manual chain "to double check" — the contract is
   that the structured tool replaces the chain.
3. **Fall back.** If the probe fails, walk the manual command
   chain documented in the same schema section. Synthesize the
   answer by combining the manual command outputs in the order the
   chain lists them. If a manual command is unavailable on the
   host, surface that as a gap and route the user to the matching
   skill (typically [`doca-setup`](../doca-setup/SKILL.md)).
4. **Report.** Open the answer with one of:
   - *"Using structured `<tool>` (path: `<path>`)."* — when the
     probe succeeded.
   - *"Falling back to manual chain (no structured `<tool>` on
     this host)."* — when the probe failed. Plus a one-line note
     pointing the user at how to install the helpers when they
     become available.

The report step is the bundle's signal to the user that the agent
*tried* the helpers, not just the manual chain. Without it, an
agent that always falls back looks identical to an agent that
never knew the helpers existed.

## Schemas

Each subsection below names ONE structured tool the bundle expects
to interoperate with, gives its detection probe, names its top-level
JSON shape, and lists the manual command chain the agent walks when
the probe fails.

### doca-env --json schema

**Detection probe:** `command -v doca-env`. The structured tool, if
installed, lives at the same `$PATH` location as `doca_caps` (i.e.
under the DOCA install tree's `bin/`).

**Top-level shape (JSON object):**

| Field | Type | Notes |
| --- | --- | --- |
| `version` | object | `pkg_config` / `applications_version` / `doca_caps` / `bfb` (string \| null) / `consistent` (bool) |
| `devices` | array of object | one entry per visible PCIe function: `pcie_address` (e.g. `0000:03:00.0`), `kind` (`PF` \| `VF` \| `SF`), `name`, `representor_of` (string \| null), `state` (`active` \| `down` \| `unknown`), `mtu` (number) |
| `libraries` | array of object | one entry per public DOCA library: `pkg_config_name`, `installed` (bool), `pc_path` (string \| null) |
| `sample_paths` | array of object | one entry per library: `library`, `path` (the on-disk samples root) |
| `drivers` | object | `mlx5_core_loaded` (bool), `mlx5_ib_loaded` (bool), `kernel_version` (string) |
| `hugepages` | object | `available_2m` (number), `available_1g` (number), `mount_point` (string \| null) |
| `host_kind` | string | one of `host` \| `bluefield` \| `unknown` |
| `bf_mode` | string \| null | one of `smartnic` \| `dpu` \| `switch` \| `null` (when `host_kind != bluefield`) |

**Manual fallback chain** (run in order; combine the outputs to
synthesize the same answer):

1. `pkg-config --modversion doca-common` → `version.pkg_config`
2. `cat /opt/mellanox/doca/applications/VERSION` → `version.applications_version`
3. `doca_caps --version` → `version.doca_caps`
4. `doca_caps --list-devs` → `devices` array (parse PCIe address + kind + representor)
5. `PCDIR="$(dirname "$(find /opt/mellanox/doca -name doca-common.pc -print -quit)")"; for pc in "$PCDIR"/*.pc; do pkg-config --exists "$(basename "$pc" .pc)" && echo "$pc"; done` → `libraries` array. `PCDIR` is commonly `/opt/mellanox/doca/lib/<arch>-linux-gnu/pkgconfig` on DOCA 3.3+, or `/opt/mellanox/doca/infrastructure/lib/pkgconfig` on legacy / split-profile installs; deriving it from the live `doca-common.pc` (equivalently `pkg-config --variable=pcfiledir doca-common`) avoids the empty-glob trap of hard-coding `infrastructure/`.
6. `ls /opt/mellanox/doca/samples/` → `sample_paths` array
7. `lsmod | grep -E '^mlx5_(core|ib)'` and `uname -r` → `drivers` object
8. `cat /proc/meminfo | grep -i Huge` → `hugepages` object
9. `dmidecode -s system-product-name` (or `cat /proc/device-tree/model` on BlueField) → `host_kind`
10. `mlxconfig -d <pcie> q INTERNAL_CPU_MODEL` → `bf_mode` (when `host_kind == bluefield`)

### version-matrix.json schema

**Detection probe:** `test -f /opt/mellanox/doca/share/version-matrix.json` (or a sibling location documented at install time by the file that ships it).

**Top-level shape (JSON object):**

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | string | semver of THIS contract; bumps on schema changes |
| `generated_at` | string | ISO-8601 timestamp of when the matrix was generated |
| `entries` | array of object | one row per (library, capability) pair |

**Per-entry shape:**

| Field | Type | Notes |
| --- | --- | --- |
| `library` | string | `pkg-config` module name (`doca-flow`, `doca-rdma`, `doca-comch`, …) |
| `capability` | string | machine-readable cap name; the per-library skill's Command appendix lists which `doca_<lib>_cap_*` query this maps to |
| `display_name` | string | human-readable label; the agent quotes this when reporting |
| `min_doca_version` | string | first DOCA release in which the capability was available (semver) |
| `max_doca_version` | string \| null | last DOCA release in which the capability was available (null = still available) |
| `source_url` | string | the public docs URL the row was derived from |
| `source_quote` | string | the exact prose from the public docs that established the row |

**Manual fallback chain:**

1. Identify the library + capability the user asked about (via the
   matching library skill's CAPABILITIES.md ## Capabilities and modes
   table).
2. Fetch the matching per-library doc page via
   [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
3. Search the page for the capability name; extract the *"available
   since"* prose; quote it verbatim.
4. Cross-check against `pkg-config --modversion doca-<library>` on
   the user's host; if the installed version is older than the
   *"available since"* line, the capability is not on this install
   regardless of what the public docs say.

### capability-snapshot schema

**Detection probe:** `command -v doca-capability-snapshot`. The
structured tool, if installed, lives at the same `$PATH` location as
`doca_caps`.

**Top-level shape (JSON object):**

| Field | Type | Notes |
| --- | --- | --- |
| `snapshot_at` | string | ISO-8601 timestamp |
| `doca_version` | string | `doca_caps --version` at snapshot time |
| `host_kind` | string | `host` \| `bluefield` |
| `devices` | array of object | one entry per `doca_devinfo`: `pcie_address`, `library_capabilities` (map of library → list of capability flags) |

**Manual fallback chain:**

1. `doca_caps --list-devs` → device enumeration
2. For each device + each library of interest: invoke the
   library-specific `doca_<lib>_cap_*` query family from a small
   test program (the agent walks the user through writing it, per
   the library's own `## test` workflow).

### validate-before-commit schema

**Detection probe:** `command -v doca-validate` (cross-library) OR the
library-specific constructor-time validation surface (e.g. for Flow,
the `DOCA_ERROR_INVALID_VALUE` / `DOCA_ERROR_NOT_SUPPORTED` return
from `doca_flow_pipe_create` — the public Flow header at this
release does not ship a separate `doca_flow_pipe_validate` symbol;
the agent uses the constructor-failure path as the validate signal).
The structured tool wraps the library-specific calls and returns a
uniform JSON result.

**Top-level shape (JSON object):**

| Field | Type | Notes |
| --- | --- | --- |
| `library` | string | which DOCA library the spec is for |
| `spec_path` | string | path on disk to the spec being validated |
| `result` | string | `pass` \| `fail` \| `skip` |
| `checks` | array of object | per-check breakdown: `name`, `status` (`pass` \| `fail`), `details` (string), `remediation` (string \| null) |

**Manual fallback chain:**

1. Find the library-specific validate surface in the matching skill's
   `## test` workflow. For some libs this is a dedicated `_validate`
   call; for others (notably DOCA Flow) it is the
   constructor-time validation embedded in the `_create` / `_cfg`
   call family (e.g. `doca_flow_pipe_create` for Flow) whose
   `DOCA_ERROR_INVALID_VALUE` / `DOCA_ERROR_NOT_SUPPORTED` return
   is the validate signal. The lib's per-skill `## test` anchor
   names which.
2. Invoke it before any commit / create / submit call.
3. Treat a non-`DOCA_SUCCESS` return as `result: fail`; map the
   `doca_error_get_descr()` text into a single `checks` entry.

### collect-host-state and collect-dpu-state schemas

**Detection probes:** `command -v doca-collect-host-state` (run on
the host side) and `command -v doca-collect-dpu-state` (run on the
BlueField side).

**Top-level shape (JSON object), shared by both:**

| Field | Type | Notes |
| --- | --- | --- |
| `side` | string | `host` \| `dpu` |
| `doca_version` | string | `doca_caps --version` |
| `firmware_version` | string | output of `flint -d <pcie> q` (sudo) |
| `kernel_version` | string | `uname -r` |
| `mlx5_modules` | array of string | which `mlx5_*` modules are loaded |
| `bf_mode` | string \| null | `smartnic` \| `dpu` \| `switch` \| null |
| `devices` | array of object | per-PCIe-function record: `pcie_address`, `kind` (`PF` \| `VF` \| `SF`), `state`, `mtu`, `representor_of` |

**Manual fallback chain (per side):**

1. `doca_caps --version` → `doca_version`
2. `flint -d <pcie> q` (sudo) → `firmware_version`
3. `uname -r` → `kernel_version`
4. `lsmod | grep mlx5` → `mlx5_modules`
5. `mlxconfig -d <pcie> q INTERNAL_CPU_MODEL` → `bf_mode`
6. `devlink dev show` + `ip -j link` → `devices` array

The two sides are deliberately symmetric so the agent can diff them
trivially when diagnosing host ↔ BlueField mismatches.

## Relationship to PR2 executables

The schemas above describe **contracts**. The implementations that
satisfy each contract are deferred to a subsequent PR on the
maintainer roadmap.

The reason this skill exists *before* the executables ship is so
every other skill in the bundle can be infra-aware from PR1 onward:
each library / service / tool skill's Command appendix references
this contract in its first row, so when the executables land in PR2
the skills do not need any retrofit.

Concrete consequence for contributors writing a new library skill:
when you build the Command appendix, do not duplicate the manual
fallback chain — link to this skill's matching schema section and
add only the *per-library overlay* (e.g. "Flow-specific: treat the
`DOCA_ERROR_INVALID_VALUE` / `DOCA_ERROR_NOT_SUPPORTED` return
from `doca_flow_pipe_create` as the validate signal; the public
Flow header at this release does not ship a separate
`doca_flow_pipe_validate` symbol — do not invent one"). The
fallback chain itself lives here.

## URL audit

This skill references the following external URLs. All MUST be
public and MUST resolve. The lint runs the URL check in CI.

| URL | Owner | Last verified | DOCA version | Notes |
| --- | --- | --- | --- | --- |
| (none — this skill is a contract, the substantive URLs are owned by `doca-public-knowledge-map` and the per-library skills) | n/a | 2026-05-17 | 3.3.0 | The agent reaches public docs via `doca-public-knowledge-map`; this skill stays vendor-neutral on URLs |
