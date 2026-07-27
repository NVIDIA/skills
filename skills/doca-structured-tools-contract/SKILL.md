---
license: Apache-2.0 AND CC-BY-4.0
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

See [`references/examples.md`](references/examples.md) for the five
worked routing examples. Keep this loader focused on detection,
fallback behavior, and the authoritative schemas below.

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

Running probes and fallbacks requires shell access to the target host,
either directly by the agent or through commands the user runs.

## Ground rules for any agent using this skill

1. **Detect first; never assume the tool is present.** Each schema
   below names the *probe command* that decides whether the
   structured tool is installed on this host. Run the probe before
   reading the schema's output as authoritative.
2. **Prefer structured when present; fall back to manual when not.**
   When the probe succeeds and the output validates against the
   selected schema, the structured JSON is the source of truth. When
   the probe fails or the output is invalid, walk the manual command
   chain in the same schema section and synthesize the equivalent
   answer.
3. **Report which path you took.** Always tell the user at the
   start of the answer: *"using structured `<helper>` (path:
   `<path>`)"* OR *"falling back to manual chain (structured
   `<helper>` probe failed: `<reason>`)"*, substituting the helper
   selected by the schema and the actual probe failure. Never report
   a helper different from the one the schema selected.
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
7. **Privilege is never implicit.** A manual fallback command that
   requires `sudo` is emitted for the user to run or executed only
   through an already approved privileged channel. Never silently
   elevate merely because the structured helper was absent. Do not
   assume such a channel exists; if it does not, ask the user to run
   the command or report the privileged-data gap.

## The agent behavior contract

The contract is a four-step loop the agent runs every time a skill's
Command appendix references this contract:

1. **Detect.** Run the probe command listed in the schema section
   for the relevant tool. Examples: `command -v doca-env`,
   `test -f /opt/mellanox/doca/share/version-matrix.json`,
   `command -v doca-capability-snapshot`. Probes are read-only and
   safe to run on any host. The executable helpers are deferred to
   PR2, so until they ship, failed command probes are expected and
   the manual chains are the operative path unless a helper was
   installed separately.
2. **Prefer.** If the probe succeeds, invoke the structured tool
   and parse its JSON per the schema in [`## Schemas`](#schemas).
   It is authoritative only when parsing succeeds and every required
   field has the documented type. Malformed JSON, missing fields, or
   type mismatches make the structured path fail: report that exact
   validation failure and use step 3. Ignore unexpected extra fields
   as advisory per ground rule 5; do not expose them as contract
   output. Do NOT run the manual chain merely "to double check" valid
   structured output — valid structured output replaces the chain.
3. **Fall back.** If the probe fails, walk the manual command
   chain documented in the same schema section. Synthesize the
   answer by combining the manual command outputs in the order the
   chain lists them. If a manual command is unavailable on the
   host, surface that as a gap and route the user to the matching
   skill (typically [`doca-setup`](../doca-setup/SKILL.md)). If that
   route cannot resolve the gap, name the missing commands or
   artifacts, state that the consolidated answer cannot be completed,
   and stop rather than presenting partial data as complete.
4. **Report.** Open the answer with one of:
   - *"Using structured `<tool>` (path: `<path>`)."* — when the
     probe succeeded and its output validated.
   - *"Falling back to manual chain (structured `<tool>` probe
     failed: `<reason>`)."* — when the probe failed. Include the
     actual probe command and failure reason; for `command -v`, note
     that failure means the helper was not found on `PATH`, not that
     it is definitively absent. Plus a one-line note pointing the
     user at how to install the helpers when they become available.
   - *"Falling back to manual chain (`<tool>` output failed schema
     validation: `<reason>`)."* — when the helper exists but its
     output is malformed, missing required fields, or has invalid
     field types.

The report step proves the agent tried the helper before falling back.

## Schemas

Select the schema from the question shape: environment/install state
uses `doca-env`; capability minimum-version lookup uses
`version-matrix`; per-device library capabilities use
`capability-snapshot`; spec validation uses
`validate-before-commit`; and a host-versus-DPU state comparison uses
the two collect-state schemas. When a per-skill Command appendix names
a schema, use that schema directly.

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
5. Find `doca-common.pc` first. If `find /opt/mellanox/doca -name doca-common.pc -print -quit` returns empty, stop this row, surface the partial-install gap, and route to `doca-setup`; do not expand an empty directory glob. Otherwise derive `PCDIR` from that result and run `for pc in "$PCDIR"/*.pc; do pkg-config --exists "$(basename "$pc" .pc)" && echo "$pc"; done` → `libraries` array. If `PCDIR` is not a directory or no module resolves through `pkg-config --exists`, surface that gap and route to `doca-setup`. `PCDIR` is commonly `/opt/mellanox/doca/lib/<arch>-linux-gnu/pkgconfig` on DOCA 3.3+, or `/opt/mellanox/doca/infrastructure/lib/pkgconfig` on legacy / split-profile installs.
6. `ls /opt/mellanox/doca/samples/` → `sample_paths` array
7. `lsmod | grep -E '^mlx5_(core|ib)'` and `uname -r` → `drivers` object
8. `cat /proc/meminfo | grep -i Huge` → `hugepages` object
9. `dmidecode -s system-product-name` (or `cat /proc/device-tree/model` on BlueField) → `host_kind`
10. `mlxconfig -d <pcie> q INTERNAL_CPU_MODEL` → `bf_mode` (when `host_kind == bluefield`)

### version-matrix.json schema

**Detection probe:** `test -f /opt/mellanox/doca/share/version-matrix.json`.
If absent, use the manual fallback; do not guess another install path.

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
   test program by following the library's own `## test` workflow
   and modifying its named shipped sample. Do not invent test code.

### validate-before-commit schema

**Detection probe:** `command -v doca-validate`. If that command is
absent, the structured helper is absent and the agent uses the manual
fallback below. A library-specific constructor-time validation
surface is not a detection probe and may mutate state. For example,
the public Flow header at this release does not ship a separate
`doca_flow_pipe_validate` symbol: never invent one and never use
`doca_flow_pipe_create` as a read-only probe. The structured tool
wraps only library-specific validation calls that are safe for its
contract and returns a uniform JSON result.

**Top-level shape (JSON object):**

| Field | Type | Notes |
| --- | --- | --- |
| `library` | string | which DOCA library the spec is for |
| `spec_path` | string | path on disk to the spec being validated |
| `result` | string | `pass` \| `fail` \| `skip` |
| `checks` | array of object | per-check breakdown: `name`, `status` (`pass` \| `fail` \| `skip`), `details` (string), `remediation` (string \| null) |

**Manual fallback chain:**

1. Find the library-specific validate surface in the matching skill's
   `## test` workflow. For some libs this is a dedicated `_validate`
   call. Constructor-time checks embedded in a mutating `_create`
   call are not read-only validators. In particular,
   `doca_flow_pipe_create` belongs to `doca-flow TASKS.md ## test`
   after that skill's snapshot/safety preconditions; do not invoke it
   as a pre-commit probe. If the caller requires read-only validation
   and the installed API exposes no dedicated validator, report
   `result: skip` and route to the per-library `## test` workflow.
2. Invoke it before any commit / create / submit call.
3. Map validation outcomes deliberately: `DOCA_ERROR_INVALID_VALUE`
   and `DOCA_ERROR_NOT_SUPPORTED` are `result: fail`. Permission,
   transport, unavailable-device, and other operational errors are
   `result: skip`, with the exact `doca_error_get_descr()` text and
   remediation in a `checks` entry. Never turn an inability to run
   validation into a claim that the spec itself failed.

### collect-host-state and collect-dpu-state schemas

Select the helper for the side where commands execute: host uses
`doca-collect-host-state`; BlueField uses `doca-collect-dpu-state`.
The presence of both binaries does not imply cross-side access. For a
diff, collect independently on each side and then compare the outputs.

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
2. `uname -r` → `kernel_version`
3. `lsmod | grep mlx5` → `mlx5_modules`
4. `devlink dev show` + `lspci | grep Mellanox` + `ip -j link` →
   enumerate the `devices` array and its PCIe addresses
5. For each discovered target PCIe address, `flint -d <pcie> q`
   through the approved privileged channel → `firmware_version`
6. On a discovered BlueField target, `mlxconfig -d <pcie> q
   INTERNAL_CPU_MODEL` → `bf_mode`

The two sides are deliberately symmetric so the agent can diff them
trivially when diagnosing host ↔ BlueField mismatches.

## Relationship to PR2 executables

The schemas above describe **contracts**. The implementations that
satisfy each contract are deferred to a subsequent PR on the
maintainer roadmap.

This skill ships first so other skills can reference the contract
without later retrofit when PR2 executables land.

Concrete consequence for contributors writing a new library skill:
when you build the Command appendix, do not duplicate the manual
fallback chain — link to this skill's matching schema section and
add only the *per-library overlay*. For Flow, the public header at
this release has no separate `doca_flow_pipe_validate` symbol: do not
invent one or use `doca_flow_pipe_create` as this contract's read-only
pre-commit probe. Report `result: skip` and route to `doca-flow
TASKS.md ## test`, where constructor-time checks may run only after
that workflow's snapshot and safety preconditions. The fallback chain
itself lives here.

## URL audit

This skill references the following external URLs. All MUST be
public and MUST resolve. The lint runs the URL check in CI.

| URL | Owner | Last verified | DOCA version | Notes |
| --- | --- | --- | --- | --- |
| (none — this skill is a contract, the substantive URLs are owned by `doca-public-knowledge-map` and the per-library skills) | n/a | 2026-05-17 | 3.3.0 | The agent reaches public docs via `doca-public-knowledge-map`; this skill stays vendor-neutral on URLs |
