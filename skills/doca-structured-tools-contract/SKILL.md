---
license: Apache-2.0 AND CC-BY-4.0
name: doca-structured-tools-contract
description: >
  Not for writing library code (name that library skill). Use this
  skill whenever another DOCA skill says "prefer the structured
  tool per doca-structured-tools-contract", or when the user wants
  a one-shot answer consolidating what multiple manual commands
  would produce — DOCA env / version / devices / capabilities /
  validate / host vs DPU state. Trigger even without the phrases
  "structured tool" or "doca-env --json" — typical phrasings
  include "is there one command that tells me everything about my
  DOCA install", "what version is X capability available since",
  "every PF/VF/SF visible on this BlueField with PCIe address", or
  "will this pipe pass validate before commit". Refuse and route
  elsewhere for general DOCA orientation, library API how-to, or
  install-from-scratch guidance — those belong to the per-library
  skill, doca-public-knowledge-map, or doca-setup.
metadata:
  author: NVIDIA
  kind: knowledge
  doca_version: "3.5"
compatibility: >
  No DOCA install required to read this skill (it is an overlay
  loaded against any DOCA artifact skill); its validation steps DO
  require a live DOCA install at /opt/mellanox/doca.
---

# DOCA structured-tools contract

**Where to start:** Reach for this skill whenever a workflow says
*“prefer the structured tool per `doca-structured-tools-contract`”*.
Read [`## The agent behavior contract`](#the-agent-behavior-contract)
first, then the matching schema in [`## Schemas`](#schemas). If the
host has the structured tool, prefer its output; if not, fall back to
the manual chain in the same section and **always report which path
was taken**.

**Scope boundary.** Writing library code is not a structured-tools
question: name the library skill and stop. Checking a spec before
commit IS in scope: apply the validate-before-commit schema.

## Example questions this skill answers well

See [`references/examples.md`](references/examples.md).

## When to load this skill


Load this skill whenever another skill's workflow says *prefer the
structured tool*, OR the user wants a one-shot answer consolidating
multiple manual commands. Concretely:

- A per-artifact skill's Command appendix references this skill.
- The user asks "is there one command that tells me X about my
  DOCA install".
- The user asks "how do I know X is valid before I commit" for a
  library with a validate-before-commit call.
- The agent computed the manual answer and wants to surface the
  structured-tool one-liner for next time.

Do **not** load for general DOCA orientation, library API questions,
or install-from-scratch guidance — use the matching library skill +
`doca-public-knowledge-map` + `doca-setup`.
Running probes and fallbacks requires shell access to the target
host, directly or via user-run commands.

## Ground rules for any agent using this skill

1. **Detect first; never assume the tool is present.** Each schema
   names the *probe command* deciding whether the tool is installed;
   run it before treating schema output as authoritative.
2. **Prefer structured when present; fall back to manual when not.**
   When the probe succeeds and output validates against the selected
   schema, the structured JSON is the source of truth. When the probe
   fails or output is invalid, walk the manual chain and synthesize
   the equivalent answer.
3. **Report which path you took.** Open the answer with *“using
   structured `<helper>` (path: `<path>`)”* OR *“falling back to
   manual chain (structured `<helper>` probe failed: `<reason>`)”*,
   substituting the schema-selected helper and the actual probe
   failure; never report a different helper than selected.
4. **Schemas are locked here; per-skill overlays are NOT.** A
   per-artifact skill MAY add a Command-appendix row that *uses* a
   schema; it MUST NOT redefine it. Schema growth happens here;
   consuming appendixes inherit it.
5. **Never invent a JSON field not in the schema.** Extra fields
   in pasted JSON are advisory; quote the schema as the boundary.
6. **Schemas describe contracts, not implementations.** The
   helpers are optional and are not part of the DOCA packages;
   a site may provide its own. So **never assume a helper named
   here is installed** — never present one as available or send
   the user to install it.
   Probe, report absent in one line, then answer from the manual
   fallback.
7. **Privilege is never implicit.** A fallback command requiring
   `sudo` is emitted for the user to run or executed only through an
   approved privileged channel — never silently elevate because the
   helper was absent. If none exists, ask the user to run it or
   report the gap.

## The agent behavior contract

A four-step loop run every time a Command appendix references this
contract:

1. **Detect.** Run the schema's probe command. Examples:
   `command -v doca-env`,
   `test -f <site version-matrix.json path>`,
   `command -v doca-capability-snapshot`. Probes are read-only and
   safe. The helpers are optional, so failed probes are expected.
2. **Prefer.** If the probe succeeds, invoke the structured tool and
   parse its JSON per the schema in [`## Schemas`](#schemas). It is
   authoritative only when parsing succeeds and every required field
   has the documented type. Malformed JSON, missing fields, or type
   mismatches fail the structured path: report that failure and use
   step 3. Ignore extra fields as advisory (rule 5). Do NOT run the
   manual chain "to double check" valid structured output — it
   replaces the chain.
3. **Fall back.** If the probe fails, walk the manual chain in the
   same schema section, combining outputs in the listed order. If a
   manual command is unavailable, surface the gap and route to the
   matching skill (typically `doca-setup`). If that cannot
   resolve it, name the missing commands, state the answer cannot be
   completed, and stop rather than presenting partial data as
   complete.
4. **Report.** Open the answer with one of:
   - *“Using structured `<tool>` (path: `<path>`).”* — probe succeeded
     and output validated.
   - *“Falling back to manual chain (structured `<tool>` probe
     failed: `<reason>`).”* — probe failed. Include the probe command
     and reason; for `command -v`, failure means not on `PATH`, not
     definitively absent. Do not suggest installing it (rule 6).
   - *“Falling back to manual chain (`<tool>` output failed schema
     validation: `<reason>`).”* — the helper exists but its output is
     malformed, missing required fields, or invalid types.

The report step proves the agent tried the helper first.

## Schemas

Select the schema from the question shape: env/install state uses
`doca-env`; capability minimum-version lookup uses `version-matrix`;
per-device library capabilities use `capability-snapshot`; spec
validation uses `validate-before-commit`; host-vs-DPU comparison uses
the two collect-state schemas. When a per-skill Command appendix names
a schema, use it directly. Each subsection names ONE structured
tool, its probe, its top-level JSON shape, and the manual chain
walked when the probe fails.

### doca-env --json schema

**Detection probe:** `command -v doca-env`.

**Top-level shape (JSON object):**

| Field | Type | Notes |
| --- | --- | --- |
| `version` | object | `pkg_config` / `applications_version` / `doca_caps` / `bfb` (string\|null) / `consistent` (bool) |
| `devices` | array of object | per visible PCIe function: `pcie_address` (e.g. `0000:03:00.0`), `kind` (`PF`\|`VF`\|`SF`), `name`, `representor_of` (string\|null), `state` (`active`\|`down`\|`unknown`), `mtu` (number) |
| `libraries` | array of object | one per public DOCA library: `pkg_config_name`, `installed` (bool), `pc_path` (string\|null) |
| `sample_paths` | array of object | one per library: `library`, `path` (on-disk samples root) |
| `drivers` | object | `mlx5_core_loaded` (bool), `mlx5_ib_loaded` (bool), `kernel_version` (string) |
| `hugepages` | object | `available_2m` (number), `available_1g` (number), `mount_point` (string\|null) |
| `host_kind` | string | `host` \| `bluefield` \| `unknown` |
| `bf_mode` | string \| null | `smartnic` \| `dpu` \| `switch` \| `null` (when `host_kind != bluefield`) |

**Manual fallback chain** (in order; combine outputs to synthesize
the same answer):

1. `pkg-config --modversion doca-common` → `version.pkg_config`
2. `cat /opt/mellanox/doca/applications/VERSION` → `version.applications_version`
3. `doca_caps --version` → `version.doca_caps`
4. `doca_caps --list-devs` → `devices` array (PCIe address + kind + representor)
5. Find `doca-common.pc` first: if `find /opt/mellanox/doca -name doca-common.pc -print -quit` is empty, stop, surface the partial-install gap, and route to `doca-setup`; never expand an empty glob. Otherwise set `PCDIR` to its directory and run `for pc in "$PCDIR"/*.pc; do m="${pc##*/}"; pkg-config --exists "${m%.pc}" && echo "$pc"; done` → `libraries` array. If `PCDIR` is not a directory or no module resolves, surface the gap and route to `doca-setup`. `PCDIR` is commonly `/opt/mellanox/doca/lib/<arch>-linux-gnu/pkgconfig` on DOCA 3.3+, or `/opt/mellanox/doca/infrastructure/lib/pkgconfig` on legacy / split-profile installs.
6. `ls /opt/mellanox/doca/samples/` → `sample_paths` array
7. `lsmod | grep -E '^mlx5_(core|ib)'` + `uname -r` → `drivers` object
8. `cat /proc/meminfo | grep -i Huge` → `hugepages` object
9. `dmidecode -s system-product-name` (or `cat /proc/device-tree/model` on BlueField) → `host_kind`
10. `mlxconfig -d <pcie> q INTERNAL_CPU_MODEL` → `bf_mode` (when `host_kind == bluefield`)

### version-matrix.json schema

**Detection probe:** `test -f <site version-matrix.json path>`.

**Top-level shape (JSON object):**

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | string | semver of THIS contract; bumps on schema changes |
| `generated_at` | string | ISO-8601 generation timestamp |
| `entries` | array of object | one row per (library, capability) pair |

**Per-entry shape:**

| Field | Type | Notes |
| --- | --- | --- |
| `library` | string | `pkg-config` module name (`doca-flow`, `doca-rdma`, `doca-comch`, …) |
| `capability` | string | machine-readable cap name; the per-library Command appendix lists which `doca_<lib>_cap_*` query it maps to |
| `display_name` | string | human-readable label; quoted when reporting |
| `min_doca_version` | string | first DOCA release in which the capability was available (semver) |
| `max_doca_version` | string \| null | last release it was available in (null = still available) |
| `source_url` | string | the public docs URL the row was derived from |
| `source_quote` | string | the exact public-docs prose that established the row |

**Manual fallback chain:**

1. Identify the library + capability via the matching skill's
   CAPABILITIES.md table.
2. Fetch its per-library doc page via `doca-public-knowledge-map`.
3. Search it for the capability name; extract and quote the
   *“available since”* prose verbatim.
4. Cross-check `pkg-config --modversion doca-<library>` on the
   user's host; if it is older than the *“available since”* line,
   the capability is not on this install regardless of the docs.

### capability-snapshot schema

**Detection probe:** `command -v doca-capability-snapshot`.

**Top-level shape (JSON object):**

| Field | Type | Notes |
| --- | --- | --- |
| `snapshot_at` | string | ISO-8601 timestamp |
| `doca_version` | string | `doca_caps --version` at snapshot time |
| `host_kind` | string | `host` \| `bluefield` |
| `devices` | array of object | one per `doca_devinfo`: `pcie_address`, `library_capabilities` (map of library → capability flags) |

**Manual fallback chain:**

1. `doca_caps --list-devs` → device enumeration
2. Per-device detail: `doca_caps --pci-addr <bdf>` (side-effect free;
   see `doca-caps`). Only if a capability it does not print is
   required, use the library's own `## test` workflow from its
   shipped sample; never write new test code for a snapshot.

### validate-before-commit schema

**Detection probe:** `command -v doca-validate`. A library-specific
constructor-time validation surface is not a detection probe and may
mutate state. The public Flow header ships no `doca_flow_pipe_validate`
symbol: never invent one, and never use `doca_flow_pipe_create` to test
for the helper.

**Top-level shape (JSON object):**

| Field | Type | Notes |
| --- | --- | --- |
| `library` | string | which DOCA library the spec is for |
| `spec_path` | string | on-disk path to the spec being validated |
| `result` | string | `pass` \| `fail` \| `skip` |
| `checks` | array of object | per-check: `name`, `status` (`pass` \| `fail` \| `skip`), `details`, `remediation` (string \| null) |

**Manual fallback chain:**

1. Find the library-specific validate surface in the matching
   skill's `## test` workflow. For some libs this is a dedicated
   `_validate` call; constructor-time checks inside a mutating
   `_create` are not read-only validators. In particular,
   `doca_flow_pipe_create` belongs to `doca-flow TASKS.md ## test`
   after its preconditions; never invoke it as a pre-commit probe.
   If the caller requires read-only validation and the API exposes
   no dedicated validator, report `result: skip`, route to the
   per-library `## test` workflow, and stop (skip steps 2 and 3).
2. Invoke that read-only validator (never a mutating `_create`)
   before any commit / create / submit call.
3. Map outcomes deliberately: `DOCA_ERROR_INVALID_VALUE` and
   `DOCA_ERROR_NOT_SUPPORTED` are `result: fail`; permission,
   transport, unavailable-device, and other operational errors are
   `result: skip`, with the `doca_error_get_descr()` text and
   remediation in a `checks` entry. Never turn an inability to run
   validation into a spec failure.

### collect-host-state and collect-dpu-state schemas

Select the helper for the side where commands execute: host uses
`doca-collect-host-state`; BlueField uses `doca-collect-dpu-state`.
Both being present does not imply cross-side access; for a diff,
collect independently on each side and compare.

**Detection probes:** `command -v doca-collect-host-state` (host) /
`command -v doca-collect-dpu-state` (BlueField).

**Top-level shape (JSON object), shared by both:**

| `side` | string | `host` \| `dpu` |
| `doca_version` | string | `doca_caps --version` |
| `firmware_version` | string | `flint -d <pcie> q` output (sudo) |
| `kernel_version` | string | `uname -r` |
| `mlx5_modules` | array of string | loaded `mlx5_*` modules |
| `bf_mode` | string \| null | `smartnic` \| `dpu` \| `switch` \| null |
| `devices` | array of object | per-PCIe-function: `pcie_address`, `kind` (`PF` \| `VF` \| `SF`), `state`, `mtu`, `representor_of` |

**Manual fallback chain (per side):**

1. `doca_caps --version` → `doca_version`
2. `uname -r` → `kernel_version`
3. `lsmod | grep mlx5` → `mlx5_modules`
4. `devlink dev show` + `lspci | grep Mellanox` + `ip -j link` →
   enumerate `devices` and PCIe addresses
5. For each discovered PCIe target, `flint -d <pcie> q` via the
   approved privileged channel → `firmware_version`
6. On a discovered BlueField target, `mlxconfig -d <pcie> q
   INTERNAL_CPU_MODEL` → `bf_mode`

The two sides are symmetric so the agent can diff them when
diagnosing host ↔ BlueField mismatches.

## Relationship to helper executables

The helpers are optional and site-provided (ground rule 6); any helper
that implements a schema above emits it unchanged, so no other skill
needs a change.
