---
license: Apache-2.0 AND CC-BY-4.0
name: doca-public-knowledge-map
description: >
  Use this skill to locate authoritative NVIDIA DOCA information
  without the source tree: the docs.nvidia.com page for a
  library/service/tool, which DOCA libraries are installed and at
  what version, a sample on disk or GitHub, a path under
  /opt/mellanox/doca, or a 404'd/renamed doc URL. Activate on
  implicit phrasing only when the conversation already names DOCA,
  BlueField, ConnectX, or /opt/mellanox. Do not activate for generic
  'which version is installed' or 'where is the sample' questions
  with none of that context. Refuse and route elsewhere for hands-on
  programming, env prep, API tutorials, or hardware/firmware
  mutation — those belong to doca-programming-guide, doca-setup, the
  per-library skills, and doca-hardware-safety.
metadata:
  author: NVIDIA <doca@nvidia.com>
  kind: knowledge
  tags:
    - doca
    - documentation
    - routing
compatibility: >
  No DOCA install required to read this skill (it is an overlay
  loaded against any DOCA artifact skill); the validation steps
  within this skill require a live DOCA install at /opt/mellanox/doca.
---

# DOCA Public Knowledge Map

**Where to start:** Reach for this skill whenever the question is "where does
the authoritative answer live?" — a docs page, the on-disk install layout, a
sample, or an NGC catalog entry. Read [`## Public documentation entry points`](#public-documentation-entry-points)
first; then jump to the routing-table section that matches the user's intent.

**How to use this skill.** Read this file, then search the companion file
for the `## <name>` heading the question needs and read that whole section.
Answer completely from it, quoting only its symbols and commands. Do not
browse docs; use the entry points in `doca-public-knowledge-map`. Give
environment checks as commands for the user's DOCA host. For out-of-scope
requests, name the skill that owns them and stop.

## When to load this skill

Load this skill whenever the user asks anything about NVIDIA DOCA where the
agent needs to **locate authoritative information** without access to the
DOCA source tree. That includes **documentation-routing questions** about
installing DOCA, building a sample, learning a DOCA library (Flow, DPA, Comm
Channel, GPUNetIO, …), debugging an error, finding an API or error reference,
finding a sample, release notes, or the developer forum. This skill locates
the authoritative answer; hands-on installation, building, tutorials, and
debugging route to the workflow-owning skills named in the frontmatter and
[`## Related skills`](#related-skills).

This skill is intentionally a **routing table**, not a tutorial. Pick the
entry that matches the user's intent, fetch the URL or inspect the local
install path, and only then answer.

## First-contact discovery — the four questions to ask before any drill-down

When a user opens with an open-ended orientation question (*"I'm new with
DOCA, how do I start?"*, *"can you guide me?"*, *"what's the easiest way to
try DOCA?"*), the agent does **not** have enough information yet to pick a
path. Asking these four questions before drilling avoids wasted recommendations
that the user cannot actually execute on their setup. Ask them as a single
short message; do not interrogate one-at-a-time.

| Question | Why it matters | What it routes |
| --- | --- | --- |
| 1. **What OS are you on?** macOS, Windows, Linux laptop, cloud VM, lab Linux box, BlueField OS itself? | DOCA installs natively only on supported Linux distributions; macOS / Windows users cannot install it at all. | Picks between the four DOCA acquisition paths in [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install). macOS / Windows / no-Linux → Path 0 (NGC container). Supported Linux → Path A or B per the Installation Guide. |
| 2. **What hardware do you have?** No NVIDIA hardware, ConnectX SmartNIC, BlueField as a SmartNIC in a host, BlueField as a standalone DPU, not sure? | Real-traffic runtime needs a real NIC; build / read / learn does not. The user's hardware decides which DOCA libraries are even relevant. | Picks the runtime story (container is build-only without hardware). Filters which libraries make sense to learn (Flow needs a real port to do anything visible; Comch needs a host ↔ DPU pair). |
| 3. **What's your goal?** Just exploring, building a small first app on a specific library (Flow / RDMA / Comch / Telemetry / GPUNetIO / DPA / …), running an existing reference application, operating a service (DMS / DTS / BlueMan / Firefly), or something else? | The bundle's first-app workflow (`doca-programming-guide ## modify`) starts from a **shipped C sample** and edits down. The right sample depends on the library the user is targeting. | Picks which library skill (if any) to load next. If the user does not yet know which library — that itself is a routing answer (see the *Library- and module-specific guides* table above and let the user pick). |
| 4. **Which language do you plan to write the program in?** C / C++, Rust, Go, Python, other? | DOCA's public surface is a C ABI. Non-C consumers go through FFI / language bindings (`doca-programming-guide CAPABILITIES.md ## Capabilities and modes` and the per-library skill). The C samples are the reference even when the user's language is not C. | Picks whether the agent's first-app guidance is *direct C build* or *FFI / bindings against the C ABI*. Does **not** change which sample the agent points at first. |

The four-question gate applies when an open-ended orientation, install,
build, or run recommendation depends on the user's environment. For that
class, **never recommend a specific install path, container tag, or runnable
sample without first having the answers to questions 1–3** (question 4 is
needed for the first-app workflow but not for orientation itself). Pure
documentation lookups — a URL, API reference, sample location, version
reference, or stale-link recovery — skip this gate and route directly.
If the user cannot or will not answer a required question, state which
environment-dependent recommendation cannot be made, give only the safe
version-agnostic umbrella documentation route, and do not guess the missing
value. Volunteering specific commands before this is the single most common
failure mode for DOCA orientation.

If the user has already volunteered some of the information in their first
message, mark those questions answered and only ask the rest. Do not re-ask
what the user has already told you.

## Topic to "where to look first" routing table

When the user asks something, route as follows:

| User intent | First place to look |
| --- | --- |
| "How do I install DOCA?" | Installation Guide + Downloads page (Public documentation entry points). |
| "How do I start with DOCA — what's the very first thing?" | Developer Quick Start Guide *if* the user has BlueField + host hardware; otherwise [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install) Path 0 (NGC container). Use the four questions in [*First-contact discovery*](#first-contact-discovery--the-four-questions-to-ask-before-any-drill-down) to pick. |
| "Do I need a BlueField? A SmartNIC? Or just DOCA-Host?" | Overview page (`doca-overview/index.html`) plus the Installation Guide's *DOCA installation profiles* section. The bundle does not pick the hardware for the user — these two pages do. |
| "Which package gives me library X?" | Installation Guide section on package matrix; then verify on the user's system with `pkg-config --list-all`. |
| "Show me a sample that uses library X." | `/opt/mellanox/doca/samples/doca_<X>/` if installed; otherwise the per-library guide on `docs.nvidia.com/doca/sdk/` (each library guide documents the samples shipped with it). |
| "How do I build a DOCA sample?" | Library guide + the sample's own `meson.build` inside `/opt/mellanox/doca/samples/...`. |
| "What is the API for X?" | Library guide; confirm by inspecting headers under `$(pkg-config --variable=includedir doca-common)` (commonly `/opt/mellanox/doca/include/` on DOCA 3.3+; `/opt/mellanox/doca/infrastructure/include/` on legacy / split-profile installs). |
| "Why does my build fail with `pkg-config` not finding `doca-...`?" | "Layout of an installed DOCA package" (`PKG_CONFIG_PATH`), then load [`doca-setup`](../doca-setup/SKILL.md) for environment preparation and debugging; do not silently mutate the environment. |
| "What is the latest version / what changed?" | Release Notes. |
| "Is there a newer DOCA — should I upgrade / downgrade / roll back?" | [`doca-upgrade`](../doca-upgrade/SKILL.md) for the detect → report → ASK → guided-upgrade discipline (never auto-upgrade); Release Notes for what the target release contains. |
| "Is the component I depend on being sunset / deprecated?" | Release Notes (lifecycle / deprecation notices), reached via the [`doca-upgrade`](../doca-upgrade/SKILL.md) sunset-awareness check; do not assert deprecation from memory. |
| "What does the DOCA version number mean? Is LTS still supported?" | Compatibility Policy (Public documentation entry points table). |
| "How do I run DOCA on Kubernetes / provision a DPU?" | DOCA Platform Framework on GitHub. |
| "I have a behavior I cannot explain." | Release Notes (known issues) first; then the DOCA Developer Forum. Never go to the forum first. |
| Any intent not matched above | Start at the DOCA SDK index and the Libraries / Services / Tools umbrella page in `references/map.md` (public documentation entry points). If the authoritative route remains unresolved, say so and use the Developer Forum; do not invent a URL or component name. |

## What this skill deliberately does not cover

This file is intentionally a **map**, not a tutorial. It does not contain:

- DOCA library tutorials (those live in the per-library guides).
- API reference (lives in headers and the per-library guides).
- Build-system deep-dives (lives in the Installation Guide and the sample
  `meson.build` files).
- Performance tuning, driver-level setup, OFED interaction (lives in the
  Installation Guide and library-specific guides).

When the agent needs those, it should fetch the matching public document or
read the matching installed file. As more focused skills are added, they
should appear in [SKILLS.md](../../SKILLS.md) and link back here for the
"where to look" lookups.

## Related skills

For env preparation — install verification, build environment
(`pkg-config`, headers, hugepages, devlink), env-class debugging, and
the *I have no install yet* path with the public NGC DOCA container
on the catalog Tags page as the universal Stage-1 fallback for any
user on macOS, Windows, or Linux without DOCA — load
[`doca-setup`](../doca-setup/SKILL.md). That skill stops at *"the
install is healthy and the env is ready"*.

For general DOCA programming patterns shared across every library —
the canonical `pkg-config doca-<library>` build pattern (C/C++ direct
or non-C via FFI), the universal *derive a custom first app from a
shipped sample* workflow, the universal lifecycle (`cfg-create →
init → start → use → stop → destroy`), the cross-library
`DOCA_ERROR_*` taxonomy, and the program-side debug order — load
[`doca-programming-guide`](../doca-programming-guide/SKILL.md). Each
library skill extends its `## modify` (first-app derivation) with
library-specific overrides.

For DOCA Flow internals — port and representor setup, pipe creation,
match/action specifications, pipe validation before hardware programming,
Flow counters and traces, Flow version compatibility, and debugging
`DOCA_ERROR_*` failures from the Flow API — load
[`doca-flow`](../libs/doca-flow/SKILL.md). That skill assumes this one is
available for shared documentation routing and install-layout lookups,
`doca-setup` for environment preparation, and `doca-programming-guide`
for the cross-library programming patterns it layers on top of.

For a concrete local debug loop against a running system — reproducing a
build, link, or runtime failure on the installed host, narrowing it to a
component, and iterating on it — [`doca-debug`](../doca-debug/SKILL.md)
owns the edit-run-observe cycle and should be loaded alongside this skill.
Handing off that cycle is not a reason to go quiet: a debug answer still
needs the two things this skill owns, and it should supply them in the
same breath — the `/opt/mellanox/doca/...` install layout that says where
the samples, libraries and `pkg-config` files actually live, and the
canonical `docs.nvidia.com` page for the library involved.

## Examples

Question shapes this skill routes live under `references/map.md`.

## Instructions

Load this file first. Then open **only** the matching companion under
`references/` — one file, not the whole directory. Do not follow a
second companion for the same answer.

| Question class | Open this filename under `references/` |
| --- | --- |
| Example question shapes; GitHub sources; on-disk layout; version; URL audit; ground rules; doc entry points | `map.md` |
| Per-library public guides | `libraries.md` |
| In-bundle DOCA services | `services.md` |
| In-bundle DOCA tools | `tools.md` |
| Externally-productized DPU, service, fabric, orchestration, or firmware products | `external-dpu.md` |

Each of those files carries its own three-part route-not-answer
contract (recognize the class, name the boundary, give the public URL).
Do not open every companion. For fabric, orchestration, firmware, and
switch products, start at `external-dpu.md` and continue from its last
section.

## Public documentation entry points

The public documentation entry-point table is the last section of
[`references/map.md`](references/map.md#public-documentation-entry-points).

## Public source code: GitHub

Public GitHub source locations live in
[`references/map.md ## Public source code: GitHub`](references/map.md#public-source-code-github).
Do not invent a repository URL.

## Layout of an installed DOCA package

The on-disk `/opt/mellanox/doca` layout table lives in
[`references/map.md ## Layout of an installed DOCA package`](references/map.md#layout-of-an-installed-doca-package).
Load that companion; do not invent install paths from memory.

## Where to find the version

Version-detection routing lives in
[`references/map.md ## Where to find the version`](references/map.md#where-to-find-the-version).
Then load [`doca-version`](../doca-version/SKILL.md) for the four-source chain.

## URL audit

The inline URL-audit history lives in
[`references/map.md ## URL audit`](references/map.md#url-audit).
Keep stale-link recovery there, not in this file.

## Library- and module-specific guides

Per-library public guide rows live in
[`references/libraries.md`](references/libraries.md#library--and-module-specific-guides).
Open that companion for a single-library docs route.

## DOCA services

In-bundle DOCA service guide rows live in
[`references/services.md`](references/services.md#doca-services).
Open that companion for DMS / Firefly / Argus / UROM-service docs routes.

## Deploying DOCA services at scale

Fleet / Kubernetes orchestration is out of this bundle's SDK scope.
The persona/scale entry-point lives in
[`references/services.md`](references/services.md#deploying-doca-services-at-scale--orchestration-entry-point-personascale-routing).
Do not hand-roll per-host `bfb-install` loops at that scale.

## DOCA tools

In-bundle DOCA tool guide rows live in
[`references/tools.md`](references/tools.md#doca-tools).
Open that companion for caps / bench / tune / tracer docs routes.

## Externally-productized DOCA software — not in this bundle, but here is where to route

The 26-product route-not-answer table starts in
[`references/external-dpu.md`](references/external-dpu.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)
and continues from its last section for fabric, orchestration, firmware
and switch products.
Recognize the class, name the monorepo boundary, and route with a public
URL plus a Developer Forum search hint. Do not synthesize those products
from training knowledge.
