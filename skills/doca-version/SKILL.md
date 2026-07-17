---
license: Apache-2.0
name: doca-version
description: >
  Use this skill when the user is doing DOCA version handling —
  detecting the installed release, validating the four-way match
  across pkg-config doca-common, applications/VERSION, doca_caps
  --version, and bfver/mlnx-release on BlueField, reasoning about
  NGC container tags, looking up whether a capability is on the
  installed release, or diagnosing build-vs-runtime drift. Trigger
  even when the user does not explicitly say "DOCA version" or
  "four-way match" — typical implicit phrasings include "program
  built but does nothing on the wire", "undefined reference to a
  symbol the docs claim exists", "DOCA_ERROR_NOT_SUPPORTED at
  runtime", "counter didn't increment", "what does `latest` mean
  for this tag", or "is my LTS still supported". Refuse and route
  elsewhere for installing or choosing DOCA packages (doca-setup),
  per-library API/capability questions (matching library skill),
  the cross-library DOCA_ERROR_* taxonomy (doca-programming-guide),
  or the general debug ladder (doca-debug) — those belong to other
  skills.
metadata:
  kind: library
compatibility: >
  No DOCA install required to read this skill (it is an overlay
  loaded against any DOCA artifact skill); the validation steps
  within DO require a live DOCA install at /opt/mellanox/doca.
---

# DOCA version

**Where to start:** This skill is the bundle's single source of
truth for DOCA *version handling*. Open
[`TASKS.md`](TASKS.md) if the user wants to *do* something with the
version (detect / validate / diagnose mismatch); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what does
version handling cover* (the four-way match, the detection chain,
NGC semantics, the per-library overlay pattern). Every other skill
in the bundle that touches version routes here — they MUST NOT
redefine the rules.

## Example questions this skill answers well

The CLASSES of version-handling questions this skill is built to
answer, each with one worked example. The agent should treat the
*class* as the load-bearing piece — the worked example is a single
instance.

- **"What DOCA version do I actually have installed?"** — worked
  example: *"the docs say 3.3 but I'm not sure what's on this
  host"*. Answered by the canonical detection chain in
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  source-of-truth table.
- **"My program built but does nothing on the wire — is my install
  consistent?"** — worked example: *"`pkg-config --modversion`
  says 3.3.0; `doca_caps --version` says 3.2.0"*. Answered by the
  four-way match rule in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  + the partial-install diagnosis in
  [`TASKS.md ## debug`](TASKS.md#debug).
- **"Is this DOCA capability / API / sample on the version I
  have?"** — worked example: *"is the symmetric-RSS hash mode in
  Flow 2.6.0"*. Answered by the version-matrix lookup procedure in
  [`TASKS.md ## test`](TASKS.md#test) (which uses the
  `version-matrix.json` schema defined in
  [`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md#schemas)
  with fallback to per-library docs via
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)).
- **"Can I run my host package version X against BFB version Y?"** —
  worked example: *"host is 3.3.0 LTS, BlueField BFB is 3.1.0"*.
  Answered by the routing to the
  [DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html)
  documented in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
- **"I'm inside the NGC DOCA container — what does the version
  match look like?"** — worked example: *"do I still need to check
  pkg-config / applications/VERSION / doca_caps separately?"*.
  Answered by the NGC container rule in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  + the container path in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do I write a per-library version-compatibility section
  for a new skill?"** — worked example: *"adding `doca-comch` to
  the bundle, what does its `## Version compatibility` look
  like?"*. Answered by the per-library overlay pattern in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the worked-example template in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **"My `apt list` shows DOCA `3.3.0109`, but `/etc/apt/sources.list.d/doca.list`
  is pinned at `latest` / a different release — is the next
  `apt install doca-*` going to silently upgrade me?"** — worked
  example: *"I rolled back BFB to 3.1.0105 but my sources still
  point at the latest channel."* Answered by the apt-source
  consistency precheck in
  [`TASKS.md ## apt-source consistency`](TASKS.md#apt-source-consistency),
  which enumerates the three legitimate shapes of a configured
  DOCA apt source (network URL, local file-repo, RHEL/OEL
  equivalent) and the *do-not-install-until-the-source-matches*
  rule that protects pinned installs from accidental drift.

## When to load this skill

Load this skill whenever version handling is the load-bearing
concern. The decision must be made **before** the agent composes
its first sentence — the activation checklist below is the same
one referenced from
[`AGENTS.md ## Cross-cutting overlay activation triggers`](../../AGENTS.md#cross-cutting-overlay-activation-triggers),
mirrored here so the activation rule is at hand whenever this skill
is consulted.

## Agent activation checklist — load this skill at the START of the answer when any cell below is true

| Trigger class | Concrete prompt-side signals (any one fires the overlay) |
| --- | --- |
| Direct version question | *"what DOCA version do I have"*, *"is X consistent"*, *"is feature Y supported on version Z"*, *"can I mix host package version A with BFB version B"*, *"is my LTS still supported"*, *"what does the version string mean"* |
| Container tag question | any prompt that mentions a specific NGC container tag, or asks about `latest`, or asks how to pin a tag in a Dockerfile / pod spec / Compose file. The agent MUST also cite the *"never invent a tag string from memory, never quote `latest` without confirming it"* rule from [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy). |
| Build vs runtime drift | any debug session where the symptom is *"the program built fine but DOCA_ERROR_NOT_SUPPORTED at runtime"*, *"undefined reference to a symbol the docs say exists"*, *"my code does nothing on the wire"*, *"counter didn't increment"* — these are the canonical partial-install symptoms |
| Upgrade / downgrade plan | the user is planning to upgrade or downgrade DOCA on a host already running other DOCA workloads, or to refresh the BFB on a BlueField pair already attached to a host |
| Per-artifact cross-link | a per-artifact skill's `## Version compatibility` section cross-links here for the rule body, OR the agent is about to author a new per-artifact skill and needs the overlay template |

When any cell above fires, the agent MUST:

1. Cite the **four-source detection chain** from [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) explicitly in the answer — `pkg-config --modversion doca-common` → `cat /opt/mellanox/doca/applications/VERSION` → `doca_caps --version` → `bfver` plus `cat /etc/mlnx-release` (BlueField hosts). Do not paraphrase or summarize the chain; cite the commands by name. Do NOT substitute `mlxprivhost` or `bfb-info` for the BFB leg — those are common hallucinations and the bundle explicitly bans them in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
2. State the **four-way match rule** from [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) verbatim if the prompt could possibly involve a mismatch (every deploy-shape question and every debug-shape question can; orientation-shape questions usually cannot).
3. Refuse to invent a version string. If the agent doesn't have the actual `pkg-config --modversion` output from the user's host, the answer must say so and route to the detection chain — not assert a version from training-data recall.

## Universal version-coherence trigger

Whenever ANOTHER overlay (e.g. [`doca-setup`](../doca-setup/SKILL.md), [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md), [`doca-container-deployment`](../doca-container-deployment/SKILL.md), [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md)) calls a `## test` / `## configure` / `## modify` step that requires *"the install is healthy"* or *"versions are consistent"*, that step MUST resolve to a citation of this skill's four-source detection chain and four-way match rule. The agent does NOT redefine the rule per-overlay — every step that needs a version verification must route here. This is the *only* place in the bundle that owns the rule body.

Do **not** load this skill for general DOCA orientation, for
install procedures (use [`doca-setup`](../doca-setup/SKILL.md)),
or for library-specific API questions (use the matching library
skill).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
version-handling material lives in two companion files:

- `CAPABILITIES.md` — the version-handling surface: the canonical
  source-of-truth table for version detection, the four-way match
  rule, NGC container semantics, the per-library overlay pattern,
  the routing to the DOCA Compatibility Policy, the error
  taxonomy for version-related failures (pkg-config missing,
  partial install, BFB/host mismatch, NGC mixing), the
  observability surface (which command to read for which version
  source), and the safety policy ("never invent a version, never
  quote `latest`").
- `TASKS.md` — step-by-step workflows for the six in-scope
  version verbs: `configure` (detect on this host), `build`
  (build-time match), `modify` (update a version pin in a build
  manifest), `run` (runtime check), `test` (four-way validation +
  version-matrix lookup), `debug` (diagnose mismatch / partial
  install). Plus a `Deferred task verbs` block.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope.
2. **For the version-detection sources, four-way match rule, NGC
   semantics, per-library overlay pattern, error taxonomy,
   observability, and safety policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

## Related skills

- [`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md) —
  the JSON schemas for the helper tools the agent should prefer
  when present. This skill's `## test` workflow uses the
  `version-matrix.json` schema defined there; do not redefine the
  schema here.
- [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md) —
  the routing table to public DOCA docs, including the
  Compatibility Policy. This skill cites the Compatibility Policy
  URL once via that map; it does not duplicate the routing.
- [`doca-setup`](../doca-setup/SKILL.md) — env-side install /
  verify / NGC container path. This skill assumes its
  preconditions are satisfied (i.e., something is installed
  somewhere; the version question is *what was installed and is
  it consistent*).
- [`doca-programming-guide`](../doca-programming-guide/SKILL.md) —
  program-side guidance (quote the version observed, header-wins,
  capability-discovery rules). The program-side `## Version
  compatibility` section there is now a 3-5 line redirect to this
  skill plus the program-side overlay (quote vs assume; never use
  agent-memory version).
- [`doca-debug`](../doca-debug/SKILL.md) — the cross-cutting
  debug ladder. Layer 2 (*version mismatch*) of that ladder is
  owned by this skill's `## debug` workflow.
