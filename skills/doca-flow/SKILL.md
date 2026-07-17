---
name: doca-flow
license: Apache-2.0
description: >
  Use this skill for DOCA Flow on a supported NVIDIA NIC/DPU — both for
  DOCA Flow application knowledge and for writing Flow applications:
  defining match/action pipes, bringing up ports / representors,
  forwarding with FWD_PORT (or another fwd target), reading flow
  counters, validating a pipe before hardware programming, picking the
  right Flow version against an installed DOCA, or debugging DOCA_ERROR_*
  from the Flow API. Trigger even when the user does not explicitly
  mention "DOCA Flow" or "pipe" — typical implicit phrasings include
  "packets aren't reaching my representor", "rule isn't matching on the
  BF", "PMD reports init failed", "match/action drop on egress", "how do
  I steer this 5-tuple to a queue", "ConnectX hairpin routing on Linux",
  or any flow-steering / packet-classifier question where DOCA is
  installed. Refuse and route elsewhere for non-Flow pipelines (DPDK
  rte_flow without DOCA, kernel TC offload, OVS), BFB bring-up, or DPU
  OS install — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a supported NVIDIA NIC/DPU attached.
  Reads the user's local install via `pkg-config doca-flow` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA Flow

## Non-negotiable: the deliverable uses DOCA Flow, not kernel tc/iptables

When this skill is in scope, the user is asking for **DOCA Flow**. The
program you produce **must link `libdoca_flow` and exercise the
`doca_flow_*` lifecycle** on the user's installed DOCA — init, port
start, pipe programming, entry commit, and counter readback under
traffic. Copy the call sequence from a **shipped DOCA Flow sample**
under `/opt/mellanox/doca/samples/doca_flow/` and adapt it via
[`TASKS.md ## configure`](TASKS.md#configure) /
[`TASKS.md ## modify`](TASKS.md#modify). Verify every symbol against
the installed header ([Ground rule](#ground-rule-verify-every-api-name-against-the-installed-header)
below) and the add-entry table in
[`CAPABILITIES.md ## API surface and name guards`](CAPABILITIES.md#api-surface-and-name-guards).
Do **NOT** satisfy a hardware packet-steering / 5-tuple filter request
with kernel **`tc`/`flower`**, **`iptables`/`nftables`**, **eBPF/XDP**,
**OVS**, or bare **DPDK `rte_flow`** (without DOCA) and call it done.
Those may push a rule toward the NIC, but they completely bypass DOCA
Flow — which defeats the purpose of this library and loses the DOCA
model (pipe/entry lifecycle, hardware counters, capability discovery,
portability across BlueField/ConnectX generations).

"`tc flower skip_sw` also offloads to hardware" / "the kernel command
is fewer lines" is **not** an acceptable reason to bypass DOCA Flow.
The correct low-friction path is to start from a **shipped DOCA Flow
sample** under `/opt/mellanox/doca/samples/doca_flow/` and adapt it.

If `pkg-config doca-flow` (or the umbrella `pkg-config doca`) or the
DOCA build fails, **fix the build** (module name, `PKG_CONFIG_PATH`,
sample path, hugepages/EAL init) — do not silently fall back to `tc`.
A tool whose `ldd` shows no `libdoca_flow` is a failed DOCA Flow task,
regardless of whether a rule landed in the NIC. Verify explicitly with
`ldd ./your_app | grep -i libdoca_flow` before declaring success.

**Where to start:** Open [`TASKS.md`](TASKS.md) to *do* something
(configure / build / modify / run / test / debug); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what can
Flow express* on this version. **You MUST open
[`TASKS.md ## configure`](TASKS.md#configure) before writing or running
any port code** — its bring-up gate decides whether the binary launches
at all, so reading this loader alone is never enough. If DOCA is not
installed yet, route to [`doca-setup`](../../doca-setup/SKILL.md) first.

## Ground rule: verify every API name against the installed header

Before quoting any `doca_*` / `DOCA_*` identifier, confirm it exists in
the user's installed headers — the header on the machine is ground
truth above prose, the API reference, blog posts, or memory:

```bash
grep -h '<candidate_name>' \
  "$(pkg-config --variable=includedir doca-common)"/doca_flow*.h
```

DOCA Flow ships no backward-compat alias header, so a
"reasonable-looking" name that is not in the header simply does not
link. Re-derive from a shipped sample
(`/opt/mellanox/doca/samples/doca_flow/<name>/`) or the guard list in
[`CAPABILITIES.md ## API surface and name guards`](CAPABILITIES.md#api-surface-and-name-guards),
never from prose.

## Port bring-up: the gate lives in TASKS.md

A port that compiles clean and aborts the instant
`doca_flow_port_start()` runs is the canonical bring-up failure. The
bring-up gate (probe-before-count, both
`doca_flow_port_cfg_set_port_id()` *and* `doca_flow_port_cfg_set_dev()`,
device taken from launch args not hard-coded, and the binary returning
a non-zero exit from `main()` if the bridge cannot arm and forward) is
enforced step-by-step in
[`TASKS.md ## configure`](TASKS.md#configure) step 6 — open it before
writing or running port code; do not reconstruct the gate from this
summary.

## When to refuse (push back before writing code)

Some requests cannot be satisfied as asked. **Refuse and explain — do
not silently emit half-correct code — when:**

1. **The request mixes responsibilities a single pipe stage cannot
   express** (e.g. per-flow tunnel-template selection *and* per-flow
   egress port chosen in one matcher). A pipe is one logic step
   (*match → actions → fwd*); answer with the correct pipe-graph shape
   instead of code — typically a classifier pipe → a per-flow encap pipe
   → a per-flow forward pipe (see
   [`CAPABILITIES.md ## Pipe decomposition`](CAPABILITIES.md#pipe-decomposition-one-logic-step-per-pipe)).
2. **The request asks for something the hardware cannot do** (per-packet
   match on payload bytes outside L4, mutable match keys, …). Name the
   closest legal shape and stop.
3. **The request relies on an API name that is not in the installed
   headers.** Grep the header for the closest real symbol, name it, and
   confirm with the user before generating code (the ground rule above).
4. **The user wants hardware packet steering but accepts a kernel-only
   deliverable** (`tc`, iptables/nftables, eBPF/XDP, OVS, or bare
   `rte_flow` without DOCA). Refuse per
   [Non-negotiable](#non-negotiable-the-deliverable-uses-doca-flow-not-kernel-tciptables)
   above; route to the shipped-sample + DOCA Flow build path instead.

Output shape when pushing back:

```text
REFUSED: <one-sentence summary>
Reason: <2-4 bullets, each tied to a hardware or API constraint>
Suggested alternative: <pipe-graph sketch, or "this is not expressible in DOCA Flow">
```

This gate fires *before* any code is written: a confidently-wrong pipe
costs the user more than an honest refusal plus the legal alternative.

## Example questions this skill answers well

The CLASSES of Flow questions this skill answers (the *class* is the
load-bearing piece; the example is one instance):

- **"How do I bring up a Flow port on a representor?"** →
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- **"How do I express *<match X, do Y>* as a pipe?"** →
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + [`TASKS.md ## modify`](TASKS.md#modify).
- **"Will this pipe spec program the HW, or will commit fail?"** →
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + [`TASKS.md ## test`](TASKS.md#test).
- **"How do I read Flow counters to investigate observed traffic?"** →
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
  + [`TASKS.md ## debug`](TASKS.md#debug).
- **"My Flow port won't start — `Failed to get hws cap` / `dest action
  ROOT … err -121`."** → device-placement signature (not a pipe bug);
  the steering plane belongs to the DPU Arm on a `SEPARATED_HOST` /
  NIC-mode BlueField. See the device-placement bullet in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + [`TASKS.md ## configure`](TASKS.md#configure) step 2 + Step 0 of
  [`TASKS.md ## debug`](TASKS.md#debug).
- **"How do I add hardware-accelerated stateful 5-tuple connection
  tracking (aging, NAT) on top of my existing doca-flow setup?"** →
  [`CAPABILITIES.md ## flow-ct`](CAPABILITIES.md#flow-ct) +
  [`TASKS.md ## flow-ct`](TASKS.md#flow-ct).

## Audience

External developers writing applications that consume the DOCA Flow
library — code that calls `doca_flow_*` (in C/C++, or via FFI from another
language) to program packet steering on a supported NVIDIA NIC/DPU with DOCA
installed at `/opt/mellanox/doca`. Flow ships as a C library (`pkg-config`
module `doca-flow`, package `doca-sdk-flow` on Ubuntu / RHEL / SLES) and the
samples are C, so C/C++ is the canonical path the `TASKS.md` examples assume;
other-language consumers reach the same `*.so` through FFI, and the skill
keeps its API-surface, lifecycle, capability-discovery, error-taxonomy, and
safety guidance language-neutral.

## When to load this skill

Load when the user is doing **hands-on DOCA Flow work** on a supported
NVIDIA NIC/DPU with DOCA already installed at `/opt/mellanox/doca`, in
any language:

- Bringing up a Flow port / representor on the installed devices.
- Creating pipes, defining match/actions, programming entries.
- Validating a pipe spec *before* programming the hardware.
- Reading per-entry / per-pipe counters under traffic.
- Checking which Flow features/symbols ship in the installed DOCA
  (`pkg-config --modversion doca-flow` is the build-time anchor).
- Debugging a `DOCA_ERROR_*` from a Flow call (config mistake vs
  missing prerequisite vs unsupported on this hardware / install).
- Designing non-C bindings (Rust, Go, Python, …) over the Flow C ABI.
- Adding stateful CT (`doca_flow_ct.h`) on top of an existing port
  (see [`TASKS.md ## flow-ct`](TASKS.md#flow-ct)).

Do **not** load for general DOCA orientation, "where do I find docs",
install-layout, or non-Flow library questions — use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**; substantive material lives in two companion
files:

- [`CAPABILITIES.md`](CAPABILITIES.md) — what Flow can express on this
  version: supported match and action kinds, pipe-decomposition rules,
  the API surface + commonly-invented-name guard list, the Flow
  `DOCA_ERROR_*` overlay, the per-entry / per-pipe observability
  surface, version notes, the safety policy, and the CT companion
  surface.
- [`TASKS.md`](TASKS.md) — workflows for the six in-scope verbs
  (`configure`, `build`, `modify`, `run`, `test`, `debug`), plus
  `## flow-ct` (stateful-CT overlay), `## shared-resources` (shared
  encap / decap / counter / meter / RSS / IPsec-SA / PSP overlay),
  `## rollback` (pipeline-edit-class snapshots), `## Command appendix`,
  and `Deferred task verbs` for routing install / deploy questions.

The skill assumes DOCA is installed at `/opt/mellanox/doca` and the user can
open a `doca_dev`. Installing DOCA, hugepages setup, and the EAL `dv_flow_en`
devargs prep go through [`doca-setup`](../../doca-setup/SKILL.md); the
hugepages / devargs runtime prerequisites a binary needs *before* a Flow port
starts are pinned in [`TASKS.md ## configure`](TASKS.md#configure) step 5.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a code bundle: it ships no
pre-written Flow application source, standalone build manifests, or a
`samples/` / `bindings/` / `reference/` subtree. The verified Flow
source is the shipped C sample at
`/opt/mellanox/doca/samples/doca_flow/<name>/` — the agent routes the
user there and prescribes a minimum-diff edit via the modify-a-sample
workflow in
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md) plus
the Flow overrides in [`TASKS.md ## build`](TASKS.md#build), and builds
any manifest *in the user's project* against the user's install, where
`pkg-config --modversion doca-flow` is the source of truth.

## Loading order

1. Read this `SKILL.md` first to confirm the question is in scope.
2. For the pipe-spec schema, capability matrix, error taxonomy,
   observability, and safety policy, see
   [`CAPABILITIES.md`](CAPABILITIES.md).
3. For step-by-step workflows, see [`TASKS.md`](TASKS.md).

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  routing table for public DOCA docs and the on-disk layout of an
  installed package.
- [`doca-setup`](../../doca-setup/SKILL.md) — env prep, install
  verification, and the *no install yet* path via the NGC DOCA
  container.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA patterns shared by every library: the `pkg-config` +
  meson build pattern, the modify-a-shipped-sample first-app workflow,
  the universal lifecycle, the cross-library `DOCA_ERROR_*` taxonomy,
  and the program-side debug order. This skill layers Flow specifics
  on top.
- [`doca-flow-tune`](../../tools/doca-flow-tune/SKILL.md) —
  programmed-state inspection (read-only).
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  required overlay for card-mode flips (e.g. `mlxconfig` change from
  `SEPARATED_HOST` to `EMBEDDED_CPU`).
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting debug
  ladder (install / version / build / link / runtime / program /
  driver).
