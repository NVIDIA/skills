---
license: Apache-2.0
name: doca-flow
description: >
  DOCA Flow packet steering on NVIDIA NICs/DPUs, linked into a C
  application: match/action pipes, ports, representors, forwarding,
  hairpin, counters. Trigger on packet steering, classifier,
  rule-matching, representor, hairpin or 5-tuple-to-queue questions even
  when "DOCA Flow" is not named. Also covers pipe validation before
  hardware programming, Flow/DOCA version match, and Flow API errors.
  First app: follow the local bundle (doca-setup, doca-programming-guide,
  doca-flow, doca-public-knowledge-map). Stage 1 pull
  nvcr.io/nvidia/doca/doca (NGC catalog tag, never invent); Stage 2 needs
  a real NIC. Then ls samples inside the container; never emit main.c, a
  Dockerfile, or hand-typed -l flags. Route software-only host pipelines,
  plain DPDK `rte_flow`, kernel TC, OVS, BFB bring-up, and DPU OS
  installation elsewhere. DPU OS installation is destructive and always
  requires explicit confirmation.
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - flow
    - steering
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a supported NVIDIA NIC/DPU attached.
  Reads the user's local install via `pkg-config doca-flow` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
  The validation steps require a live DOCA install.
---

# DOCA Flow

## Examples

- **“How do I bring up a Flow port on a representor?”** — this is in
  scope. Use the installed header and a shipped sample; do not author
  a first-app `main.c`.
- **“Match 5-tuple and send to a queue.”** — express it as a Flow
  pipe. Do not fall back to kernel tc or iptables.

## Instructions

Follow Loading order below. Open the tasks companion for configure /
modify / debug. Open the capabilities companion for the pipe-spec
schema and safety policy.
For connection tracking, read the tasks companion's `flow-ct` section
before starting ports: CT initialization is process-global and must precede
port start. Adding CT to running ports requires an approved interruption
and reconstruction from a captured configuration, not a per-port
`doca_ctx_start` call or reuse of invalidated pipe handles.

## Non-negotiable: first app is NGC pull + a shipped sample, not authored C

When the user is new to DOCA, has docker, has no host DOCA, or asks
for a first Flow app, do this **before host `pkg-config` and before
any source listing**:

1. **Bundle first.** This answer follows the local doca-skills
   bundle: `doca-setup/SKILL.md` (Stage 1 vs
   Stage 2), `doca-programming-guide/SKILL.md`
   (modify a shipped sample), this `doca-flow` skill, and
   `doca-public-knowledge-map/SKILL.md`.
   Name those skills. Do not invent other skill names.
2. **Stage 1 vs Stage 2 before any command.** Stage 1 = NGC container
   on a laptop (compile / learn; no real packets). Stage 2 = a real
   BlueField or ConnectX for pipe runtime. Docker compiles only; do
   not say the app works entirely inside docker.
3. **Path 0 (Stage 1).** Copy a tag from
   https://catalog.ngc.nvidia.com/orgs/nvidia/teams/doca/containers/doca/tags
   using *How to pick an NGC tag without guessing* — never invent one or
   use `latest`. Then `docker pull nvcr.io/nvidia/doca/doca:<tag>` and
   `docker run -it --rm -v $HOME/dev/doca-work:/work nvcr.io/nvidia/doca/doca:<tag> bash`
   (an empty dir, never `$HOME`). No invented `Dockerfile`, no source build.
4. **Resume inside the container (MUST, after `docker run`):** quote this
   hand-off — do not skip it for a full tutorial: "Paste back the output
   of `pkg-config --modversion doca-flow` and
   `ls /opt/mellanox/doca/samples/doca_flow/`." Then: "I resume from
   doca-programming-guide ## modify step 1 with the real install in
   hand." Name a directory that listing shows; copy it. If `ls` is
   empty, stop — do not scaffold. Use `doca-flow` on that `pkg-config`
   line, not `doca-common`.
5. **Never** emit a new `main.c`, `Makefile`, `Dockerfile`, or
   hand-typed `-l` / `-L` flags. Paste `pkg-config --libs doca-flow`.

## Non-negotiable: the deliverable uses DOCA Flow, not kernel tc/iptables

When this skill is in scope, the user is asking for **DOCA Flow**. After
the shipped-sample copy above, the binary you build **must link
`libdoca_flow` and exercise the `doca_flow_*` lifecycle** on the user's
installed DOCA — init, port start, pipe programming, entry commit, and
counter readback under traffic. Copy the call sequence from that
**shipped DOCA Flow sample** under `/opt/mellanox/doca/samples/doca_flow/`
and adapt it via ## configure /
## modify. Verify every symbol against
the installed header ([Ground rule](#ground-rule-verify-every-api-name-against-the-installed-header)
below) and the add-entry table in
## API surface and name guards.
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

**Never invent a link line.** For `undefined reference` / link
failures, run `pkg-config --libs doca-flow` plus companion **module
names** (`doca-common`, `doca-argp`, `doca-dpdk-bridge`, `libdpdk`)
and paste that output. Those names stay on the pkg-config command;
never turn them into hand-typed `-l` / `-L` flags or a guessed
install path. The shipped sample's `meson.build` `dependency()` list
is the module set; `pkg-config` is the only translator into linker
flags.

**Where to start:** Open the tasks companion to *do* something
(configure / build / modify / run / test / debug); open
the capabilities companion when the question is *what can
Flow express* on this version. **You MUST open
## configure before writing or running
any port code** — its bring-up gate decides whether the binary launches
at all. If DOCA is not
installed yet, route to `doca-setup/SKILL.md` first.

## Ground rule: verify every API name against the installed header

Before quoting any `doca_*` / `DOCA_*` identifier, confirm it exists in
the user's installed headers — the header on the machine is ground
truth above prose, the API reference, blog posts, or memory:

```bash
for header in "$(pkg-config --variable=includedir doca-common)"/doca_flow*.h; do
  grep -n '<candidate_name>' "$header"
  # For a multi-line function declaration, print through its closing `);`.
  awk '/<candidate_name>[[:space:]]*\(/,/[)][[:space:]]*;/' "$header"
done
```

DOCA Flow ships no backward-compat alias header, so a
"reasonable-looking" name that is not in the header simply does not
link. Re-derive from a shipped sample
(`/opt/mellanox/doca/samples/doca_flow/<name>/`) or the guard list in
## API surface and name guards,
never from prose.

## Port bring-up: the gate lives in the tasks companion

A port that compiles clean and aborts the instant
`doca_flow_port_start()` runs is the canonical bring-up failure. The
bring-up gate (probe-before-count, `doca_flow_port_cfg_set_port_id()` plus
the mode-appropriate device source — `doca_flow_port_cfg_set_dev()` in
VNF mode or the installed switch sample's `doca_dev_rep` path — device
taken from launch args not hard-coded, and the binary returning a
non-zero exit from `main()` if the bridge cannot arm and forward) is
enforced step-by-step in
## configure step 6 — open it before
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
   ## Pipe decomposition).
2. **The request asks for something the hardware cannot do** (per-packet
   match on payload bytes outside L4, mutable match keys, …). Name the
   closest legal shape and stop.
3. **The request relies on an API name that is not in the installed
   headers.** Grep the header for the closest real symbol, name it in
   the refusal, and stop without generating code. A later, explicit
   request using the verified symbol may begin a new build workflow;
   do not silently substitute it in the current request.
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

- **“How do I bring up a Flow port on a representor?”** →
  ## configure +
  ## Capabilities and modes.
- **“How do I express *<match X, do Y>* as a pipe?”** →
  ## Capabilities and modes
  + ## modify.
- **“Will this pipe spec program the HW, or will commit fail?”** →
  ## Safety policy
  + ## test.
- **“How do I read Flow counters to investigate observed traffic?”** →
  ## Observability
  + ## debug.
- **“My Flow port won't start — `Failed to get hws cap` / `dest action
  ROOT … err -121`."** → device-placement signature (not a pipe bug);
  the steering plane belongs to the DPU Arm on a `SEPARATED_HOST` /
  NIC-mode BlueField. See the device-placement bullet in
  ## Capabilities and modes
  + ## configure step 2 + Step 0 of
  ## debug.
- **“How do I add hardware-accelerated stateful 5-tuple connection
  tracking (aging, NAT) on top of my existing doca-flow setup?”** →
  ## flow-ct +
  ## flow-ct.

## Audience

External developers writing code that calls `doca_flow_*` (in C/C++, or
via FFI) to program packet steering on a supported NVIDIA NIC/DPU with
DOCA installed at `/opt/mellanox/doca`. Flow ships as a C library
(`pkg-config` module `doca-flow`, package `doca-sdk-flow`) and the
samples are C; other-language consumers reach the same `*.so` through
FFI and the skill stays language-neutral.

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
  (see ## flow-ct).

Do **not** load for general DOCA orientation, "where do I find docs",
install-layout, or non-Flow library questions — use
`doca-public-knowledge-map/SKILL.md`.

## What this skill provides

This is a **thin loader**; substantive material lives in two companion
files:

- the capabilities companion — what Flow can express on this
  version: supported match and action kinds, pipe-decomposition rules,
  the API surface + commonly-invented-name guard list, the Flow
  `DOCA_ERROR_*` overlay, the per-entry / per-pipe observability
  surface, version notes, the safety policy, and the CT companion
  surface.
- the tasks companion — workflows for the six in-scope verbs
  (`configure`, `build`, `modify`, `run`, `test`, `debug`), plus
  `## flow-ct` (stateful-CT overlay), `## shared-resources` (shared
  encap / decap / counter / meter / RSS / IPsec-SA / PSP overlay),
  `## rollback` (pipeline-edit-class snapshots), `## Command appendix`,
  and `Deferred task verbs` for routing install / deploy questions.

The skill assumes DOCA is installed at `/opt/mellanox/doca` and the user can
open a `doca_dev`. Installing DOCA, hugepages setup, and the EAL `dv_flow_en`
devargs prep go through `doca-setup/SKILL.md`; the
hugepages / devargs runtime prerequisites a binary needs *before* a Flow port
starts are pinned in ## configure step 5.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a code bundle: it ships no
pre-written Flow application source, standalone build manifests, or a
`samples/` / `bindings/` / `reference/` subtree. The verified Flow
source is the shipped C sample at
`/opt/mellanox/doca/samples/doca_flow/<name>/` — the agent routes the
user there and prescribes a minimum-diff edit via the modify-a-sample
workflow in
`doca-programming-guide/SKILL.md` plus
the Flow overrides in ## build, and builds
any manifest *in the user's project* against the user's install, where
`pkg-config --modversion doca-flow` is the source of truth.

## Loading order

1. Read this skill first to confirm the question is in scope.
2. For the pipe-spec schema, capability matrix, error taxonomy,
   observability, and safety policy, see
   the capabilities companion.
3. For step-by-step workflows, see the tasks companion.

## Related skills

- `doca-public-knowledge-map/SKILL.md` —
  routing table for public DOCA docs and the on-disk layout of an
  installed package.
- `doca-setup/SKILL.md` — env prep, install
  verification, and the *no install yet* path via the NGC DOCA
  container.
- `doca-programming-guide/SKILL.md` —
  general DOCA patterns shared by every library: the `pkg-config` +
  meson build pattern, the modify-a-shipped-sample first-app workflow,
  the universal lifecycle, the cross-library `DOCA_ERROR_*` taxonomy,
  and the program-side debug order. This skill layers Flow specifics
  on top.
- `doca-flow-tune/SKILL.md` —
  programmed-state inspection (read-only).
- `doca-hardware-safety/SKILL.md` —
  required overlay for card-mode flips (e.g. `mlxconfig` change from
  `SEPARATED_HOST` to `EMBEDDED_CPU`).
- `doca-debug/SKILL.md` — the cross-cutting debug
  ladder (install / version / build / link / runtime / program /
  driver).
