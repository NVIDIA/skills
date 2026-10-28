---
license: Apache-2.0
name: doca-dts
description: >
  Deploy and configure the NVIDIA DOCA Telemetry Service (DTS) as it
  is actually run: (A) BlueField (built-in static pod), (B) Host
  (docker) on x86_64 / aarch64, or (C) DPF (DPUService / helm chart)
  — plus
  the shared configuration layer (`dts_config.ini`, providers,
  exporters, level labels) that all three share. The user's
  DEPLOYMENT TYPE decides everything; pick it first. Trigger even
  without "DTS" for "prometheus on the DPU shows nothing on 9100",
  "telemetry container exits right after start", "add ethtool
  counters to the telemetry DPUService", "how do I upgrade the
  built-in telemetry on BlueField", or "level labels per rack/pod".
  Refuse and route elsewhere for installing DOCA itself, the
  `doca_telemetry` / `doca_telemetry_exporter` library API, installing
  the DPF cluster, Grafana / Prometheus server setup or dashboard /
  PromQL authoring, and BFB / BSP imaging — those belong to other
  skills or to external products.
metadata:
  kind: service
  doca_version: "3.5"
compatibility: >
  Container-shaped DOCA service. On BlueField Arm it ships inside
  the BlueField OS image (upgradable from the NGC image); on an
  x86_64 or aarch64 host it runs from the NGC `-host` image under
  docker; under DPF it is a `DPUService` rendered from the NGC
  `doca-telemetry` helm chart. The version anchor is the container
  tag / chart version, not the host DOCA install. Never fabricate
  a tag or chart version — read it from the NGC page or
  `helm show values`.
---

# DOCA Telemetry Service (DTS)

**Scope note.** DTS is an *externally productized* DOCA service: it is
not one of the DOCA SDK libraries, and its source of
truth is the public DTS guide plus the NGC container / helm-chart pages
and the DOCA Platform (DPF) public docs. This skill covers DTS **as
deployed and configured** — which manifest or invocation starts it,
which `dts_config.ini` keys shape it, how to verify it exports. It does
not cover the library side: reading telemetry in your own program is
`libs/doca-telemetry` (`doca-telemetry/SKILL.md`), publishing
into DTS over IPC from your own program is
`libs/doca-telemetry-exporter` (`doca-telemetry-exporter/SKILL.md`).

**Where to start:** decide the **deployment type** before anything
else — it selects the manifest, the config-directory path, the restart
mechanism and the debug commands. Route through
`doca-setup ## recognize` (`doca-setup/TASKS.md#recognize`)
(`uname -m`, `lspci -d 15b3:`, `bf-release`) to learn whether you are
on a BlueField Arm image (shape A), a plain host (shape B), or inside a
DPF-managed cluster (shape C), then open
[`TASKS.md ## configure`](TASKS.md#configure) and follow that shape's
column. State the shape in the answer before the first command, even
when the question implies it ("on my BlueField" is shape A: the
built-in static pod under kubelet, inspected with `crictl`, not
`docker`). Every configure, change or debug answer ends with the
verification contract, whose green signal is two `/metrics` scrapes at
least two `update` intervals apart showing a counter that advanced
([`TASKS.md ## test`](TASKS.md#test) step 3). If the question is *what
can DTS collect and export, and in which mode*, start at
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
If DOCA / the BlueField OS is not installed yet, route to
`doca-setup/SKILL.md` first.

**Host docker container exits right after start (shape B).** Check the
documented host shape and copy the exact invocations from the
version-matched guide's "Host Deployment" section instead of inventing
flags: (1) a `-host` image tag copied from the NGC `doca_telemetry` tag
list, never guessed; (2) the one-shot init container
(`DTS_CONFIG_DIR=host /usr/bin/telemetry-init.sh`) runs and exits 0
before the long-running `telemetry-run.sh` container; (3) the guide's
host-namespace flags and the `config`, `ipc_sockets`, `data` and `log`
mounts under `/opt/mellanox/doca/services/telemetry/`; (4) the documented
run uses `--rm`, so `docker logs` finds nothing: read the host-mounted
`.../telemetry/log` directory or re-run once without `-d`.

**BlueField built-in DTS (shape A) serves nothing on port 9100.** Inspect
with `crictl pods`, `crictl ps -a` and `crictl logs <id>` (not docker);
check `prometheus=` and `enable-provider=` in
`/opt/mellanox/doca/services/telemetry/config/dts_config.ini` (edit the file
it links to and allow about 60 s for the respawn). Green signal: two
`curl <bluefield-ip>:9100/metrics` scrapes at least two update intervals
apart show a counter that advanced. Change one variable per iteration.

**DPF-managed DTS (shape C).** Change providers and keys in the
`DPUService` helm values (`configMapData`), confirm the keys with
`helm show values`, apply with `kubectl apply -f`, and check the service
conditions with `dpfctl`. Never hand-edit the INI on a DPF-managed DPU.

## Example questions this skill answers well

The CLASSES of DTS questions this skill is built to answer, each with
one worked example. The class is the load-bearing piece; the worked
example is one instance.

- **"Which deployment shape am I in, and what does that change?"** —
  worked example: *"I have a BlueField with DTS already running and a
  separate x86 host with no DPU; do I configure them the same way?"*.
  Answered by the shape table in
  [`CAPABILITIES.md ## Pattern overview`](CAPABILITIES.md#pattern-overview)
  + the shape-first step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do I enable or disable a provider / component?"** — worked
  example: *"turn on `ethtool` counters and drop `sysfs.eth`"*.
  Answered by the provider inventory in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the `enable-provider=` / `disable-provider=` step in
  [`TASKS.md ## modify`](TASKS.md#modify) (in DPF the same option is
  set as `key: value` under `configMapData`, not in the INI).
- **"Which exporter do I pick and where does the endpoint live?"** —
  worked example: *"Prometheus should scrape the DPU, but the fleet
  also pushes to a Fluent Bit forwarder"*. Answered by the pull-vs-push
  exporter table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the exporter step in [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do I upgrade or pin the DTS version on a given shape?"** —
  worked example: *"the BlueField OS bundles an older DTS; I want the
  NGC one"*. Answered by the tag / chart-version rule in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  + the NGC-yaml download pattern in
  [`TASKS.md ## build`](TASKS.md#build), with the generic kubelet
  static-pod manifest mechanics in
  `doca-container-deployment/TASKS.md#configure`.
- **"How do I attach rack / pod / SU labels to every metric?"** —
  worked example: *"per-rack level labels for a DPF-managed fleet"*.
  Answered by the level-labels model in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the `level_labels.ini` / `DPUFlavorTemplate` step in
  [`TASKS.md ## modify`](TASKS.md#modify), with the reprovision
  warning from
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- **"The container is up but nothing is exported — which layer
  failed?"** — worked example: *"`curl :9100/metrics` on the DPU
  returns nothing, `crictl ps` shows the pod running"*. Answered by
  the layer split in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in [`TASKS.md ## debug`](TASKS.md#debug).
- **"How do I prove a config change actually took effect?"** — worked
  example: *"I edited `dts_config.ini`; did DTS restart and pick it
  up?"*. Answered by the restart-within-60-s contract and the green
  signal in [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
  + the bounded loop in [`TASKS.md ## test`](TASKS.md#test).

## When to load this skill

Load this skill for **hands-on DTS deployment or configuration work**:

- **(A) BlueField (built-in static pod):** understanding the kubelet static
  pod under `/etc/kubelet.d/doca_telemetry_standalone.yaml`, its
  `initContainers` (`telemetry-init.sh`, `enable-fluent-forward.sh`),
  disabling auto-start, or upgrading to a newer NGC image than the
  BlueField OS bundles.
- **(B) Host (docker):** running the NGC `-host` image with the
  documented namespace / device / volume flag families, the one-shot
  init container, and deciding whether `--gpus all` or
  the MST device mapping apply to this host.
- **(C) DPF (DPUService / helm chart):** authoring a `DPUService` for the `doca-telemetry`
  chart, mapping `dts_config.ini` keys onto `configMapData`, keeping
  the Prometheus port consistent between `configMapData` and
  `configPorts`, and reading `dpfctl describe` conditions.
- **Configuration (all shapes):** sampling interval, providers and
  components, exporters (Prometheus pull; Fluent Bit / OpenTelemetry /
  Prometheus Remote Write push; storage to `/data`), IPC with external
  applications, level labels and device labels, the HFT HTTP API.
- **Debugging:** the container exits at start or restart-loops, the
  Prometheus endpoint is empty, a provider shows no counters, the Fluent
  Bit destination receives nothing, or a DPF `DPUService` never reaches
  `ApplicationsReady`.

Do **not** load this skill for general DOCA orientation, installing
DOCA or the BlueField OS, installing DPF itself, library-API questions,
or building Grafana dashboards. For those, route via
`doca-public-knowledge-map/SKILL.md`,
`doca-setup/SKILL.md`, or the matching `libs/`
skill.

## What this skill provides

This is a **thin loader**. Substantive material lives in two companion
files:

- `CAPABILITIES.md` — the three deployment shapes and what each fixes
  (manifest, config path, restart mechanism, debug tooling); the
  inventory-discovery procedure (generated config / rendered ConfigMap
  over the example tables) with the provider and exporter catalogues
  (pull / push / storage); the
  version anchor (image tag / chart version, never the host DOCA
  release); the layered error taxonomy (image → manifest → init /
  config → providers → exporter → consumer); the observability surface
  with its named green signal; the safety policy; and the
  public-source pointer.
- `TASKS.md` — step-by-step workflows for the in-scope verbs:
  `configure`, `build` (assembling the deployment artifact — there is
  nothing to compile), `modify`, `run`, `test`, `debug`, plus a
  `Command appendix`, a `Deferred task verbs` block, and the
  `Cross-cutting` contracts every change-recommending answer ends with.

The skill assumes the operator has the root / `sudo` or cluster
credentials the public DTS guide expects, and a target that already
runs a BlueField OS image, a docker-capable host, or a working DPF
cluster. It does not provide any of those prerequisites.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a templates or sample-config
bundle. To keep the boundary clean, it deliberately does not contain —
and pull requests should not add:

- **Copy-paste manifests, helm values files, or `dts_config.ini`
  bodies.** The kubelet yaml is published per version on the NGC
  container page; the chart's keys come from `helm show values`; the
  INI options come from the live guide for the running version. The
  agent quotes the documented flag / key *families* with their "why"
  and points at the exact guide section for the verbatim recipe.
- **Container image tags or chart versions as facts.** Tags follow the
  `<dts-version>-doca<doca-version>[-host]` shape and are read from the
  NGC tag list at answer time; a memorized tag is the canonical
  hallucination for a service skill. See
  `doca-version/SKILL.md`.
- **Grafana dashboards or Prometheus / Fluent Bit server configs.**
  The skill names the DTS-side endpoint and the consumer-side target
  shape and stops there.
- **Credentials, image pull secrets, or cluster credential files.**
  `imagePullSecrets` is named as a chart key family; its contents are
  the user's.
- **A `samples/`, `templates/`, or `reference/` subtree** of any kind.
  A mock or incomplete artifact in this skill's tree, even one labeled
  "reference", is misleading: operators will read it as
  production-ready.

## Loading order

1. Read this file first to confirm the question is about DTS *as
   deployed* (not the library API, not DPF installation), and fix the
   deployment shape via
   `doca-setup ## recognize` (`doca-setup/TASKS.md#recognize`).
2. **For the shape table, providers, exporters, version anchor, error
   taxonomy, observability and safety policy, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run, test,
   debug — see [TASKS.md](TASKS.md)**, reading only the column for the
   chosen shape.
4. **Apply the safety overlay in every phase.** privileged mode with host
   namespaces, `systemctl start dpe`, fwctl / mst device exposure, and a
   DPF flavor-template change that reprovisions DPUs are the
   safety-relevant actions — see
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).

## Related skills

- `doca-setup/SKILL.md` — the front door:
  `## recognize` (`doca-setup/TASKS.md#recognize`) decides the
  deployment shape;
  `## no-install` (`doca-setup/TASKS.md#no-install`) carries the
  tag-selection rule this skill reuses.
- `doca-container-deployment/SKILL.md`
  — generic kubelet standalone / static-pod manifest mechanics on
  BlueField and the pod-spec do-not-invent rule; the DTS-specific
  facts (`/etc/kubelet.d/doca_telemetry_standalone.yaml`, the NGC
  `configs/<version>/` yaml download, the `crictl pods` / `crictl ps` /
  `crictl logs` inspection) come from the DTS guide and the NGC
  container page and live in this skill. Shape A delegates only the
  generic mechanics there via
  `## configure` (`doca-container-deployment/TASKS.md#configure`)
  and `## run` (`doca-container-deployment/TASKS.md#run`).
- `doca-version/SKILL.md` — canonical
  version-handling rules; DTS overlays "the anchor is the container
  tag / chart version, not the host DOCA release".
- `doca-debug/SKILL.md` — the
  universal debug-loop contract (`doca-debug/CAPABILITIES.md#universal-debug-loop-contract`)
  and shared
  command appendix (`doca-debug/TASKS.md#command-appendix`) that
  `TASKS.md ## debug` overlays.
- `doca-hardware-safety/CAPABILITIES.md#safety-policy`
  — DTS is read-only telemetry, but its deployment touches privileged
  containers, `dpe`, firmware-facing devices and DPF reprovisioning.
- `libs/doca-telemetry` (`doca-telemetry/SKILL.md`) — the
  reader library; "consume telemetry in my program" goes there.
- `libs/doca-telemetry-exporter` (`doca-telemetry-exporter/SKILL.md`)
  — the publisher library; "my app reports but DTS sees nothing" is
  split: IPC socket / shared-memory mounts are this skill's, the
  `doca_telemetry_exporter_*` calls are that skill's.
- `doca-public-knowledge-map/SKILL.md`
  — URL routing for the public DOCA documentation set;
  [`CAPABILITIES.md ## Public-source pointer`](CAPABILITIES.md#public-source-pointer)
  carries the DTS guide, NGC and DPF URLs.
