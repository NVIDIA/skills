# DOCA Telemetry Service — Tasks

**Where to start:** The order is `configure → build → modify → run →
test → debug`. The `## test` verb is a bounded iterative loop, not a
one-shot pass — see the eval-loop overlay in `## test` below. For DTS,
`build` is *assembling the deployment artifact* for the user's shape
(a static-pod YAML, a docker invocation, or a `DPUService`), not
compiling anything; `modify` is *adapting the documented recipe with
the minimum change*.

Every step assumes the operator has the live public DTS guide on
`docs.nvidia.com` open as the authoritative reference; this file
prescribes the *order*, *which shape's column to follow*, and *what to
look up where*, not a copy-paste runbook. The one organizing decision
is the **deployment shape**: (A) BlueField (built-in static pod),
(B) Host (docker), (C) DPF (DPUService / helm chart). The configuration layer (`dts_config.ini`,
providers, exporters, level labels) is shared; only its *delivery*
differs per shape. If the version-matched guide section cannot be
reached, stop and report the source gap; never fall back to remembered
flags, tags, or chart keys.

## configure

Deciding the shape, the version anchor, and the configuration plan
before any artifact is written.

1. **Pin the deployment shape first.** Route through
   `doca-setup ## recognize` (`doca-setup/TASKS.md#recognize`)
   (auto-detect `uname -m`, `lspci -d 15b3:`, `bf-release`) and map
   onto the shape table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   - **(A) BlueField (built-in static pod)** — `bf-release` present, `aarch64`; DTS ships
     in the BlueField OS image, auto-started by the static pod
     `/etc/kubelet.d/doca_telemetry_standalone.yaml`.
   - **(B) Host (docker)** — `x86_64` or `aarch64` server; DTS is a docker
     container from the NGC image with a `-host` tag.
   - **(C) DPF (DPUService / helm chart)** — DPUs managed by the DOCA Platform Framework; DTS is
     a `DPUService` (or `DPUServiceTemplate` + `DPUServiceConfiguration`
     referenced from `DPUDeployment.spec.services.dts`).
   The `## recognize` scale gate applies: a fleet signal points at (C),
   a single DPU or host at (A)/(B). State the shape in one line and get
   acknowledgment before continuing.
2. **Pin the version anchor — never fabricate it.** The anchor is the
   container tag or chart version, *not* the host DOCA install
   ([`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
   `doca-version/SKILL.md`):
   - (A): the image tag already in the installed
     `/etc/kubelet.d/doca_telemetry_standalone.yaml` (backup copies:
     `doca_telemetry_service_${version}_arm64.tar.gz` and the yaml on
     the DPU). Upgrading beyond the bundled version is the guide's
     "DOCA Service on NGC" section.
   - (B): a `-host` tag copied verbatim from the NGC container page
     tag list for `nvcr.io/nvidia/doca/doca_telemetry:<tag>`. Example of
     the *rule only*, observed on NGC 2026-09-16: `1.26.5-doca3.5.0-host`.
     If the page is unreachable, say so and ask the user to paste the
     tag list.
   - (C): the chart version from the NGC helm chart page, confirmed
     with `helm show chart` / `helm show values`. Chart version and
     image tag are different numbers; the chart pins its own image tag.
3. **Decide the configuration strategy (initialize vs reuse).** The
   config dir `/opt/mellanox/doca/services/telemetry/config/` starts
   empty; `telemetry-init.sh` generates defaults on first launch **only
   if the dir is empty**. Non-empty => init skips — *why* upgrades keep
   operator edits, and *why* new defaults never appear after a version
   bump. Reset = delete all files in the config dir and restart DTS
   ("Resetting Configuration"). Record which the user wants: fresh
   defaults, keep existing, or reset.
4. **Pick providers and check prerequisites.** `enable-provider=<name>`
   / `disable-provider=<name.component>` (e.g. `disable-provider=sysfs.eth`);
   read the inventory of *this* install first — the `enable-provider=`
   lines of the generated `dts_config.ini` on (A)/(B), the rendered
   ConfigMap on (C)
   ([`CAPABILITIES.md ### Discovering the inventory`](CAPABILITIES.md#discovering-the-inventory));
   the names documented when this skill was written are (sysfs, ethtool, tc, ifconfig, vnic, ppcc_eth, nvidia_smi,
   dcgm, bfperf, storage, amber, diagnostic_data_low_freq,
   diagnostic_data_high_freq). Gating prerequisites: remote collection
   `grpc.sysfs.fw_pages` (BlueField only) needs `systemctl start dpe`;
   `diagnostic_data_*` needs fwctl (`lsmod | grep fwctl`,
   `ls -l /dev/fwctl/fwctl*`); `amber` / `ppcc_eth` need MFT devices
   (the MST device mapping on host); `nvidia_smi` / `dcgm` are **host shape (B) only, and only
   when the host has NVIDIA GPUs** (`nvidia-smi -L` lists them) plus the
   NVIDIA container toolkit (`--gpus all`); never on BlueField (A)/(C). Each is a
   safety-relevant exposure —
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
5. **Pick the exporter and the port plan.** From "Data Export Options"
   / "Enabling Data Output" — the generated config's banner section
   headers are the exporter inventory of the deployed version, same
   discovery rule as step 4: Prometheus pull
   (`prometheus=<scheme>://0.0.0.0:9100`, `prometheus-fset-indexes=name`),
   Fluent Bit push (`fluent_bit_configs/`; set via
   `enable-fluent-forward.sh -i=<host> -p=<port>` on (A), the
   `fluent.forward` group on (C)), OpenTelemetry push
   (`open-telemetry-receiver=<url>`; `configMapData.otlp.*` on (C)),
   Prometheus Remote Write (`remote-write-receiver=<url>`; on (C) via
   the free-form `configMapData.general` map), or storage (`output=/data`). On (C) `configMapData.prometheus.port` **must
   equal** the port in `spec.configPorts` (and the chart's
   `exposedPorts`, if the pinned chart exposes it — confirm with
   `helm show values`) — see the port-mismatch entry in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
   On (B) host networking binds the `prometheus=` port directly on the
   host: `<addr>` is the host IP and the Prometheus server's
   `static_configs` target is `<host-ip>:<port>`. If the HFT HTTP API is
   wanted (`enable-http-api=true`, default `http-port=9117`), add that
   port.
6. **Plan labels.** Level labels:
   `/opt/mellanox/doca/services/telemetry/config/level_labels.ini`,
   enabled with `level-labels-file=/config/level_labels.ini` ((A)/(B));
   on (C) either `configMapData.levelLabels` (template-only key — not
   in `values.yaml`; verify with `helm template`, not `helm show values`)
   or per-DPU generation by a `DPUFlavorTemplate`
   into the DPU's `level_labels.ini` —
   which **reprovisions DPUs** when template or `DPUDevice` values
   change. Confirm with `helm show values` and the DPF level-labels page
   which carrier the pinned chart wires into `level-labels-file=`; do
   not set both. Device labels: `add-pci-address-label=true`,
   `add-numa-node-label=true`.

## build

There is nothing to compile. The image comes from the BlueField OS
image (A) or from NGC (B, C). "Build" means **assembling the deployment
artifact for the pinned shape** by copying the documented recipe for
the pinned version — never hand-writing it.

- **(A) BlueField (built-in static pod).** The artifact is the per-version
  `doca_telemetry_standalone.yaml`, already at `/etc/kubelet.d/`. For
  an upgrade, download the matching yaml from the container's NGC page
  (`configs/<version>/` in the File Browser); do not hand-write it and
  do not `docker pull` the DPU image (the NGC page discourages it — the
  image is K8s-shaped). The yaml mounts
  `/opt/mellanox/doca/services/telemetry/{config,data,ipc_sockets}` and
  runs the `initContainers` command
  `/usr/bin/telemetry-init.sh && /usr/bin/enable-fluent-forward.sh -i=127.0.0.1 -p=24224`;
  the `-i=`/`-p=` pair (Fluent Bit destination) is the only routine
  edit. The yaml path, the NGC `configs/<version>/` download and the
  `crictl` inspection commands come from the DTS guide and the NGC
  container page and are this skill's; generic kubelet static-pod
  manifest mechanics:
  `doca-container-deployment ## configure` (`doca-container-deployment/TASKS.md#configure`).
- **(B) Host (docker).** The artifact is the two invocations from the
  guide's "Host Deployment" section — copy them verbatim for your
  version: the one-shot init (`docker run` with auto-remove, interactive,
  `--name doca-telemetry-init`, the `.../telemetry/config:/config` bind
  mount, and `/bin/bash -c` running `/usr/bin/telemetry-init.sh` with
  the `DTS_CONFIG_DIR=host` prefix), then the long-running
  `telemetry-run.sh` container. Flag families:

  | Flag family | Why |
  | --- | --- |
  | detached, auto-remove, named `doca-telemetry` | addressable by name for `docker ps` / `docker logs` |
  | host network, UTS, IPC and PID namespaces | providers must see the host's NICs and processes; IPC clients share namespaces |
  | privileged mode | provider access to sysfs / devices; safety-relevant |
  | the MST device mapping + the `/usr/lib64/mft` bind mount | MFT devices and libs for `amber` / `ppcc_eth`; drop if MFT is absent |
  | `--gpus all` | `nvidia_smi` / `dcgm` — only if `nvidia-smi -L` shows GPUs and the NVIDIA container toolkit is installed; otherwise omit the flag and keep both providers disabled |
  | `-v .../telemetry/config:/config` | persisted `dts_config.ini`, `level_labels.ini`, `fluent_bit_configs` |
  | `-v .../telemetry/ipc_sockets:/tmp/ipc_sockets` | IPC socket dir for `doca-telemetry-exporter` clients |
  | `-v .../telemetry/data:/data`, `-v .../telemetry/log:/var/log` | `output=/data` storage exporter; DTS logs on the host |

  `$DTS_IMAGE` is `nvcr.io/nvidia/doca/doca_telemetry:<-host tag>` from
  configure step 2; `DTS_CONFIG_DIR=host` is the only documented env var.
- **(C) DPF (DPUService / helm chart).** The artifact is a `DPUService`
  (`apiVersion: svc.dpu.nvidia.com/v1alpha1`, `namespace: dpf-operator-system`,
  `spec.helmChart.source {repoURL, chart: doca-telemetry, version: <chart ver>}` —
  `repoURL` is the NGC DOCA helm repository; copy it from the public
  `dpuservices/dts/DPUService.yaml`)
  or, under `DPUDeployment`, a `DPUServiceTemplate` +
  `DPUServiceConfiguration` (both `spec.deploymentServiceName: dts`).
  Start from the public example in `github.com/NVIDIA/doca-platform`
  (`dpuservices/dts/DPUService.yaml`; `DPUServiceSPX.yaml` is the
  Spectrum-X variant). Configuration is delivered through
  `spec.helmChart.values.configMapData` groups — `providers` (CSV),
  `aggregator_providers` (CSV, aggregator-side providers — template-only
  key, shown commented in the public `DPUService.yaml`; verify with
  `helm template`), `general` (free-form
  options), `fluent` (`forward` / `kafka` / `influxdb` /
  `elasticsearch`), plus `counterset.{cset,fset,xcset}` for data
  filtering (chart key — confirm with `helm show values`) — with the
  documented mapping: the guide's `key=value` in `dts_config.ini`
  becomes `key: value` under `general`. Confirm every other key
  (`image.*`, `dtsConfigDir`, `hostVolumePrefix`, `shareHostSharedMemory`,
  `exposedPorts`, `configMapData.prometheus.*`, `configMapData.otlp.*`,
  `serviceDaemonSet.*`, `tolerations`) with `helm show values` for the
  pinned chart version; do not assert defaults from memory.

## modify

The DTS analog of "modify a sample" is **adapt the documented recipe
to the environment with the minimum change**.

1. **Start from the documented recipe** for the pinned shape and
   version (build above). Quote it; do not author a new one.
2. **Diff against the environment.** Typical forced substitutions:
   (A) the `-i=`/`-p=` destination in the yaml's `initContainers`;
   (B) dropping `--gpus all` or the MST device mapping when the host
   lacks a GPU / MFT, choosing the `-host` tag; (C) chart `version`,
   `configMapData.providers`, `general.*`, `fluent.*`, and
   `configPorts` (and the chart's `exposedPorts`, if present — confirm
   with `helm show values`) kept equal to `prometheus.port`.
3. **Apply minimum-change.** Every extra deviation (a mount, a widened
   namespace, a new privileged device) widens exposure; on (C) every
   value override is a chart-upgrade liability. Config edits go into
   `dts_config.ini` (host `/opt/mellanox/doca/services/telemetry/config/dts_config.ini`,
   `/config/dts_config.ini` in-container) — edit the symlink target
   `clx_config.ini` (or use `sed -i --follow-symlinks`), then expect the
   collector to respawn within ~60 s (documented for `level_labels.ini`,
   observed for `dts_config.ini` too); apply the shape's restart rule
   only if no respawn appears;
   on (C) edit the `DPUService` values instead: the chart delivers
   `configMapData` to the pod as a ConfigMap and DTS regenerates
   `/config/dts_config.ini` from it (the `/configmap` mount path and the
   regeneration mechanism are chart internals — confirm with
   `helm show values` for the pinned chart version).
4. **Re-validate the safety-relevant knobs** from configure step 4
   (privileged mode, `/dev/mst`, fwctl, `systemctl start dpe`, DPF
   flavor-template changes that reprovision DPUs). Any widening the
   user did not ask for is a stop condition
   (`doca-hardware-safety ## Safety policy` (`doca-hardware-safety/CAPABILITIES.md#safety-policy`)).

## run

Start / stop / restart / upgrade, per shape.

- **(A) BlueField (built-in static pod).** *Start*: the yaml present at
  `/etc/kubelet.d/doca_telemetry_standalone.yaml` → kubelet starts the
  pod at boot; confirm with `crictl pods`, `crictl ps`, read
  `crictl logs <container-id>`. *Stop / disable auto-start*: remove the
  yaml (keep a copy — it is the rollback artifact). *Restart after a
  yaml change*: "applied on system restart" — reboot, or the
  kubelet-side mechanics in
  `doca-container-deployment ## run` (`doca-container-deployment/TASKS.md#run`);
  a `dts_config.ini` edit is normally picked up without it (collector
  respawn within ~60 s; `crictl stop <container-id>` is the no-reboot
  restart when it is not) — only the `level_labels.ini` auto-reload is
  documented. *Upgrade*: replace the yaml with the per-version one from NGC
  (build (A)); a non-empty config dir keeps operator config — but the
  downloaded yaml carries the default `-i=127.0.0.1 -p=24224`, and
  `enable-fluent-forward.sh` overwrites `fluent_bit_configs` on every
  init even when the config dir is non-empty (the non-empty rule
  protects only `telemetry-init.sh` output), so re-apply your Fluent
  Bit destination in the new yaml before dropping it into
  `/etc/kubelet.d`, and after the restart confirm the regenerated
  `fluent_bit_configs/` exp-file names your `-i=`/`-p=` destination,
  not `127.0.0.1:24224` — the reset is silent otherwise. "After any
  server configuration changes (firmware, networking, firewall),
  restart DTS."
- **(B) Host (docker).** *Start*: the documented init container once,
  then the documented `telemetry-run.sh` container; `docker ps` shows
  `doca-telemetry` Up, `docker logs doca-telemetry` shows bring-up.
  *Stop*: removing the `doca-telemetry` container (config persists on the host
  volume). *Restart*: stop and re-run; skip init unless the config dir
  was reset. *Upgrade*: new `-host` tag in `$DTS_IMAGE`, re-run.
- **(C) DPF (DPUService / helm chart).** *Start / update*: `kubectl apply -f DPUService.yaml`
  (or the `DPUServiceTemplate` / `DPUServiceConfiguration` /
  `DPUDeployment` set); watch
  `kubectl -n dpf-operator-system exec deploy/dpf-operator-controller-manager -- /dpfctl describe all --show-resources=dpuservice/doca-telemetry-service --show-conditions=all`
  (conditions such as `ApplicationPrereqsReconciled`,
  `ApplicationsReady`, `ApplicationsReconciled`,
  `DPUServiceInterfaceReconciled` — every condition printed must be
  True), then `kubectl get pods -n dpf-operator-system` and the rendered
  DaemonSet pods in the DPU cluster. *Stop / rollback*:
  `kubectl delete -f DPUService.yaml`; on the `DPUDeployment` route the
  controller owns the `DPUService`, so remove the `dts` entry from
  `DPUDeployment.spec.services` (or revert the `DPUServiceConfiguration`
  values) and re-apply instead. *Restart*: a values change rolls per
  `spec.serviceDaemonSet.updateStrategy` (RollingUpdate,
  `maxUnavailable`); DTS also self-restarts on the regenerated ini
  (chart mechanics — confirm with `helm show values`).
  *Upgrade*: bump `spec.helmChart.source.version` after re-reading
  `helm show values` (keys can move); flavor-template label changes
  reprovision DPUs — confirm with the operator first.

Smoke before scale: after the first start on any shape, walk `## test`
steps 1–3 once before adding providers or pointing consumers at it.

## test

DTS has no "compile and unit-test" workflow — testing is operational.

**`## test` is a bounded iterative loop, not a one-shot pass.** Every
configuration mutation (provider enabled, exporter changed, port moved,
tag bumped, label file edited) re-opens the sweep, because the restart that
applies it (automatic for `level_labels.ini`, otherwise the shape's
restart rule) can fail silently (provider missing
its prerequisite, port already bound). The bound is per invocation:
the first sweep plus at most one mutation followed by one complete
second sweep — regardless of which back-edge consumes it. If the
second sweep reveals any failure, including a different
class, do not take another back-edge: return a diagnostic bundle (all
smoke results, debug-layer captures, the capability snapshot if step 6
was reached) and mark the result `requires_human_decision`. Rollback is
the shape's stop action (remove the yaml from `/etc/kubelet.d`,
removing the container, `kubectl delete -f DPUService.yaml` — or, on the
`DPUDeployment` route, removing the `dts` entry from
`DPUDeployment.spec.services` / reverting the `DPUServiceConfiguration`
values; reverting a flavor-template label change reprovisions DPUs
again, so schedule it like the forward change).

**Green criteria:** the DTS process is up with a flat restart count;
`dts_config.ini` exists in the config dir and lists the chosen
providers; the Prometheus `/metrics` endpoint answers twice across at
least two `update` intervals with a counter or timestamp advancing
versus the baseline scrape; the chosen push destination (Fluent Bit /
OTel / storage) receives records; the host's own networking and SSH are
unaffected (negative side); the capability snapshot matches the pinned
shape and version. The auditable "done" sentence names all six.

The eval-loop overlay (rows apply to every shape):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 6 → ## modify → 1 | Snapshot drift (a provider not listed, a port not the planned one) needs a documented config change in `## modify`; then restart the sweep at step 1 | [`## test`](#test) step 6 |
| 1 → ## debug | A restart loop or an exited container is never "probably fine" — `## debug` layers 1–3 before any later step | [`## debug`](#debug) |
| 3 → ## debug layer 4 | `/metrics` answers but nothing advances: process up, provider not sampling — prerequisites or `update` interval | [`## debug`](#debug) |
| 4 → ## run → 1 | Destination receives nothing: re-check `-i=/-p=`, `fluent.*`, or the OTel URL in `## run`, then restart at step 1 | [`## run`](#run) |
| 1..6 → ## run | All six pass → hand off to live consumers (Prometheus server scrape, Grafana) | [`## run`](#run) |

1. **Process up, no restart loop.** (A) `crictl ps` shows the DTS
   container Running with a stable restart count across two reads a
   minute apart; (B) `docker ps` shows `doca-telemetry` Up and
   `docker logs doca-telemetry` has no repeating error; (C) all
   `dpfctl describe` conditions True and DPU-side pods flat.
2. **Config generated.** `/opt/mellanox/doca/services/telemetry/config/dts_config.ini`
   exists ((C): `/config/dts_config.ini` in the pod) and its
   `enable-provider=` lines match configure step 4. If the dir was
   non-empty before first start, this is the *pre-existing* file — say so.
   Cross-check what DTS actually loaded with
   `curl -s <addr>:<port>/management/config_file` and which providers
   really run with `/management/provider` (`run_status: running`) —
   that is the live truth if the file and the process disagree.
3. **Prometheus scraped twice — baseline first.**
   `curl -s <addr>:<port>/metrics | head`, wait at least two
   `update` intervals (default `update=1000` ms), scrape again, and show
   one counter or timestamp that advanced. `<port>` is from configure
   step 5 (documented default `9100`). On (B) `<addr>` is the host IP
   (host networking binds the port on the host itself). On (C) the public
   example sets `spec.configPorts.serviceType: None`, so `<addr>` is
   whatever address the operator's chosen `serviceType` exposes;
   whether the chart's `exposedPorts` adds a NodePort Service is a chart
   detail — confirm with `helm show values`. When level labels are
   configured, both scrapes must also show the expected label
   key=value (e.g. `rack`) on the metric lines — a metric without the
   label is not green for a labels change. Headers with no metric
   lines is not green.
4. **Exporter destination receives.** Fluent Bit: the receiver at the
   `-i=/-p=` (or `fluent.forward.host/port`) destination logs records
   for the configured tag (DPF default `fluent.forward.tag:
   doca_telemetry`; on (A)/(B) read the tag from the generated
   `fluent_bit_configs`); OTel: the receiver's ingest counter moves;
   storage: files appear under `/opt/mellanox/doca/services/telemetry/data`
   and decode with `clx_read` (package `collectx-clxapidev`). Before
   touching the destination, read `/management/exporter` twice: the
   exporter's `statistics` counters increasing prove DTS is pushing;
   a destination that still sees nothing is then a layer-5/6 problem,
   not a DTS one. Skip only if the plan is Prometheus-only, and say so.
5. **Negative side.** SSH to the host/DPU still works from outside and
   its networking is unchanged; `crictl ps` / `docker ps` / pod restart
   counts are flat since step 1. DTS is read-only telemetry — any
   observed side effect is a stop condition.
6. **Capability snapshot.** Record: shape, image tag / chart version,
   providers enabled (from `dts_config.ini`), exporters and their ports,
   label files in use. Future debug sessions start here.

## debug

Layered diagnosis mirroring
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
Walk the layers in order; each is *symptom → read-only capture →
single-variable change*; clear a layer before dropping below it. One
change per iteration, re-capture; if still not green after one
iteration, roll back per `## test` and return
`requires_human_decision`. Generic loop contract:
`doca-debug ## Universal debug-loop contract` (`doca-debug/CAPABILITIES.md#universal-debug-loop-contract`).

**Every debug answer opens with the deployment shape and closes with
the `/metrics` proof.** State the shape before the first command, even
when the question implies it: "on my BlueField" means shape A, the
built-in static pod under kubelet, inspected with `crictl` rather than
`docker`; a plain host is shape B; a DPF cluster is shape C. Then walk
the layers below for that shape. End with the verification contract
from [`## test`](#test), using step 3 as the green signal: scrape
`curl -s <addr>:<port>/metrics` twice, at least two `update` intervals
apart, and show a counter or timestamp that advanced.

1. **Image / tag layer.** *Symptom*: pull failure,
   immediate exit, or providers behaving "wrong" for the box.
   *Capture*: the tag in the yaml / `$DTS_IMAGE` / `helm show chart`
   versus the NGC tag list; on (B) the documented run uses auto-remove, so
   an exited container is gone and `docker logs` has nothing — read the
   host `/opt/mellanox/doca/services/telemetry/log` dir (the `/var/log`
   bind mount) or re-run the documented command once without `-d` to
   capture the exit output. *Gotcha*: a DPU tag on a host or a `-host`
   tag on the DPU — different images. *Change*: the one tag.
2. **Manifest / pod scheduling layer.** *Symptom*: (A) no pod in
   `crictl pods`; (C) conditions not True or no DPU-side pods.
   *Capture*: (A) is the yaml really in `/etc/kubelet.d/` and was the
   system restarted since; (C) `kubectl get pods -n dpf-operator-system`,
   the DPUService conditions, `serviceDaemonSet.nodeSelector` /
   `tolerations`. Kubelet mechanics:
   `doca-container-deployment ## debug` (`doca-container-deployment/TASKS.md#debug`).
   *Change*: one manifest field or one restart.
3. **Init container / config generation layer.** *Symptom*: DTS runs
   but the expected config is missing — no new defaults after an
   upgrade, `enable-fluent-forward.sh` destination not applied, no
   `dts_config.ini`. *Capture*: `ls` the config dir; the init
   container's log (`crictl logs <init-container-id>`,
   on (B) the terminal output of the auto-remove, interactive init run). *Gotcha*:
   the config dir was **non-empty**, so `telemetry-init.sh` skipped
   generation by design; conversely, on (A) a freshly downloaded
   upgrade yaml carries the default `-i=127.0.0.1 -p=24224` and
   `enable-fluent-forward.sh` overwrites `fluent_bit_configs` on every
   init, so the Fluent Bit destination resets to localhost unless the
   edit was re-applied to the new yaml. *Change*: reset (delete config dir contents,
   restart) or hand-merge the one missing key.
4. **DTS process and providers layer.** *Symptom*: `/metrics` answers
   but a provider's counters are absent or frozen, or DTS restart-loops
   after an ini edit. *Capture*: the provider's error line in the
   container log; prerequisite probes — `lsmod | grep fwctl`,
   `ls -l /dev/fwctl/fwctl*` (diagnostic data), `systemctl status dpe`
   (remote collection), `/dev/mst` and the MFT lib mount (amber /
   ppcc_eth), `--gpus all` (nvidia_smi / dcgm). *Gotcha*: ini edited
   and nothing changed — first `ls -l dts_config.ini`: a plain file
   instead of the `-> clx_config.ini` symlink means a `sed -i` broke the
   link and DTS never saw the edit; then look for the respawn
   (`exited: clx` / `spawned: 'clx'` in the container log) and restart
   only if it is absent; on (A)/(B) confirm you edited the
   host-mounted `/opt/mellanox/doca/services/telemetry/config/dts_config.ini`,
   not a copy; on (C) the file is regenerated from the ConfigMap the
   chart delivers (chart mechanics — confirm with `helm show values`),
   so edit the `DPUService` values. *Change*: one provider or one prerequisite.
5. **Exporter reachability layer.** *Symptom*: nothing on `/metrics`
   from outside, or the push destination sees nothing. *Capture*:
   `curl -s <addr>:<port>/metrics | head` on-box and off-box;
   `ss -tlnp` for the listener; on (C) `configMapData.prometheus.port`
   versus `spec.configPorts` (and the chart's `exposedPorts`, if present
   — confirm with `helm show values`). *Gotcha*: **port mismatch
   on DPF** — chart default and example (9100) can differ, so the
   Service forwards to a port DTS does not bind. *Change*: set both
   ports to one value, or fix a single firewall rule.
6. **Consumer side layer.** *Symptom*: DTS green end-to-end but the
   Prometheus server / Grafana / IPC client shows nothing. *Capture*:
   the consumer's own target status (`static_configs` target
   `<dpu-ip>:9100` per the guide's Grafana section); for IPC, whether
   the client has **write access to the socket dir**
   `/opt/mellanox/doca/services/telemetry/ipc_sockets` (root-created;
   the client may need root) and sees `/dev/shm/telemetry` (on (C),
   `shareHostSharedMemory` for bare-metal clients — confirm with
   `helm show values`). *Change*: one consumer-side setting; API
   questions go to
   `libs/doca-telemetry-exporter` (`doca-telemetry-exporter/SKILL.md`).

### Path-specific ladders

Three paths cut across the six layers; walk them as their own
symptom → capture → change sequence, still one change per iteration.

- **Storage read-back (`output=/data`).** (i) *No files*: layer 3/4 —
  is `output=/data` uncommented in the generated `dts_config.ini`, is
  the `data/` host dir mounted at `/data` (yaml `volumes:` / the
  `-v …/telemetry/data:/data` run flag), does the start-up block list
  any enabled provider? (ii) *Files but no growth*: layer 4 — the
  providers are not collecting; same captures as layer 4. (iii) *Files
  grow but cannot be read*: the guide's "Enabling Data Output" reader
  is `clx_read` (`/opt/mellanox/collectx/bin/clx_read` inside the
  container via `crictl exec` / `docker exec`, or the
  `collectx-clxapidev` package on the host) with the schema directory
  `-s /data/schema`; a decode failure with files present means the
  schema dir is missing or from another version — point `-s` at the
  schema written next to those files, never at another install.
  Validation: a file timestamp advancing across two `update` intervals
  **and** one record decoded.
- **Remote collection (`enable-provider=grpc.<provider>`, BlueField
  only).** (i) *Provider rejected / absent in the start-up block*:
  `systemctl status dpe` on the BlueField — DPE not running is the
  layer-4 prerequisite; starting it is privileged (Safety policy gate).
  (ii) *DPE running, provider still silent*: the `grpc.`-prefixed
  option lines (`grpc.<provider>.<component>=…`) must mirror the
  non-grpc spelling exactly; compare against the commented examples in
  the generated config. (iii) *Works, then stops*: `dpe` restarted or
  stopped — re-check (i) before touching the ini. Validation: the
  provider's counters on `/metrics` across two scrapes.
- **Boundary answers ("DTS does not do X").** A refusal still owes the
  user a validation and a debug step for the alternative it recommends
  (e.g. the exporter library's own NetFlow API → that skill's `## test`
  / `## debug`); link the owning skill's anchors instead of stopping at
  the route.

## Command appendix

DTS-specific commands the verbs above reach for, grouped so the agent
picks the right family without searching prose. Every row is a class —
do not invent flags beyond what the row names; the guide section for
the pinned version is the only source for the verbatim form. `<addr>`,
`<port>`, `<tag>`, `<chart ver>` come from `## configure`.

| Purpose | Command (class shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Shape detection | `uname -m`, `lspci -d 15b3:`, `cat /etc/mlnx-release` (BlueField bundle) or `/etc/nvidia/bf-release` (if present) | [`## configure`](#configure) step 1 | Output pins exactly one of (A)/(B)/(C); user acknowledges. |
| (A) pod state | `crictl pods`, `crictl ps` | [`## run`](#run), [`## test`](#test) step 1 | DTS pod Ready, container Running, restart count flat across two reads. |
| (A) logs | `crictl logs <container-id>` (init container id for layer 3); persisted copy under the host dir the yaml's `volumes:` maps to `/var/log` (`/var/log/doca/telemetry` in the shipped yaml) | [`## debug`](#debug) layers 3–4 | Start-up block lists the intended providers as enabled; no repeating provider / exporter error. |
| Management API (all shapes, needs `prometheus=`) | `curl -s <addr>:<port>/management` (route table), `/management/provider`, `/management/provider/<name>`, `/management/exporter`, `/management/exporter/<name>`, `/management/config_file`, `/management/all_configs`, `/management/(cset\|fset\|xcset)[/<name>]` | [`## test`](#test), [`## debug`](#debug) layers 4–5 | Providers report `run_status: running`; exporter `statistics` counters increase between two calls; `config_file` shows your edit. |
| In-container log (all shapes) | `crictl exec <id> tail -n 200 /var/log/clx.log` / `docker exec doca-telemetry tail -n 200 /var/log/clx.log` / `kubectl exec <pod> -- tail -n 200 /var/log/clx.log` | [`## debug`](#debug) | Start-up block (`# of enabled providers`) present; no repeating provider / exporter error. |
| Log verbosity (all shapes) | `verbose=` in the generated `dts_config.ini` (scale in its inline comment: error / info / debug); raise for one restart cycle, then restore | [`## debug`](#debug) | Debug-level lines appear after the restart; value restored afterwards. |
| (A) enable / disable auto-start | place / remove `/etc/kubelet.d/doca_telemetry_standalone.yaml`, then system restart | [`## run`](#run) | Pod appears / disappears after restart; removed yaml kept as rollback. |
| (A) upgrade artifact | download `configs/<version>/doca_telemetry_standalone.yaml` from the NGC container page | [`## build`](#build) | Tag in the yaml equals a tag on the NGC list. |
| (B) init (one-shot) | the guide's one-shot `docker run` (auto-remove, interactive, `--name doca-telemetry-init`, the `.../config:/config` bind mount, `/usr/bin/telemetry-init.sh` under the `DTS_CONFIG_DIR=host` prefix) — verbatim form from "Host Deployment" | [`## build`](#build), [`## run`](#run) | Exits 0; `dts_config.ini` present (generated only if the dir was empty). |
| (B) run | the guide's detached `docker run` of `/usr/bin/telemetry-run.sh` as `doca-telemetry` (host namespaces, privileged mode, the documented mounts) — verbatim | [`## run`](#run) | `docker ps` shows `doca-telemetry` Up. |
| (B) state / logs / stop | `docker ps`, `docker logs doca-telemetry` (while running — with auto-remove an exited container is gone; read the host `.../telemetry/log` dir or re-run once without `-d`), removing the `doca-telemetry` container | [`## run`](#run), [`## debug`](#debug) | Up with no restarts; no repeating errors; stop removes the container, config persists. |
| (C) chart introspection | `helm show chart` / `helm show values` for `doca-telemetry` at `<chart ver>` from the NGC DOCA helm repository (the `repoURL` in the public `dpuservices/dts/DPUService.yaml`; chart page in [`CAPABILITIES.md ## Public-source pointer`](CAPABILITIES.md#public-source-pointer)) | [`## configure`](#configure) step 2, [`## build`](#build) | Keys you plan to override exist; image tag read, not guessed. |
| (C) apply / delete | `kubectl apply -f DPUService.yaml` / `kubectl delete -f DPUService.yaml` | [`## run`](#run) | Resource created; conditions turn True within the rollout window. |
| (C) conditions | `kubectl -n dpf-operator-system exec deploy/dpf-operator-controller-manager -- /dpfctl describe all --show-resources=dpuservice/doca-telemetry-service --show-conditions=all` | [`## run`](#run), [`## test`](#test) step 1 | Every condition printed is True (including `ApplicationPrereqsReconciled`, `ApplicationsReady`, `ApplicationsReconciled`, `DPUServiceInterfaceReconciled`). |
| (C) pods | `kubectl get pods -n dpf-operator-system` (host cluster); DaemonSet pods in the DPU cluster | [`## test`](#test) step 1 | Running, restarts flat. |
| (C) level-label preview | `dpfctl flavor-template render -f template.yaml --dpudevice-file device.yaml` | [`## configure`](#configure) step 6 | Rendered INI has `[global_labels] [device_labels] [device_mapping] [data_types_mapping]`; applying it reprovisions DPUs. |
| Config edit | edit the symlink target `/opt/mellanox/doca/services/telemetry/config/clx_config.ini` (`dts_config.ini -> clx_config.ini`; `sed -i` without `--follow-symlinks` breaks the link) ((A)/(B)) or the `DPUService` `configMapData` values ((C)) | [`## modify`](#modify) | After the respawn (~60 s; `exited: clx` / `spawned: 'clx'` in the container log, pod attempt count unchanged), new `enable-provider=` visible on the next `/metrics`. |
| Config reset | delete all files in `/opt/mellanox/doca/services/telemetry/config/`, restart DTS | [`## configure`](#configure) step 3, [`## debug`](#debug) layer 3 | Fresh `dts_config.ini` regenerated by `telemetry-init.sh`. |
| Prometheus probe | `curl -s <addr>:<port>/metrics \| head` — twice, >= 2 `update` intervals apart | [`## test`](#test) step 3 | Metric lines present; a counter/timestamp advanced between scrapes. |
| Provider prerequisites | `lsmod \| grep fwctl`, `ls -l /dev/fwctl/fwctl*`, `systemctl status dpe`, `ls /dev/mst` | [`## configure`](#configure) step 4, [`## debug`](#debug) layer 4 | Prerequisite present *before* the provider is enabled. |
| Storage exporter check | `ls /opt/mellanox/doca/services/telemetry/data`, `clx_read <file>` (`collectx-clxapidev`) | [`## test`](#test) step 4 | Files grow across intervals; `clx_read` decodes them. |
| Cross-cutting health | `ss -tlnp` for `<port>`; restart counts from the shape's state command | [`## debug`](#debug) layer 5, [`## test`](#test) step 5 | Listener bound on the planned port; restarts flat; host SSH unaffected. |

Three cross-cutting rules for this appendix:

- **Never invent a DTS option, tag, or chart key.** The guide section
  named in each verb is the contract; the NGC tag list and
  `helm show values` are the only secondary sources.
- **Init log before run log.** When the config is "wrong", read the
  init container's output and the config dir first; only then the
  running container's log.
- **Cross-link instead of duplicate.** The read-only triple, `dmesg`,
  and generic container-runtime commands live in
  `doca-debug ## Command appendix` (`doca-debug/TASKS.md#command-appendix`)
  and `doca-container-deployment ## Command appendix` (`doca-container-deployment/TASKS.md#command-appendix`).

## Deferred task verbs

- **Installing DOCA / preparing the host or BlueField** — route to
  `doca-setup ## configure` (`doca-setup/TASKS.md#configure`) and
  `doca-setup ## no-install` (`doca-setup/TASKS.md#no-install`).
  (B) and (C) need no host DOCA install — the anchor is the container
  tag / chart version.
- **Installing or operating the DPF cluster itself** (operator install,
  DPU provisioning, `DPUSet` rollout) — the DPF documentation
  (`docs.nvidia.com/networking/display/dpf/`) and the public
  `github.com/NVIDIA/doca-platform` repo; this skill owns only the DTS
  `DPUService` and its values.
- **Library API questions** (`doca_telemetry_*`,
  `doca_telemetry_exporter_*`, IPC client code, custom schemas) —
  `libs/doca-telemetry` (`doca-telemetry/SKILL.md`) and
  `libs/doca-telemetry-exporter` (`doca-telemetry-exporter/SKILL.md`).
  This skill stops at "the socket dir exists and the client can write".
- **Prometheus server / Grafana administration** — the guide's
  "Deploying with Grafana Monitoring" section names the DTS-side
  prerequisites (`prometheus=<scheme>://0.0.0.0:9100`,
  `enable-provider=sysfs`, a `static_configs` target `<dpu-ip>:9100`);
  everything beyond that belongs to the Prometheus / Grafana projects.
- **BFB / BSP / firmware** — the BlueField OS image that bundles DTS is
  owned by the BlueField platform docs; a DTS upgrade without a BFB
  change is the "DOCA Service on NGC" path in `## run` (A).

## Cross-cutting

- The public DTS guide is the single source of truth for option names,
  paths, ports, and the init/run recipes; the NGC pages for tags and
  yaml downloads; `helm show values` for chart keys. Anything not
  traceable to one of those is not written.
- The deployment shape decides everything after configure step 1; do
  not mix columns (a `docker pull` of the DPU image, a `-host` tag in a
  static pod, an ini edit on a DPF-managed DPU).
- Configuration is shared across shapes; only delivery differs. Two
  invariants hold everywhere: init generates defaults **only into an
  empty config dir**, and an ini change (documented for
  `level_labels.ini`, observed for `dts_config.ini`) respawns the
  collector **within ~60 s** — restart manually only if no respawn appears.
- DTS is read-only telemetry, but privileged mode, the MST device mapping,
  fwctl exposure, `systemctl start dpe`, and DPF flavor-template changes
  (DPU reprovision) are the safety-relevant actions —
  `doca-hardware-safety ## Safety policy` (`doca-hardware-safety/CAPABILITIES.md#safety-policy`)
  and [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- Every change-recommending answer ends with the bundle's five-step
  verification contract (preconditions → smoke spawn → smoke probe →
  scale → observability), with the Prometheus double-scrape as the
  named green signal and flat restarts + unaffected host SSH as the
  negative side. Not-green fires `## debug` for one bounded iteration,
  then rollback.
- For URL routing see
  `doca-public-knowledge-map/SKILL.md`;
  [`CAPABILITIES.md ## Public-source pointer`](CAPABILITIES.md#public-source-pointer)
  carries the guide, NGC, and DPF URLs.
