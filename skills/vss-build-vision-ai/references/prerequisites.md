# VSS Prerequisites Check
<a id="preflight"></a>

Verifies system readiness for any VSS developer profile. For NGC CLI setup specifically, see [`ngc.md`](ngc.md).

## Preflight — quick reference

Run the GPU, Docker, NVIDIA runtime, network, and NGC gates below before
resolution or deployment. For DGX Spark / IGX Thor / AGX Thor, also run the
cache-cleaner install and verification block in
[`edge.md`](edge.md#cache-cleaner-every-edge-deploy).

Also verify the runner for the build helper scripts. `uv` is preferred, but a
host with `python3` and `PyYAML` is supported. Do this before generating
`resolved.yml`; otherwise a host can pass every system prerequisite and still
fail during normalization.

```bash
if command -v uv >/dev/null 2>&1; then
  VSS_SKILL_PY=(uv run)
elif python3 - <<'PY' >/dev/null 2>&1
import yaml
PY
then
  VSS_SKILL_PY=(python3)
else
  cat >&2 <<'EOF'
Missing Python runner for vss-build-vision-ai helper scripts.

Install uv:
  curl -LsSf https://astral.sh/uv/install.sh | sh

Or install PyYAML for the active Python 3 environment and use the direct
Python fallback:
  python3 -m pip install PyYAML
EOF
  exit 1
fi

printf 'VSS_SKILL_PY=%q' "${VSS_SKILL_PY[@]}"
printf '\n'
```

## Repo detection
<a id="repo-detect"></a>

Auto-detect the `video-search-and-summarization/` checkout and export it as
`$REPO` before asking the user. Probe the git root first, then common paths,
accepting a candidate only if it carries `deploy/docker/compose.yml` and
`skills/vss-build-vision-ai/`:

```bash
REPO="${REPO:-}"
if [ -z "$REPO" ]; then
  git_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  candidates=()
  [ -n "$git_root" ] && candidates+=("$git_root")
  candidates+=(
    "$PWD"
    "$PWD/.."
    "$PWD/../.."
    "$HOME/video-search-and-summarization"
    "$HOME/VSS/vss-oss/video-search-and-summarization"
    "$HOME/VSS/video-search-and-summarization"
  )

  for candidate in "${candidates[@]}"; do
    candidate="$(cd "$candidate" 2>/dev/null && pwd -P || true)"
    if [ -n "$candidate" ] \
      && [ -f "$candidate/deploy/docker/compose.yml" ] \
      && [ -d "$candidate/skills/vss-build-vision-ai" ]; then
      REPO="$candidate"
      break
    fi
  done
fi

if [ -z "$REPO" ]; then
  echo "Could not auto-detect video-search-and-summarization; ask the user for the checkout path."
else
  echo "REPO=$REPO"
fi
```

## When to Use

Use this reference when:

- A VSS deploy failed and you need to diagnose why
- User asks to verify GPU, Docker, or system setup
- After a driver or Docker update
- Called from BOOTSTRAP during first-time setup

---

## Sudo Access

Most prerequisite steps require `sudo` (Docker install, NVIDIA toolkit, kernel settings, systemctl, edge cache-cleaner). On cloud instances (Brev, Colossus, DGX Cloud) the default user typically has passwordless sudo. On bare-metal machines, the user may need to enter a password or be in the `sudo` group.

Check first — every subsequent step branches on this result:

```bash
sudo -n true 2>/dev/null && SUDO_NOPASSWD=1 || SUDO_NOPASSWD=0
echo "SUDO_NOPASSWD=${SUDO_NOPASSWD}"
```

**Run that probe on its own.** Batched with the GPU, Docker, and addressing
checks it takes all of them down whenever an agent's `sudo` token is refused,
and a preflight that came back empty reads as a broken host rather than as one
unasked question.

**Branch — passwordless sudo (`SUDO_NOPASSWD=1`):** the skill can run
the install snippets in this document directly (`sudo modprobe`,
`sudo apt-get install`, `sudo tee`, `sudo -b`, etc.).

**Branch — password-required sudo (`SUDO_NOPASSWD=0`):** **do not**
attempt `sudo -n` installs. They will fail silently (exit 1, no
`askpass`) and leave the host half-configured — most visibly with the
edge cache-cleaner (`sudo -b /usr/local/bin/sys-cache-cleaner.sh`):
the install no-ops, deploy proceeds, and first-frame inference OOMs
on edge platforms with no obvious cause.

Instead, surface the failing command block verbatim to the user with
a handoff like:

> *"Sudo requires a password on this host. Please run the block
> below in your shell, then confirm so I can continue."*
> *(then paste the relevant install snippet from this doc)*

Resume only after the user confirms the command succeeded. Do not
re-run `sudo -n` checks in a loop — they won't change without user
action.

**Branch — `sudo` denied to the agent:** the probe above can be refused
*before it executes*, by the agent's own permission policy rather than by
`sudoers` (`blocked by administrator policy`). It inherits the previous branch's
rule against attempting `sudo -n` installs and nothing else — in particular, not
its handoff: **offer the approved run below before handing anything over.** And
since the host's sudo state was never determined, report that **this agent** may
not run `sudo`, not that the host requires a password.

**Take this branch only on a refusal you actually saw.** Run the probe and
quote what came back; never predict the refusal from the environment, the
platform, or a previous session. The same rule applies one level down: a
`sudo -n true` refusal is the *probe's* result, and a `sudoers` rule scoped
to specific commands (`NOPASSWD: /usr/bin/apt-get`) can still admit the
operation the step actually needs — so report the probe as what failed, and
say which command was never attempted rather than implying it would fail.

**Offer the run under an explicit approval.** Where the harness can prompt, say
what the command changes on the host and that it needs passwordless `sudo` to
succeed — a password prompt has no terminal to appear on and fails immediately.

When the prompt is declined or unavailable, give the user the exact command to
run in their terminal, in the [Handoff form](#handoff) below. Resume after they
confirm success, then repeat only the failed check. Do not work around the
restriction through scripts or containers.

### Handoff form
<a id="handoff"></a>

One block, one ask. A handoff competing with unrelated findings is the noise
that gets it skipped.

- Give **the command**, copy-pasteable, with paths already resolved — not a
  description of it and not a pointer to this file.
- Say in one line what it fixes and what breaks without it.
- Keep what the user cannot act on out of that message. Batch several
  commands only when they run in one sitting, and then as consecutive blocks
  with nothing between them.
- Do not re-diagnose, restate the version matrix, or recount what already
  passed.
- **Resume by re-running only the check that failed**, not the whole
  preflight.

## Kernel Settings

Required for Elasticsearch and Kafka. Apply before deploying:

```bash
sudo sysctl -w vm.max_map_count=262144
sudo sysctl -w net.core.rmem_max=5242880
sudo sysctl -w net.core.wmem_max=5242880
```

To persist across reboots, write to `/etc/sysctl.d/99-vss.conf`:

```bash
cat <<'EOF' | sudo tee /etc/sysctl.d/99-vss.conf
vm.max_map_count = 262144
net.core.rmem_max = 5242880
net.core.wmem_max = 5242880
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
EOF
sudo sysctl --system
```

## Network addressing — HOST_IP / EXTERNAL_IP
<a id="addressing"></a>

VST and the NIMs bind their configured container ports; host-published port
conflicts are handled through the `*_HOST_PORT` overrides. Internal VST defaults
live in `services/vios/vst.env` (`VST_INTERNAL_URL=http://vst-ingress:30888`,
`VST_INGRESS_ENDPOINT=vst-ingress:30888/vst`), while model URLs and UI/report
links still use `HOST_IP` / `EXTERNAL_IP` where those clients dial the host.

**`HOST_IP` — the in-cluster dial address.** Must be reachable from the docker
bridge (VLM→VST), the host-net agent, and (since `EXTERNAL_IP` inherits it) LAN
browsers. Detect it with `ip route get 1.1.1.1`, which is correct
on bare-metal LAN **and** cloud VMs (returns the primary **private** IP). The one
exception is a host whose **default route is a VPN/tunnel** (`gpd*`, `tun*`, `wg*`,
`tailscale*`) — there `ip route` returns the VPN IP, which the bridge and LAN
clients **cannot** reach. Detect that and fall back to the LAN IP:

```bash
HOST_IP=$(ip route get 1.1.1.1 | awk '/src/{for(i=1;i<=NF;i++)if($i=="src")print $(i+1)}')
IFACE=$(ip route get 1.1.1.1  | awk '/dev/{for(i=1;i<=NF;i++)if($i=="dev")print $(i+1)}')
case "$IFACE" in gpd*|tun*|tap*|wg*|ppp*|tailscale*|utun*)
  echo "default route is VPN ($IFACE → $HOST_IP) — not bridge-reachable. LAN candidates:"
  ip -4 -o addr show scope global up \
    | awk '$2 !~ /^(gpd|tun|tap|wg|ppp|tailscale|utun|docker|br-|veth)/{print $2, $4}' ;;
esac
```

If the VPN branch fires — or the host is multi-NIC and the right IP is ambiguous —
**prompt the user for the LAN IP instead of guessing.** Verify the pick with the
bridge→host probe in [`troubleshooting.md`](troubleshooting.md#vlm-500--fetch_video_async-timeouterror--bridge-nim-cant-reach-host-vst).

**Detect Brev before choosing an address below — file check, not a question.**
A Brev instance publishes an environment context file mapping each exposed port
to its FQDN, and sets `BREV_ENV_ID` in `/etc/environment`. On a hit,
[`brev.md`](brev.md) owns the access-mode decision. For a browser-facing deployment,
its secure-link values are mandatory, and no LAN or public/elastic IP is offered
as an option. For an explicitly authorized host-side Search Harbor eval,
select [`the host-local eval branch`](brev.md#host-local-search-eval) during
preflight when the 7777 link is not published or unreadable: keep the deployment's local
host settings and skip the browser-only address selection below. Public links
are not a prerequisite for that branch; local reachability and readiness still
apply.

```bash
BREV_CTX="${BREV_ENVIRONMENT_CONTEXT_PATH:-/etc/brev/environment-context.json}"
jq -e '.ports' "$BREV_CTX" 2>/dev/null \
  || grep -h '^BREV_ENV_ID=' /etc/environment 2>/dev/null \
  || echo "not a Brev instance"
```

For browser-facing access, take the browse host from that file's `fqdn` for the ingress port
([`brev.md`](brev.md) → *Resolving a secure link*). Never assemble one from
`BREV_ENV_ID` and a domain.

A public/elastic IP is wrong here: the browser arrives over the secure link, so
a host-IP origin serves `http://…:7777` URLs into an `https://` page and the
browser blocks them as mixed content.

**`EXTERNAL_IP` — the browser-facing address.** Keep it equal to `HOST_IP` for
a plain LAN box. Set it explicitly only when the browser path differs from the
internal one:

| Environment | `EXTERNAL_IP` |
|---|---|
| Plain LAN | same as `HOST_IP` (the LAN IP) |
| Cloud VM (AWS/GCP/Azure) | the **public/elastic IP** — **not on the NIC** (provider NAT, so `ip route`/`ip addr` can't see it). Read from instance metadata, e.g. AWS IMDSv1: `curl -s --max-time 2 http://169.254.169.254/latest/meta-data/public-ipv4` (`--max-time` so it fails fast off-AWS; IMDSv2-only instances must first fetch an `X-aws-ec2-metadata-token`). **Prompt the user** to confirm the public IP and that the security group opens the port. |
| Brev, browser-facing | the secure-link FQDN the environment context file publishes for the ingress port — resolved, not constructed (`brev.md`) |
| Brev, explicitly authorized host-side Search Harbor eval without published links | retain the deployment's local host setting; follow [`the host-local eval branch`](brev.md#host-local-search-eval) |
| Reach over a tunnel | the tunnel address (Tailscale `100.x`, cloudflared/ngrok hostname) |

A private `192.168.x` / `10.x` `EXTERNAL_IP` (including a GlobalProtect VPN IP) is
only reachable on that LAN/VPN, never the public internet — and corp VPNs usually
block client-to-client, so a VPN IP rarely works even for VPN peers. For real remote
access use the cloud public IP or a mesh VPN (Tailscale). **When unsure where the
user will browse from, ask before setting `EXTERNAL_IP` — but only when the Brev
check above came back empty.**

## Firewall — Docker bridge → host services
<a id="firewall"></a>

Pick `HOST_IP` / `EXTERNAL_IP` first — see [Network addressing](#network-addressing--host_ip--external_ip).

VSS runs a mixed network topology: VST and `vss-agent` use host networking, but
the VLM/LLM NIMs run on the Compose Docker bridge (`<project>_default`, default `vss_default`). The agent hands the VLM a
`http://$HOST_IP:30888/...` VST URL, so the bridge must reach host ports. If `ufw`
is active it blocks the bridge subnet by default — the VLM then can't download
clips and `video_understanding` returns HTTP 500 (`fetch_video_async TimeoutError`).

Allow the Docker bridge subnets before deploying (skip if `ufw` is inactive). Use
the specific `/16`s, **not** a broad `172.16.0.0/12` (it overlaps corporate-VPN
ranges); **do not disable ufw**:

```bash
if sudo ufw status 2>/dev/null | grep -q "Status: active"; then
  sudo ufw allow from 172.17.0.0/16   # docker default bridge
  sudo ufw allow from 172.18.0.0/16   # vss_default (first compose bridge, unless COMPOSE_PROJECT_NAME is changed)
  sudo ufw reload
fi
```

If the Compose project network already exists and landed on a different subnet
(multiple Docker stacks on the host), allow that one instead:
`docker network inspect "${COMPOSE_PROJECT_NAME:-vss}_default" -f '{{range .IPAM.Config}}{{.Subnet}}{{end}}'`.
This applies to every profile on an active-ufw host, including Brev.

**Browser access from another machine.** The bridge rule above only lets *containers*
reach the host — it does **not** open ports to other devices. The `HAPROXY_HOST_PORT`
ingress (default `7777`) reverse-proxies the UI, agent API, and VST, so a single
allow covers all three:

```bash
sudo ufw allow 7777/tcp        # HAProxy ingress — fronts UI + agent + VST. Or scope to your LAN:
# sudo ufw allow from 192.168.0.0/16 to any port 7777 proto tcp
sudo ufw reload
```

`nvstreamer` is the exception — its host-published port (`NVSTREAMER_HTTP_HOST_PORT`, default `31000`) is **not** behind
the ingress, so reaching its UI / RTSP directly needs its own allow:

```bash
sudo ufw allow 31000/tcp
sudo ufw reload
```

(Reachability still depends on `EXTERNAL_IP` — see [Network addressing](#network-addressing--host_ip--external_ip).)

## GPU Module Loading

If `nvidia-smi` fails with "NVIDIA-SMI has failed" but the driver is installed, load the kernel modules:

```bash
sudo modprobe nvidia && sudo modprobe nvidia_uvm
```

This works without a reboot on Brev and Colossus instances.

## Confinement vs. host blocker
<a id="confinement"></a>

An agent sandbox produces failures indistinguishable from host faults. **Re-probe
unconfined before reporting any of them, and report the unconfined result.** A
false blocker sends the user to repair a host that was never broken, and it
discredits the real blockers reported alongside it.

| Symptom | Confinement cause | Real-host test |
|---|---|---|
| `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`, with no `/dev/nvidia*` | the device nodes are not exposed to the sandbox, while `lsmod` still lists `nvidia` and the GPUs are on the PCI bus | re-run `nvidia-smi` unconfined before touching the driver table in check 1 |
| `Permission denied` on a root-owned path that `ls -ld` reports as `nobody:nogroup` | a user namespace leaves host `root` unmapped | re-read it unconfined, and compare the ownership the two probes print |
| `Permission denied` as uid 0 on a path whose mode already grants its owner access | the DAC-override capabilities are gone — `grep CapBnd /proc/self/status` has bits 1 and 2 clear (`…f9`). Cite the bounding set, not `CapEff`: a zero `CapEff` alone is regainable by a setuid exec, while a cleared bound is a ceiling `sudo` cannot lift | none from this process. Read it through a privileged peer the user approves, or ask for the value |

That last row is the one to state precisely, because it is the one that invites a
wrong conclusion. The path is readable **on the host** and unreadable **from
here**, so the finding is *"I need this value from you"* — never *"this file is
unreadable"*, and never an inference about what a notebook, script, or service
running outside the agent will manage to read. `/etc/brev/environment-context.json`
is the usual instance of it, and it has a working path rather than a dead end:
[`brev.md`](brev.md#confined-read) reads it through `dockerd`'s capabilities under
an approval the user gives. Take that before concluding the deployment cannot
resolve a secure link.

## Checks

Run in order, report pass/fail for each.

### 1. GPU Detection

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
```

If `nvidia-smi` fails, rule out [confinement](#confinement) first — inside a sandbox it fails on a perfectly healthy host. Once the unconfined probe fails too, the driver is not installed or not loaded. Each platform has its own minimum — a floor, not an exact pin, so a newer build passes:

| Platform | Minimum driver | Where it comes from |
|---|---|---|
| x86 dGPU — RTX PRO 4500, RTX PRO 6000 SE | **`595.58.03`** | https://www.nvidia.com/en-us/drivers/ |
| x86 dGPU — H100 (SXM / PCIe), H200 | **`595.58.03`** | same x86_64 baseline; the SXM parts also need Fabric Manager below |
| x86 Brev Cloud — NemoClaw, RTX 6000 launchables | **`595.91.07`** | the instance image |
| Kubernetes — RTX 6000, H200 | **`595.58.03`** | the GPU Operator's driver setting |
| ARM SBSA — DGX Station GB300 | **`595.58.03`** | the DGX OS 7.6.0 driver; a newer bundled driver passes |
| ARM SBSA — GH200 / GB200 | **`580.159.03`** | bundled in the container, not installed on the host |
| DGX-SPARK | **`580.173.02`** | the bundled DGX OS driver |
| Jetson AGX Orin / AGX Thor / Orin NX | **`595.78`** | AGX Thor: JetPack 7.2 / L4T r39.2; Orin and Orin NX: public JetPack 7.2.2 release |
| IGX Thor T7000 / T5000 | **`580.00`** | IGX-SW / IGX OS 2.0 |

After install, load the kernel modules instead of rebooting:

```bash
sudo modprobe nvidia && sudo modprobe nvidia_uvm
```

> **Multi-GPU H100 SXM / H200 SXM only — NVIDIA Fabric Manager matching the installed driver** is also required to host a local LLM. Unlike the driver table above, this is an exact match and not a floor: install `cuda-drivers-fabricmanager-<branch>` at the running driver's version (`cuda-drivers-fabricmanager-595` against `595.58.03`), because the service refuses to initialize the fabric against a driver it does not match. Single-GPU systems and the PCIe form factors of H100/H200 do **not** need Fabric Manager — installing it will conflict with the standard driver package.

> **Workaround:** If GPU is present but detection fails during a deploy, prepend `SKIP_HARDWARE_CHECK=true` — but investigate root cause.

### 2. Docker

```bash
docker --version        # need 28.3.3+ and earlier than 29.5.0
docker compose version  # need v2.39.1+
docker ps               # verify runs without sudo
```

If Docker needs to be installed: https://docs.docker.com/engine/install/ubuntu/

> **Docker upper bound — `< 29.5.0`.** Docker Engine `29.5.0` and later fail to pull some NGC-hosted image tags after the layers download with `error from registry: Incorrect Repository Format`. Pin a supported version with the script below. If the host is locked above the bound and cannot be downgraded, disable the containerd snapshotter daemon-side — see [Docker 29.5.0+ workaround](#docker-2950-workaround) below.
>
> **DGX Station GB300 ships above the bound.** DGX OS 7.6.0 carries Docker `29.6.1`, so the pin script's downgrade branch is what runs there — try it rather than assuming it fails. Fall back to the override only on a *version not found*, meaning the DGX-OS arm64 repo does not carry the epoch-versioned packages. Then apply the [cgroup driver](#cgroup-driver) separately: the override settles the snapshotter alone, and a host that skipped the script never got `cgroupfs`.

#### Pin the tested Docker versions
<a id="docker-pin"></a>

```bash
bash "$REPO/deploy/docker/scripts/pin_docker_version.sh"
```

The script owns the pinned versions, the tested range, the `apt-mark hold`
that keeps unattended-upgrades from undoing them, and the
[cgroupfs driver](#cgroup-driver) the deploy requires. It skips the downgrade
when the installed engine is already in range, which is what makes it safe on
DGX Spark / DGX-OS arm64 — that apt repo may not carry the exact epoch-versioned
packages, and re-pinning there fails with *version not found*. Run it on every
host rather than only one that failed the check above; it is idempotent.

**Run it here, at Step 3, and not later.** A downgrade or a cgroup-driver
change restarts `dockerd`, which costs nothing before Step 9 and costs every
container in the deployed build after it. The Step 9 image pulls are also what
trigger the `Incorrect Repository Format` failure, so a pin that lands after
them prevents nothing. `deploy_nemoclaw.ipynb` section 2.1 runs the same script
at Step 10, so a host pinned here only re-applies the holds and finds the driver
already set when the harness comes up.

Needs `sudo` and `apt` — Ubuntu/Debian only. It sudoes internally, so an agent
that may not run `sudo` should **offer to run this script under an approval**
rather than hand it over; it is checked in, which is what makes that run
legitimate ([Sudo Access](#sudo-access)). Only a declined or unavailable prompt
makes it a handoff.

**A refused `sudo` probe is not a refusal of this script.** The refusal matches
a `sudo` token on the submitted command line, and the line above carries none —
the calls are inside the script — so its "only a user can run it" wording
governs the probe rather than this step. Submit the line above for approval
first, and reach for the handoff only once that approval comes back declined or
the harness cannot prompt for one.

If `docker ps` requires sudo → add user to docker group:
```bash
sudo usermod -aG docker $USER && newgrp docker
```

#### Cgroup driver — `cgroupfs`
<a id="cgroup-driver"></a>

Required: under the `systemd` driver, long-running containers stop responding
after hours. Read the live driver rather than the file — a `daemon.json` edit
that never reached a `dockerd` restart still greps clean:

```bash
docker info --format '{{.CgroupDriver}}'   # must print cgroupfs
```

On anything else, **run the pin script above** — it merges
`exec-opts: ["native.cgroupdriver=cgroupfs"]` into `daemon.json`, keeping the
file's other keys (installing `jq` first if the host lacks it), and restarts
`dockerd`. It is the same approved run as the
version pin, so try it before writing a `daemon.json` edit out to the user, and
re-run the probe above rather than the whole preflight afterwards.

Hand over this block only once that run is declined or unavailable, per
[Handoff form](#handoff). It replaces every `exec-opts` entry rather than
merging into it, so prefer the script:

```bash
command -v jq >/dev/null || { sudo apt-get update && sudo apt-get install -y jq; }
test -f /etc/docker/daemon.json || echo '{}' | sudo tee /etc/docker/daemon.json >/dev/null
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
# Stage the merge: jq's exit status is the gate, so a rejected file leaves the
# live config as it was instead of truncating it to empty.
sudo jq '.["exec-opts"] = ["native.cgroupdriver=cgroupfs"]' \
  /etc/docker/daemon.json.bak > /tmp/daemon.json.new \
  && sudo install -m 0644 /tmp/daemon.json.new /etc/docker/daemon.json \
  && sudo systemctl restart docker
```

#### Docker 29.5.0+ workaround

If the host is locked to Docker `29.5.0` or later (e.g. distro-managed), add or merge the following daemon-side override and restart Docker to fall back to the legacy graphdriver image store.

> ⚠ **The snippet below overwrites `/etc/docker/daemon.json` in full.** If the host already has other keys there (`registry-mirrors`, `log-driver`, `dns`, `insecure-registries`, etc.), back up first and merge them manually — otherwise they'll be silently dropped.

**Inspect first, then back up:**

```bash
# Inspect any existing config (root-owned 0600 on many hosts, so read it as root:
# an unprivileged cat reports "no existing daemon.json" over a file that has keys)
sudo test -f /etc/docker/daemon.json && sudo cat /etc/docker/daemon.json || echo "no existing daemon.json"

# Backup (safe no-op if the file doesn't exist)
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak 2>/dev/null || true
```

**If `daemon.json` was empty or only contained the `exec-opts` cgroup line**, the `cat >` snippet below is safe verbatim:

```bash
sudo bash -c 'cat > /etc/docker/daemon.json << EOF
{
  "exec-opts": ["native.cgroupdriver=cgroupfs"],
  "features": {
    "containerd-snapshotter": false
  }
}
EOF'
sudo systemctl daemon-reload && sudo systemctl restart docker
```

**If `daemon.json` had other keys**, merge `features.containerd-snapshotter: false` into the existing file (jq is the easiest):

```bash
sudo jq '.features."containerd-snapshotter" = false' \
  /etc/docker/daemon.json.bak | sudo tee /etc/docker/daemon.json >/dev/null
sudo systemctl daemon-reload && sudo systemctl restart docker
```

The `exec-opts` cgroup driver line must remain present either way — it's required by the deploy regardless of the snapshotter override.

### 3. NVIDIA Container Toolkit

Required minimum: **`1.17.8+`**.

```bash
# Check installed version
dpkg -s nvidia-container-toolkit 2>/dev/null | grep -E '^Version:' || \
  rpm -q nvidia-container-toolkit 2>/dev/null

# Check runtime is registered
docker info 2>/dev/null | grep -i "runtimes"

# Check it works end-to-end
docker run --rm --gpus all ubuntu:22.04 nvidia-smi 2>&1 | head -8
```

Should print GPU info from inside the container. If `runtimes` line doesn't show `nvidia`, or the run fails with `unknown or invalid runtime name: nvidia`:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure Docker and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Re-run the `docker run` check to confirm before continuing.

> Full guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

### 4. NGC CLI + Access

Required minimum: **`4.10.0+`**. Follow [`ngc.md`](ngc.md) to check NGC CLI
and API-key access.

**A NemoClaw build needs no host install.** Both harness images —
[`.openclaw/Dockerfile`](../../../.openclaw/Dockerfile) and
[`.hermes/Dockerfile`](../../../.hermes/Dockerfile) — ship the CLI
unconditionally, and the sandbox is where the operation skills run. Verify it
there if anything looks off (`ngc --version` in the sandbox), not here.

**On every other harness, install it on this host**, since the `vss` CLI and the
operation skills' sample-data and fixture bootstraps run here:

```bash
command -v ngc >/dev/null 2>&1 && ngc --version \
  || echo "NGC CLI missing — install it per ngc.md before continuing"
```

Attempt [`ngc.md`](ngc.md)'s install when it is missing. That install needs
`sudo`, so when this host's branch is `SUDO_NOPASSWD=0` (see
[Sudo Access](#sudo-access)) or the agent's own permissions forbid `sudo`, do
not retry it and do not improvise an install path: surface `ngc.md`'s block
verbatim with the handoff wording from that section, and resume once the user
confirms it succeeded.

---

## Canonical version matrix

Single source of truth for **every** dependency the deploy assumes. Sourced from the [VSS prerequisites page](https://docs.nvidia.com/vss/3.4.0/prerequisites.html); update this table when the upstream blueprint docs change.

| Component | Required version | Notes |
|---|---|---|
| OS — x86 host | Ubuntu 24.04 recommended | Ubuntu 22.04 does **not** block deployment. Warn the user that it is not the recommended OS and unforeseen issues may occur, then continue |
| OS — DGX Station GB300 | DGX OS 7.6.0 | |
| OS — DGX-SPARK | DGX OS 7.5.0 | |
| OS — IGX-THOR | IGX-SW 2.0 Production (LTS), including IGX OS 2.0 (Ubuntu 24.04) | |
| OS — AGX-THOR | JetPack 7.2 (Jetson Linux/L4T r39.2) | |
| OS — AGX Orin / Orin NX | public JetPack 7.2.2 release | |
| NVIDIA Driver | per platform — [GPU Detection](#1-gpu-detection) owns the table | a minimum, not an exact pin. `595.58.03` on x86 dGPU and Kubernetes; `595.91.07` on Brev Cloud; `595.58.03` on DGX Station GB300; `595.78` on Jetson; `580.173.02` on DGX-SPARK; `580.00` on IGX Thor |
| NVIDIA Fabric Manager | exact match to the installed driver | **only** for multi-GPU NVLink/NVSwitch hosts running local LLM (H100 SXM, H200 SXM, NVSwitch, HGX). Not a floor like the driver row — the service will not initialize the fabric on a mismatch, so it moves with whatever [GPU Detection](#1-gpu-detection) selected |
| NVIDIA Container Toolkit | `1.17.8+` | |
| Docker | `28.3.3+` **and** `< 29.5.0` | pin with [`pin_docker_version.sh`](#docker-pin), which owns the exact versions. Upper bound: `29.5.0`+ breaks NGC image pulls — on a host that cannot be downgraded, see [Docker 29.5.0+ workaround](#docker-2950-workaround) |
| Docker Compose | `v2.39.1+` | |
| Kubernetes — where it is supported at all | GPU Operator `v26.3.3`, K8s `v1.29`–`v1.36` | x86_64 and ARM SBSA only. Every Tegra platform (IGX Thor T7000/T5000, Jetson AGX Orin, AGX Thor, Orin NX), DGX Spark, GH200/GB200 and DGX Station GB300 are **Docker Compose only** — Helm paths are not validated there |
| NGC CLI | `4.10.0+` | follow `ngc.md` |

---

## Summary

- All pass → "System ready. You can deploy base, lvs, search, or alerts."
- Any fail → report the item, provide the fix, re-run that check before continuing. Ubuntu 22.04 is a warning, not a failure.
