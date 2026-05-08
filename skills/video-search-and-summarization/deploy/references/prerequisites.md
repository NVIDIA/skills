---
name: vss-prerequisites
description: Check VSS system prerequisites — GPU driver, Docker, NVIDIA Container Toolkit, and NGC access. Use when troubleshooting a deploy failure, after a system change, or to verify the system is ready for VSS.
---

# VSS Prerequisites Check

Verifies system readiness for any VSS developer profile. For NGC CLI setup specifically, use the `ngc` skill.

## When to Use

Use this skill when:

- A VSS deploy failed and you need to diagnose why
- User asks to verify GPU, Docker, or system setup
- After a driver or Docker update
- Called from BOOTSTRAP during first-time setup

## Read TOOLS.md First

Check `TOOLS.md` for the VSS section. If missing, the environment isn't configured yet — run BOOTSTRAP first.

---

## Sudo Access

Most prerequisite steps require `sudo` (Docker install, NVIDIA toolkit, kernel settings, systemctl). On cloud instances (Brev, Colossus, DGX Cloud) the default user typically has passwordless sudo. On bare-metal machines, the user may need to enter a password or be in the `sudo` group.

Check before proceeding:

```bash
sudo -n true 2>/dev/null && echo "passwordless sudo" || echo "sudo requires password"
```

If sudo requires a password, ask the user to run privileged commands manually or configure passwordless sudo for the session.

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

## GPU Module Loading

If `nvidia-smi` fails with "NVIDIA-SMI has failed" but the driver is installed, load the kernel modules:

```bash
sudo modprobe nvidia && sudo modprobe nvidia_uvm
```

This works without a reboot on Brev and Colossus instances.

## Checks

Run in order, report pass/fail for each.

### 1. GPU Detection

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
```

Expected for this machine: 2× RTX PRO 6000 Blackwell, devices 0 and 1.

If `nvidia-smi` fails → driver not installed or not loaded. Guide the user:

- Ubuntu 24.04: install driver `580.105.08` from https://www.nvidia.com/en-us/drivers/
- Ubuntu 22.04: install driver `580.65.06`
- After install: load the kernel modules instead of rebooting:
  ```bash
  sudo modprobe nvidia && sudo modprobe nvidia_uvm
  ```

> **Workaround:** If GPU is present but detection fails during a deploy, prepend `SKIP_HARDWARE_CHECK=true` — but investigate root cause.

### 2. Docker

```bash
docker --version        # need 27.2.0+
docker compose version  # need v2.29.0+
docker ps               # verify runs without sudo
```

If Docker needs to be installed: https://docs.docker.com/engine/install/ubuntu/

If `docker ps` requires sudo → add user to docker group:
```bash
sudo usermod -aG docker $USER && newgrp docker
```

Also verify cgroupfs driver:
```bash
cat /etc/docker/daemon.json | grep cgroupfs
# Should contain: "exec-opts": ["native.cgroupdriver=cgroupfs"]
```

### 3. NVIDIA Container Toolkit

```bash
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

Use the `ngc` skill to check NGC CLI and API key access.

---

## Summary

- All pass → "System ready. You can deploy base, lvs, search, or alerts."
- Any fail → report the item, provide the fix, re-run that check before continuing.
