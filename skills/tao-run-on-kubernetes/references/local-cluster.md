# Local Kubernetes cluster (development, CI, and agent-driven evals)

A throwaway local cluster exercises manifest admission, the four verbs,
job-record wiring, and log plumbing without touching cluster quota. It is what
an agent-driven eval should provision for itself.

## Installing the tools

`kubectl` and `minikube` are single static binaries, so neither needs root — a
package manager is not required, which matters in a CI or eval container that
runs as a non-root user and has no `sudo`. Install to any directory on `$PATH`
that you can write (e.g. `~/.local/bin`):

```bash
mkdir -p "$HOME/.local/bin"; export PATH="$HOME/.local/bin:$PATH"

# kubectl
curl -fsSLo "$HOME/.local/bin/kubectl" \
  "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x "$HOME/.local/bin/kubectl"

# minikube
curl -fsSLo "$HOME/.local/bin/minikube" \
  https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
chmod +x "$HOME/.local/bin/minikube"
```

Swap `linux/amd64` for your platform (`darwin-arm64` on Apple Silicon).

## Starting and tearing down

```bash
# Linux / Intel with a Docker daemon — the usual CI case.
minikube start --driver=docker --cpus=2 --memory=3g

# Apple Silicon, or any host where the Docker daemon is unavailable or
# restricted: vfkit needs no daemon (brew install vfkit).
minikube start --driver=vfkit --cpus=2 --memory=3g

kubectl get nodes          # STATUS Ready before submitting
...
minikube delete            # the cluster is disposable — always tear it down
```

`kind` is an alternative (`kind create cluster` / `kind delete cluster`) but
requires a working Docker daemon; minikube's vfkit/qemu drivers do not.

**Inside a container with the host Docker socket mounted** (the skill-eval
setup), `--driver=docker` makes minikube create its cluster as a *sibling*
container on the host. That works, and `kubectl` can reach the API server
**only because the eval container uses host networking** — minikube binds the
API server on the host, so a bridge-networked container would fail to connect.
Do not switch that container to bridge networking.

## GPUs on a local cluster

A default local cluster advertises **no** `nvidia.com/gpu`, so a Job requesting
one waits forever. Three options, in increasing fidelity:

1. **Lifecycle only** — render with `NUM_GPUS=0`. Validates admission, the four
   verbs, and logs. Cannot say anything about GPU scheduling; say so explicitly
   rather than implying GPU coverage.
2. **Scheduling without hardware** — install a fake device plugin that
   advertises `nvidia.com/gpu` capacity. The Job then schedules and you exercise
   the resource key, limits, and node selection, though nothing CUDA runs.
3. **Real GPUs** — on a Linux host with a GPU and the NVIDIA Container Toolkit,
   minikube passes them through:
   ```bash
   minikube start --driver=docker --container-runtime=docker --gpus all
   kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'
   ```
   This is why a single GPU box is enough for a GPU-real k8s smoke — no managed
   cluster required. Verify the allocatable count is non-zero before concluding
   the passthrough worked.

## Two things that keep a rendered TAO Job `Pending`

Both are prerequisites rather than bugs, and both were reproduced against
minikube. The order matters — the first masks the second:

1. **The PVC must exist.** The templates mount `@@PVC_CLAIM@@`; without it the
   scheduler reports `persistentvolumeclaim "<name>" not found`, and that fires
   *before* any GPU complaint. minikube's default StorageClass binds a plain
   claim immediately:
   ```bash
   kubectl create -f - <<'EOF'
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata: {name: edgeai-datasets}
   spec: {accessModes: [ReadWriteOnce], resources: {requests: {storage: 1Gi}}}
   EOF
   ```
2. **GPU capacity must be advertised.** With the PVC satisfied, a Job requesting
   `nvidia.com/gpu` on a GPU-less cluster reports
   `0/1 nodes are available: 1 Insufficient nvidia.com/gpu` and waits forever.

Size any smoke test or eval against which of the three GPU options above is
actually in play, and report that honestly — a GPU-less pass is not GPU
coverage.
