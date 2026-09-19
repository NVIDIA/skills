---
name: tao-run-on-kubernetes
description: Kubernetes execution platform — submits TAO container jobs as k8s Jobs with NVIDIA GPU scheduling; single-pod
  for one node, Indexed Jobs for multi-node distributed training. Use when running on EKS / GKE / AKS / on-prem clusters
  with the NVIDIA GPU Operator installed, or when integrating TAO into an existing k8s-native ML platform.
license: Apache-2.0
compatibility: Requires GPU worker nodes with NVIDIA driver branch 580, CUDA Toolkit 13.0, and NVIDIA Container Toolkit 1.19.0; a `kubectl` client authenticated to the cluster; and the NVIDIA GPU Operator or device plugin. No nvidia-tao-sdk required — jobs are submitted with plain `kubectl`.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- kubernetes
- k8s
- gpu
- compute
- container
---

# Kubernetes

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Submits TAO container jobs as Kubernetes Jobs. Works on any cluster reachable via kubeconfig (EKS / GKE / AKS / on-prem) or in-cluster service account (when running inside a pod).

Single-pod by default; opt into multi-node distributed training via `num_nodes > 1` (uses Indexed Job + headless Service, see [Multi-node training](#multi-node-training-distributed) below).

## Preflight

Three checks: GPU host runtime ready, cluster reachable via `kubectl`, GPU
Operator/device plugin present.

```bash
# 0. GPU node host runtime.
# Run this on each self-managed GPU worker node or in the node image build.
# Set TAO_K8S_SKIP_NODE_RUNTIME_CHECK=1 only when using managed GPU nodes whose
# driver/toolkit lifecycle is owned by the cloud provider or GPU Operator policy.
if [ "${TAO_K8S_SKIP_NODE_RUNTIME_CHECK:-0}" != "1" ]; then
  TAO_SKILL_BANK_ROOT="${TAO_SKILL_BANK_ROOT:-$PWD}"
  SETUP_SCRIPT="${TAO_SKILL_BANK_ROOT}/skills/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh"

  bash "$SETUP_SCRIPT" --backend kubernetes --check-only || {
    echo "MISSING: TAO Kubernetes GPU node runtime is not ready."
    echo "For self-managed GPU nodes, run after user approval:"
    echo "  bash \"$SETUP_SCRIPT\" --backend kubernetes --install --yes"
    echo "For managed clusters, verify the node image/GPU Operator policy installs driver 580 and toolkit 1.19.0, then set TAO_K8S_SKIP_NODE_RUNTIME_CHECK=1."
    exit 1
  }
fi

# 1. Cluster reachable (kubeconfig OR in-cluster service account)
command -v kubectl >/dev/null 2>&1 || {
  echo "MISSING: kubectl not found on PATH. Install kubectl to submit Jobs."
  exit 1
}
kubectl cluster-info >/dev/null 2>&1 || {
  echo "MISSING: no reachable cluster (kubeconfig at ~/.kube/config, \$KUBECONFIG, or in-pod service account)."
  echo "Configure kubectl for your cluster, or set \$KUBECONFIG:"
  echo "  EKS: aws eks update-kubeconfig --name <cluster> --region <region>"
  echo "  GKE: gcloud container clusters get-credentials <cluster> --region <region>"
  echo "  AKS: az aks get-credentials --resource-group <rg> --name <cluster>"
  echo "  local: minikube start   (see 'Local cluster' below)"
  exit 1
}

# 2. NVIDIA GPU Operator present (soft check — warn, don't fail)
gpu=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | grep -v '^$' | head -1)
if [ -z "$gpu" ] || [ "$gpu" = "0" ]; then
  echo "WARN: no nvidia.com/gpu allocatable on this cluster."
  echo "Install the NVIDIA GPU Operator before submitting GPU jobs:"
  echo "  https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html"
fi
```

The GPU node runtime check is mandatory for self-managed nodes. For managed
clusters where the client is not running on a GPU worker, verify the provider
node image or GPU Operator policy and set `TAO_K8S_SKIP_NODE_RUNTIME_CHECK=1`
instead of running the installer on the client. The GPU-capacity warning here is
a soft check; the `submit` verb re-checks allocatable `nvidia.com/gpu` and
hard-fails before applying the manifest (there is no gang scheduling, so a
too-big Job would sit `Pending` forever).

## Credentials & configuration

- **Kubeconfig** (one of):
  - `~/.kube/config` — default discovery path
  - `$KUBECONFIG` — alternate path
  - In-cluster service account — used when running inside a pod (no kubeconfig needed)
- **TAO_K8S_NAMESPACE** (optional): default namespace for Job submission. Defaults to `default`.
- **TAO_K8S_CONTEXT** (optional): kubeconfig context name to switch clusters.
- **NGC_KEY** (optional): for nvcr.io image pulls. If you've pre-created an image-pull secret in the target namespace, reference its name in the rendered manifest's `imagePullSecrets`.
- **AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / S3_BUCKET_NAME / S3_ENDPOINT_URL** (optional): for S3 dataset I/O (storage tier C), injected into the pod via the per-job Secret (`envFrom.secretRef`), never inline. Legacy `ACCESS_KEY`/`SECRET_KEY` are mapped by `tao-data-io`.

Do not ask for Brev or SLURM credentials for Kubernetes runs. Ask for
S3 credentials only when the selected workflow uses `s3://` inputs or outputs,
and ask for model-specific credentials such as `HF_TOKEN` only when the selected
model requires them. Before launch, verify the selected namespace can create
Jobs, dataset/result paths are visible from the pod, and PVC/mounted filesystem
paths are proven to be mounted into the job container; an agent-host local path
is not sufficient proof.

## Execution — the four verbs

`tao-run-on-kubernetes` is a platform **consumer**: it runs a spec-bundle via
`kubectl`, mutating only the job-record. No nvidia-tao-sdk, no `tao_sdk` import —
jobs are submitted with plain `kubectl apply`.
`$BANK` = `${TAO_SKILL_BANK_PATH}`.

### submit

1. **GPU-capacity gate — hard-fail first** (no gang scheduling → a too-big Job
   sits `Pending` forever):
   ```bash
   ALLOC=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' | awk '{s+=$1} END{print s+0}')
   [ "${ALLOC:-0}" -ge "$NUM_GPUS" ] || { echo "insufficient GPUs: need $NUM_GPUS, allocatable $ALLOC"; exit 1; }
   ```
2. **Storage tier** (via `tao-data-io`): **A** = mount a bound PVC/NFS holding the
   data (author the mount paths, no fetch — the air-gap answer, and what the
   packaged template does); **C** = ephemeral: an initContainer fetches from S3
   into a shared `emptyDir` and a final step uploads results to S3 before TTL.

   **Tier C holds the GPU while it downloads.** A pod reserves `nvidia.com/gpu`
   for its whole lifetime, initContainers included, so a large tier-C fetch —
   or a first-time multi-GB image pull — is billed and reaper-eligible idle GPU
   time, exactly like pulling inside a SLURM allocation. Prefer tier A when the
   data is already on a PVC; choose tier C knowingly, for small inputs.

   A producer action request may declare several mounts, including duplicate-
   source aliases for logical and embedded absolute paths. Read
   `references/action-request.md` and use its staging map plus packaged
   renderer. For `mode=config`, materialize the producer's nested spec first,
   stage that exact generated file, and pass it back to the renderer so the
   pod receives a verified read-only config mount. The legacy single-root
   template cannot represent that contract.
3. **Open the record — mints the id, binds `results_dir`, before launch:**
   ```bash
   JOB_ID=$("$BANK/scripts/tao_job_record.py" open --platform kubernetes --image "$IMAGE" \
     --network-arch "$ARCH" --action "$ACTION" --storage-tier "$TIER" --results-dir "$RESULTS_DIR")
   ```
   `results_dir` must be a **mounted (surviving) volume path or an S3 prefix** —
   `ttlSecondsAfterFinished` deletes the Job and its logs after it ends, so
   nothing is recoverable from the Job object later.
4. **Render, gate, apply, and record RUNNING.** For a producer action request,
   follow `references/action-request.md`; it owns backend-name normalization,
   conditional Secret references, native argv rendering, server dry-run, and
   binding the applied object name to the job-record. For a simple one-root
   spec-bundle, render `templates/k8s/single-pod-job.yaml.tmpl`, run
   `redact_secrets.py lint` plus `kubectl apply --dry-run=server`, apply it, and
   mark the record with `backend-ref=<namespace>/<actual-object-name>`.

A submit that skipped the gate or the open has no id — so it cannot launch.

### status

Keep `K8S_JOB_NAME` from submit. On reattach, read the job-record's
`backend_ref=<namespace>/<name>` and recover both values from that field; do
not assume the Kubernetes name equals the record id.

```bash
kubectl get job "$K8S_JOB_NAME" -n "$NAMESPACE" \
  -o jsonpath='{.status.conditions[0].type} {.status.active} {.status.succeeded} {.status.failed}'
```

| kubectl signal | vocab |
|---|---|
| no pods scheduled | `PENDING` (`kubectl get pods -n "$NAMESPACE" -l job-name="$K8S_JOB_NAME"` → `ImagePullBackOff` / `Insufficient nvidia.com/gpu` in `message`) |
| `active` ≥ 1 | `RUNNING` |
| condition `Complete` | `COMPLETE` |
| condition `Failed` | `ERROR` (classify from the pod's terminated reason — `OOMKilled` → `ERR_INFRA`) |
| Job/pod not found | `UNKNOWN` (may be TTL-deleted — the job-record is the source of truth) |

### logs

```bash
kubectl logs -n "$NAMESPACE" -l "job-name=$K8S_JOB_NAME" --tail "${N:-200}"
```

### cancel

```bash
kubectl delete job "$K8S_JOB_NAME" -n "$NAMESPACE" --cascade=foreground
if [ -n "${CRED_SECRET:-}" ]; then
  kubectl delete secret "$CRED_SECRET" -n "$NAMESPACE" --ignore-not-found
fi
"$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state CANCELED --source agent
```

### Multi-node (nodes > 1)

Same four verbs, plus:

1. **Version gate:** require k8s ≥ 1.28 (`kubectl version -o json`) — the pod
   hostname `<job>-<index>` (PodIndexLabel) that `MASTER_ADDR=<job>-0.<svc>`
   resolves to needs it; on older clusters rank-0 hangs at rendezvous.
2. **Capacity gate ×nodes:** hard-fail unless allocatable GPUs ≥ `gpus_per_node ×
   nodes` (no gang scheduling → a partial start leaves rank-0 waiting forever).
3. **Render `templates/k8s/indexed-job.yaml.tmpl`** — the headless Service +
   Indexed Job + rendezvous env (`WORLD_SIZE` = node count, `NODE_RANK` from
   `JOB_COMPLETION_INDEX`, `MASTER_ADDR=<job>-0.<svc>`, `/dev/shm` 16Gi so NCCL
   doesn't silently hang). `kubectl apply -f` creates the Service and Job together;
   `cancel` deletes the Job (Foreground) and the Service.
4. **NCCL probe first** (as SLURM) — a 2-node all-reduce with a timeout; on hang,
   set the cluster NCCL env and re-probe; cache per cluster.

## Local cluster (development, CI, and evals)

A throwaway minikube/kind cluster exercises admission, the four verbs,
job-record wiring, and log plumbing without cluster quota — and is what an
agent-driven eval should provision for itself. `kubectl` and `minikube` are
single static binaries needing no root, so a non-root CI container can install
them itself.

Two prerequisites keep a rendered Job `Pending`, and the first masks the second:
the PVC the template mounts must exist (`persistentvolumeclaim "<name>" not
found` fires before any GPU complaint), then a Job requesting `nvidia.com/gpu`
on a GPU-less cluster reports `Insufficient nvidia.com/gpu` and waits forever.
Render `NUM_GPUS=0` for a lifecycle-only run and say GPU scheduling was not
verified; on a Linux GPU host, `minikube start --driver=docker --gpus all`
passes real GPUs through, so one GPU box suffices for a GPU-real smoke.

Install commands, driver choice, the container/host-networking caveat, and the
fake-device-plugin middle option: `references/local-cluster.md`.

## Container shell

The simple single-pod template invokes its command via `/bin/sh -c` (POSIX sh,
present in busybox/distroless as well as TAO images). For producer action
requests, an args-mode command and its arguments are native container argv; a
producer that needs a shell declares the shell and its script explicitly. A
simple config-mode command also becomes native argv after `{config_path}` is
substituted. A producer-owned config command that is itself a multi-line shell
script is preserved verbatim under `/bin/sh -c`; the renderer never constructs
shell text from config values.

## GPU Operator dependency

The `submit` verb refuses to launch GPU jobs on a cluster with no `nvidia.com/gpu` allocatable. For self-managed clusters, first run the `tao-setup-nvidia-gpu-host` install action on every GPU worker node or bake the same package set into the node image:

```bash
bash skills/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh --backend kubernetes --install --yes
```

Then install the NVIDIA GPU Operator or device plugin:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install --wait gpu-operator -n gpu-operator --create-namespace nvidia/gpu-operator
```

Full guide: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html

## Multi-node training (distributed)

Set `num_nodes > 1` (see the [Multi-node (nodes > 1)](#multi-node-nodes--1) verb
steps above) to run distributed training across N pods. Rendering
`templates/k8s/indexed-job.yaml.tmpl` provisions:

1. A **headless Service** named after the Job (selector: `job-name=<job-name>`, `clusterIP: None`, `publishNotReadyAddresses: true` so pods can rendezvous before they're all Ready).
2. An **Indexed Job** with `parallelism = completions = num_nodes`, `completionMode: Indexed`. Each pod gets `JOB_COMPLETION_INDEX` injected by k8s automatically (= the node rank).
3. A **command wrapper** that exports the rendezvous env vars before invoking the user command. Two naming conventions are exported simultaneously:

   | Env var | Value | Read by |
   |---|---|---|
   | `WORLD_SIZE` | `num_nodes` | TAO PyTorch container's `nvidia_tao_pytorch/core/entrypoint.py` (uses this to mean *node count*, even though PyTorch's own convention is *total processes*) |
   | `NUM_GPU_PER_NODE` | `gpu_count` | TAO PyTorch container's entrypoint |
   | `NNODES` | `num_nodes` | `torchrun` and PyTorch-standard rendezvous |
   | `NPROC_PER_NODE` | `gpu_count` | `torchrun` |
   | `NODE_RANK` | `$JOB_COMPLETION_INDEX` | both |
   | `MASTER_ADDR` | `<job-name>-0.<job-name>` (pod-0's DNS) | both |
   | `MASTER_PORT` | `29500` | both (TAO's default) |

   Both naming conventions are set so TAO entrypoints (`dino train`, etc.) and raw `torchrun` commands work without modification.

For a TAO entrypoint, the container reads `spec.train.num_nodes` and the wired
env vars — e.g. `dino train -e /tmp/spec.yaml` with `gpu_count=8`, `num_nodes=4`
(4 × 8 = 32 GPUs total).

For raw `torchrun`-based commands (non-TAO containers), the wrapper invokes:

```bash
torchrun --nnodes=$NNODES --nproc-per-node=$NPROC_PER_NODE --node-rank=$NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT train.py
```

The capacity check sums across nodes: `gpu_count × num_nodes` ≤ cluster's allocatable `nvidia.com/gpu`.

### Cluster requirements for multi-node

- **k8s 1.28+** is required for stable pod hostnames in Indexed Jobs (the `PodIndexLabel` feature). On older clusters the `MASTER_ADDR=<job>-0.<svc>` DNS lookup fails. Verify with `kubectl version`.
- **Pod-to-pod networking** must be open on port 29500 (PyTorch default; configurable via `MASTER_PORT` env var). Most CNIs (Calico, Cilium, AWS VPC CNI) allow this by default; restrictive NetworkPolicies must be relaxed.
- **NCCL** in the container talks GPU-to-GPU; if the cluster has multi-NIC nodes or RDMA, set `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` in the container `env` of the rendered manifest.

### Reference reading

- Kubernetes Indexed Job: <https://kubernetes.io/docs/concepts/workloads/controllers/job/#completion-mode>
- Indexed Job for batch ML: <https://kubernetes.io/blog/2022/06/01/indexed-jobs-mpi/>
- PyTorch distributed (env-var rendezvous): <https://pytorch.org/docs/stable/elastic/run.html>
- NCCL networking tuning (NCCL_SOCKET_IFNAME, NCCL_IB_HCA): <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>

### Kubernetes operator alternatives

For more sophisticated topologies (gang scheduling, PyTorch elastic / fault-tolerant training, MPI / Horovod, RDMA setup), reach for an operator instead of plain Indexed Job:

- **MPI Operator** — <https://github.com/kubeflow/mpi-operator> — for MPI / Horovod workloads.
- **Kubeflow Training Operator** (`PyTorchJob`, `TFJob`) — <https://www.kubeflow.org/docs/components/training/> — for elastic PyTorch training with built-in restart logic.
- **Volcano** — <https://volcano.sh/> — gang scheduling, queues, fair-share. Useful in shared multi-tenant clusters.
- **Kueue** — <https://kueue.sigs.k8s.io/> — quota / queue layer on top of any of the above.

This skill's Indexed Job path is intentionally simple and dependency-free; if you need elastic restart or gang scheduling, layer one of these on top and submit jobs through the operator's CRD instead.

## Common error patterns

**`No nvidia.com/gpu resources allocatable on the cluster`** — the GPU Operator (or NVIDIA Device Plugin) isn't installed. Install per the link above; verify with `kubectl get nodes -o jsonpath='{.items[*].status.allocatable}'`.

**`ImagePullBackOff` / `ErrImagePull`** — the cluster can't pull the image. For nvcr.io: pre-create an image-pull secret in the namespace and reference it as the pod's `imagePullSecrets` in the rendered manifest:
Feed the key over stdin — `--docker-password=$NGC_KEY` would put the secret in
argv, where it is visible in the host's process table and shell history:
```bash
set -a; source /path/to/.env; set +a   # omit if already exported
kubectl create secret generic ngc-pull-secret -n tao-jobs \
  --type=kubernetes.io/dockerconfigjson \
  --from-file=.dockerconfigjson=/dev/stdin <<EOF
{"auths": {"nvcr.io": {"username": "\$oauthtoken", "password": "${NGC_KEY}"}}}
EOF
# Verify without reading the secret back:
kubectl get secret ngc-pull-secret -n tao-jobs >/dev/null && echo SECRET_OK
```

**Pod stays `Pending` forever** — `kubectl describe pod -l job-name=$JOB_ID` shows the scheduling reason in the `Events`. Common causes: insufficient GPU capacity (`Insufficient nvidia.com/gpu`), no node matches the pod's `nodeSelector`, missing image-pull secret, or PVC mount failure.

**`OOMKilled` (exit 137)** — container exceeded memory. Reduce batch size, lower max_length, or add a memory request/limit and target a larger node.

**`CredentialError: Could not authenticate to a Kubernetes cluster`** — neither kubeconfig nor in-cluster auth worked. Run `kubectl get nodes` to verify your config, or set `$KUBECONFIG` to the right path.

## What this skill does NOT support (yet)

- **Elastic / fault-tolerant training.** Indexed Job has `backoff_limit=0` — failures fail the whole training run. For elastic restart (e.g., resume from checkpoint after a node death), use Kubeflow's `PyTorchJob` operator instead.
- **Gang scheduling.** Indexed Job pods are scheduled independently — no all-or-nothing. Multi-node training will *partially* start if only some pods can be scheduled (rank-0 will hang waiting for peers). For all-or-nothing scheduling on shared clusters, use Volcano or Kueue.
- **MPI / Horovod.** Use the MPI Operator. The Indexed Job path here is PyTorch-distributed-shaped (env-var rendezvous on `MASTER_ADDR:MASTER_PORT`).
- **Auto-creating image-pull secrets from `$NGC_KEY`.** You pre-create the secret in the target namespace and pass the name. K8s namespace conventions vary widely, so we keep secret creation explicit.
