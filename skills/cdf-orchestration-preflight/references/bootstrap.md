# Set up the Linux environment

Use this reference when prerequisites are missing. Work on the user's selected Linux host and
repository checkout. Reuse existing tools, clusters, GPU management and storage. Inspect first,
present the required changes, and run them within the user's approved scope. If the user is
executing commands, provide them and check the results before continuing. Commands below use Bash.

## Tools and Python environment

Inspect the distribution, architecture, available disk, GPU access and installed tools:

```bash
cat /etc/os-release
uname -m
df -h
nvidia-smi
for tool in git make curl python3 uv docker containerd nvidia-ctk kubectl helm aws envsubst jq; do
  command -v "$tool" || true
done
```

Install missing base tools using the distribution's package manager: Git, Make, curl, CA
certificates, GnuPG, Python 3, unzip, jq and gettext (`gettext-base` on Debian/Ubuntu provides
`envsubst`). Require working `nvidia-smi` before GPU setup; driver installation is outside this
procedure.

Use the linked Linux installation commands for the detected distribution and architecture.
Select compatible versions before installation; keep healthy versions that meet the repo's
requirements. Do not use an x86_64 download on an ARM host.

| Component | Requirement | Installation |
| --- | --- | --- |
| Docker Engine | 29.6.0+ for controller builds | [Docker Engine](https://docs.docker.com/engine/install/) |
| containerd | CRI v1 support, compatible with Kubernetes | [Container runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/) |
| NVIDIA Container Toolkit | Compatible with the installed driver/runtime | [Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| kubeadm, kubelet, kubectl | Matching supported Kubernetes versions | [kubeadm installation](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/) |
| Helm | 3.x | [Helm installation](https://helm.sh/docs/intro/install/) |
| uv | Python/environment management | [uv installation](https://docs.astral.sh/uv/getting-started/installation/) |
| AWS CLI | v2 | [AWS CLI installation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |

For an existing cluster, install the client tools only; do not install kubeadm or replace its
runtime. For a fresh cluster host, follow the vendor package-repository instructions, inspect the
package transaction, then install. Hold the selected Kubernetes packages using the distribution's
version-lock mechanism. Installing the AWS CLI does not provide AWS credentials.

From the repo root, install the Python runtime dependencies used by workflow payload checkers:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv sync --frozen --all-packages
source .venv/bin/activate
python --version
helm version --short
aws --version
```

An empty virtual environment is insufficient for checkers that import Airflow. If frozen sync
fails, resolve the reported dependency/access problem without changing `uv.lock`.

## Container runtime

Check service health and access as the user who will build the controller:

```bash
sudo systemctl is-active docker containerd
docker info --format '{{.ServerVersion}}'
sudo docker ps -a --format 'table {{.Names}}\t{{.Status}}'
id -nG
```

If Docker works with sudo only, inspect the socket permissions and Docker context. Where Docker
group access is approved, run `sudo usermod -aG docker "$(id -un)"`, then reconnect or run
`newgrp docker`. Membership in this group grants root-equivalent access. Do not make the socket
world-writable. Restore PATH and venv activation in the new shell.

Before runtime edits, back up `/etc/containerd/`, including imported configuration files, and
identify running containers that a restart could affect. Inspect the effective configuration
locally; do not print registry credentials into the conversation.

- Enable CRI by removing `cri` from `disabled_plugins` if present.
- Configure the runc runtime's `options.SystemdCgroup = true` to match kubelet's systemd driver.
  The plugin is `io.containerd.grpc.v1.cri` in containerd 1.x and
  `io.containerd.cri.v1.runtime` in containerd 2.x. Preserve unrelated settings and imports.
- Configure the NVIDIA runtime after installing the toolkit. Review the dry run before applying:

```bash
sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default --dry-run
sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default
sudo containerd config dump >/dev/null
sudo systemctl restart containerd
sudo systemctl is-active containerd
sudo crictl --runtime-endpoint unix:///run/containerd/containerd.sock version
sudo docker ps -a --format 'table {{.Names}}\t{{.Status}}'
```

Require `RuntimeApiVersion: v1`; running CRI subplugins alone do not prove a usable CRI API.
Inspect `crictl info` locally to confirm the effective default runtime and systemd cgroup setting.
A missing `/etc/crictl.yaml` warning is harmless when the endpoint is supplied explicitly.

## Kubernetes and networking

For an existing cluster, export its supplied `KUBECONFIG` and verify `kubectl get nodes`. For a
fresh host selected to run its own cluster, use kubeadm below. Do not initialize over an existing
`/etc/kubernetes/admin.conf` or etcd data directory. K3s is not supported by this repository.

Inspect `ip -4 address`, `ip -4 route show table all`, `swapon --show` and `sudo ss -lntp`.
Choose a routable API address and Pod/Service CIDRs that do not overlap each other or host,
Docker, VPN or upstream network ranges. Disable swap and its persistent configuration if active;
verify enough memory is available first. Enable persistent IPv4 forwarding:

```bash
printf 'net.ipv4.ip_forward = 1\n' | sudo tee /etc/sysctl.d/99-cdf-kubernetes.conf >/dev/null
sudo sysctl -p /etc/sysctl.d/99-cdf-kubernetes.conf
```

Set the variables below to the selected node, network and installed Kubernetes version. Keep
these values for CNI configuration; do not substitute another machine's addresses.

```bash
: "${CDF_NODE_NAME:?Set the target node name}"
: "${CDF_API_IP:?Set the routable API address}"
: "${CDF_POD_CIDR:?Set the Pod CIDR}"
: "${CDF_SERVICE_CIDR:?Set the Service CIDR}"
: "${CDF_K8S_VERSION:?Set the installed kubeadm version, including v prefix}"
sudo kubeadm init --kubernetes-version="$CDF_K8S_VERSION" \
  --node-name="$CDF_NODE_NAME" --apiserver-advertise-address="$CDF_API_IP" \
  --cri-socket=unix:///run/containerd/containerd.sock \
  --pod-network-cidr="$CDF_POD_CIDR" --service-cidr="$CDF_SERVICE_CIDR" --skip-token-print
mkdir -p "$HOME/.kube"
# Choose a new destination; preserve any existing kubeconfig.
test ! -e "$HOME/.kube/cdf.yaml"
sudo install -m 600 -o "$(id -u)" -g "$(id -g)" \
  /etc/kubernetes/admin.conf "$HOME/.kube/cdf.yaml"
export KUBECONFIG="$HOME/.kube/cdf.yaml"
```

Install one compatible CNI using its Linux/kubeadm procedure, such as
[Calico](https://docs.tigera.io/calico/latest/getting-started/kubernetes/self-managed-onprem/onpremises).
Set its IP pool to `CDF_POD_CIDR` before applying the manifest. Do not reuse a default CIDR that
conflicts with Docker. For a single-node development cluster only, allow workload scheduling:

```bash
kubectl taint nodes "$CDF_NODE_NAME" node-role.kubernetes.io/control-plane:NoSchedule-
kubectl wait --for=condition=Ready "node/$CDF_NODE_NAME" --timeout=300s
kubectl rollout status deployment/coredns -n kube-system --timeout=180s
kubectl get pods -n kube-system
```

Node `NotReady` and CoreDNS `Pending` before CNI installation are expected. If readiness does not
recover, inspect CNI Pods/events before retrying installation or changing network configuration.

## GPU and shared storage

Reuse existing GPU management. Otherwise install the
[GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html)
with a chart version compatible with the cluster. For the preinstalled driver/toolkit and NVIDIA
default runtime configured above:

```bash
: "${CDF_GPU_OPERATOR_VERSION:?Select a compatible chart version}"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update nvidia
helm install gpu-operator nvidia/gpu-operator --namespace gpu-operator --create-namespace \
  --version "$CDF_GPU_OPERATOR_VERSION" \
  --set driver.enabled=false --set toolkit.enabled=false --set cdi.enabled=false \
  --wait --timeout 10m
kubectl wait --for=jsonpath='{.status.state}'=ready clusterpolicy/cluster-policy --timeout=300s
kubectl get nodes -o 'custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
```

Preserve existing MIG configuration and monitoring; disable Operator components that would
conflict with them. Require a successful GPU test Pod using the Operator's CUDA sample procedure.
GPU advertisement alone does not prove computation. For video workflows, also verify
`libnvcuvid.so.1`; host library presence alone does not prove its availability inside a GPU Pod.

Prefer an existing RWX StorageClass. If absent, read [NFS remediation](nfs-remediation.md) and use
the repository installer only on a compatible self-managed host:

```bash
kubectl get storageclass
kubectl get pvc --all-namespaces
kubectl get node "$CDF_NODE_NAME" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}{"\n"}'
# Set CDF_NFS_SERVER to the confirmed node InternalIP above, not a load balancer/API address.
: "${CDF_NFS_SERVER:?Set the confirmed NFS node InternalIP}"
CDF_NFS_HOSTNAME="$(kubectl get node "$CDF_NODE_NAME" -o jsonpath='{.metadata.labels.kubernetes\.io/hostname}')"
: "${CDF_NFS_HOSTNAME:?Selected NFS node must have a hostname label}"
make install nfs NFS_NODE_HOSTNAME="$CDF_NFS_HOSTNAME" NFS_SERVER="$CDF_NFS_SERVER" \
  NFS_EXPORT_PATH=/srv/nfs/k8s NFS_NAMESPACE=nfs-system NFS_STORAGE_CLASS=nfs
kubectl rollout status deployment/nfs-subdir-external-provisioner -n nfs-system --timeout=180s
```

Keep the chart's PostgreSQL, log and model-cache StorageClass settings aligned. Budget storage
for the configured PVCs and workflow data; NFS claim sizes do not reserve host disk. Verify a
file exchange between two Pods mounting a disposable RWX claim, and test Pod DNS plus TCP access
to the kubeconfig API endpoint. Delete only the test resources after collecting results. A Bound
PVC or a Ready node alone is insufficient evidence for these checks.

## Secrets

Create `secrets.env` with the user's own credentials by following
[Secrets](../../../docs/getting-started.md#secrets), then run the [access checks](#check-access)
below. Optional API keys and OSMO credentials depend on the workflows being used. The product does
not support temporary AWS session credentials. Keep `secrets.env` out of Git and logs.

### Check access

Require `NGC_API_KEY`, `HF_TOKEN` and all four `AWS_S3_*` values in `secrets.env`.
Authenticate to NGC using `docker login nvcr.io --username '$oauthtoken'` and its hidden
password prompt. Check the selected image manifest and S3 bucket access; login or populated
credential names alone do not prove resource access. Test S3 read/write only at a user-approved
prefix, using a unique temporary object and deleting only that object afterward.

## Verify readiness

Reuse the existing preflight at both checkpoints below. Restore the repo venv, PATH and the
recorded `KUBECONFIG` after reconnecting. Read the reported statuses; command exit alone is not a
readiness verdict. These checks supplement the GPU, storage and network tests above.

### After bootstrap

Select the controller image path. For the repo's build target, use an NGC repository: it logs in
with `NGC_API_KEY` and `$oauthtoken`. With approval, replace the placeholder
`AIRFLOW_IMAGE_REPOSITORY` and the `AIRFLOW_IMAGE_TAG` values in `make/common.mk` with the selected
repository and a unique tag, as the
[getting-started guide](../../../docs/getting-started.md#start-airflow-and-run-workflows) describes.
Do not commit that edit. Only after the user confirms push access with those credentials, set the
audit arguments from the configured values (an exported variable takes precedence over the file):

```bash
: "${AIRFLOW_IMAGE_REPOSITORY:=$(sed -n 's/^AIRFLOW_IMAGE_REPOSITORY ?= *//p' make/common.mk)}"
: "${AIRFLOW_IMAGE_TAG:=$(sed -n 's/^AIRFLOW_IMAGE_TAG ?= *//p' make/common.mk)}"
case "$AIRFLOW_IMAGE_REPOSITORY" in
  ''|*'<'*|*'>'*) echo "Replace the placeholder AIRFLOW_IMAGE_REPOSITORY in make/common.mk" >&2 ;;
  *) CDF_IMAGE_ARGS=(--confirmed-push-repository "$AIRFLOW_IMAGE_REPOSITORY") ;;
esac
```

Alternatively, set `CDF_IMAGE_ARGS=(--prebuilt-controller-image "$CDF_CONTROLLER_IMAGE")` using
the user's published image. Login alone does not prove push permission. If permission is unknown,
leave `CDF_IMAGE_ARGS=()` and report the unresolved access check.

When no toolkit Pod exists, use `--probe-pod` with a ready Python 3 container. If none exists,
create a temporary GPU probe within the approved setup scope. Select an accessible glibc-based
Python 3 image compatible with the node architecture (for example `python:3.11-slim`). This example uses the
NVIDIA default runtime configured above and reserves one GPU:

```bash
: "${CDF_PROBE_IMAGE:?Select the Python 3 probe image}"
CDF_GPU_HOSTNAME="$(kubectl get node "$CDF_NODE_NAME" -o jsonpath='{.metadata.labels.kubernetes\.io/hostname}')"
: "${CDF_GPU_HOSTNAME:?Selected node must have a hostname label}"
CDF_PROBE_POD="$(kubectl create -n default -f - -o jsonpath='{.metadata.name}' <<EOF
apiVersion: v1
kind: Pod
metadata:
  generateName: cdf-preflight-
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: "$CDF_GPU_HOSTNAME"
  containers:
    - name: check
      image: "$CDF_PROBE_IMAGE"
      command: [python3, -c, "import time; time.sleep(3600)"]
      env:
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: compute,utility,video
      resources:
        limits:
          nvidia.com/gpu: 1
EOF
)"
kubectl wait -n default --for=condition=Ready "pod/$CDF_PROBE_POD" --timeout=180s
CDF_PROBE_ARGS=(--probe-pod "default/$CDF_PROBE_POD/check")
```

For multiple GPU nodes, select a GPU container per node and repeat `--probe-pod`. Probe containers
must have video driver libraries exposed; a CPU container cannot verify GPU library injection.
When toolkit Pods provide coverage, set `CDF_PROBE_ARGS=()`. From the repo root, run:

```bash
python3 skills/cdf-orchestration-preflight/scripts/preflight.py audit \
  --repo-root "$PWD" --kubeconfig "$KUBECONFIG" --format json \
  "${CDF_IMAGE_ARGS[@]}" "${CDF_PROBE_ARGS[@]}"
```

Save the report and carry `CDF_IMAGE_ARGS`, `CDF_PROBE_ARGS` and the image exports into the setup
handoff. Keep the temporary probe through setup's fresh pre-install audit and the post-deployment
audit below. This example exits after one hour; if it expires, recreate it and update the probe
arguments before auditing. Restore these values when changing shells or agents.

Require `install_readiness: READY` before handing off to the
[controller setup skill](../../cdf-orchestration-setup/SKILL.md). Resolve installation blockers
and missing evidence first. Before the first deployment, `controller.state` and `run_readiness`
can be `BLOCKED` because the controller is absent; this is expected.

### After controller deployment

Use the setup skill's Airflow connection/authentication instructions to populate `AIRFLOW_URL`
and `AIRFLOW_TOKEN` without printing the token or placing it in command arguments. Restore the image
arguments and select live probe containers as above before rerunning:

```bash
: "${AIRFLOW_URL:?Set the reachable Airflow URL}"
: "${AIRFLOW_TOKEN:?Authenticate through the setup skill first}"
export AIRFLOW_URL AIRFLOW_TOKEN
python3 skills/cdf-orchestration-preflight/scripts/preflight.py audit \
  --repo-root "$PWD" --kubeconfig "$KUBECONFIG" --format json \
  "${CDF_IMAGE_ARGS[@]}" "${CDF_PROBE_ARGS[@]}"
```

Verify controller health, DAG availability, zero import errors and the required pools. Inspect the
requested DAG's mode before proceeding to its workflow skill. `run_readiness: PARTIAL` means
some workflows remain blocked or unverified; list them rather than declaring all workflows ready.
Missing OSMO access or external endpoints must not be inferred from Kubernetes health.

Save the post-deployment report, then delete only the temporary probe(s) created for these checks:

```bash
kubectl delete pod -n default "${CDF_PROBE_POD:?}"
```

On a single-GPU node, the probe reserves the only GPU, so GPU workflow modes can report `BLOCKED`
while API/library probes pass. After deleting it, use `kubectl describe node "$CDF_NODE_NAME"` to
compare allocatable GPUs with the remaining requests. Report the saved probe evidence and current
free capacity separately; this is not an all-`READY` automated audit. The same reservation matters
on larger clusters near a workflow's GPU requirement. A deleted probe cannot supply live evidence
to a later audit; recreate it or select a suitable running container when rechecking.

Preserve `UNVERIFIED` where evidence is missing. The audit remains read-only; creating probe Pods
is a separate setup action, and selecting one does not bypass a failed probe.
