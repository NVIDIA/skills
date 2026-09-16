# Remote setup topologies

Controller placement and model-service placement are independent. A local controller does not
imply local inference services, and external inference endpoints do not imply a remote controller.

| Controller | Workflow compute | Model services | Valid use |
| --- | --- | --- | --- |
| Existing/remote | Remote Kubernetes | External | Low-footprint first trial |
| Same Kubernetes cluster | Same cluster | External | Recommended for a one-H100 all-in-one host |
| Existing or local | Kubernetes | Internal | Requires at least three GPUs for Image Attribute Augmentation or four for Event Video Generation |

## Existing controller plus remote GPU Kubernetes

This is the recommended first-trial topology. Airflow already runs somewhere reachable. The GPU
host may be a cloud VM, bare metal, or a managed cluster, but it must expose a Kubernetes API and
`nvidia.com/gpu` resources.

Each workflow's K8s manifest expects:

- Airflow connection: `kubernetes_remote`
- Namespace: `sdg-workflow`
- Pools: `k8s_gpu_1`, `default_pool`, plus the workflow's own image-service pools —
  `external_image_edit_service_pool` / `iaa_internal_image_edit_service_pool` for
  `image-attribute-augmentation-workflow`, or `external_image2video_service_pool` /
  `internal_image2video_service_pool` for `event-video-generation-workflow`
- Image-pull secret: `ngc-docker-registry-secret`
- Internal-mode PVC: `ngc-model-cache`

Use a kubeconfig whose API server address is reachable from the Airflow controller. A kubeconfig
that points at `127.0.0.1` on the GPU host is not remotely usable.

## All-in-one Kubernetes

The controller and augmentation worker may share one cluster while VLM, LLM, and image-edit
services remain external. This is a valid one-H100 topology. `make setup` reads the cluster
kubeconfig from `$KUBECONFIG` and embeds its contents into the generated Helm values, for the
`kubernetes_remote` Airflow connection.

## Local controller deployment

Use this path when the controller must be deployed into a new cluster. External model endpoints are
fully compatible and preferred for a single-GPU node.

Confirm the following tools are on PATH before running `make setup`:

| Tool | Min version | Install |
|---|---|---|
| `kubectl` | 1.26+ | https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/ |
| `helm` | 3.x | https://helm.sh/docs/intro/install/ |
| `uv` | 0.11.21+ | https://docs.astral.sh/uv/getting-started/installation/ |
| `docker` | 29.6.0+ | https://docs.docker.com/engine/install/ubuntu/ |
| `aws` | 2.x | https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html |

Install `aws` v2 from the official installer — `pip install awscli` gives the older v1 CLI.

If `helm` is on PATH but fails with `Permission denied`, install a user-local copy instead:

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 \
  | HELM_INSTALL_DIR=~/.local/bin USE_SUDO=false bash
```

Also inspect the Kubernetes context and storage classes, verify an S3 bucket and credentials, and
check the presence — not contents — of `secrets.env` and the Helm values `make setup` generates.

### How the worker kubeconfig is supplied

`setup_airflow_secrets.sh` reads the `KUBECONFIG` environment variable to find the kubeconfig
file. Set it in `secrets.env` (or your shell) before running `make setup`:

```bash
# In secrets.env — an absolute path to a kubeconfig you supply:
KUBECONFIG=/absolute/path/to/your/kubeconfig.yaml
```

The kubeconfig is never committed — it contains a cluster address and admin credentials.

The script embeds the kubeconfig's contents into the generated Helm values as the
`kubernetes_remote` Airflow connection (`in_cluster: false`). If `KUBECONFIG` is not set and the
shell is non-interactive, `make setup` will fail with an error.

Key requirement: **the embedded server must be pod-reachable.** The controller *pod* uses this
connection. If the kubeconfig's `server:` is `https://127.0.0.1:...` or `localhost` (common for
K3s and kubeadm), rewrite it to the node's cluster-reachable IP (for example
`https://<node-internal-ip>:6443`) before running `make setup`.

### Deploy steps

See [deploy-controller.md](deploy-controller.md) for the full `make setup` /
`make install sdg-controller` walkthrough.

### Storage

`deploy/values.yaml` uses `storageClassName: "nfs"` for its PVCs. If the cluster does not already
provide an `nfs` StorageClass, install one first:

```bash
make install nfs
```

This provisions a host NFS export and registers the StorageClass. Configure NFS parameters through
Makefile variables (`NFS_NODE_HOSTNAME`, `NFS_SERVER`, etc.) before running, or set them in
`deploy/values.yaml`. If the cluster already provides a different `ReadWriteMany` StorageClass,
change the `storageClassName` entries in `deploy/values.yaml` to match.

For internal service mode the shared model cache PVC is created as part of `helm upgrade --install`
when `modelCache.enabled: true` in `deploy/values.yaml`. External endpoint mode does not need it.

### In-cluster authentication

The `make`-supported path uses `in_cluster: false` with the embedded kubeconfig reaching the
cluster API. A true `in_cluster: true` connection requires the Airflow service account to have
RBAC to create pods, deployments, and services in `sdg-workflow` and cannot be produced by
`make setup`. To use it, manually remove the embedded kubeconfig from the generated Helm values
before `make install sdg-controller`.

After deployment, verify pods/jobs, Airflow service-account RBAC, API health, the target DAG(s)
(`image_attribute_augmentation_dag_k8s` and/or `event_video_generation_dag_k8s`), the required
pools, and storage mappings before reporting readiness.

## Remote controller deployment

Use this path to deploy the controller into a remote Kubernetes cluster from a workstation.
`make` targets run locally; the cluster does not need local tools.

1. On the **workstation**, put `kubectl`, `helm`, `uv`, `docker`, and `aws` on PATH.
2. Point `KUBECONFIG` at the remote cluster's kubeconfig. Its `server:` must be reachable both from
   your workstation and from inside the cluster's pods. Confirm `kubectl get nodes` works first.
3. Fill in `secrets.env`, then run `make setup`, then `make install sdg-controller`.
4. `make port-forward` forwards the controller's port 8080 (Airflow) to your workstation.

Verify the same readiness items as the local recipe against the remote cluster before reporting
ready.

## Plain remote host

An SSH-accessible GPU host is inventory, not yet an SDG compute backend. Install and configure an
appropriate Kubernetes distribution and NVIDIA integration using the provider's supported method,
then rerun the audit. The skill deliberately does not choose a distribution or install privileged
GPU components automatically.
