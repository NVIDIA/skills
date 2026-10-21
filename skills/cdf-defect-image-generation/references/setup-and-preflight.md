# Setup and preflight

Use these checks for execution, not for offline explanations or JSON preparation.
Resolve the product checkout, controller URL, namespace/release, and authorized
cluster context from the task. Use the cluster credential file already exported
in the shell environment; otherwise ask the user for its path. Never guess a
credential path or print the file's contents.

## Controller and storage

The DIG manifest runs task pods in the controller's own cluster
(`in_cluster: true`) in namespace `sdg-workflow`, so no remote-cluster Airflow
connection is needed. Inspect the overriding manifest when
`DEFECT_IMAGE_GENERATION_SMT_K8S_MANIFEST_PATH` is configured: it can change
resources and placement.

For the default installation, these are read-only checks:

```bash
kubectl get pods -n sdg-workflow -l release=sdg-workflow-controller
kubectl get svc -n sdg-workflow sdg-workflow-controller-api-server
kubectl get pvc -n sdg-workflow
kubectl get secret -n sdg-workflow -o name
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
```

Check readiness as well as Running state for the controller containers. A missing
namespace/controller calls for setup. A failed task pod alone does not imply a
broken controller. For local API access, prefer an established reachable URL;
ClusterIP works only from hosts that can route to the cluster service network.

Confirm the DAG is loaded/unpaused through Airflow. Verify only the relevant pools:

| Resource | Pool / storage |
|---|---|
| Simulation GPU | `k8s_rtx6000_1`, one slot per task |
| Internal image-edit | `k8s_rtx6000_1`, one held slot per replica; `ngc-model-cache` PVC |
| Internal augmentation workers | `default_pool`, capped by the manifest's `max_concurrency` |
| External augmentation workers | `external_image_edit_service_pool` |
| AnomalyGen | `k8s_rtx6000_1`, two slots per task in the checked-in profile; `ngc-model-cache` PVC |
| CPU setup tasks | `default_pool` |

Use API pool responses for actual slot counts; do not copy the starter payloads'
eight image-edit replicas or assume configured slots equal physical free GPUs. If
`num_gpus` exceeds two, flag the manifest's fixed two-slot AnomalyGen reservation
for review so Airflow accounting matches Kubernetes requests.

## GPU capacity by stage

Inspect pod resource requests across namespaces on the target nodes, including
Pending reservations and other users' workloads; report free versus allocatable
GPUs, plus node placement constraints. Augmentation workers make remote calls and
request no GPU. The default stage sequence permits GPU reuse:

| Mode | Per-stage GPU requirement for one active job |
|---|---|
| `simulation` | One simulation GPU |
| `sim2real`, external | One simulation GPU; external image-edit has no local service GPU |
| `sim2real`, internal | One simulation GPU, then one per image-edit replica |
| `full`, external | One simulation GPU, then at least two for default AMG/AnomalyGen |
| `full`, internal | One simulation GPU, then image-edit replicas, then at least two for default AMG/AnomalyGen |
| `defect` | At least two for default AMG/AnomalyGen |

The runner requests `max(2, num_gpus)` when `amg` is among the stages; otherwise it
requests `num_gpus`. Multiple GPUs for a single AnomalyGen pod must fit on one
eligible node. Two separate one-GPU nodes are insufficient. Internal services shut
down before AnomalyGen starts. Mapped jobs may run concurrently within a stage;
pools and capacity constrain that concurrency. Do not remove `amg` just to fit
hardware: such a change requires a compatible prepared-mask workflow.

## Inputs, model access and credentials

- Use S3 locations that the deployment's storage credentials can read and write:
  `make setup` configures a single S3 credential set, which every DIG task uses.
  Verify scene ZIP and asset readability and a writable output prefix with those
  credentials. For defect-only, inspect the class folders; for full, check
  selected component types against every defect configuration's mapping.
- Check each variant's effective `staging.fetch.input`, `recipe` and `checkpoint`,
  which the checker prints. The checked-in AnomalyGen template names example
  locations, so require the user's values as payload overrides; per-run asset
  changes do not need a controller rebuild.
- For internal image-edit, confirm model-cache storage, registry pull access and
  `HF_TOKEN` access to the SMT checkpoint. The checkpoint download runs after
  simulation. AnomalyGen also uses `HF_TOKEN` to fetch model weights that are not
  yet on the model-cache volume.
- For external image-edit, verify the endpoint from the compute network and its
  served model. Requests authenticate with the deployment's NGC API key, so the
  endpoint must accept that key or require none. Neither a VLM nor an LLM service
  is needed for the fixed-prompt golden-image route.
- For full and defect runs, check the AMG endpoint and model configuration
  separately, and that `VLM_API_KEY` (or the NGC API key fallback) is available to
  the task. Confirm credential presence without reading Kubernetes Secret bodies
  into the conversation.

If a check fails, report the blocker and resolve it within the requested scope.
Use the setup skill or the product's `docs/getting-started.md` for missing
infrastructure. Do not install or redeploy a healthy controller just to check a
run.
