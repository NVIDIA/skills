# SDG controller connection

## Kubernetes connection

The controller's Airflow deployment needs a connection named `kubernetes_remote`. Its kubeconfig
must authenticate without interactive prompts and use an API server address reachable from Airflow.
Store kubeconfig content through the controller's secret-management path (Helm values); never paste
it into chat.

The standard path is to point `KUBECONFIG` at the kubeconfig the user supplied and run
`make setup`. The setup script embeds its contents into the generated Helm values under
`global.secrets.kubernetesRemote.kubeconfig`, yielding an `in_cluster: false` `kubernetes_remote`
connection. The embedded `server:` must be pod-reachable — use a node IP or the cluster's external
API endpoint, not `127.0.0.1`/`localhost`. K3s is not supported.

## DAG and plugin delivery

DAGs and plugins are **baked into the Airflow image** at build time (`docker/Dockerfile.airflow`
`COPY`s `airflow/dags` and `airflow/plugins` in) — there is no S3 sync, no dag-synchronizer
sidecar, and no `make sync-dag` target. `make install sdg-controller` only deploys whichever image
tag `deploy/values.yaml` currently points at; it does not rebuild or republish it. To pick up
DAG/plugin changes, rebuild and redeploy:

```bash
make build sdg-controller     # rebuilds and pushes the image with current DAGs/plugins
make install sdg-controller   # redeploys the controller with that image
```

A Helm install can report `STATUS: deployed` with every pod `Running` while the DagProcessor still
finds zero DAGs, if the deployed image lacks the current DAG source. Always confirm via the Airflow
API (`GET /api/v2/dags`, check `total_entries`) rather than trusting Helm success or pod health
alone. On the build path, rebuild and redeploy before checking the DagProcessor logs for
manifest-parsing errors. On the prebuilt path, never rebuild the selected image; report it as
unsuitable and stop.

## Pools and DAG registration

`default_pool` is Airflow's own built-in pool, always present. Every other pool below is
workflow-specific, created by the `airflowPools` job during Helm install/upgrade from
`deploy/values.yaml`'s `airflowPools.pools`. Verify the pools for whichever workflow(s) you intend
to run have positive slots:

**Never hardcode pool names.** They are deployment-specific: each DAG manifest names the pools it
needs in `deployment.profiles.*.pool`, and `deploy/values.yaml`'s `airflowPools.pools` creates them.
A customer on different hardware, or one who renamed them, will not match what this repository ships
today. Derive the required set from the manifests (see the snippet in SKILL.md's controller checks)
and verify each derived name exists with open slots. For orientation, the checked-in manifests
currently resolve to:

- `default_pool` — Airflow's built-in pool, shared by every workflow
- the GPU-task pool named by the K8s manifests' GPU profiles — `k8s_rtx6000_1` today (RTX PRO 6000
  Blackwell); an H100-based deployment would name it differently
- the OSMO pools for OSMO manifests — `osmo_rtx6000_1`, `osmo_h100_1`
- `external_image_edit_service_pool` (external mode) — `cdf-image-attribute-augmentation` and
  `cdf-defect-image-generation`
- `external_image2video_service_pool` (external mode) — `cdf-event-video-generation`
- `cosmos_transfer_3_service_pool` (external mode) — `cdf-video-data-augmentation` and
  `cdf-autonomous-vehicle-dataset-generation`

Internal-mode augmentation has no dedicated pool: it runs on `default_pool` and is capped by the
manifest's `max_concurrency`.

A pool missing from Airflow after install indicates the `create-pools` job failed — check its logs
in namespace `sdg-workflow`.

Verify the target DAG is loaded and `is_paused: false` — for example
`image_attribute_augmentation_dag_k8s` or `event_video_generation_dag_k8s`. A DAG is only
registered when its manifest file exists under that workflow's `configs/` directory (see
[topologies.md](topologies.md) for the full DAG-to-manifest mapping).

## Storage and secrets

Verify controller-side configuration includes:

- `multistorageclient_configuration_secret` mappings for input and output URLs
- `nvcf_default` when task containers receive NVIDIA credentials
- `hf_token` only when internal model deployment needs HuggingFace access (set via `HF_TOKEN` in
  `secrets.env` before `make setup`)

### Separate input and output buckets

`make setup` requires `AWS_S3_BUCKET`, `AWS_S3_REGION`, `AWS_S3_ACCESS_KEY_ID`, and
`AWS_S3_SECRET_ACCESS_KEY`; it generates one bucket profile and does not read per-purpose
`AWS_S3_INPUT_*`, `AWS_S3_OUTPUT_*`, or `AWS_S3_DAG_*` variables. DAGs are baked into the image.
For separate input and output buckets, add both profiles and their path mappings using the
Secret patch below. Preserve existing mappings needed by other workflows.

### S3-compatible storage (SwiftStack, MinIO, and similar)

`make setup` derives the endpoint from the region, so it always targets AWS. When a user asks to
hook up a bucket on an S3-compatible service — "use my bucket `xyz` at endpoint `abc`, creds are in
this file" — wire it by writing the storage config directly into the secret the DAGs read
(`Variable.get("multistorageclient_configuration_secret")`).

Collect the bucket, region, and endpoint URL. Read the credentials from the file path the user
gives you; never ask them to paste keys into the chat, and never echo the values back.

Build the Secret patch below, adding a second profile and mapping when input and output differ.
The string value replaces the entire storage configuration; include all mappings that must remain.
Write it to `$SCRATCH/msc-patch.yaml` with mode `600` in your private scratchpad directory
(not the repository), apply it, then delete the file.

```yaml
stringData:
  multistorageclient_configuration_secret: |-
    {
      "profiles": {
        "<bucket>": {
          "storage_provider": {
            "type": "s3",
            "options": {"base_path": "<bucket>", "region_name": "<region>", "endpoint_url": "<endpoint>"}
          },
          "credentials_provider": {
            "type": "S3Credentials",
            "options": {"access_key": "<key>", "secret_key": "<secret>"}
          }
        }
      },
      "path_mapping": {"s3://<bucket>/": "msc://<bucket>/", "<endpoint>/<bucket>/": "msc://<bucket>/"}
    }
```

```bash
: "${SCRATCH:?Set SCRATCH to the private directory containing the patch file for this operation}"
kubectl patch secret multistorageclient-configuration-secret -n sdg-workflow \
  --type merge --patch-file "$SCRATCH/msc-patch.yaml"
# Remove only the temporary credential patch created for this operation.
rm -- "$SCRATCH/msc-patch.yaml"

# The config reaches Airflow as an env var, so running pods keep the old value until restarted.
kubectl rollout restart -n sdg-workflow \
  deployment/sdg-workflow-controller-scheduler \
  deployment/sdg-workflow-controller-api-server \
  deployment/sdg-workflow-controller-dag-processor \
  statefulset/sdg-workflow-controller-triggerer
```

Wait for the rollout to finish, then confirm the DAG resolves the bucket before triggering a run.
Tell the user this patch is replaced whenever `make setup` + `make install sdg-controller` or a
`helm upgrade` regenerates the secret, so it must be re-applied after a redeploy.

Verify the worker cluster can resolve and reach the storage endpoints and external inference URLs.
Obtain explicit user approval before creating connectivity pods in the remote cluster.

External service mode still runs the attribute-search worker on Kubernetes. It does not deploy the
VLM, LLM, or image-edit services and does not require `ngc-model-cache` for those services.

## Existing API health

`make port-forward` exposes Airflow on port 8080. Verify Airflow health by following the preflight
checks in the Image Attribute Augmentation skill's
`references/airflow-direct-api.md`. Treat the Airflow-level checks as client configuration only
unless authorized controller checks also verified DAG access, pool capacity, and
Airflow-to-Kubernetes connectivity.
