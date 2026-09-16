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
API endpoint, not `127.0.0.1`/`localhost`.

## DAG and plugin file sync

After `make install sdg-controller`, DAGs and plugins are pulled from S3 by the dag-synchronizer
sidecar at the interval configured under `global.s3Sync.intervalSeconds` in `deploy/values.yaml`
(default 30 s). To push updated DAG files without reinstalling the Helm chart, run:

```bash
make sync-dag
```

Then wait for the dag-synchronizer interval before expecting the DagProcessor to pick up the
changes. If a DAG disappears after a sync, check the DagProcessor logs for manifest-parsing
errors.

## Pools and DAG registration

`default_pool` is Airflow's own built-in pool, always present. Every other pool below is
workflow-specific, created by the `airflowPools` job during Helm install/upgrade from
`deploy/values.yaml`'s `airflowPools.pools`. Verify the pools for whichever workflow(s) you intend
to run have positive slots:

- `k8s_gpu_1`, `default_pool` — shared by every workflow
- `external_image_edit_service_pool`, `iaa_internal_image_edit_service_pool` (internal mode) —
  `image-attribute-augmentation-workflow`
- `external_image2video_service_pool`, `internal_image2video_service_pool` (internal mode) —
  `event-video-generation-workflow`

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

Set the per-purpose `AWS_S3_INPUT_*` and `AWS_S3_OUTPUT_*` variables in `secrets.env` instead of
the unified `AWS_S3_*` shorthand from `secrets.env.example`, then re-run `make setup` and
`make install sdg-controller`.

### S3-compatible storage (SwiftStack, MinIO, and similar)

`make setup` derives the endpoint from the region, so it always targets AWS. When a user asks to
hook up a bucket on an S3-compatible service — "use my bucket `xyz` at endpoint `abc`, creds are in
this file" — wire it by writing the storage config directly into the secret the DAGs read
(`Variable.get("multistorageclient_configuration_secret")`).

Collect the bucket, region, and endpoint URL. Read the credentials from the file path the user
gives you; never ask them to paste keys into the chat, and never echo the values back.

Build the config below, adding a second profile and mapping when input and output differ. Write it
to a patch file in your scratchpad directory (not the repository), apply it, then delete the file.

```json
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
kubectl patch secret multistorageclient-configuration-secret -n sdg-workflow \
  --type merge --patch-file "$SCRATCH/msc-patch.json"

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
Do not run connectivity pods without approval because that creates remote resources.

External service mode still runs the attribute-search worker on Kubernetes. It does not deploy the
VLM, LLM, or image-edit services and does not require `ngc-model-cache` for those services.

## Existing API health

`make port-forward` exposes Airflow on port 8080. Verify Airflow health by following the preflight
checks in the Image Attribute Augmentation skill's
`references/airflow-direct-api.md`. Treat the Airflow-level checks as client configuration only
unless authorized controller checks also verified DAG access, pool capacity, and
Airflow-to-Kubernetes connectivity.
