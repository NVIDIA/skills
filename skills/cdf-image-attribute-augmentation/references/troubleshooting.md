# Troubleshooting

## Payload rejected locally

- Confirm `num_augmentation` is singular and at least one.
- Match nested output directories and service modes to the top-level fields; the server raises if
  `cosmos.*` or `event_and_person_attribute_search.*` disagree with the top level.
- External mode requires `cosmos.vlm_service_url`, `cosmos.llm_service_url`,
  `cosmos.image_edit_service_url`, and `event_and_person_attribute_search.llm_service_url`.
- Conditional distributions must cover every possible parent value.
- `payload.py render` and `validate` reject a *partial* `variable_distribution`. Supply all six
  clothing attributes or omit the field entirely. The DAG itself does not check this at trigger
  time, so a partial distribution submitted without `payload.py` is accepted and fails at runtime.
  See [payload-contract.md](payload-contract.md).
- Use `payload.py validate` to normalize the payload before submission.

## Dataset rejected locally

- Put images exactly one directory below the root: `<person_id>/<image>.jpg`.
- Remove nested directories, root files, symlinks, empty files, and non-image files.
- Fix files whose extension does not match JPEG/PNG content.
- Use `--skip-content-check` only when a storage/export tool produces valid images with signatures
  the lightweight detector cannot recognize.

## DAG not found

- Confirm the DAG ID is `image_attribute_augmentation_dag_k8s`, the only one this repository
  registers.
- A DAG is only registered when its manifest exists in
  `airflow/dags/workflows/image_attribute_augmentation_dag/configs/`. If the manifest is present but
  the DAG is missing, check the DagProcessor logs — a manifest load failure is logged as a warning
  and the DAG is skipped silently.
- DAGs and plugins are baked into the Airflow image at build time — there is no S3 sync and no
  dag-synchronizer. A stale deployed image is the usual cause of a DAG that's present in the repo
  but missing from Airflow; rebuild and redeploy as described in
  [DAG not updating after code change](#dag-not-updating-after-code-change).

## Run failed

Fetch the per-task breakdown before speculating about a cause — see
[airflow-direct-api.md#per-task-breakdown-useful-for-diagnosing-failures](airflow-direct-api.md#per-task-breakdown-useful-for-diagnosing-failures).
Report the run ID, platform, terminal state, and the first failing task. Do not claim a failed stage
or auto-restart without evidence.

Common failure points:

- `cosmos_augmentation.validate_outputs` failing with missing `output.jpg` usually means the
  augmentation container exited early — a partial `variable_distribution` is the most common cause,
  and the pod itself reports success.
- `wait_for_vlm` / `wait_for_image_edit` hanging in internal mode means the service deployments
  never became ready; check GPU availability in the manifest's GPU pool (whatever
  `deployment.profiles.k8s_gpu_endpoint.pool` names — `k8s_rtx6000_1` in the currently checked-in
  manifest) and the pods in the `sdg-workflow` namespace. Internal mode requires at least **two**
  free GPUs — one for `vlm_service` and one for `image_edit_service`. There is no `wait_for_llm`
  task and no separate LLM endpoint: the one deployed Qwen VLM also serves the text-only LLM
  traffic.
- Service pods restarting with `Orchestrator did not become ready within <N>s` mean model
  initialization exceeded its `--init-timeout`. Follow
  [model startup timeouts](../../cdf-orchestration-setup/references/model-startup-timeouts.md).
  Propose a timeout increase to the user.
- Tasks stuck `queued` mean the relevant pool has no open slots. Check `default_pool`
  for internal mode or `external_image_edit_service_pool` for external mode.

## Run processes fewer images than requested / uses wrong clothing distribution

If the run completes but processed only 1 image (the default) or generated single-outfit results
despite a richer `variable_distribution` in your payload, the payload was not picked up — the DAG
ran with model defaults instead.

**Cause:** the trigger body must wrap the payload under `conf.payload`, not at the top level of
`conf`. A trigger body of `{"conf": {...payload fields...}}` is silently ignored.

**Verify** what the DAG actually received after triggering:

```bash
RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$DAG_RUN_ID")
RESPONSE="$RESPONSE" python3 -c "
import json, os
c = json.loads(os.environ['RESPONSE']).get('conf', {})
payload = c.get('payload') or c
print('max_imgs:', payload.get('max_imgs'))
print('image_edit_url:', (payload.get('cosmos') or {}).get('image_edit_service_url'))
"
```

**Fix:** always trigger as `{"conf": {"payload": { ...your payload fields... }}}`.

## Image-edit 404 — function not found for account

```
Generation failed ... 404 {'detail': "Function '<id>': Not found for account '<account>'"}
```

The image-edit endpoint URL belongs to a different NGC organization. Supply a URL from an endpoint
the user's NGC account owns or has been granted access to.

## DAG not updating after code change

DAGs are baked into the Airflow image, so a code change requires a rebuild and redeploy — there is
no lighter-weight sync step. Rebuild first; this writes the new image tag into
`deploy/values.yaml`:

```bash
make build sdg-controller     # rebuilds and pushes the image with the current DAG files
```

**`make install sdg-controller` is not an isolated repair.** It runs `make uninstall` first, which
force-deletes every Job, Deployment, StatefulSet, Service, and Pod in the `sdg-workflow`
namespace — including other users' active runs and their GPU services. PVCs are kept. Use it only
when the user confirms that nothing else is running in the namespace or explicitly asks for a full
reinstall; that path belongs to the environment-setup skill.

On a shared installation, roll out the rebuilt image with an in-place Helm upgrade instead. It
uses the same chart and value files as `make install` but skips the namespace cleanup, so running
tasks keep their pods. Confirm the command with the user before running it:

```bash
REPO_ROOT=/path/to/cdf-orchestration
cd "$REPO_ROOT"
helm upgrade sdg-workflow-controller deploy \
  --namespace sdg-workflow \
  --values deploy/values.yaml \
  --values deploy/secret_values_dev.yaml \
  --wait --timeout 10m
```

`deploy/secret_values_dev.yaml` is generated by `make setup`; if it is missing, route to the
environment-setup skill rather than recreating it. Re-trigger only after the controller's pods are
`Running` and `GET /api/v2/dags` reflects the change. Redeploying an unrebuilt image tag will not
pick up the edit even though Helm reports success.

## Port-forward fails

If `make port-forward` fails or the Airflow UI is unreachable, find the ClusterIP directly:

```bash
kubectl get svc -n sdg-workflow sdg-workflow-controller-api-server -o jsonpath='{.spec.clusterIP}'
AIRFLOW_URL="http://<cluster-ip>:8080"
```

The ClusterIP answers only from hosts that route the cluster service network, such as a cluster
node. From a workstation it usually times out; use a port-forward there instead.
