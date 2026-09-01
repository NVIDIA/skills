# Troubleshooting

## Payload rejected locally

- Confirm `num_augmentation` is singular and at least one.
- Match nested output directories and service modes to the top-level fields; the server raises if
  `cosmos.*` or `event_and_person_attribute_search.*` disagree with the top level.
- External mode requires `cosmos.vlm_service_url`, `cosmos.llm_service_url`,
  `cosmos.image_edit_service_url`, and `event_and_person_attribute_search.llm_service_url`.
- Conditional distributions must cover every possible parent value.
- A *partial* `variable_distribution` fails at runtime, not at validation — supply all six clothing
  attributes or omit the field entirely. See [payload-contract.md](payload-contract.md).
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
- After changing DAG files, run `make sync-dag` and wait for the dag-synchronizer to pull the
  updated artifacts from S3 (check interval is configured in `deploy/values.yaml` under
  `global.s3Sync.intervalSeconds`, default 30 seconds).

## Run failed

Fetch the per-task breakdown before speculating about a cause — see
[airflow-direct-api.md#per-task-breakdown-useful-for-diagnosing-failures](airflow-direct-api.md#per-task-breakdown-useful-for-diagnosing-failures).
Report the run ID, platform, terminal state, and the first failing task. Do not claim a failed stage
or auto-restart without evidence.

Common failure points:

- `cosmos_augmentation.validate_outputs` failing with missing `output.jpg` usually means the
  augmentation container exited early — a partial `variable_distribution` is the most common cause,
  and the pod itself reports success.
- `wait_for_vlm` / `wait_for_llm` / `wait_for_image_edit` hanging in internal mode means the
  service deployments never became ready; check GPU availability in the `k8s_gpu_1` pool and the
  pods in the `sdg-workflow` namespace. Internal mode requires at least three free GPUs (one each
  for VLM, LLM, and image-edit).
- Tasks stuck `queued` mean the relevant pool has no open slots. Check `iaa_internal_image_edit_service_pool`
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
  "$AIRFLOW_URL/api/v2/dags/$DAG_ID/dagRuns/$RUN_ID")
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

After editing DAG files, push the changes to S3 with:

```bash
make sync-dag
```

The dag-synchronizer pulls from S3 on a configurable interval (default 30 s). Wait for the
DagProcessor to pick up the new files before re-triggering. Do not reinstall the Helm chart to
update DAGs — `make sync-dag` is sufficient.

## Port-forward fails

If `make port-forward` fails or the Airflow UI is unreachable, find the ClusterIP directly:

```bash
kubectl get svc -n sdg-workflow sdg-workflow-controller-api-server -o jsonpath='{.spec.clusterIP}'
AIRFLOW_URL="http://<cluster-ip>:8080"
```

The REST API is reachable from the host via the ClusterIP even without port-forward.
