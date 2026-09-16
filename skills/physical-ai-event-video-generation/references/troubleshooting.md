# Troubleshooting

## Payload rejected locally

- Confirm `num_augmentation` is singular and at least one.
- Match `cosmos.output_directory` and `cosmos.external_services` to the top-level fields; the
  server raises if they disagree.
- External mode requires `cosmos.vlm_service_url`, `cosmos.llm_service_url`, and
  `cosmos.image2video_service_url`.
- `variable_distribution.variables` must contain **exactly** `anomaly_type` and `env_type` — no
  more, no fewer. Unlike the Image Attribute Augmentation DAG, there is no partial-set runtime
  failure mode here: the exact-two-key rule is enforced by Pydantic (`extra="forbid"` plus a
  model validator) at payload validation time, so a malformed distribution fails fast rather than
  producing a misleading `success`.
- Each distribution (`anomaly_type`, `env_type`) must have at least one entry with a positive
  total weight.
- Use `payload.py validate` to normalize the payload before submission.

## Dataset rejected locally

- Do not nest images in subdirectories — every accepted file directly under `input_path` (or the
  single file it names) is treated as one input image; there is no person-ID or other grouping
  convention.
- Remove symlinks, empty files, and non-image files.
- Fix files whose extension does not match JPEG/PNG content.
- Use `--skip-content-check` only when a storage/export tool produces valid images with signatures
  the lightweight detector cannot recognize.

## DAG not found

- Confirm the DAG ID is `event_video_generation_dag_k8s`, the only one this repository registers.
- A DAG is only registered when its manifest exists in
  `airflow/dags/workflows/event_video_generation_dag/configs/`. If the manifest is present but
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

- `cosmos_augmentation.validate_outputs` failing with a missing `output.mp4`, `caption.txt`, or
  `metadata.json` usually means the image-to-video generation container exited early or the
  endpoint returned an error before writing all artifacts.
- A Cosmos augmentation that exhausts its Airflow retries is silently excluded from
  `anomaly_dataset/dataset.json` rather than failing the whole run — cross-check
  `dataset.json`'s `total_scenes` against `processed images * num_augmentation` if the count looks
  short; do not assume every requested augmentation is present just because the run reports
  `success`.
- `validated_output.validate_outputs` failing with a missing `sidecars/person_attribute_search/*`
  or `contextual/person_attributes.json` file for a scene that **does** have
  `sidecars/detection_and_tracking/tracks.json` indicates a real person-attribute-search failure.
  If `tracks.json` is absent for that scene, those files are correctly skipped — check
  `trackless_scenes` in the task's return value before treating it as an error.
- `wait_for_vlm` / `wait_for_llm` / `wait_for_image2video` hanging in internal mode means the
  service deployments never became ready; check GPU availability in the `k8s_gpu_1` pool and the
  pods in the `sdg-workflow` namespace. Internal mode requires at least four free GPUs (one each
  for VLM and LLM, two for image2video).
- Tasks stuck `queued` mean the relevant pool has no open slots. Check
  `internal_image2video_service_pool` for internal mode or `external_image2video_service_pool`
  for external mode.

## Run processes fewer images than requested / uses wrong anomaly distribution

If the run completes but processed only 10 images (the default `max_images`) or generated a
single-anomaly result despite a richer `variable_distribution` in your payload, the payload was
not picked up — the DAG ran with model defaults instead.

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
print('max_images:', payload.get('max_images'))
print('image2video_url:', (payload.get('cosmos') or {}).get('image2video_service_url'))
print('variable_distribution:', (payload.get('cosmos') or {}).get('variable_distribution'))
"
```

**Fix:** always trigger as `{"conf": {"payload": { ...your payload fields... }}}`.

## Image2video endpoint errors

If the image2video endpoint returns a 404 or account-mismatch error, the URL belongs to a
different NGC organization or the deployed function ID is stale. Supply a URL from an endpoint the
user's NGC account owns or has been granted access to.

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
