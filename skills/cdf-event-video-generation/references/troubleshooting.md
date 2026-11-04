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
- Use `payload.py validate` as a check, preserving the original payload; the normalizer drops
  evaluator fields. See [payload validation](payload-contract.md#validation).

## Dataset rejected locally

- Do not nest images in subdirectories — every accepted file directly under `input_path` (or the
  single file it names) is treated as one input image; there is no person-ID or other grouping
  convention.
- Remove symlinks, empty files, and non-image files.
- Fix files whose extension does not match JPEG/PNG content.
- Use `--skip-content-check` only when a storage/export tool produces valid images with signatures
  the lightweight detector cannot recognize.

## DAG not found

- Confirm the DAG ID is `event_video_generation_dag_k8s` or `event_video_generation_dag_osmo` —
  the two this repository registers.
- A DAG is only registered when its manifest exists in
  `airflow/dags/workflows/event_video_generation_dag/configs/`. If the manifest is present but
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

- `cosmos_augmentation.validate_outputs` skips, with a warning, each candidate that is missing
  `config.yaml`, `output.mp4`, `caption.txt`, or `metadata.json`; whose `config.yaml` or
  `metadata.json` cannot be read as a mapping; whose Cosmos Evaluator metadata is missing while the
  evaluator is enabled, or reports that the evaluator did not complete; or that fails a Cosmos
  Evaluator gate. When no candidate is accepted, the task fails with `No complete Event Video
  Generation augmentations passed output validation`. Check the task log for the per-candidate
  `Skipping ...` warnings to see why each candidate was dropped.
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
- `wait_for_vlm` / `wait_for_image2video` hanging in internal mode means the service deployments
  never became ready; check GPU availability in the manifest's GPU pool (whatever
  `deployment.profiles.k8s_gpu_endpoint.pool` names — `k8s_rtx6000_1` in the currently checked-in
  manifest) and the pods in the `sdg-workflow` namespace. Internal mode requires at least **three**
  free GPUs for the inference endpoints (one `vlm_service`, two `image2video_service`), plus two
  more when the Cosmos Evaluator checks are enabled. There is no `wait_for_llm` task and no
  separate LLM endpoint: the one deployed Qwen VLM also serves the text-only LLM traffic.
- `image2video_service` pods restarting with `Orchestrator did not become ready within <N>s`
  mean model initialization exceeded its `--init-timeout`. Follow
  [model startup timeouts](../../cdf-orchestration-setup/references/model-startup-timeouts.md).
  Propose a timeout increase to the user.
- Tasks stuck `queued` mean the relevant pool has no open slots. Check `default_pool` for
  internal mode or `external_image2video_service_pool` for external mode.

## Run processes fewer images than requested / uses wrong anomaly distribution

If the run completes but processed only 10 images (the default `max_images`) or generated a
single-anomaly result despite a richer `variable_distribution` in your payload, the payload was
not picked up — the DAG ran with model defaults instead.

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
