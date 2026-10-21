# Internal model startup timeouts

Applies to every workflow run with `external_services: false`. Each in-cluster `vllm` or
`vllm-omni` endpoint must finish loading its model within the `--init-timeout` in that endpoint's
`container_args` in the workflow's k8s manifest under
`airflow/dags/workflows/<workflow>/configs/`. `vllm-omni` endpoints also accept
`--stage-init-timeout` for each pipeline stage. An endpoint without `--init-timeout` uses the
server's 600-second default; read the deployed manifest rather than assuming a value.

These values are upper bounds, not fixed waits. The DAG polls the endpoint's `health_uri` and
continues as soon as it reports ready. Separately, the DAG waits up to
`deployment_ready_timeout_seconds` in the endpoint profile for the whole deployment.

## Diagnose

A server that exceeds its timeout exits with
`TimeoutError: Orchestrator did not become ready within <N>s` and Kubernetes restarts the pod.
Confirm it from the previous container's log:

```bash
kubectl -n sdg-workflow logs <pod> --previous
```

- Inspect each restarting replica; do not attribute one log's cause to every replica.
- A single restart that then recovers does not fail the run. Report it, but do not intervene.
- Do not claim a storage or cache root cause without evidence.

## Recommend

If startup repeatedly exceeds the timeout, for example when many replicas load from the shared
model cache at once or the cache is cold, suggest raising it:

- Raise `--init-timeout` and `--stage-init-timeout` together on `vllm-omni` endpoints.
- Keep both below `deployment_ready_timeout_seconds`, or the DAG stops waiting first.
- Increase in steps rather than to the maximum. A higher value also delays failure of an
  endpoint that is genuinely stuck.

This is a manifest change baked into the Airflow image. Propose it to the user with the observed
log evidence; do not edit the manifest or redeploy without their approval, and never during an
active run. Once approved, rebuild and roll the controller as described in
[deploy-controller.md](deploy-controller.md).
