---
name: cdf-orchestration-setup
description: Use when the user asks to deploy, install, reinstall, redeploy, upgrade, or connect to the CDF Orchestration Airflow controller. Owns the end-to-end deploy sequence and runs the cdf-orchestration-preflight audit first. Covers make setup, controller image builds, Helm installation, controller connection, port-forwarding, and post-install verification. Not for standalone prerequisite audits or remediation, K3s environments, authoring DAGs, or running workflow payloads.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  owner: NVIDIA
  service: cosmos-data-factory
  version: 1.1.3
  reviewed: '2026-10-09'
  author: NVIDIA
  tags:
    - physical-ai
    - cdf-orchestration
    - kubernetes
    - airflow
---

# CDF Orchestration Setup

Deploy and verify the Airflow controller after CDF orchestration preflight has established that the
checkout and target environment are ready. This skill owns deployment and post-install operations;
it does not repeat prerequisite discovery or remediation.

## Safety boundary

Before any Helm install, image push, Airflow connection change, or remote mutation, summarize the
exact changes and obtain user approval. Never print kubeconfig, NGC keys, Hugging Face tokens, AWS
secrets, OSMO tokens, or Kubernetes Secret bodies.

## Verify CDF preflight first

Before `make setup`, a controller image build, or any Helm installation or redeployment, invoke and
follow `$cdf-orchestration-preflight` for the same checkout and target environment.

Require a fresh preflight report with `install_readiness: READY`. If installation readiness is
`BLOCKED` or `UNVERIFIED`, stop and return to preflight to identify and resolve the prerequisite gap.
Do not independently rerun prerequisite probes, including local-tool, Kubernetes, GPU, storage/NFS,
registry, credential, OSMO, or workflow-mode checks.

A `WARNING` on a tool version means the installed version is outside the tested configuration. It
does not block installation: tell the user which tool and version, note that it is untested, and
continue once they acknowledge it.

`run_readiness` may still be blocked before the first installation solely because the controller
is absent. That is expected and does not block deployment when `install_readiness` is `READY`.

If preflight identifies K3s, stop. This setup skill does not support or invoke K3s. Carry forward
the preflight report's repository root, kubeconfig, namespace, storage class, service mode, and
image repository and tag rather than rediscovering them.

Preserve the preflight invocation's image and `--probe-pod` arguments, including any exact
`--confirmed-push-repository` confirmation, for fresh and post-install audits. Keep bootstrap's
temporary probes through those checks; if one has expired, return to preflight to recreate it.
Follow its cleanup and single-GPU capacity guidance after the post-install audit.

## Select two independent axes

1. Select controller placement: use an existing controller, or deploy Airflow into the current
   Kubernetes cluster.
2. Select model-service placement separately: use external VLM/LLM/image-edit endpoints, or deploy
   those services in-cluster. Never infer one choice from the other.
3. Prefer external endpoints on a one-GPU node. The controller and augmentation worker may
   still run in that node's cluster.
**Compute backend: Kubernetes, plus OSMO for whichever DAGs ship an OSMO manifest.** A DAG is
registered per platform only when its manifest exists, so determine OSMO support by looking, not by
memory:

```bash
ls airflow/dags/workflows/*/configs/*osmo*manifest*.yaml
```

**OSMO workflows accept external endpoints only.** This is structural, not a policy preference:
every OSMO manifest defines batch tasks but an empty `components.endpoints`, so there is nothing
in-cluster for a service lifecycle to deploy. Any OSMO payload must therefore set
`external_services: true` and supply every inference URL the pipeline needs (plus an external
evaluator/arbitrator URL when a quality check is enabled). A payload with
`external_services: false` is invalid against a deployed OSMO manifest even though the shared
payload schema will happily parse it — never submit one, and never reuse a K8s internal-mode
starter payload for an OSMO DAG. Internal/in-cluster model services are a Kubernetes-only option.

Read [topologies.md](references/topologies.md) before changing infrastructure.

## Deploy a controller on the current cluster

When the user requests setup (not audit-only), choose the command sequence that matches the
preflight report, present the exact substituted commands, and obtain explicit approval.

For the build/push path, the image location is `AIRFLOW_IMAGE_REPOSITORY`/`AIRFLOW_IMAGE_TAG` in
`make/common.mk`, which preflight already verified. If that file still holds the placeholder
repository, return to preflight; do not substitute a path. Name the configured image in the prompt:

> "I will run the following commands in order:
> 1. `make setup` — validates secrets from `secrets.env` and generates the Helm values for install
> 2. `make build sdg-controller` — builds the Airflow image with DAGs and plugins baked in
>    (`docker/Dockerfile.airflow` copies `airflow/dags` and `airflow/plugins`), pushes it to
>    `<AIRFLOW_IMAGE_REPOSITORY>:<AIRFLOW_IMAGE_TAG>` from `make/common.mk`, and writes that
>    repository/tag into `deploy/values.yaml`
> 3. `make install sdg-controller` — installs/upgrades the Helm release using that image
>
> Installation first uninstalls the release and deletes all cronjobs, jobs, statefulsets,
> deployments, replicasets, services, and pods in the target namespace, including unrelated
> workloads. Most workload deletions are forced. Uninstall can remove Helm-managed PVCs,
> including model-cache and log claims. Preserve needed data before reinstall.
>
> Proceed?"

For an exact image verified by `--prebuilt-controller-image`:

> "I will run the following commands in order:
> 1. `make setup` — validates secrets from `secrets.env` and generates the Helm values for install
> 2. Create a temporary copy of `deploy/values.yaml` and set its Airflow repository/tag to
>    `<VERIFIED_REPOSITORY>:<VERIFIED_TAG>`; no image will be built or pushed
> 3. `make install sdg-controller VALUES_DEV=<TEMPORARY_VALUES>` — installs the verified prebuilt image
> 4. Run the required cluster and Airflow post-install checks, then remove the temporary values file
>
> Installation first uninstalls the release and deletes all cronjobs, jobs, statefulsets,
> deployments, replicasets, services, and pods in the target namespace, including unrelated
> workloads. Most workload deletions are forced. Uninstall can remove Helm-managed PVCs,
> including model-cache and log claims. Preserve needed data before reinstall.
>
> Proceed?"

DAGs and plugins are **baked into the Airflow image**, not synced from S3 at runtime — there is no
dag-synchronizer and no `make sync-dag` target. On the build/push path, any time
DAG/plugin/runtime-dependency source changes, or the target image tag has never been published,
`make build sdg-controller` must run before `make install sdg-controller`. A selected prebuilt image
is immutable input to this flow: never rebuild or push it. If its installed DAG set is empty or
stale, stop and report that the supplied image is unsuitable. Confirm DAGs actually loaded via the
Airflow API (`GET /api/v2/dags`, `total_entries`) rather than trusting Helm success alone.

There are exactly three controller install-related targets: `make setup`,
`make build sdg-controller`, and `make install sdg-controller`. There is no bare `make install`.

When preflight verified an exact `--prebuilt-controller-image`, do not rebuild or push it. Run
`make setup`, create a temporary copy of `deploy/values.yaml` with
`airflow.defaultAirflowRepository` and `airflow.defaultAirflowTag` set to the verified reference,
then run `make install sdg-controller VALUES_DEV=/path/to/temporary-values.yaml`. Command-line
`AIRFLOW_IMAGE_REPOSITORY`/`AIRFLOW_IMAGE_TAG` are build inputs and do not override the image used
by `make install`. Keep the temporary values file outside the repository and remove it after the
post-install checks.

**Always run `make setup` first** on any deploy, install, or redeploy request — even if a previous
run already generated the Helm values. Secrets rotate. `make setup` is quick, but it regenerates
`secrets.env` and `deploy/secret_values_dev.yaml` from `secrets.env`: variables it does not manage
are removed and the `global.secrets.aws` block is reset. If the user keeps custom values in either
file, ask them to back it up first. Only skip it mid-session when the agent itself just ran it
moments earlier.

The OSMO settings (`OSMO_TOKEN`, `OSMO_URL`, `OSMO_POOL` and the optional
`OSMO_HF_CACHE_SEED_DIR`) are optional at deploy time and **not** something to prompt for here.
OSMO manifests ship with the repository, so their presence says nothing about whether this user
intends to run an OSMO workflow; `secrets.env.example` already documents the variables for those
who do. A Kubernetes-only deployment never needs them. The workflow skills check them at trigger
time, which is the only point where their absence actually matters — see
[deploy-controller.md](references/deploy-controller.md) for what `make setup` does with it.

**Missing namespace** — if `kubectl get ns sdg-workflow` returns `NotFound`, this is a normal
first-install condition, not an error to diagnose. On the build path, build first if the tag has
not been published; on the verified prebuilt path, proceed directly to the prebuilt install
sequence after approval.

Read [deploy-controller.md](references/deploy-controller.md) for the full `make setup` /
`make build sdg-controller` / `make install sdg-controller` walkthrough and post-invocation
cluster-state verification. Any missing prerequisite discovered during deployment returns to
`$cdf-orchestration-preflight`; do not add an alternate preparation path here.

When internal model endpoints restart with a startup timeout, read
[model-startup-timeouts.md](references/model-startup-timeouts.md) before proposing a manifest
change.

## Connect the SDG controller

Read [controller-connection.md](references/controller-connection.md). Before the post-install checks,
establish a reachable Airflow endpoint with a bounded probe; if the ClusterIP is unreachable, fall
back to `make port-forward` and `http://localhost:8080`. Then obtain an authentication token via
`/auth/token`, following [Connect to the deployed controller](#connect-to-the-deployed-controller),
and use it for the checks below.

```bash
# 1. kubernetes_remote connection exists.
# It is injected as an env var, not stored in the metadata database, so
# GET /api/v2/connections/kubernetes_remote returns 404 on a healthy controller.
# Check the env var instead — a 404 here is not a failure.
kubectl exec -n sdg-workflow deploy/sdg-workflow-controller-scheduler -c scheduler -- \
  printenv AIRFLOW_CONN_KUBERNETES_REMOTE >/dev/null 2>&1 \
  && echo "kubernetes_remote: present" \
  || echo "kubernetes_remote: MISSING"

# 2. Required pools have slots. default_pool is Airflow's built-in pool (not chart-created); the
# rest are created from deploy/values.yaml's airflowPools.pools and consumed by each DAG manifest's
# deployment.profiles.*.pool. Pool names are deployment-specific — never hardcode them. Derive the
# set that must exist by reading the manifests for the workflows you intend to run, so this check
# stays correct on a cluster whose pools were renamed or whose hardware differs.
REQUIRED_POOLS=$(python3 - airflow/dags/workflows/*/configs/*_manifest.yaml <<'PY'
import sys, yaml
pools = set()
for path in sys.argv[1:]:
    try:
        m = yaml.safe_load(open(path)) or {}
    except Exception:
        continue
    for prof in (m.get("deployment", {}).get("profiles", {}) or {}).values():
        if prof.get("pool"):
            pools.add(prof["pool"])
# External-mode augmentation pools are set in DAG wiring, not the manifests; internal mode runs on
# default_pool. Include the external ones the chart creates so a mode switch doesn't hit a missing pool.
pools |= {"default_pool", "external_image_edit_service_pool",
          "external_image2video_service_pool", "cosmos_transfer_3_service_pool"}
print(" ".join(sorted(pools)))
PY
)
echo "manifest-derived pools: $REQUIRED_POOLS"
POOLS_JSON=$(curl -s -H "Authorization: Bearer $TOKEN" "$AIRFLOW_URL/api/v2/pools")
REQUIRED_POOLS="$REQUIRED_POOLS" python3 -c "
import sys,json,os
required = tuple(os.environ['REQUIRED_POOLS'].split())
pools = {p['name']: p for p in json.load(sys.stdin).get('pools',[])}
for n in required:
    p = pools.get(n)
    print(n, '- OK slots:', p['slots'] if p else 'MISSING')
" <<< "$POOLS_JSON"

# 3. The DAG(s) you intend to run are loaded and unpaused
IAA_DAG_JSON=$(curl -s -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/image_attribute_augmentation_dag_k8s")
python3 -c "import sys,json; d=json.load(sys.stdin); print('is_paused:', d.get('is_paused'), '| found:', 'dag_id' in d)" <<< "$IAA_DAG_JSON"
EVG_DAG_JSON=$(curl -s -H "Authorization: Bearer $TOKEN" \
  "$AIRFLOW_URL/api/v2/dags/event_video_generation_dag_k8s")
python3 -c "import sys,json; d=json.load(sys.stdin); print('is_paused:', d.get('is_paused'), '| found:', 'dag_id' in d)" <<< "$EVG_DAG_JSON"

# 4. Multistorage config secret exists
if kubectl get secret -n sdg-workflow multistorageclient-configuration-secret -o name; then
  echo "multistorage config secret exists"
else
  echo "multistorage config secret missing or inaccessible; controller readiness unverified"
fi
```

Return `controller readiness: unverified` unless these were checked. For a newly deployed
controller, run all four checks above before reporting ready.

## Connect to the deployed controller

After `make install sdg-controller` succeeds, establish the `AIRFLOW_URL`. The ClusterIP is
routable from the host machine on some clusters (e.g. when the agent host sits on the cluster's
pod network) but not on others — a `curl` to it may simply hang and time out with no error. Try it
first since it needs no port-forward, but do not assume it will work; if it doesn't connect within
a few seconds, fall back to `make port-forward` and use `localhost:8080` instead (see below):

```bash
AIRFLOW_URL="http://$(kubectl get svc -n sdg-workflow \
  sdg-workflow-controller-api-server \
  -o jsonpath='{.spec.clusterIP}'):8080"
echo "AIRFLOW_URL=$AIRFLOW_URL"
```

Then obtain a JWT token. Credentials are in `deploy/values.yaml` under
`airflow.createUserJob.defaultUser` (default `admin`/`admin` — change before production use). Note
the path is `createUserJob`, not `webserver`, which does not exist in this chart:

```bash
AUTH_RESPONSE=$(curl -s -X POST "$AIRFLOW_URL/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin"}')
TOKEN=$(AUTH_RESPONSE="$AUTH_RESPONSE" python3 -c "import sys,json,os; print(json.loads(os.environ['AUTH_RESPONSE'])['access_token'])")

test -n "$TOKEN" && echo "auth OK" || echo "auth FAILED"
```

To also expose the UI in a browser from another machine, start a port-forward. It binds
`0.0.0.0:8080` on the host, so the UI is reachable at the host's own address on port 8080:

```bash
make port-forward   # blocks until interrupted — run it in a terminal you own
HOST_IP=$(hostname -I | awk '{print $1}')
echo "Airflow UI: http://$HOST_IP:8080"
```

The Kubernetes ClusterIP and the host's own network address are separate address spaces. The
ClusterIP is reachable from the host but is not externally routable; the host address via
port-forward is what a browser on another machine should use. Resolve both at runtime — never
assume or hard-code either.

`make port-forward` never exits. The agent may start it as a background job using the harness's
native background-job mechanism (not a raw shell `&`) to verify connectivity or serve a short-lived
need — this keeps the shell responsive for follow-up commands. Tell the user it will stop when the
agent session ends, and prefer a terminal the user owns for anything that must persist beyond this
conversation. Before starting a new forward, check for and clean up any stray prior
`make port-forward` / `kubectl port-forward ... 8080` processes so they don't compete for the port:

```bash
ps -ef | grep "port-forward" | grep -v grep
kill <pid>   # or kill -9 if it doesn't respond
```

Verify with a bounded probe against both addresses:

```bash
curl -s -o /dev/null -w "%{http_code}\n" --max-time 5 http://localhost:8080
curl -s -o /dev/null -w "%{http_code}\n" --max-time 5 http://<host-ip>:8080
```

To update DAGs or plugins after the initial install, DAGs/plugins are baked into the image, so
rebuild and redeploy — there is no separate sync step:

```bash
make build sdg-controller     # rebuilds and pushes the image with current DAGs/plugins
make install sdg-controller   # redeploys the controller with that image
```

## Handoff to a workflow

Produce a deployment report containing the preflight report used, installation-readiness result,
topology, commands executed, controller image tag, Airflow URL, controller checks, and unresolved
deployment warnings. If the original request also asks to run a workflow, continue with that
workflow's own skill only after the post-install checks pass. Do not submit a workflow solely
because Helm installation succeeded.
