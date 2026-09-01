---
name: orchestration-setup
description: Audit, prepare, and deploy PAIDF Orchestration on a Kubernetes GPU cluster - single-GPU H100/L40S hosts, managed Kubernetes, kubeadm, and similar. Select for requests to set up, install, deploy, configure, or check a PAIDF Orchestration environment; run a workflow on a new or unverified GPU host; connect via kubeconfig; validate GPU compute; deploy the Airflow controller; or choose external versus in-cluster model services. A plain SSH host is not a supported backend.
---

# PAIDF Orchestration Environment Setup

Prepare a Kubernetes GPU environment for PAIDF Orchestration without assuming a cloud provider.
Treat Kubernetes—not the host vendor—as the integration contract.

## Safety boundary

Start with a read-only audit. Before any `prepare`, Helm install, Airflow connection change, or
remote mutation, summarize the exact changes and obtain user approval. Never print kubeconfig,
NGC keys, Hugging Face tokens, AWS secrets, or Kubernetes Secret bodies.

## Select two independent axes

1. Select controller placement: use an existing controller, or deploy Airflow into the current
   Kubernetes cluster.
2. Select model-service placement separately: use external VLM/LLM/image-edit endpoints, or deploy
   those services in-cluster. Never infer one choice from the other.
3. Prefer external endpoints on a one-GPU H100 node. The controller and augmentation worker may
   still run in that node's cluster.
4. Reject a Docker-only or SSH-only host until a supported Kubernetes distribution and NVIDIA
   device plugin expose `nvidia.com/gpu`.

Read [topologies.md](references/topologies.md) before changing infrastructure.

## Audit the compute cluster

The cluster is reached only through a kubeconfig the user supplies. It carries a cluster address
and admin credentials, so it is never part of the repository. Resolve it in this order:

1. Use `$KUBECONFIG` if it is already set in the environment.
2. Otherwise **ask the user for the path** and export it.

```bash
echo "${KUBECONFIG:-unset}"   # ask the user for a path when this is unset
export KUBECONFIG=/path/the/user/gave
```

Never guess a path, assume a repository-relative location, or fall back to `~/.kube/config`. If
the path the user names does not exist, say so and ask again.

Run locally when the agent already has the kubeconfig (`remote_k8s.py audit` has no
`--kubeconfig` flag; pass it via the env var):

```bash
python scripts/remote_k8s.py audit --service-mode external --json
```

Alternatively, pass it inline through `--kubectl-command`:

```bash
python scripts/remote_k8s.py audit \
  --kubectl-command "kubectl --kubeconfig $KUBECONFIG" \
  --service-mode external --json
```

Run through SSH when Kubernetes tooling exists only on the remote host:

```bash
python scripts/remote_k8s.py audit \
  --ssh-target ubuntu@host \
  --kubectl-command "kubectl" \
  --service-mode external --json
```

`remote_k8s.py` is bundled with this skill — run it from the skill directory, not the repository
`scripts/` directory. Use `--kubectl-command "k3s kubectl"` when appropriate. Do not pass SSH
passwords or private-key contents in the prompt; use SSH configuration or an agent.

Before a Helm install the audit exits non-zero with `ready: false` and blocker
`NGC image-pull secret is missing`. That is the expected first-install state, because the chart
creates that secret itself — read the `facts` block and continue. Do not resolve it with
`--create-registry-secret`, which makes the subsequent install fail on ownership metadata.

Interpret capacity conservatively:

- **Image Attribute Augmentation:** External endpoints deploy no in-cluster inference services, and
  the checked-in augmentation and attribute-search tasks use CPU profiles. Internal mode deploys
  VLM, LLM, and image-edit services, each claiming one GPU; require at least three allocatable GPUs
  for one replica of each, plus one for each additional replica.
- **Event Video Generation:** External endpoints deploy no in-cluster inference services, but
  detection/tracking, captioning, and visual-QA auto-labeling task pods each claim one GPU while
  active. Internal mode additionally needs one GPU per VLM and LLM replica plus two per image-to-video
  replica — at least four allocatable GPUs for one replica of each service.
- A single-GPU node (for example, one H100) can use external model endpoints, subject to Event Video
  Generation's GPU auto-labeling capacity.

**The compute cluster is shared.** Other users' DAG runs may be active in the same namespace.
Always report GPUs as free-versus-total (check running pods for GPU requests, not just node
allocatable), and never issue broad destructive commands (`delete pods --all`) against the compute
namespace without first checking pod ownership via `dag_id` and `run_id` labels.

## Deploy a controller on the current cluster

When the user requests setup (not audit-only), confirm which steps to run before executing
anything. Present the exact commands you plan to run and obtain explicit approval:

> "I will run the following commands in order:
> 1. `make setup` — validates secrets from `secrets.env` and generates the Helm values for install
> 2. `make install sdg-controller` — packages Airflow runtime dependencies, uploads DAGs and
>    plugins to S3, and installs/upgrades the Helm release
>
> Proceed?"

There are exactly two install-related targets: `make setup` and `make install sdg-controller`.
There is no bare `make install` and no `make install nfs` unless NFS storage is also needed (see
Storage below).

**Always run `make setup` first** on any deploy, install, or redeploy request — even if a previous
run already generated the Helm values. Secrets rotate; `make setup` is cheap and safe. Only skip it
mid-session when the agent itself just ran it moments earlier.

**Missing namespace** — if `kubectl get ns sdg-workflow` returns `NotFound`, this is a normal
first-install condition, not an error to diagnose. Route directly to `make install sdg-controller`.

### `make setup`

This is a lightweight secrets-validation step, not a full environment bootstrap. It:

- Reads `secrets.env` at the repository root (copy `secrets.env.example` and fill in values before
  running);
- Validates that `NGC_API_KEY`, `HF_TOKEN`, and all required AWS S3 credentials are present;
- Generates the Helm values used by the install and refreshes `secrets.env` with normalized values.

**Required variables in `secrets.env`:**

| Variable | Purpose |
|---|---|
| `NGC_API_KEY` | NGC image pulls and NVCF access |
| `HF_TOKEN` | HuggingFace model downloads (internal service mode) |
| `AWS_S3_BUCKET` | Shorthand; fills DAG, input, and output bucket settings |
| `AWS_S3_REGION` | Region for the S3 bucket |
| `AWS_S3_ACCESS_KEY_ID` | S3 access key |
| `AWS_S3_SECRET_ACCESS_KEY` | S3 secret key |

These are the variables `make setup` validates. `secrets.env.example` ships the unified
`AWS_S3_*` shorthand above, which fills the DAG, input, and output settings alike; set the
per-purpose `AWS_S3_DAG_*`, `AWS_S3_INPUT_*`, and `AWS_S3_OUTPUT_*` variables instead when those
differ.

`KUBECONFIG` is **not** one of them and is not in `secrets.env.example`. `make setup` reads it
from the environment, prompts for a path when it is unset and the terminal is interactive, and
fails when it is unset non-interactively. Export the path the user gave you before running:

```bash
export KUBECONFIG=/path/the/user/gave
```

`make setup` then writes that path into `secrets.env` for later `make install` / `make sync-dag`.

`make setup` does **not** install Kubernetes, Docker, CLI tools, or any system software. Ensure
`kubectl`, `helm`, `uv`, `docker`, and `aws` (v2) are already on PATH before running.

### `make install sdg-controller`

This target performs five steps: packages Airflow runtime Python dependencies from the locked
`airflow-runtime` group in `pyproject.toml`/`uv.lock`, uploads DAGs, plugins, and dependencies to
S3, runs `helm dependency update`, uninstalls any prior Helm release, then
`helm upgrade --install` into namespace `sdg-workflow` with a 10-minute timeout.

Before running, verify S3 connectivity (the install fails immediately at Step 2 if S3 is
unreachable):

```bash
source secrets.env
# Redirect output to /dev/null so head-pipe doesn't mask aws exit code
aws s3 ls "s3://$AWS_S3_DAG_BUCKET/" --region "$AWS_S3_DAG_REGION" >/dev/null 2>&1 \
  && echo "S3 OK" \
  || echo "S3 UNREACHABLE — check credentials and network before proceeding with make install sdg-controller"
```

**Run every S3-touching step in a shell that is fully outside the agent sandbox.** This applies to
`make setup`, `make install sdg-controller`, `make sync-dag`, and the pre-flight above. The agent
executes these itself — do **not** hand them to the user; automating this deployment is the whole
point of the skill.

The sandbox failure signature is DNS, not egress: the sandbox network namespace has no
`/etc/resolv.conf`, so nothing under `amazonaws.com` resolves — including the generic regional
endpoint, not just the bucket subdomain.

```
fatal error: Could not connect to the endpoint URL: "https://<bucket>.s3.<region>.amazonaws.com/..."
curl: (6) Could not resolve host: s3.<region>.amazonaws.com     # curl exits 6
getent hosts s3.<region>.amazonaws.com                          # returns nothing
```

On this signature, escalate to an unsandboxed shell and retry. Do not re-diagnose credentials,
bucket names, or regions — they are not the cause. Requesting broader *network* permission alone
does not help: the sandbox will report unrestricted network access while DNS still fails, because
the missing resolver is a property of the namespace rather than the egress policy.

After the install invocation returns — **whatever its exit status, including an interrupt or abort**
— always verify against cluster state, which is the only source of truth:

```bash
helm --kubeconfig "$KUBECONFIG" list -n sdg-workflow
kubectl --kubeconfig "$KUBECONFIG" get pods -n sdg-workflow
kubectl --kubeconfig "$KUBECONFIG" get pvc  -n sdg-workflow
kubectl --kubeconfig "$KUBECONFIG" get events -n sdg-workflow --sort-by=.lastTimestamp | tail -20
```

An interrupted invocation frequently still applies every change. Interpret the results as:

| Signal | Meaning |
|---|---|
| Helm `deployed`, all controller pods `Running` | Success — proceed |
| Helm `failed` with `context canceled`, pods `Running` | Interrupted mid-wait. Re-run `make install sdg-controller` to reconcile to `deployed`. Do **not** run `make uninstall` |
| Any PVC `Pending` | Capacity or provisioner problem — check `get events`; pods alone will look healthy |
| Pods `ImagePullBackOff` | Registry secret missing or NGC key invalid |

The install takes roughly 2–5 minutes on a warm cache and longer on a first install. Report a
status line as each of the five steps completes so the user never has to ask whether it is still
running.

Capacity pre-flight — the chart requests ~536Gi of PVCs by default, 500Gi of it `modelCache`. On a
single-node or small-disk host this claim can never bind, and because no controller pod mounts it
the whole deployment still looks healthy:

```bash
df -h /   # free space must exceed the sum of PVC requests in deploy/values.yaml
```

Lower `modelCache.size`, or set `modelCache.enabled: false` when every model service is external
and nothing downloads models in-cluster.

Key points to state to the user before running:

- It **does** upload Python packages and DAG files to S3, so AWS credentials and S3 connectivity
  must be in place.
- It **destroys** the running deployment first (Helm uninstall) before reinstalling.
- DAGs and plugins are pulled from S3 by the dag-synchronizer rather than mounted from the local
  working tree; run `make sync-dag` afterward whenever DAG files change.
- It does **not** build or push Docker images; container images are pulled from NGC.

**Pre-existing unmanaged resources:** the uninstall step preserves PVCs. If the namespace contains
PVCs, Secrets, or other resources not created by a prior Helm release, Helm may refuse to install.
Delete them first, confirming with the user because PVCs may hold cached model weights.

### Storage

See [Storage](references/topologies.md#storage) in topologies.md for the `nfs` StorageClass
requirement, `make install nfs`, and model-cache PVC behavior.

## Prepare missing cluster prerequisites

After explicit approval, create only the requested resources. `remote_k8s.py prepare` reads
`NGC_API_KEY` from the **environment** — use `set -a` to export variables from `secrets.env`
before running, otherwise `source` alone does not export them to child processes:

```bash
set -a && source secrets.env && set +a   # sets KUBECONFIG when `make setup` has already run
export KUBECONFIG=/path/the/user/gave    # otherwise set it explicitly, after the source above
python skills/orchestration-setup/scripts/remote_k8s.py prepare \
  --create-registry-secret
```

The secret is sent as a manifest over stdin; `NGC_API_KEY` never appears in command arguments or
output.

**Do not pre-create `ngc-docker-registry-secret` when you intend to run `make install
sdg-controller`.** The Helm chart manages that secret itself, and a manually created one has no
Helm ownership metadata, so the install aborts before deploying anything:

```
Error: unable to continue with install: Secret "ngc-docker-registry-secret" in namespace
"sdg-workflow" exists and cannot be imported into the current release: invalid ownership
metadata; label validation error: missing key "app.kubernetes.io/managed-by"...
```

Use `--create-registry-secret` only to validate NGC credentials against a cluster that will not be
Helm-managed. If the conflict occurs, delete the secret and let Helm recreate it:

```bash
kubectl --kubeconfig "$KUBECONFIG" delete secret ngc-docker-registry-secret -n sdg-workflow
```

For internal services, create the model-cache PVC only after selecting a valid storage class:

```bash
python scripts/remote_k8s.py prepare \
  --create-registry-secret \
  --create-model-cache-pvc \
  --storage-class nfs \
  --pvc-access-mode ReadWriteMany
```

`prepare` has no `--kubeconfig` flag (neither does `audit`); like `audit`, it relies on ambient
`kubectl` picking up `$KUBECONFIG` from the environment.

Do not install a GPU operator, device plugin, or Kubernetes distribution automatically. Report
those as infrastructure prerequisites, because the correct installation is provider- and
distro-specific.

## Connect the SDG controller

Read [controller-connection.md](references/controller-connection.md). After install, verify using
the Airflow token obtained in [Connect to the deployed controller](#connect-to-the-deployed-controller):

```bash
# 1. kubernetes_remote connection exists.
# It is injected as an env var, not stored in the metadata database, so
# GET /api/v2/connections/kubernetes_remote returns 404 on a healthy controller.
# Check the env var instead — a 404 here is not a failure.
kubectl exec -n sdg-workflow deploy/sdg-workflow-controller-scheduler -c scheduler -- \
  printenv AIRFLOW_CONN_KUBERNETES_REMOTE >/dev/null 2>&1 \
  && echo "kubernetes_remote: present" \
  || echo "kubernetes_remote: MISSING"

# 2. Required pools have slots. default_pool is Airflow's built-in pool (not chart-created);
# the rest come from deploy/values.yaml airflowPools.pools and are workflow-specific — include
# every workflow you intend to run, not just one.
POOLS_JSON=$(curl -s -H "Authorization: Bearer $TOKEN" "$AIRFLOW_URL/api/v2/pools")
python3 -c "
import sys,json
required = ('k8s_gpu_1','default_pool',
    'external_image_edit_service_pool','iaa_internal_image_edit_service_pool',  # image-attribute-augmentation-workflow
    'external_image2video_service_pool','internal_image2video_service_pool')   # event-video-generation-workflow
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
SECRET_JSON=$(kubectl get secret -n sdg-workflow multistorageclient-configuration-secret \
  -o jsonpath='{.data}' 2>/dev/null)
if [ -n "$SECRET_JSON" ]; then
  python3 -c "import sys,json; print('keys:', list(json.load(sys.stdin).keys()))" <<< "$SECRET_JSON"
else
  echo "multistorageclient-configuration-secret NOT FOUND"
fi
```

Return `controller readiness: unverified` unless these were checked. For a newly deployed
controller, run all four checks above before reporting ready.

## Connect to the deployed controller

After `make install sdg-controller` succeeds, establish the `AIRFLOW_URL`. The ClusterIP is
always routable from the host machine (even without port-forward) and is the most reliable choice
for agent use:

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
TOKEN=$(curl -s -X POST "$AIRFLOW_URL/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

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

To update DAGs or plugins after the initial install without reinstalling (the dag-synchronizer
picks up S3 changes within the configured interval, default 30 s):

```bash
make sync-dag
```

## Handoff to the augmentation run

Produce a readiness report containing topology, Kubernetes context, ready GPU count, service mode,
Airflow URL, missing resources, controller checks, and safe remediation. Before continuing to a
workflow run, present the planned install commands (`make setup`, `make install sdg-controller`)
and wait for explicit approval — even if controller pods appear healthy. If the original request
also asks to run a workflow, continue with that workflow's own skill procedure (for example
`image-attribute-augmentation-workflow` or `event-video-generation-workflow`) only after the user
approves or declines the install steps and compute and controller readiness are established; do
not ask the user to name or re-invoke another skill. Never submit a workflow solely because
`kubectl get nodes` succeeds.
