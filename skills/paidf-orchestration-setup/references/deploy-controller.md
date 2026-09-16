# Deploy a controller on the current cluster

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

## `make setup`

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

## `make install sdg-controller`

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

## Storage

See [Storage](topologies.md#storage) in topologies.md for the `nfs` StorageClass
requirement, `make install nfs`, and model-cache PVC behavior.
