# Deploy the controller

Start only after `$cdf-orchestration-preflight` reports `install_readiness: READY` for the same
checkout and target environment. Preflight owns tool, cluster, GPU, storage/NFS, image-registry,
credential, OSMO, and workflow-mode checks; do not repeat them here.

Confirm which deployment steps are needed, present the exact substituted commands, and obtain
explicit approval. Use this sequence only for the build/push path:

> "I will run the following commands in order:
> 1. `make setup` — validates secrets from `secrets.env` and generates the Helm values for install
> 2. `make build sdg-controller` — builds the Airflow image with DAGs and plugins baked in, pushes
>    it to the NGC registry, and writes the published repository/tag into `deploy/values.yaml`
> 3. `make install sdg-controller` — installs/upgrades the Helm release using that image
>
> Proceed?"

For an image verified by `--prebuilt-controller-image`, use a separate approval sequence containing
`make setup`, creation of a temporary image-adjusted values file, and
`make install sdg-controller VALUES_DEV=<TEMPORARY_VALUES>`. Include the same namespace-wide
deletion warning in the approval prompt. State that no build or push occurs, uninstall can remove Helm-managed PVCs,
and the temporary file is removed only after post-install checks. Preserve needed data before reinstall.

There are exactly three controller install-related targets: `make setup`,
`make build sdg-controller`, and `make install sdg-controller`. There is no bare `make install`.

If preflight verified an exact prebuilt controller image, skip `make build sdg-controller`. Copy
`deploy/values.yaml` to a temporary file, set `airflow.defaultAirflowRepository` and
`airflow.defaultAirflowTag` there, and pass it as `VALUES_DEV` to `make install sdg-controller`.
Passing `AIRFLOW_IMAGE_REPOSITORY`/`AIRFLOW_IMAGE_TAG` directly to the install target does not
select the deployed image; those variables are consumed only by the build target.

**Always run `make setup` first** on any deploy, install, or redeploy request — even if a previous
run already generated the Helm values. Secrets rotate, but `make setup` regenerates `secrets.env`
and overwrites `deploy/secret_values_dev.yaml`: unmanaged variables are removed and the
`global.secrets.aws` block is reset. If the user keeps custom values in either file, ask them to
back it up before running setup. Only skip setup mid-session when the agent itself just ran it
moments earlier.

**Missing namespace** — if `kubectl get ns sdg-workflow` returns `NotFound`, this is a normal
first-install condition, not an error to diagnose. Build first only on the build/push path when no
image has been published for the target tag. A verified prebuilt image proceeds directly to the
prebuilt install sequence after approval.

**DAGs and plugins are baked into the image, not synced at runtime.** `make install sdg-controller`
only deploys whatever image tag `deploy/values.yaml` currently points at — it does not rebuild or
republish it. On the build/push path, an unpublished tag or changed DAG/plugin source requires
`make build sdg-controller` before install. On the prebuilt path, never rebuild or push the selected
image; if the DAG processor finds zero or stale DAGs, report the supplied image as unsuitable and
stop. Always confirm `GET /api/v2/dags` reports the expected `total_entries` after install — do not
assume a green Helm status means DAGs loaded.

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
| `AWS_S3_BUCKET` | Bucket used for workflow input and output |
| `AWS_S3_REGION` | Region for the S3 bucket |
| `AWS_S3_ACCESS_KEY_ID` | S3 access key |
| `AWS_S3_SECRET_ACCESS_KEY` | S3 secret key |

These are the variables `make setup` validates. The helper uses the four unified `AWS_S3_*`
variables above to generate one bucket profile. It does not read per-purpose bucket variables;
DAGs are baked into the image. For separate buckets, follow
[Separate input and output buckets](controller-connection.md#separate-input-and-output-buckets).

**Optional, but silently empty when omitted:**

| Variable | Purpose |
|---|---|
| `OSMO_TOKEN` | OSMO API credential, surfaced to DAGs as the `AIRFLOW_VAR_OSMO_TOKEN` env var from the chart-managed `osmo-token-secret` (`deploy/values.yaml`) |
| `OSMO_POOL` | OSMO compute pool the OSMO DAGs submit to (`AIRFLOW_VAR_OSMO_POOL`, same secret). OSMO tasks fail at run time with a clear error while it is empty |
| `OSMO_URL` | OSMO service URL the token logs in to (`AIRFLOW_VAR_OSMO_URL`, same secret). OSMO tasks with a token fail at run time with a clear error while it is empty |
| `OSMO_HF_CACHE_SEED_DIR` | Pre-populated Hugging Face cache directory mounted on the OSMO pool (`AIRFLOW_VAR_OSMO_HF_CACHE_SEED_DIR`, same secret). Empty makes OSMO video augmentation download models with `HF_TOKEN` |

`make setup` does **not** validate the OSMO settings. It reports "All required secrets are set"
without them and writes empty `global.secrets.osmo.*` values into the generated Helm values, so the
install succeeds and the OSMO DAGs register normally; the gap only surfaces at run time. That is
the intended behaviour — a Kubernetes-only deployment does not need them, and OSMO manifests ship
with the repository regardless of what this user plans to run, so do **not** prompt for them
during setup. The workflow skills check them when an OSMO run is actually requested.

If a user does want them, they belong in `secrets.env` (already listed in `secrets.env.example` as
optional), followed by `make setup` and a redeploy. Never write them into
`deploy/secret_values_dev.yaml` — that file is generated and the next `make setup` overwrites it.

`KUBECONFIG` is **not** one of them and is not in `secrets.env.example`. `make setup` reads it
from the environment, prompts for a path when it is unset and the terminal is interactive, and
fails when it is unset non-interactively. Export the path the user gave you before running:

```bash
export KUBECONFIG=/path/the/user/gave
```

`make setup` then writes that path into `secrets.env` for later `make build sdg-controller` /
`make install sdg-controller`.

`make setup` does **not** install Kubernetes, Docker, CLI tools, or system software. Their presence
and versions are established by preflight.

## `make build sdg-controller`

This target builds `docker/Dockerfile.airflow` (which `COPY`s `airflow/dags` and `airflow/plugins`
into the image, plus installs the locked `airflow-runtime` Python dependencies), pushes the image
to the NGC registry (`AIRFLOW_IMAGE_REPOSITORY:AIRFLOW_IMAGE_TAG`; the repository default
`nvcr.io/<your-org>/<your-team>/sdg-workflow-airflow` is a placeholder the target refuses to
build until the user replaces it with an NGC org/team path they can push; set both values in
`make/common.mk` as the [getting-started guide](../../../docs/getting-started.md#start-airflow-and-run-workflows)
describes, or export them for a single shell), then rewrites
`deploy/values.yaml`'s `defaultAirflowRepository`/`defaultAirflowTag` to match. It requires
`NGC_API_KEY` (from `secrets.env`) and local `docker`. It does **not** touch S3 or Kubernetes at
all — it only builds and publishes an image.

Run this whenever DAG, plugin, or `airflow-runtime` dependency source has changed since the last
build, and always on a true first install (no image has ever been published for this
repository/tag). `make install sdg-controller` will happily redeploy a stale or DAG-less image
without complaint, so treat "did DAGs actually change" as the trigger, not "did the install fail."

## `make install sdg-controller`

This target performs three steps: `helm dependency update`, uninstalls any prior Helm release
(`make uninstall`, which deletes all cronjobs, jobs, statefulsets, deployments, replicasets,
services, and pods in the target namespace, including unrelated workloads, and can remove Helm-managed PVCs),
then `helm upgrade --install`
into namespace `sdg-workflow` with the image from `deploy/values.yaml`, with a 10-minute timeout.
It does **not** build, push, or upload anything — it only deploys whatever image tag
`deploy/values.yaml` already points at (see `make build sdg-controller` above).

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
| Helm `failed` with `context canceled`, pods `Running` | Inspect `helm status`, `helm history`, and controller readiness first. When an upgrade is appropriate, use the [in-place Helm procedure](#in-place-controller-recovery) with the intended chart and values. `make install` invokes namespace-wide cleanup; reserve full reinstall for an agreed maintenance window. |
| Any PVC `Pending` | Capacity or provisioner problem — check `get events`; pods alone will look healthy |
| Pods `ImagePullBackOff` | Registry secret missing or NGC key invalid |

The Helm chart does not create the `shared-ffmpeg` claim that the Autonomous Vehicle
Dataset Generation workflow mounts read-only at `/opt/ffmpeg`. Its absence does not affect
controller health. If the user will run that workflow, check `kubectl get pvc
shared-ffmpeg -n sdg-workflow`. A missing claim is a workflow prerequisite: report it and
hand off to the [preflight skill](../../cdf-orchestration-preflight/SKILL.md). The operator fills
it once, following the
[FFmpeg sidecar prerequisite](../../../docs/autonomous-vehicle-dataset-generation/prerequisites.md#shared-ffmpeg-installation).
Never bake FFmpeg or PyAV into the Airflow image instead; the repository must not distribute FFmpeg.

The install takes roughly 2–5 minutes on a warm cache and longer on a first install. Report a
status line as each of the three `make install sdg-controller` steps completes (and separately for
`make build sdg-controller`'s three steps, when run) so the user never has to ask whether it is
still running.

Key points to state to the user before running `make install sdg-controller`:

- It does **not** touch S3, build images, or push anything — it only deploys the selected values
  file. On the build/push path, build first whenever that image may be stale or DAG-less. On the
  prebuilt path, stop and report an unsuitable image instead of rebuilding it.
- It first uninstalls the release and deletes all cronjobs, jobs, statefulsets, deployments,
  replicasets, services, and pods in the target namespace, including unrelated workloads. Most
  workload deletions are forced. Uninstall can remove Helm-managed PVCs, including model-cache
  and log claims. Preserve needed data before reinstall; include this warning in the install approval prompt.
- Container images are pulled from NGC using the `ngc-docker-registry-secret` the Helm chart
  manages.

**First-install PVC conflicts are common, not exceptional.** A namespace that already has PVCs from
a prior partial install/teardown — `ngc-model-cache`, `data-<release>-postgresql-0`, or similar —
but no Helm release owning them causes the install to abort with an "invalid ownership metadata"
error, or in Postgres's case, to install "successfully" against a data
volume with mismatched credentials or a half-initialized database. Check `kubectl get pvc -n
sdg-workflow` for anything older than the current install attempt before troubleshooting deeper,
and confirm with the user before deleting any of them (they may hold cached model weights or prior
run history).

**Postgres can crash-loop on first boot under NFS-backed storage.** The bundled Postgres chart's
default liveness probe (`initialDelaySeconds: 30`, `failureThreshold: 6` — ~90s budget) can be
shorter than `initdb` + the custom `airflow`/`sdg_workflow` database-creation script take to finish
on NFS. If it fires mid-init, Postgres is killed, restarts, and — because it now considers its data
directory already initialized — never retries the interrupted step (creating the database, or
appending the `pg_hba.conf` host-access rule), so every dependent pod stays stuck in `Init:0/1`
forever with `no pg_hba.conf entry` or `database ... does not exist` in the migration job's logs.
Confirm from `kubectl get pods -n sdg-workflow` whether `*-postgresql-0` restarted before its
`READY` state stabilized; if so, delete the stale `data-*-postgresql-0` PVC and retry rather than
re-running the install repeatedly hoping to win the timing race. If it recurs, raise
`postgresql.primary.livenessProbe`/`readinessProbe` `initialDelaySeconds`/`failureThreshold` in
`deploy/values.yaml` (confirm with the user first — it's a checked-in file) rather than retrying
indefinitely.

**Pre-existing unmanaged resources:** resources outside the Helm release can remain after uninstall. If the namespace contains
PVCs, Secrets, or other resources not created by a prior Helm release, Helm may refuse to install.
Delete them first only after identifying ownership, explaining the data-loss risk, and obtaining
approval.

Return to `SKILL.md` for controller connection and post-install verification. Helm success alone
does not establish controller readiness.

## In-place controller recovery

For an existing release, inspect Helm status/history and controller events before retrying an
interrupted deployment. Once the cause is understood and an upgrade is appropriate, confirm the
intended chart, image, values, and controller rollout with the user. From the repository root:

```bash
helm --kubeconfig "$KUBECONFIG" upgrade sdg-workflow-controller deploy \
  --namespace sdg-workflow \
  --values deploy/values.yaml \
  --values deploy/secret_values_dev.yaml \
  --wait --timeout 10m
```

This skips the namespace cleanup performed by `make install`; controller workloads may still
restart. Use the existing deployment's values, including any overrides. If the generated secret
values file is missing, restore it through the approved setup procedure before proceeding.
Verify controller readiness and DAG registration after the upgrade. A full reinstall requires an
agreed maintenance window because it deletes workloads throughout the namespace.
