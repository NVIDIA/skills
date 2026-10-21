---
name: cdf-orchestration-preflight
description: Set up missing Linux prerequisites and audit whether a Cosmos Data Factory (CDF) Orchestration checkout and its Kubernetes or OSMO environment are ready to install the controller and run every packaged DAG. Use for prerequisite, pre-setup, Kubernetes, GPU, RWX/NFS, credential, controller-readiness, and quick-start checks. Not for deploying the controller (use cdf-orchestration-setup, which calls this audit first), authoring payloads, or running workflows.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  owner: NVIDIA
  service: cosmos-data-factory
  version: 1.1.2
  reviewed: '2026-10-09'
  tags:
    - physical-ai
    - cdf-orchestration
    - preflight
    - kubernetes
---

# Cosmos Data Factory Orchestration Preflight

Determine what the current environment can install and which packaged DAGs it can run. Report
partial readiness precisely; do not collapse a mixed environment into a single pass/fail result.

## Safety and scope

Start with a read-only audit. Never print kubeconfig content, API keys, tokens, AWS credentials,
Kubernetes Secret bodies, or raw command output that may contain them. Do not silently use
`~/.kube/config`: use the supplied `KUBECONFIG` path for an existing cluster. For a new cluster,
bootstrap creates the kubeconfig.

This skill may offer prerequisite remediation after the audit. Before changing a host, cluster, or
tracked configuration, show the exact target and commands and obtain approval. Stop after
prerequisites are ready: do not run `make setup`, build or push the Airflow image, install the
controller Helm release, or trigger a DAG. Those are deployment or workflow tasks.

## Set up missing prerequisites

For a fresh Linux environment, read [bootstrap.md](references/bootstrap.md). It covers local
tools, the Python environment, container runtime, Kubernetes, GPU/storage setup and secrets.
Reuse working prerequisites and follow the user's execution preference. Installing the Helm
client is prerequisite setup; deploying the controller belongs to the setup skill.

## Run the audit

Run the bundled script from the repository root:

```bash
python3 skills/cdf-orchestration-preflight/scripts/preflight.py audit \
  --repo-root "$PWD" \
  --kubeconfig "$KUBECONFIG"
```

Use `--format json` when a machine-readable report will help. The script is read-only. If
`KUBECONFIG` is unset, ask for its path when a cluster exists; otherwise follow bootstrap.
Omit `--kubeconfig` to run local checks before a cluster exists. `AIRFLOW_URL` and
`AIRFLOW_TOKEN` may be exported for authenticated post-install checks; never put the token on the
command line.

When the user supplies an already-published controller image, pass its exact registry-qualified
reference with `--prebuilt-controller-image REGISTRY/REPOSITORY:TAG`. This switches installation
readiness from build/push prerequisites to a read-only `docker manifest inspect` check. For a
private registry, authenticate Docker before the audit without exposing credentials; the audit
does not log in, pull the image, or persist registry credentials. A missing or inaccessible
manifest leaves `install_readiness` `UNVERIFIED`.

For a local build, the image location comes from `AIRFLOW_IMAGE_REPOSITORY` and `AIRFLOW_IMAGE_TAG`
in `make/common.mk`. The checked-in repository is the placeholder
`nvcr.io/<your-org>/<your-team>/sdg-workflow-airflow`, which blocks the audit and
`make build sdg-controller` until it is replaced. Follow the
[getting-started guide](../../docs/getting-started.md#start-airflow-and-run-workflows): ask for an
NGC org/team path that `NGC_API_KEY` can push and a unique tag, then, with approval, set both
`?=` values in `make/common.mk`. Tell the user not to commit that edit. Exported variables of the
same names override the file for one shell; prefer the file edit because it persists across shells
and agents. After the user confirms push permission with the build credentials, pass
`--confirmed-push-repository` with the configured repository. This records user confirmation for
that exact repository, not an audit-performed push test. Without confirmation, access remains
`UNVERIFIED`. The setup skill's build step reads the same values.

For a non-loopback kubeconfig API endpoint, the audit runs a bounded `/busybox/nc` TCP probe from
an existing ready NVIDIA GPU Operator toolkit pod. A successful connection is `READY`; a
conclusive connection failure is `BLOCKED`. A loopback endpoint is always `BLOCKED`. A missing
toolkit pod or probe tool, denied `pods/exec`, or an inconclusive execution remains `UNVERIFIED`.
The audit does not create a diagnostic pod.

With a host-installed toolkit, pass `--probe-pod NAMESPACE/POD/CONTAINER` for an existing Running,
ready, non-hostNetwork container with Python 3. The first selected container tests API TCP access;
selected containers requesting `nvidia.com/gpu` also test loading `libnvcuvid.so.1` on their nodes.
Repeat the option to cover each GPU node without a toolkit Pod. Invalid selections, missing Python
or denied exec remain `UNVERIFIED`. Bootstrap includes a temporary probe Pod example.

For every node that advertises NVIDIA GPU capacity, the audit verifies `libnvcuvid.so.1`
through that node's running NVIDIA GPU Operator toolkit pod. This uses a constant, read-only
`kubectl exec` probe and does not create cluster resources. A missing library blocks workflow modes
whose readiness profile requires GPU video decode. A missing or non-running toolkit pod, a
non-Ready GPU node, or denied `pods/exec` access leaves those modes `UNVERIFIED`; do not report the
library as absent without a conclusive probe. Explicit GPU probe containers take precedence over
toolkit probes on their nodes; loading failures block decode-dependent modes.

Some modes also require operator-provisioned PVCs in `sdg-workflow` that the Helm chart does not
create. The Autonomous Vehicle Dataset Generation modes require `shared-ffmpeg`, the
read-only FFmpeg sidecar for Curator and its media probe pods. A missing or unbound claim blocks
those modes, and the report points to the
[FFmpeg sidecar prerequisite](../../docs/autonomous-vehicle-dataset-generation/prerequisites.md#shared-ffmpeg-installation).
Filling the claim is the operator's step: they choose the FFmpeg build and provision it
according to those requirements. Creating or filling the claim changes the cluster;
perform those actions only when the user requests setup. The workflow verifies the claim's contents when
its tasks start; never propose bundling FFmpeg or PyAV into the Airflow image as a substitute.

Interpret the report as two independent gates:

- `install_readiness` covers the workstation, repository configuration, credentials, Kubernetes,
  and storage needed to install the controller.
- `run_readiness` covers the live controller and every packaged DAG. `PARTIAL` means at least one
  DAG is ready and at least one is blocked or unverified.

Read [readiness-model.md](references/readiness-model.md) when explaining workflow or service-mode
results, reconciling live Airflow state, or investigating an unprofiled DAG.

## Check payload storage mapping

Workflow skills run this before every trigger. It finds every storage URL in a payload, whatever
the field name, and checks that the deployed `multistorageclient-configuration-secret` maps it,
using the same longest-prefix rule as DAG tasks. It is read-only and prints bucket prefixes only,
never credentials:

```bash
python3 skills/cdf-orchestration-preflight/scripts/storage_check.py --payload <payload.json>
```

`Storage mapping: ready` means every bucket resolves. Each `UNMAPPED` line names a field whose
bucket has no mapping; the run would fail at runtime. The generated configuration maps only
`AWS_S3_BUCKET` from `secrets.env`; to use another bucket, update it there, rerun `make setup` and
`make install sdg-controller`, then rerun the check. Use `--namespace` for a non-default
controller namespace.

## Resolve findings

Prefer an existing compatible ReadWriteMany StorageClass. Read
[nfs-remediation.md](references/nfs-remediation.md) before recommending or running the repository's
host-based NFS installer. That installer is not a general managed-Kubernetes solution.

Treat `UNVERIFIED` as missing evidence, not success. Common examples are external inference
endpoints, Hugging Face license acceptance, registry push access, an inconclusive pod-to-API
probe, and NFS mount behavior. Ask for the smallest missing fact or offer a scoped test; do not
invent readiness.

Do not install Kubernetes, a GPU operator, device plugins, Docker, or system packages without approval.
For requested prerequisite setup, use [bootstrap.md](references/bootstrap.md) to select the
procedure for the Linux distribution and existing infrastructure, then verify the result.

## Quick start handoff

When install readiness is satisfied, return this sequence with the audit's concrete substitutions
and unresolved warnings. Do not execute it as part of this skill:

Carry forward the exact image and probe arguments (`CDF_IMAGE_ARGS`/`CDF_PROBE_ARGS` when using
bootstrap), plus any exported image settings. Reapply them to setup's fresh preflight and the
post-install audit. Keep temporary probes until those checks finish; if a probe has expired,
recreate it and update its argument using [bootstrap.md](references/bootstrap.md#verify-readiness).

```bash
cp secrets.env.example secrets.env       # only when secrets.env does not already exist
export KUBECONFIG=/path/the/user-supplied
# Build path: set AIRFLOW_IMAGE_REPOSITORY and AIRFLOW_IMAGE_TAG in make/common.mk to a
# writable registry path and unique tag (do not commit that edit).
make setup
make build sdg-controller
make install sdg-controller
make port-forward
```

For an image verified with `--prebuilt-controller-image`, omit `make build sdg-controller`. Create
a temporary copy of `deploy/values.yaml`, set its
`airflow.defaultAirflowRepository`/`defaultAirflowTag` to the verified reference, and pass that file
as `VALUES_DEV` to `make install sdg-controller`. Do not edit the generated secret-values file and
do not assume `AIRFLOW_IMAGE_REPOSITORY`/`AIRFLOW_IMAGE_TAG` affect the install target directly.

After installation, rerun this preflight with `AIRFLOW_URL` and `AIRFLOW_TOKEN` exported. Only
report all workflows ready when every packaged DAG has at least one verified runnable mode. Always
list unavailable alternative modes and unprofiled live DAGs.
