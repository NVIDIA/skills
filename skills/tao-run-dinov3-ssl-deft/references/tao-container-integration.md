# DINOv3 SSL DEFT container lifecycle

Data Services owns the controller and includes TAO PyTorch/Core. No host TAO
installation, nested Docker, Dockerfile change, or startup package replacement
is required by the single-container design.

## Release readiness

This application uses the release-managed `images.tao_toolkit.data_services`
image from `versions.yaml`, selected by `skill_info.yaml`. There is no separate
DEFT image or DINOv3-specific image key. Use the supported release checkout;
`main` is development source and its default image need not contain pending
changes. Resolve the image from that checkout, never copy a tag from prose.

The release team builds DS with the reviewed DS/PyTorch packages and updates
the release image pins. Before rollout, validate that exact image's installed
packages with workflow preflight, packaged regression tests and GPU smoke
tests. In particular, verify held-out benchmark exclusion, rejection before
materialization seals, and fresh-start training with auto-resume disabled.
An image built before the fixes must not be reported as containing them merely
because its tag is newer or because source-overlay tests pass. Record the
resolved digest and test results in the release evidence; do not patch packages
at container startup. A rebuild and pin update belong to the release handoff,
not to a parallel application-specific image lifecycle.

Source-overlay QA demonstrates source compatibility only. It is not evidence
that the published image contains these modules. The ANN path additionally
requires an approved CUDA-compatible cuVS dependency; the exact-search profile
does not. Never silently change an approved search backend.

## Docker: preparation and execution

Read the Docker platform skill for its host preflight, secure registry login,
job-record opening, UID/GID mapping, cache variables and GPU selection.
Resolve the approved application image on the host with:

```bash
DEFT_TAG=$(python scripts/resolve_tao_image.py --skill-bank . --application tao-run-dinov3-ssl-deft \
  --action run --format json | python -c 'import json, sys; print(json.load(sys.stdin)["image"])')
```

Run it from the Skill Bank checkout; without `--skill-bank` the resolver reads
`$TAO_SKILL_BANK_PATH` or `~/tao-skill-bank`, which can be a different bank.

Show `DEFT_TAG` to the user and accept an explicit override. Then pin the
digest, because a tag can move between preflight and launch:

```bash
docker pull "$DEFT_TAG"
DEFT_IMAGE=$(docker image inspect --format '{{index .RepoDigests 0}}' "$DEFT_TAG")
```

`RepoDigests` is empty for an image that was never pulled from a registry; stop
in that case. Every `docker run` below takes `$DEFT_IMAGE`. DS does not read
Skill Bank or `versions.yaml`.

The example below assumes the platform has prepared these reviewed bindings:
`DEFT_IMAGE` is the tested image digest, `DEFT_ROOT` contains config, data,
checkpoints and outputs, `DEFT_SCRATCH` is a writable node-local directory, and
`DEFT_GPU` is the allocated host GPU identifier. The GRIT score action writes
its work files to `$TAO_LOCAL_SCRATCH`; without it the first score stage fails.
All paths in the YAML use the container's `/deft` namespace.
`DEFT_IDENTITY_ARGS` and `DEFT_CACHE_ARGS` are the Docker skill's non-root
identity and writable cache arguments. Do not inline credentials.

```bash
DEFT_MOUNTS=(-v "$DEFT_ROOT:/deft" -v "$DEFT_SCRATCH:/scratch")
DEFT_RUNTIME=(--gpus "device=$DEFT_GPU" --shm-size=8g -e CUDA_VISIBLE_DEVICES=0 -e TAO_LOCAL_SCRATCH=/scratch)
DEFT_COMMAND=(python -m nvidia_tao_ds.mining.dinov3.workflow)

docker run --rm "${DEFT_RUNTIME[@]}" "${DEFT_IDENTITY_ARGS[@]}" "${DEFT_CACHE_ARGS[@]}" "${DEFT_MOUNTS[@]}" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" preflight --gpu
docker run --rm "${DEFT_IDENTITY_ARGS[@]}" "${DEFT_CACHE_ARGS[@]}" "${DEFT_MOUNTS[@]}" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" init --recipe grit-score --output /deft/run.yaml
# Fill reviewed paths/resources and set execution.backend: local.
docker run --rm "${DEFT_IDENTITY_ARGS[@]}" "${DEFT_CACHE_ARGS[@]}" "${DEFT_MOUNTS[@]}" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" validate /deft/run.yaml
# With the config, preflight also checks each action's scratch, GPU count and,
# when execution.capabilities.gpu_faiss is set, GPU FAISS.
docker run --rm "${DEFT_RUNTIME[@]}" "${DEFT_IDENTITY_ARGS[@]}" "${DEFT_CACHE_ARGS[@]}" "${DEFT_MOUNTS[@]}" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" preflight /deft/run.yaml --gpu
docker run --rm "${DEFT_IDENTITY_ARGS[@]}" "${DEFT_CACHE_ARGS[@]}" "${DEFT_MOUNTS[@]}" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" plan /deft/run.yaml
```

Do not request approval until the config preflight passes: `validate` and
`plan` do not check the runtime allocation.

After approval, the Docker platform opens the job-record and binds
`output.run_dir` to its results directory. Its returned `JOB_ID` names the
detached controller container. Preserve the platform's record-then-launch order:

```bash
docker run -d --name "$JOB_ID" --label "tao-job=$JOB_ID" "${DEFT_RUNTIME[@]}" "${DEFT_IDENTITY_ARGS[@]}" "${DEFT_CACHE_ARGS[@]}" "${DEFT_MOUNTS[@]}" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" run /deft/run.yaml
docker exec "$JOB_ID" "${DEFT_COMMAND[@]}" status /deft/results
# Obtain CLIENT_JOB_ID from status, then:
docker exec "$JOB_ID" "${DEFT_COMMAND[@]}" logs /deft/results "$CLIENT_JOB_ID"
```

The status/logs commands are explicit `docker exec` operations against the
running controller, not separate submissions to the platform. Use the platform's
Docker inspect/log verbs as well: a persisted workflow state is not proof that
the container is alive. Cancel through `workflow cancel /deft/results` while
the container is alive; SIGTERM to its main process also records cancellation.

After exit, `docker exec` is unavailable. Inspect the retained container with
the platform verbs and inspect durable state using the same image and a
read-only mount (with separately writable temporary caches):

```bash
docker run --rm "${DEFT_IDENTITY_ARGS[@]}" -e USER=tao -e LOGNAME=tao -e HOME=/tmp -v "$DEFT_ROOT:/deft:ro" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" status /deft/results
docker run --rm "${DEFT_IDENTITY_ARGS[@]}" -e USER=tao -e LOGNAME=tao -e HOME=/tmp -v "$DEFT_ROOT:/deft:ro" "$DEFT_IMAGE" "${DEFT_COMMAND[@]}" logs /deft/results "$CLIENT_JOB_ID"
```

For a failed (not canceled) workflow, submit a new container/job-record with the
same image, mounts and resources and `resume /deft/run.yaml`. Do not restart the
old container concurrently. The application data stays in the durable mount;
removing an inspected terminal container does not remove those results.

## Allocation, storage and recovery

Use `execution.backend: local`, omit `container_images`, and pin
`CUDA_VISIBLE_DEVICES` to the assigned allocation. GPU-consuming actions must
request exactly that visible GPU count; the controller does not discover or
claim unallocated devices. The recipe's one-GPU `torch_exact` scoring avoids
a GPU FAISS dependency. Explicit `faiss_exact` must pass a GPU FAISS probe.

The run directory must support reliable POSIX `flock`, atomic rename and
`fsync` (including directories). Do not use CIFS with `nobrl`, or NFS mounts
without working cross-client locks. Inputs are immutable and mounted read-only
where practical; outputs and scratch are separate writable locations.

Cancellation is terminal; `resume` adopts failed/interrupted work, not a canceled
run. When a dead local session leader has live descendants, the runner reports
UNKNOWN and does not signal unverified process identities. Inspect the recorded
PID/start time, descendants and allocation on the host, stop only verified
survivors, then confirm termination before retrying. Never remove a lock file to
force recovery. Config or native action-code changes require a new run and the
explicit `adopt-training` handoff where appropriate.

## External execution boundary

The DS runtime retains its four-verb external adapter interface, but neither DS
nor Skill Bank ships a Slurm/Kubernetes/separate-image runner executable for
this application. That deployment is unsupported until a named, tested runner
is supplied and reviewed. Do not present a placeholder executable as a runnable
recipe. The interface and attestation requirements remain in
[adapter-contracts.md](adapter-contracts.md) for integrators.
