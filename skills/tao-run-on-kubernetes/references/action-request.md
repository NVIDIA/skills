# Producer Action Requests

Read this reference before consuming a platform-neutral action request that
contains a `mounts` array. The simple one-root spec-bundle template cannot
preserve duplicate-source aliases or per-target access modes.

The renderer accepts action-request schema version `"1"` only and fails closed
on a missing or unsupported `request.schema_version`; update the renderer and
this contract together before consuming a future version.

## Materialize config-mode bundles, then stage every source

For `spec_bundle.mode=config`, the producer supplies a nested `spec` as
content, not a launcher-local path. The Kubernetes consumer must first
serialize that content and later replace the command's `{config_path}` with
the file's in-container path:

```bash
CONFIG_RENDER_ARGS=()
if [ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["spec_bundle"]["mode"])' \
    "$ACTION_REQUEST")" = "config" ]; then
  CONFIG_SOURCE=$(python3 \
    "$BANK/skills/platform/tao-run-on-kubernetes/scripts/render_action_job.py" \
    materialize-config --request "$ACTION_REQUEST" \
    --output-dir "$WORKSPACE/.tao/action-configs")
  CONFIG_RENDER_ARGS+=(--config-source "$CONFIG_SOURCE")
fi
```

The materializer supports the contract's `yaml`, `json`, and `toml` formats.
It writes an atomic, content-addressed file and returns its absolute path.
Treat that returned path as one additional consumer-derived staging source:
copy the exact file onto the PVC and add exactly one row for it to the staging
receipt. Do not rewrite the producer bundle as `mode=args`, hand-author a
second config, or substitute a launcher path that the pod cannot see.

Use `tao-data-io` when the producer source is not already visible from the
Kubernetes compute frame. Mirror each distinct `request.mounts[].source` into
a job-owned relative path on one bound PVC. If `CONFIG_SOURCE` is set, mirror
that exact file too. Use delete semantics when the producer requires remote
freshness. Record exactly one row per distinct source, including the generated
config when present:

```json
{
  "schema_version": "1",
  "sources": [
    {"source": "/launcher/run/results", "sub_path": "jobs/action-123/results"},
    {"source": "/launcher/run/config", "sub_path": "jobs/action-123/results/config"},
    {"source": "/launcher/cache", "sub_path": "jobs/action-123/cache"},
    {"source": "/launcher/.tao/action-configs/tao-action-config-<sha256>.yaml", "sub_path": "jobs/action-123/action-config.yaml"}
  ]
}
```

Omit the last row for `mode=args`. The staged subpaths must already exist.
Verify read access and required write
access from a pod before opening the job-record. For an application freshness
contract, also verify every declared absent path and persist its staging
receipt before opening the record. Never add an undeclared source or use an
incremental copy that can retain stale outputs.

## Record, credentials, and backend name

Open the job-record only after the staging/freshness gate. Job-record ids allow
characters and lengths that Kubernetes object names do not. Derive, never
hand-edit, the collision-resistant backend name; the manifest retains the
original record id in an annotation:

```bash
K8S_JOB_NAME=$(python3 \
  "$BANK/skills/platform/tao-run-on-kubernetes/scripts/render_action_job.py" \
  name --job-id "$JOB_ID")
```

When `request.forward_env` is non-empty, create `CRED_SECRET` after the record
is open and feed exactly those approved variables through an env-file on
stdin. For example, for a request that forwards these three names:

```bash
CRED_SECRET="${K8S_JOB_NAME}-creds"
set -a; source /path/to/.env; set +a   # omit if already exported
printf 'AWS_ACCESS_KEY_ID=%s\nAWS_SECRET_ACCESS_KEY=%s\nHF_TOKEN=%s\n' \
  "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$HF_TOKEN" \
  | kubectl create secret generic "$CRED_SECRET" -n "$NAMESPACE" \
      --from-env-file=/dev/stdin
```

The manifest projects each approved `forward_env` name from the same-named key
with an individual `env.valueFrom.secretKeyRef`; unrelated keys in the Secret
are not imported. Omit both the credential Secret and `--credential-secret`
when `forward_env` is empty (the renderer rejects that mismatch). Likewise,
pass an image-pull Secret only when the namespace needs one; do not create a
dependency on a placeholder Secret.

## Render and gate

The renderer validates the immutable command/image/GPU shape, exact source
map, unique targets, safe relative PVC subpaths, writable coverage for fresh
outputs, duplicate-source aliases, read-only flags, and conditional Secret
references. For `mode=config`, it also verifies that `CONFIG_SOURCE` is the
canonical serialization of `spec_bundle.spec`, mounts that exact staged file
read-only at a content-addressed path, and substitutes the path for every
`{config_path}` occurrence. A stale, altered, unstaged, or symlinked config is
rejected before a manifest is emitted.

Simple commands are emitted as native `command`/`args` arrays. A producer-owned
config command that is itself a multi-line shell script (for example, a
Cosmos-RL action that discovers its hook inside the image) is preserved
verbatim under `/bin/sh -c`; values from the config are never interpolated as
shell text:

```bash
RENDER_ARGS=()
if [ -n "${CRED_SECRET:-}" ]; then
  RENDER_ARGS+=(--credential-secret "$CRED_SECRET")
fi
if [ -n "${IMAGE_PULL_SECRET:-}" ]; then
  RENDER_ARGS+=(--image-pull-secret "$IMAGE_PULL_SECRET")
fi

python3 "$BANK/skills/platform/tao-run-on-kubernetes/scripts/render_action_job.py" \
  render --request "$ACTION_REQUEST" --staging-map "$STAGING_MAP" \
  --job-id "$JOB_ID" --namespace "$NAMESPACE" --pvc-claim "$PVC_CLAIM" \
  "${CONFIG_RENDER_ARGS[@]}" "${RENDER_ARGS[@]}" >"$MANIFEST"

"$BANK/scripts/redact_secrets.py" lint "$MANIFEST"
kubectl apply --dry-run=server -f "$MANIFEST"
```

The renderer uses `templates/k8s/action-job.yaml.tmpl`. Do not use the legacy
`single-pod-job.yaml.tmpl` for an action request with multiple mounts.

Apply and bind the real backend name to the record:

```bash
K8S_OBJECT=$(kubectl apply -f "$MANIFEST" -o name)
[ "$K8S_OBJECT" = "job.batch/$K8S_JOB_NAME" ] || {
  echo "unexpected Kubernetes object: $K8S_OBJECT" >&2
  exit 1
}
"$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state RUNNING \
  --backend-ref "$NAMESPACE/$K8S_JOB_NAME"
```

On reattach, recover namespace and backend name from that `backend_ref`; never
assume the Kubernetes object name equals the record id. Status, logs, and
cancel use the backend name. Delete `CRED_SECRET` after terminal log/output
collection, when it was created.
