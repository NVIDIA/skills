# Deployment

Use the same build artifacts and resolved Compose lifecycle for stock and delta
builds. Before resolving:

- confirm Docker, the NVIDIA runtime, and the requested GPUs are available;
- export `NGC_CLI_API_KEY` for local NVIDIA images/models;
- export the required API key for explicitly requested remote endpoints;
- set or confirm host paths and browser-reachable ingress values;
- export the selected common container image tag, when the build selected one;
- check the selected profile reference for stock-specific knobs and readiness.

## Resolve

Never copy or edit the Foundation files. Generate the exact deployment model
from the root Compose graph, optional changed-service patches, and four ordered
env layers:

```bash
REPO="$(git rev-parse --show-toplevel)"
BUILD_DIR="$REPO/_builds/<name>"
FOUNDATION="$(sed -n 's/^FOUNDATION=//p' "$BUILD_DIR/override.env")"
FOUNDATION_DIR="$REPO/deploy/docker/developer-profiles/dev-profile-$FOUNDATION"

if command -v uv >/dev/null 2>&1; then
  VSS_SKILL_PY=(uv run)
elif python3 - <<'PY' >/dev/null 2>&1
import yaml
PY
then
  VSS_SKILL_PY=(python3)
else
  echo "Install uv or install PyYAML for python3 before normalizing resolved.yml." >&2
  exit 1
fi

env_args=(
  --env-file "$REPO/deploy/docker/containers.env"
  --env-file "$FOUNDATION_DIR/.env"
  --env-file "$FOUNDATION_DIR/overrides.env"
  --env-file "$BUILD_DIR/override.env"
)

# containers.env derives every managed image tag from VSS_CONTAINER_TAG and is
# expanded before override.env, so only an exported value reaches those tags.
VSS_CONTAINER_TAG="${VSS_CONTAINER_TAG:-$(
  sed -n 's/^VSS_CONTAINER_TAG=//p' "$BUILD_DIR/override.env"
)}"
tag_args=()
if [ -n "$VSS_CONTAINER_TAG" ]; then
  export VSS_CONTAINER_TAG
  tag_args=(--expect-container-tag "$VSS_CONTAINER_TAG")
fi

docker compose "${env_args[@]}" \
  -f "$BUILD_DIR/compose.yml" \
  config --no-consistency > "$BUILD_DIR/resolved.yml"

"${VSS_SKILL_PY[@]}" "$REPO/skills/vss-build-vision-ai/scripts/normalize_resolved_yml.py" \
  "$BUILD_DIR/resolved.yml"

"${VSS_SKILL_PY[@]}" "$REPO/skills/vss-build-vision-ai/scripts/validate_resolved_yml.py" \
  "$BUILD_DIR/resolved.yml" --repo-root "$REPO" "${tag_args[@]}"
```

Write `resolved.yml` with the `>` redirect exactly as shown — see `composition.md`
for how to keep Compose's stderr out of the file. Act on that stderr rather than
silencing it: fix any error (non-zero exit) before deploying, and treat `variable
is not set` warnings as informational.

## Review and deploy

Validate and review the exact standalone file that will be deployed:

```bash
docker compose -f "$BUILD_DIR/resolved.yml" config --quiet
docker compose -f "$BUILD_DIR/resolved.yml" config --services
docker compose -f "$BUILD_DIR/resolved.yml" config --images
```

Confirm the resolved services, fully filled environment, images, GPU placement,
model endpoints, public ingress, checked-in bind sources, and requested
capability checks. Run the mandatory check/create gate in
[`data-directory.md`](data-directory.md), then deploy that exact file:

```bash
docker compose -f "$BUILD_DIR/resolved.yml" pull --ignore-buildable \
  && docker compose -f "$BUILD_DIR/resolved.yml" up -d --build
```

Wait for this command to finish and check its exit status before proceeding.
If the execution tool returns a running session or background task, continue
waiting on that same task until it completes; an image pull still in progress
is not a completed deployment. After successful bring-up, perform the readiness
checks below before ending the deployment task. On failure, report the failing
command and diagnostics.

Deploy with **only** `-f "$BUILD_DIR/resolved.yml"` (plus optional
`-p <project>` and `--build`). Do **not** pass `--env-file` — not even the
build's own `override.env` — and do **not** pass `--profile`: `resolved.yml` is
already self-contained, and re-reading any env file or supplying a profile flag
re-injects `COMPOSE_PROFILES`/`FOUNDATION` and breaks the runtime deploy
contract.

`COMPOSE_PROFILES` has already filtered the source graph during resolution, and
`docker compose config` baked the project `name`, each service `env_file`, and
all interpolation into the file. Normalization removes the remaining service
profile gates, so no Foundation env file or profile flag is needed at deployment
time. `pull --ignore-buildable` refreshes the registry-backed images even when a
tag already exists locally, while skipping the build-backed services — those
carry a local-only `image:` tag with no registry manifest, so a blanket
`--pull always` on `up` would abort with `manifest not found`. `up -d --build`
then builds those build-backed services from their local `build:` rather than
that bare `image:` tag, and starts everything.

## Readiness

First require a non-empty expected service list and acceptable container states:

```bash
resolved_args=(-f "$BUILD_DIR/resolved.yml")

expected="$(docker compose "${resolved_args[@]}" config --services | wc -l)"
actual="$(docker compose "${resolved_args[@]}" ps --all -q | wc -l)"
[ "$expected" -gt 0 ] && [ "$actual" -ge "$expected" ]

if docker compose "${resolved_args[@]}" ps --all --format json |
   jq -e 'select((.State == "running" or
                  (.State == "exited" and .ExitCode == 0)) | not)' >/dev/null
then
  echo "A service is not ready" >&2
  exit 1
fi
```

Then run the Foundation's stock readiness checks plus checks for every added
capability owner. Allow cold NIM and RTVI model loads to finish. If a check
fails, report the failing service and its recent logs; do not declare a partial
deployment successful.

Deployment and readiness bring the backends **up**; they register no source and
serve no query. A build holding no sources is finished, not half-finished:
neither path below runs unless the request asks for it, and provisioning a
source to prove the stack works is not a readiness check.

- **Write path (provisioning), when a source was requested.** For both Agent-backed
  and headless builds, follow `vss-manage-video-io-storage`
  [`provision-vios-source.md`](../../operations/vss-manage-video-io-storage/references/provision-vios-source.md)
  to register one VIOS source. The build's mounted notification config fans it
  out, asynchronously, so registration ends the write path.
- **Read path (query), when a query was requested.** Run `vss configure --base-url <build-origin>` (the fronting
  `http://$HOST_IP:$HAPROXY_HOST_PORT`) through the project-local `vss` entry point
  (`uv run --project <repo>/libs/vss vss`; see `deployment_resolution.md`),
  not a bare `vss`, to record the deployment, then defer to `vss-search-archive` for
  the query.

## Stop

A harnessed build starts with
[`teardown.md`](teardown.md#nemoclaw-harness--before-compose) instead — the
harness is outside the Compose project, and stopping the build first leaves it
pointed at a dead origin. That covers a build holding a `sandbox` file and an
older one that carries only `nemoclaw-setup.log`; that section resolves the
name from either.

Clean the complete Compose project (`COMPOSE_PROJECT_NAME`, default `vss`) and its named volumes by default:

```bash
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-vss}"
docker compose -p "${COMPOSE_PROJECT_NAME}" -f "$BUILD_DIR/resolved.yml" down -v --remove-orphans
```

This removes data volumes and model caches. Use the cache-preserving path only
when the user explicitly requests it. Follow [`teardown.md`](teardown.md) for
leftover containers, stale volumes, and bind-mounted data cleanup.

## Sources

- `deploy/docker/README.md`
- `deploy/docker/compose.yml`
- `deploy/docker/containers.env`
- `deploy/docker/developer-profiles/dev-profile-*/.env`
- `deploy/docker/developer-profiles/dev-profile-*/overrides.env`
