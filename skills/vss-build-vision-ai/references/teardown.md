# Tear down an existing VSS deployment

## Contents

- [NemoClaw harness](#nemoclaw-harness--before-compose)
- [Default teardown](#default-teardown--clean-project-volumes)
- [Cache-preserving teardown](#cache-preserving-teardown--explicit-opt-in)
- [Bind-mounted data cleanup](#bind-mounted-data-cleanup)

Always tear down **by project name**. Profiles default `COMPOSE_PROJECT_NAME`
to `vss`, and users may change it to run multiple stacks on one host. A plain
`docker compose down` leaves named volumes and the project network behind, so
target the selected project and pass `-v --remove-orphans`.

The default removes all project volumes, including model caches. Use the
cache-preserving path only when the user explicitly asks to keep model caches.

Ask the user to confirm before running either path, and say which one you are
about to run: on the default, that it discards the NIM/RTVI model caches and the
next deploy re-downloads them.

## NemoClaw harness — before Compose

Run this whenever the build was harnessed — it holds a `sandbox` file, or, from
before that file existed, a `nemoclaw-setup.log` naming the sandbox. Tearing
down the build covers its sandbox and relay without the user naming them. Run
it first, so the sandbox is not left pointed at an origin that has stopped
answering.

A build whose harness setup **failed** has the file too — the name is recorded
before the notebook runs, because onboarding happens in its section 3.1 and a
later section can still fail over a live sandbox. So run this section for a
failed build as well; skipping it is how a sandbox gets left behind.

`nemoclaw destroy` does not stop the two host processes the setup notebook
started. The dashboard-forward watchdog would see the forward die and run
`nemoclaw <sandbox> recover` against the sandbox being destroyed, and keep
retrying it afterwards, so stop it **before** the destroy. The dashboard relay,
left running, holds `NEMOCLAW_DASHBOARD_RELAY_PORT` (default `18790`), and the
next deploy's relay cell stops on it as a foreign listener; stop it after.

```bash
REPO="$(git rev-parse --show-toplevel)"
BUILD_DIR="$REPO/_builds/<name>"
if [ -s "$BUILD_DIR/sandbox" ]; then
  SANDBOX="$(cat "$BUILD_DIR/sandbox")"
else  # a build harnessed before the skill wrote the sandbox file
  SANDBOX="$(sed -n 's/^Sandbox: //p' "$BUILD_DIR/nemoclaw-setup.log" \
    2>/dev/null | tail -1)"
fi
if [ -z "$SANDBOX" ]; then
  # Nothing was named, so nothing was created. Exit 0: the Compose teardown
  # the user asked for still has to run.
  echo "no sandbox recorded in $BUILD_DIR; nothing to destroy" >&2
  exit 0
fi

# Match the exact --sandbox argument, never the port or a name prefix: other
# sandboxes' watchdogs and relays share the scripts and may sit on neighbouring
# ports. pkill -f compiles an extended regex. Escape the recorded name so
# vision.dev selects that sandbox's processes alone.
SANDBOX_RE="$(printf '%s' "$SANDBOX" | sed 's/[][(){}.*+?^$|\\]/\\&/g')"
stop_sandbox_process() {  # <script> <label>
  local pattern="$1 --sandbox ${SANDBOX_RE}( |$)"
  pkill -f -- "$pattern"
  for _ in $(seq 50); do pgrep -f -- "$pattern" >/dev/null || return 0; sleep 0.1; done
  pgrep -af -- "$pattern"
  echo "$2 for $SANDBOX still running" >&2
  exit 1
}

stop_sandbox_process 'dashboard-forward-watchdog\.py' "dashboard-forward watchdog"
nemoclaw "$SANDBOX" destroy --yes --cleanup-gateway
stop_sandbox_process 'dashboard-relay\.py' "dashboard relay"
```

A name recorded by a run that failed *before* onboarding belongs to a sandbox
that was never created, so `destroy` reports it as not found. That is the one
non-zero here to accept and move on from; any other failure is a blocker.

A setup that failed even earlier records no name at all — an older build's log
carries the `Sandbox:` line only once the notebook has printed it. Report the
build as carrying no recorded sandbox and continue with the Compose teardown;
never leave the requested teardown unfinished over a sandbox that was never
created.

The name comes from that file and nowhere else: `nemoclaw list` names no
Compose project and the sandbox carries no project label, so neither the build
directory nor the `vss-harness-sandbox` default identifies the sandbox this
build owns.

So a sandbox no build recorded — one a user onboarded by running
`deploy_nemoclaw.ipynb` themselves, or one left by a build directory since
deleted — survives every teardown here, along with its watchdog and relay.
Nothing above goes looking for it, and the next deploy takes nothing from it
unasked: onboard stops or moves the sandbox elsewhere when it holds `18789`,
and the relay cell stops on a foreign listener on `18790`. The next harness bring-up is where that gets
settled, under [Ports the harness
claims](agent-harness.md#ports-the-harness-claims) — which reads ownership off
this same per-build record, so a sandbox named `vss-harness-sandbox` that the
next build's own record does not name is one it reports and asks about rather
than recreates.
Never destroy a sandbox this build does not own.

A harness built from a [harness source
ref](agent-harness.md#harness-source-ref) also left a detached worktree under
the build directory. Remove it here: deleting `_builds/<name>/` alone leaves
the worktree registered in the checkout.

```bash
if [ -d "$BUILD_DIR/harness-src" ]; then
  git -C "$REPO" worktree remove --force "$BUILD_DIR/harness-src"
fi
```

## Default teardown — clean project volumes

Removes containers, the project network, **and all named volumes** (including
multi-GB NIM/RTVI model caches).

```bash
REPO="$(git rev-parse --show-toplevel)"
BUILD_DIR="$REPO/_builds/<name>"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-vss}"
docker compose -p "${COMPOSE_PROJECT_NAME}" -f "$BUILD_DIR/resolved.yml" \
  down -v --remove-orphans

# Remove same-project volumes left by an older resolved model.
docker volume ls -q \
  --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}" \
  | xargs -r docker volume rm
```

- **`-p "${COMPOSE_PROJECT_NAME}"`** targets the selected VSS project. `--remove-orphans` also removes
  same-project containers omitted from the current `resolved.yml`.
- **`-f "$BUILD_DIR/resolved.yml"`** supplies the exact deployed Compose model;
  project name alone is not a Compose configuration.
- **`-v`** removes named volumes — without it ES / Kafka / Postgres / Milvus data
  **and** NIM/RTVI model caches all survive.
- **`--remove-orphans`** frees the project network from leftover or host-networked
  containers so the network is deleted too.
- The label-filtered sweep removes only volumes owned by the selected Compose
  project; never sweep every dangling volume on the host.

`-v` drops NIM/RTVI model caches (multi-GB re-download next deploy). To keep them
for an immediate redeploy or profile switch, use the cache-preserving teardown below.

`-v` removes docker **volumes**, but the bind-mounted **on-disk data dirs**
(ES/Kafka/Redis data, behavior-learning, VST/nvstreamer recordings) live on the host
filesystem and survive any teardown — they poison the next run if left. After
**either** flavor, also clear them with the sudo-gated
[bind-mounted data cleanup](#bind-mounted-data-cleanup) below.

## Cache-preserving teardown — explicit opt-in

Removes containers, the project network, and *stale data* volumes (ES indices,
Kafka offsets, Postgres, nvstreamer recordings) but **keeps** model caches so the
next deploy doesn't re-download them. Do not select this path unless the user
explicitly requests cache preservation.

### Tear down while preserving model caches

When cache preservation was explicitly requested, still stop every prior VSS
stack, especially when switching profiles (`base` → `search`, alerts
verification → alerts real-time). Compose profile flags only start selected
services; they do not stop services from a previous deployment.

```bash
# Tear down by the selected project name. This catches every
# same-project container/network regardless of which resolved.yml is on disk.
# NO -v here — the cache-preserving
# path keeps NIM/RTVI model caches; stale DATA volumes are removed explicitly
# below. --remove-orphans frees + deletes the project network.
REPO="$(git rev-parse --show-toplevel)"
BUILD_DIR="$REPO/_builds/<name>"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-vss}"
docker compose -p "${COMPOSE_PROJECT_NAME}" -f "$BUILD_DIR/resolved.yml" \
  down --remove-orphans

# Catch-all: remove every VSS-stack container the dev-profile compose
# files bring up. Without this, leftovers from a prior deploy linger
# (especially the *-smc set, which the alerts compose profile shares
# with the *-dev set on host networking and port 30000) and either:
#   - bind ports the new deploy needs → second sensor-ms fails to bind
#     → /sensor/list returns 502 (issue #151), or
#   - pass the new deploy's container-name health checks while serving
#     stale data from the prior deploy's DB.
# The patterns below cover everything declared under
# deploy/docker/services/ (agent, vios, rtvi, infra, nim, video-summarization, …)
# and deploy/docker/developer-profiles/dev-profile-*/compose files.
docker ps -a --format '{{.Names}}' \
  | grep -E '^(vss-|mdx-|perception-|rtvi-|alert-|nvstreamer-|sensor-ms-|vst-ingress-|vst-mcp-|vst-file-proxy|centralizedb-|storage-ms-|streamprocessing-ms-|sdr-(http|streamprocessing)-|envoy-(http|streamprocessing)-|rtspserver-ms-|recorder-ms-|replaystream-ms-|livestream-ms-|metropolis-vss-ui|phoenix)' \
  | xargs -r docker rm -f

# `down --remove-orphans` already deletes the project network (${COMPOSE_PROJECT_NAME}_default).
# Remove it explicitly only as a belt-and-suspenders, by EXACT name — `-f name=...`
# is a substring match and could catch unrelated networks.
docker network rm "${COMPOSE_PROJECT_NAME}_default" 2>/dev/null || true

# `down` (no -v) also leaves every named volume. Remove the stale DATA volumes
# that poison a fresh deploy — ES indices, Kafka offsets, Postgres, logstash
# libs — while KEEPING model caches (rtvi-*, *_cache).
# Names are <project>_<vol>; match on the volume-name suffix.
docker volume ls -q \
  --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}" \
  | grep -E '(elastic-(data|logs)|kafka-data|logstash-libs|phoenix-data|vios_pg_data)$' \
  | xargs -r docker volume rm
```

## Bind-mounted data cleanup

Run after **either** teardown flavor above. Removing containers/volumes does **not**
clear the bind-mounted on-disk data dirs; this step does. Ask the user to confirm
before you proceed.

Use the bundled cleanup helper. It clears every directory whose stale state can poison a fresh deploy: kafka logs, elasticsearch data + logs, redis data + log, behavior-learning data, video-analytics API state, calibration toolkit, VST/nvstreamer recordings, and any blueprint-configurator backup files.

The cleaner needs **root**. Gate on sudo the same way the SKILL.md pre-flight does:
if sudo is passwordless, run it; otherwise **do not** run it under automation —
surface the command and let the user run it once, then resume.

**Pass the Foundation's checked-in env file, not `_builds/<name>/override.env`.**
The cleaner keys its blueprint-specific cleanup off the env file's *pathname* —
a `warehouse` build additionally leaves generated VIOS assets
(`services/vios/configs/{Top.png,calibration.json,labels.txt}`) that are only
removed when that path contains `industry-profiles/warehouse-operations`.
`_builds/<name>/override.env` never does, so passing it silently skips them.
`_builds/` is this skill's own artifact; the cleaner is a shared deploy script,
so the skill adapts to its interface rather than the reverse.

The build's `VSS_DATA_DIR` / `VSS_APPS_DIR` still have to win over the
placeholders the checked-in files ship (`/path/to/...`). Export them: the
cleaner's `load_env()` saves both before sourcing and restores them afterwards,
so a pre-set value always beats the file's.

```bash
BUILD_DIR="$REPO/_builds/<name>"
ENV_FILE="$BUILD_DIR/override.env"
[ -f "$ENV_FILE" ] || {
  echo "missing build override: $ENV_FILE" >&2
  exit 1
}

FOUNDATION="$(sed -n 's/^FOUNDATION=//p' "$ENV_FILE")"
case "$FOUNDATION" in
  warehouse) CLEAN_ENV="$REPO/deploy/docker/industry-profiles/warehouse-operations/.env" ;;
  *)         CLEAN_ENV="$REPO/deploy/docker/developer-profiles/dev-profile-$FOUNDATION/.env" ;;
esac

# Take the effective paths from the build, stripping any surrounding quotes.
DATA="$(sed -n 's/^VSS_DATA_DIR=//p' "$ENV_FILE" | tr -d "\"'")"
APPS="$(sed -n 's/^VSS_APPS_DIR=//p' "$ENV_FILE" | tr -d "\"'")"

# Sudo gate: passwordless sudo → run it; otherwise surface the exact command for
# the user to run once (don't run privileged cleanup under non-interactive sudo).
# `sudo env VAR=...` rather than `sudo VAR=...`: sudo's env_reset drops inline
# assignments unless the sudoers policy grants SETENV.
if sudo -n true 2>/dev/null; then
  sudo env VSS_DATA_DIR="$DATA" VSS_APPS_DIR="$APPS" \
    bash "$REPO/deploy/docker/scripts/cleanup_all_datalog.sh" --env-file "$CLEAN_ENV"
else
  echo "sudo needs a password — run this once and confirm, then resume:"
  printf '  sudo env VSS_DATA_DIR=%q VSS_APPS_DIR=%q \\\n' "$DATA" "$APPS"
  printf '    bash %q --env-file %q\n' \
    "$REPO/deploy/docker/scripts/cleanup_all_datalog.sh" "$CLEAN_ENV"
fi
```

On a `warehouse` build, confirm the blueprint-specific block actually ran — the
output must include `Deleting warehouse VST config:` lines for the three
generated assets. Their absence means the wrong env file was passed.

Run this **before** deleting `_builds/<name>/`: the block above reads
`VSS_DATA_DIR`, `VSS_APPS_DIR` and `FOUNDATION` out of the build override.
