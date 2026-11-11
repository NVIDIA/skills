# Prepare the runtime cloud workspace

Follow the showcase composition model: the parent retains its map assets,
while the owning skills provide cloud configuration and generate per-run map
and fleet overrides. There is no parent-owned cloud Compose or defaults copy.

## Create or resume a workspace

Read the manifest-resolved cloud, map, and fleet `SKILL.md` files and the adapter
contracts before running their scripts. Resolve dependencies first. Reuse a
recorded workspace for an existing demo; create a fresh directory only for a new
run, outside both this package and the upstream checkouts:

```bash
export HUMANOID_RUN_DIR="$(mktemp -d /tmp/humanoid-cloud-demo.XXXXXXXX)"
export MC_WORKSPACE="$HUMANOID_RUN_DIR/mission-control"
export COMPOSE_PROJECT_NAME="humanoid-$(basename "$HUMANOID_RUN_DIR" | tr '[:upper:]' '[:lower:]')"
mkdir -p "$MC_WORKSPACE/app/config/maps"
cp "$SKILL_DIR/assets/carter_warehouse_navigation.png" "$MC_WORKSPACE/app/config/maps/"
cp "$SKILL_DIR/assets/carter_warehouse_navigation.yaml" "$MC_WORKSPACE/app/config/maps/"
chmod 0755 "$HUMANOID_RUN_DIR" "$MC_WORKSPACE" "$MC_WORKSPACE/app" \
  "$MC_WORKSPACE/app/config" "$MC_WORKSPACE/app/config/maps"
chmod 0644 "$MC_WORKSPACE/app/config/maps/carter_warehouse_navigation.png" \
  "$MC_WORKSPACE/app/config/maps/carter_warehouse_navigation.yaml"
```

Record the run directory, workspace, and project name in the task's runtime
state and preserve these exports in every later shell. Scope permission changes
to the newly created directories and map files above. Any credential files
created later must retain restricted permissions. Never reuse an arbitrary
shared workspace or create configuration in the caller's working directory.

## Configure through the owning skills

For the default scene/map and robot `g1_0`, preview these calls, satisfy the
owning skills' gates, then repeat without `--dry-run` to apply:

```bash
python3 "$SKILL_DIR/references/change-map/scripts/run.py" \
  --source-type file \
  --source /tmp/config/maps/carter_warehouse_navigation.png \
  --metadata-yaml /tmp/config/maps/carter_warehouse_navigation.yaml --dry-run
python3 "$SKILL_DIR/references/change-fleet-composition/scripts/run.py" \
  --fleet-json '[{"name":"g1_0","labels":["test","g1","ros"],"sim":false}]' --dry-run
```

For a selected alternative, copy the matching PNG/YAML pair and adapt both map
paths; submit the complete desired fleet with the selected robot identity.
The map paths above are inside the Mission Control container's config mount.
The G1 launch and navigation helpers instead need host/container-visible paths
to the same map bytes. Ensure the pair is mounted into the activated Isaac ROS
environment and verify agreement before launch; `/tmp` on the host is not
automatically visible inside that container.

The map skill writes runtime defaults from its upstream template. The fleet
skill also writes a runtime Compose file from its upstream template, updating
the generic simulator arguments. Keep `sim:false` and omit `--with-sim` so the
G1 launch remains the only robot source. Do not independently rewrite either
file or modify the upstream resources. Ensure generated defaults and map files
are readable by the Mission Control container user before starting it.

## Resolve the cloud configuration and start

`bring-up-cloud-stack` resolves each file from `MC_WORKSPACE` when an override
exists, otherwise from its own resources. Its `.env`, `images.conf`, and
`init-db.sh` stay upstream-owned unless the operator explicitly requests an
appropriate runtime override. Image tags come from the resolved `images.conf`,
not the humanoid release label. Report resolved image references without
printing credential contents. Never dump the full expanded Compose config or
source a credentials file into a traced shell.

Preserve the upstream broker's authentication settings. Match the G1 client's
MQTT endpoint and credentials to the resolved stack, including TCP versus
WebSocket transport. A default authenticated broker requires credentials; do
not omit them or weaken authentication because the removed overlay was anonymous.

With the recorded exports still active, follow the cloud skill's gates:

```bash
python3 "$SKILL_DIR/references/bring-up-cloud-stack/scripts/run.py" --dry-run
python3 "$SKILL_DIR/references/bring-up-cloud-stack/scripts/run.py"
```

Removing the local overlay does not prove upstream image compatibility. If the
resolved Compose requires `/bin/sh` and a selected image lacks it, report the
service, image reference, and entrypoint mismatch. Do not retry unchanged,
install a host shell, or recreate a bespoke Compose in this skill to hide the
upstream issue. The owning cloud stack must supply a compatible image/config
combination before startup can pass.
