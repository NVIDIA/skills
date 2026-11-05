# Container Networking

Use Docker bridge networking for MV3DT. DeepStream needs both Kafka metadata output and MQTT peer communication. Publishing host port 9092 does not make a localhost-only broker reachable from a container. Kafka bootstrap and advertised addresses and the MQTT broker must be reachable from the bridge. X11 uses the explicit Unix-socket mount and needs no host network namespace.

## Select The Broker Endpoint

Reuse an existing approved broker endpoint when its metadata advertises a bridge-reachable host and port. Set `MV3DT_KAFKA_HOST` and `MV3DT_KAFKA_PORT` to that endpoint. Do not silently change a shared broker.

For the dedicated native Kafka installation prepared by MV3DT setup on a trusted local development host, add a separate bridge listener while retaining the localhost listener for host-side BEV tools. This listener uses the development broker's plaintext protocol: do not apply it to a shared or production host. There, have the owner supply an authenticated/TLS bridge-reachable endpoint and a protected DeepStream Kafka adapter config via `MV3DT_KAFKA_CONFIG_FILE`. Explain the exact config path, backup, new listener, and broker restart; obtain approval from the broker owner first. Keep the existing cluster storage and controller settings.

```bash
export KAFKA_DIR="${BASE_DIR:-$HOME}/kafka_2.13-4.2.0"
export MV3DT_KAFKA_HOST="$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')"
export MV3DT_KAFKA_PORT="${MV3DT_KAFKA_PORT:-19092}"
test -n "$MV3DT_KAFKA_HOST" || { echo "ERROR: Docker bridge gateway unavailable"; exit 1; }
```

After approval, run this in an interactive shell. For a non-interactive agent call, first obtain the owner's approval of the exact config path and endpoint, then pass `MV3DT_KAFKA_LISTENER_APPROVAL` matching the string requested by the gate; do not set it without that approval. The edit preserves all existing listeners and creates a backup before adding the dedicated one; mismatching existing MV3DT listeners require investigation.

```bash
if test -t 0; then
  read -r -p "Add MV3DT bridge listener to $KAFKA_DIR/config/server.properties and restart the dedicated broker? [y/N]: " answer
  case "$answer" in y|Y) ;; *) echo "Stopped before broker changes"; exit 1 ;; esac
else
  EXPECTED_LISTENER_APPROVAL="$KAFKA_DIR/config/server.properties|$MV3DT_KAFKA_HOST:$MV3DT_KAFKA_PORT"
  test "${MV3DT_KAFKA_LISTENER_APPROVAL:-}" = "$EXPECTED_LISTENER_APPROVAL" || {
    printf 'ERROR: ask the owner to approve this target, then rerun with MV3DT_KAFKA_LISTENER_APPROVAL=%s\n' "$EXPECTED_LISTENER_APPROVAL"
    exit 1
  }
fi
python3 - <<'PY'
import ipaddress
import os
from pathlib import Path
import shutil
import tempfile

host = str(ipaddress.IPv4Address(os.environ['MV3DT_KAFKA_HOST']))
port = int(os.environ['MV3DT_KAFKA_PORT'])
if not 1024 <= port <= 65535:
    raise SystemExit('Use an unprivileged TCP port below 65536')
path = Path(os.environ['KAFKA_DIR']) / 'config/server.properties'
if path.is_symlink():
    raise SystemExit('Refusing a symlink broker config; have its owner configure the listener')
text = path.read_text()
lines = text.splitlines()
settings = {}
for index, line in enumerate(lines):
    if line.strip().startswith('#') or '=' not in line:
        continue
    key, value = line.split('=', 1)
    if key.strip() in {'listeners', 'advertised.listeners', 'listener.security.protocol.map'}:
        if key.strip() in settings:
            raise SystemExit('Duplicate Kafka listener setting; resolve it before proceeding')
        settings[key.strip()] = (index, value.strip())
required = {'listeners', 'advertised.listeners', 'listener.security.protocol.map'}
if settings.keys() != required:
    raise SystemExit('Expected explicit KRaft listener settings; ask the broker owner to configure this deployment')
updates = {'listeners': f'MV3DT://{host}:{port}',
           'advertised.listeners': f'MV3DT://{host}:{port}',
           'listener.security.protocol.map': 'MV3DT:PLAINTEXT'}
for key, addition in updates.items():
    index, value = settings[key]
    entries = value.split(',')
    existing = [entry for entry in entries if entry.startswith('MV3DT:')]
    if existing and existing != [addition]:
        raise SystemExit('Existing MV3DT listener differs; do not overwrite it')
    if not existing:
        lines[index] = f'{key}={value},{addition}'
updated = '\n'.join(lines) + '\n'
if updated != text:
    with tempfile.NamedTemporaryFile(prefix='server.properties.backup-', dir=path.parent, delete=False) as backup:
        backup_path = Path(backup.name)
    shutil.copy2(path, backup_path)
    path.write_text(updated)
    print(f'Backup: {backup_path}')
else:
    print('MV3DT listener is already configured')
PY
LISTENER_EDIT_STATUS=$?
test "$LISTENER_EDIT_STATUS" -eq 0 || exit 1
```

Stop only the identified dedicated MV3DT broker PID after the owner approves interruption; do not use a broad Kafka stop script on a shared host. Restart that broker with its existing config and storage in a persistent terminal:

```bash
cd "$KAFKA_DIR"
bin/kafka-server-start.sh config/server.properties
```

In another terminal with the exported endpoint values, verify broker metadata/topic access, then container TCP reachability:

```bash
"$KAFKA_DIR/bin/kafka-topics.sh" --bootstrap-server "$MV3DT_KAFKA_HOST:$MV3DT_KAFKA_PORT" --list | grep '^mv3dt$'
docker run --rm --network bridge --entrypoint /bin/bash \
  "${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}" \
  -c 'exec 3<>/dev/tcp/$1/$2' -- "$MV3DT_KAFKA_HOST" "$MV3DT_KAFKA_PORT"
```

Stop if either check fails. Host BEV consumers keep `localhost:9092`; the added listener advertises only the bridge gateway. Retain the broker and its config after a run; restore the recorded backup only at the owner's request when no workload depends on the added listener.

## MQTT Peer Communication

Have the broker owner provide a bridge-reachable MQTT listener and a local protocol-adapter config for it. For the dedicated native Mosquitto service, the owner can add a listener on the Docker gateway only (for example, port `11883`), retaining the existing host listener. Configure authentication and ACLs for `/trck/#`; do not expose a new unauthenticated listener on all interfaces. An example owner-managed Mosquitto fragment is:

```text
listener 11883 <Docker-bridge-gateway-IP>
allow_anonymous false
password_file <operator-managed-password-file>
acl_file <operator-managed-ACL-file>
```

The operator supplies the matching DeepStream `[message-broker]` config through `MV3DT_MQTT_CONFIG_FILE`, readable only by its owner, outside version-controlled source directories. Do not print that file or its credentials. In an explicitly approved isolated lab using an unauthenticated broker, that config can omit credentials and contain only `[message-broker]` and `set-threaded=0`. Never carry the template's dummy credentials into an authenticated deployment.

```bash
export MV3DT_MQTT_HOST="$MV3DT_KAFKA_HOST"  # or the operator's reachable broker
export MV3DT_MQTT_PORT=11883
export MV3DT_MQTT_CONFIG_FILE="/path/to/protected/mqtt-adapter.conf"
docker run --rm --network bridge --entrypoint /bin/bash \
  "${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}" \
  -c 'exec 3<>/dev/tcp/$1/$2' -- "$MV3DT_MQTT_HOST" "$MV3DT_MQTT_PORT"
```

Verify an authenticated publish/subscribe round trip on the approved `/trck/#` namespace using the operator's normal secret-handling client configuration; a TCP check alone does not prove authentication or ACLs. Stop on failure. The generated peer config is then rewritten with this endpoint and the adapter config is copied with mode `0600`. Both live and headless runs use these bridge endpoints. No host network namespace is shared.
