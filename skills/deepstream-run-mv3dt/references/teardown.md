# Cleanup And Teardown

Use when the user asks to stop MV3DT, inspect interrupted run processes, or explicitly clean up services/artifacts.

Default teardown is run-level cleanup only. MV3DT does not deploy a persistent application container: DeepStream runs in the foreground with `docker run --rm`, so the container is removed when `deepstream-test5-app` exits. Do not stop Kafka, Mosquitto, delete models, delete datasets, or remove generated artifacts after a normal run unless the user explicitly asks.

Generated `config_mqtt.txt` and optional `config_kafka.txt` can contain adapter credentials. Keep them mode `0600`, report their retained paths without contents, and exclude them from shared reports, exports, or source control. If the user requests credential cleanup after the container exits, confirm those exact run-local files and remove only them; leave the operator's original credential source files untouched. Retained credentials must not be silently included with a saved-results archive.

For display runs, tell the user how to stop the windows:

- In `DeepStreamTest5App`, press `q` to stop early.
- In the BEV visualizer window, press `q` to close it.
- If the user started BEV recording interactively, press `r` again to stop recording before closing when possible.
- For agent-managed runs, stop the BEV visualizer PID that the skill started after DeepStream exits; no extra approval is needed for that current-run PID.

For headless or interrupted runs, inspect before stopping anything:

```bash
docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Command}}\t{{.Status}}\t{{.Names}}'
pgrep -af 'kafka_bev_visualizer.py|kafka_client.py' || true
```

If a DeepStream container or Python visualizer process from an unknown, user-started, or interrupted MV3DT run is still active, show the exact container ID or PID and ask before stopping it. Stop only the selected process, for example `docker stop <container_id>` or `kill <pid>`; do not use broad process-kill patterns.

If the user asks to stop prerequisite services, explain that they may be shared by future MV3DT runs or other local workflows, then ask before stopping them:

```bash
KAFKA_DIR="${BASE_DIR:-$HOME}/kafka_2.13-4.2.0"
"${KAFKA_DIR}/bin/kafka-server-stop.sh"

sudo systemctl stop mosquitto
```

If Mosquitto was started manually instead of through systemd, identify the listener first and ask before killing only that PID:

```bash
lsof -nP -iTCP:1883 -sTCP:LISTEN
```

If the user asks to delete generated outputs, show the planned target first and ask for confirmation. Keep deletion scoped to the selected `EXPERIMENT_DIR` or generated run output directories. Do not delete custom datasets, `models/`, Kafka installations, Kafka storage, or Mosquitto packages as part of ordinary cleanup.

If the user asks for a full uninstall, treat it as destructive cleanup. Present separate confirmation for each category before removal:

- Generated MV3DT experiment outputs under the selected `EXPERIMENT_DIR`.
- Downloaded sample datasets under `${REPO_ROOT}/datasets`.
- Downloaded model files and parser builds under `${REPO_ROOT}/models`.
- Kafka installation directory `kafka_2.13-4.2.0` under `BASE_DIR` (the user's home directory when unset), and any Kafka storage the user confirms belongs only to MV3DT.
- Mosquitto service configuration such as `/etc/mosquitto/conf.d/mv3dt.conf` and optional package removal.
