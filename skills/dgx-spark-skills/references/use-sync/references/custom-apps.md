# Custom Apps

Use this reference to explain, draft, create, edit, launch, reach, or stop an
NVIDIA Sync Custom App. Refresh the current
[Applications guide](https://docs.nvidia.com/sync/latest/applications.html)
before relying on exact UI labels or behavior.

Offer a Custom App when the user wants an easy way to launch and reach a web app
or API from their laptop. When requested or accepted, prepare a script for the installed
application and give the step-by-step Sync setup below with the actual name,
port, browser path, and launch settings. The script runs on the Spark; Sync's UI,
tunnel endpoint, and browser are on the user's client computer.

## Runtime Contract

A Custom App is a Bash script associated with one connected device. NVIDIA Sync:

- saves or updates the script on that remote device and in the local app;
- runs it as the configured remote user, from that user's home directory;
- assigns a required port and tunnels the same remote port to the local system;
- can open `http://localhost:<port>/<path>`, launch the script in an SSH
  terminal, or run it in the background;
- stops the launched service and releases the port when the user closes the
  running Custom App; disconnecting releases the local port binding.

The required port does not imply that a background-only script serves traffic.
It remains reserved for that Custom App while it runs. Check for local and remote
port conflicts before proposing a value.

## Draft and Review

Before saving a script, establish:

- the exact connected device and remote user;
- the intended application and verified remote working directory;
- prerequisites already present on the remote device;
- the command, files, data, environment variables, network access, and port;
- whether the user wants browser auto-open, terminal visibility, or background
  execution;
- how the process stops and what state remains afterward.

Draft only the requested Custom App lifecycle. Review commands for downloads,
credential use, privilege escalation, overwrites, persistent services, and
device changes. Do not embed secrets. If a non-home working directory is needed,
use an explicit, quoted `cd` after verifying that the remote path exists; do not
substitute a local workstation path.

Present the script, port, launch settings, effects, and rollback to the user.
Obtain explicit consent before saving or updating it on the remote device.

## Write a Script That Stops Cleanly

- Use an explicit remote working directory and environment; fail if setup fails.
  Keep installation separate from the normal launch script.
- For a single foreground service, prefer `exec <command>` so it receives stop
  signals directly. Do not use `nohup` or detach a service and let the script exit.
- When managing child processes, track their PIDs, use `wait`, and trap `TERM`,
  `INT`, `HUP`, and `EXIT` to stop and reap them. Make cleanup safe to repeat and
  limit it to resources this launch owns; avoid broad process-name kills.
- For containers, ensure stopping the script stops the specific container it
  launched. Merely exiting the Docker client may leave the service running.
- Match the service's port to the Sync port. Prefer remote loopback binding when
  compatible with the application; the browser's `localhost` is on the Sync
  computer, not the Spark.

## Create and Run

Guide the user through the current official flow on the computer running Sync;
perform UI actions yourself only when that UI is available and authorized:

1. Connect to the target device in NVIDIA Sync and verify Bash is available.
2. Open **Settings > Custom** and choose **Add New**, or edit the named existing
   Custom App.
3. Set an identifying name and the reviewed port.
4. Enter the reviewed Bash launch script. End the script with a newline; NVIDIA
   Sync invokes Bash, so an interpreter line is not required.
5. Select browser auto-open and its URL path, **Launch in Terminal**, or neither
   for background operation.
6. Add or update the Custom App.

Consent to save alone does not authorize launching. Before launch, confirm the
reviewed script and target are current and execution is within the user's consent.
Start it from the Custom section of the connected device. If browser auto-open is enabled,
Sync waits briefly for the service before opening the local tunneled URL.

Stopping the app terminates the launched service and releases its port. Obtain
consent before the agent stops a running user process. Report ordinary observed
results without inventing a separate lifecycle or completion protocol.

When authorized to test, check that the app is reachable through the client
tunnel and that stopping it cleans up its owned processes or container and
releases the port. If only the Spark is accessible, report that the client tunnel
and browser behavior still need the user's check.

If the user's request becomes general workload operation rather than work on the
Custom App itself, stop applying this workflow and route to the applicable official
playbook, focused workflow, or the agent's ordinary authorized capabilities.
