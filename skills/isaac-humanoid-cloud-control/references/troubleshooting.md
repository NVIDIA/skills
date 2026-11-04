# Readiness and troubleshooting

Require all of these before submission:

- Isaac Sim timeline and /clock advance.
- Live joint-state and IMU topics exist.
- AGILE is balanced; the gantry was released by the simulation-only guarded
  helper (explicitly approved for this release attempt) or by an operator-requested
  **G** action; and the observed unsupported pose remains stable before mission
  submission.
- TF resolves map to the configured base frame.
- Nav2 planner, controller, and navigator lifecycle nodes are active.
- The manual /cmd_vel publisher is stopped.
- MQTT and Mission Dispatch are reachable.
- The map, robot identity, and spawn pose agree across systems.
- The Isaac ROS stack was launched immediately after Play; no uncontrolled
  playing interval or observed instability occurred.
- The browser WebRTC view is connected and visibly agrees with robot motion.
- The user has received a clickable, browser-reachable live-stream URL and was
  told that it remains available for the active demo. For remote sessions,
  require the user to confirm the rendered viewport is visible before mission
  submission; an HTTP 200 response alone proves only that the viewer page loaded.

On failure, report the first failed observation with measured values and keep
the relevant shells and logs available. Do not submit overlapping retries. Do
not raise safety velocity thresholds unless the operator explicitly requests
the upstream troubleshooting procedure and the spike is confirmed to be an
expected controller transition.

