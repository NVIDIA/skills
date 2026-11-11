<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Application-owned IWER orchestration

Generate a supervisor in the application for its web endpoint, renderer/runtime,
IWER driver, readiness gates, deadlines, logs and cleanup. This Markdown bundle
ships the implementation contract; it does not supply a runner to invoke.
The supervisor is validation infrastructure, not the production XR frame loop.

## Readiness order

For a client-gated runtime, use these state transitions:

1. Validate commands, artifact destination and port ownership before launch.
2. Start the web endpoint and attribute its listener to the owned process.
3. Start the renderer; wait for its runtime listener and attachment-start marker.
4. Start IWER so its client can connect while renderer initialization continues.
5. Wait for renderer/stage readiness, then XR-context readiness.
6. Require a submitted successful frame before accepting driver evidence.
7. Wait for the driver result, then clean up the recorded children.

Starting the driver before listener readiness creates a race. Waiting for a
client-bound XR context before starting its client creates a circular wait.
Use stable application lifecycle markers; do not infer readiness from FPS logs.

## Plan data

Keep a plan in the generated application with four named sections:

- `ports`: loopback endpoints, fixed only when required by the runtime;
- `processes`: executable/argument arrays, working directory, environment,
  owned listeners and graceful stop signal for each child;
- `steps`: ordered start, wait-HTTP, wait-log and wait-exit actions, each with
  a deadline and semantic state name;
- `cleanupOrder`: reverse dependencies, normally driver, renderer, then web.

An individual process and step can use this shape:

```json
{
  "process": {"name": "web", "command": ["web-executable", "web-entrypoint"],
              "cwd": "application-root", "ownsPorts": ["web"]},
  "step": {"action": "wait_log", "process": "renderer",
           "regex": "runtime listener ready", "timeoutSeconds": 30,
           "state": "RUNTIME_LISTENER_READY"}
}
```

These field names describe application data, not SDK exports. Resolve executable
paths and arguments explicitly. Validate any interpolation before launch and
reject unresolved placeholders. Keep credentials out of plans and logs.

## Supervisor algorithm

1. Allocate a unique run ID and new artifact directory. Snapshot existing
   listeners; reject a requested occupied endpoint without stopping its owner.
   For automatically selected ports, account for allocation/bind races and
   require the child to report the endpoint it actually bound.
2. Start each child without a shell, in its own owned process group or Windows
   Job Object. Record PID, start identity, executable, run ID and group/job.
   External renderers or services remain external even if their ports match.
3. Continuously drain stdout/stderr into separate bounded buffers and full logs;
   decode invalid bytes without stopping the reader. Readiness checks inspect
   those buffers while also watching process exit and the monotonic deadline.
4. Check HTTP status and expected body for web readiness. For native listeners,
   verify ownership and the application marker; an unrelated open port is not
   readiness. A child exit before its gate is a distinct failure.
5. On timeout, cancellation, driver failure or interrupt, record the current
   phase and enter one idempotent cleanup path. Stop new driver/input work,
   request graceful child shutdown, and continue draining output.
6. After bounded grace, terminate only a child/group whose recorded identity
   still matches. Recheck descendants and owned endpoints; never kill by a
   substring, shared process name or port alone. Stop when identity is uncertain
   and report the unresolved owner rather than terminating unrelated work.
7. Join log readers and write a final result even when initialization failed.
   Verify owned children/groups exited and owned listeners/SHM resources were
   released or intentionally retained by their owner. Otherwise report
   `cleanup-failed` and preserve diagnostic identities.

Use monotonic clocks for all deadlines. The owner must be able to cancel a wait
without waiting for its full timeout. POSIX process identity can combine PID,
start time and process group; Windows should use owned handles/Job Objects.
If reliable identity inspection is unavailable, fail that cleanup capability
explicitly instead of weakening it to a global process-name search.

## Result and hardware-free tests

Write a result containing run ID, final status, failure phase/reason, ordered
state transitions and timestamps, per-process exit codes/log paths, listener
identities, cleanup errors and remaining owned resources. A dry run resolves
and validates the plan without starting processes or reserving their services.

Before a live run, test the generated supervisor with synthetic children that
emit readiness markers: successful completion; a missing marker timeout;
child exit before readiness; an interrupt while waiting; and an unrelated
listener/process that must survive. Require a final failure result and no owned
survivors in each failed case. Synthetic success validates orchestration only;
first rendered/submitted/decoded XR pixels remain separate live gates.
