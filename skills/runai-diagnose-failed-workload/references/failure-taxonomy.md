# Ambiguous Failure Signals

Use this reference to interpret ambiguous exit codes, prioritize conflicting evidence,
and determine whether the failure recurred. Apply it after status confirms a `Failed`,
`Crashed`, or non-preempted `Stopped` workload; a `Running` workload with
CrashLoopBackOff or repeated container crashes; or a `Pending` workload with a direct
image-pull, init-container, mount, or failed-pod signal.

## Exit codes

- Exit 137 identifies SIGKILL, not its cause. `OOMKilled` confirms `OOM_KILLED`; without that reason, inspect pod events and node conditions for a drain or external termination.
- Exit 139 identifies SIGSEGV and supports `CRASH_APP_ERROR` after system causes are excluded.
- Exit 1 or 2 supports `CRASH_APP_ERROR` when no system signal explains the termination.

## Evidence priority

Prefer explicit kubelet reasons and workload-history events over exit-code inference. A
pod `k8sPhase` of `OOMKilled` is direct evidence; a generic `Error` still needs status,
history, or logs to identify the cause. Use the submitted spec to corroborate image,
command, memory-limit, mount, and restart-policy details, but never classify from
configuration alone.

For OOM, compare the observed peak with the memory limit, not the request. If a
short-lived job has no metric samples, report the missing peak and use direct
`OOMKilled` evidence plus the configured limit. Never present requested, limited, or
allocated memory as an observed peak. For every category, cite the tool or Kubernetes
command that produced the signal.

## Recurrence

Compare prior failures within `lookback`. When the same category repeats, report the number of failures and time window, and recommend a permanent correction instead of an unchanged rerun.
