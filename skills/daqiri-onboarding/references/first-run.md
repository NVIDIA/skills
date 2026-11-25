# First Benchmark

Read only the selected raw or socket/RDMA benchmark guide. Show concrete commands with host/container roles, configured inputs and a bounded duration. Start RX/server before TX/client where the roles are separate. For UDP/TCP/RoCE wire tests, verify the route and namespace path rather than assume same-host endpoints cross the cable.

Use the existing namespace/cross-host setup scripts only for a matching topology after reading their help and current behavior. Explain changes before applying them and plan cleanup for resources created by this run. A pre-existing namespace or container is not owned merely because its name matches an example. Do not delete namespaces containing moved interfaces without following the documented restoration procedure.

Before a measured run, verify the benchmark, monitor and their direct dependencies inside the actual
execution context. For `mlnx_perf`, check `ethtool`; verify link tools such as `ip` when the planned
discovery or restoration commands require them. If prerequisites are missing, report the missing
commands, required packages and affected steps before proposing installation or a rebuild.

Evaluate exit status, error diagnostics and usable counter samples together. An exit status of zero
with a dependency error or no samples is a failed measurement. Distinguish benchmark delivery
evidence from throughput measurement evidence.

Keep first-smoke success separate from optional performance tuning. Do not change firmware, boot settings, MTU, pause policy, clocks or routes solely because a tutorial lists an optimization. Resolve applicability, authorization and restoration before changes.

For a raw ibverbs run that must preserve the selected MTU, check [initialization sizing](ibverbs-init-limits.md) before launch: configured buffer capacities can cause an implicit MTU raise.


Interpret the output using [evidence.md](evidence.md). For raw logs, the optional summary script reconciles final app counts without reading a full benchmarking manual.
