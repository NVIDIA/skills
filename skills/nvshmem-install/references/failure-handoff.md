# NVSHMEM Failure Handoff

Read this reference for planning, installation, or validation failures. Preserve the exact command, output, target summary, selected transport/bootstrap, and documentation version. For post-install launch, initialization, runtime, or validation failures, invoke `$nvshmem-troubleshoot-and-report-bugs` when available; otherwise provide official support guidance with the preserved evidence.

| Error | Required response |
| --- | --- |
| Official documentation cannot be retrieved | Return mapped official links, identify each blocked claim, and do not synthesize version-specific requirements or commands. |
| A probe field is `unknown` | Ask for the smallest missing fact; do not classify it as `fail` or request elevated privileges. |
| No installation method can be selected | Resolve authority, scope, compatibility, or mandatory-requirement gaps; otherwise offer a user-owned prefix, approved container, site module, or administrator handoff. |
| An installation command fails | Stop, preserve its command and output, compare evidence with exact-version documentation, and request approval before a revised block. After three approved revised blocks fail, stop and provide escalation guidance with all evidence. |
| Same-node passes but two-node fails | Preserve both outputs and selected configuration. Do not report remote transport as working; route to troubleshooting or official support guidance. |
| `nvshmem-info` succeeds without visible output | Set <code>LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"</code> and run `"$PREFIX/bin/nvshmem-info" -n -b`. |
| `nvshmem-info` cannot load `libnvshmem_host.so*` | Set <code>LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"</code> before `nvshmem-info`, smoke tests, and archive-installed applications. |
