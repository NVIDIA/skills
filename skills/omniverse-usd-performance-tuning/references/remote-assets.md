# Remote asset access

Use before the first open of an `omniverse://` asset. Require a working resolver and
authorized server access; local USD inputs do not need this step.

The successor assessment and profiling tools require a local input with all required
dependencies available locally. Resolve remote access, then prepare that dependency-complete
copy before invoking them. A working resolver alone does not enable direct URL input
to these tools. Keep the remote source unchanged and record its relationship to the
localized input. Authorized standalone tools with resolver support can use remote assets directly.

Reuse cached credentials and verify access with a small `omni.client.stat(url)` request.
The supported Omniverse route needs `omni.client` and `omni.usd_resolver` in a configured
runtime. If browser SSO is needed, reuse prior authorization for that interaction; otherwise
ask before starting it. Do not ask for passwords/tokens in chat or write them into the repo.
Use a supplied approved service-credential route for unattended access.

Distinguish authentication, network/resolver and missing-asset failures. Preserve the URL
and bounded error details in the run log; do not repeatedly retry a full-stage open.
If remote access cannot be established, report that limitation and use an authorized mounted
or local copy when available. Keep its referenced dependencies resolvable and record the
actual input location. Comparing remote baseline I/O with local output I/O does not isolate
optimization gains. Continue through [setup](setup.md) after access is verified.
