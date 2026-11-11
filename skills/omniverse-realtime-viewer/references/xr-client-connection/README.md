# Connect and diagnose an XR viewer client

Read [library runtime-adapters](../dependencies/ovxr.md) and matching client/runtime docs for service/backend/port contracts. Historical port tables are build qualifications, not OVXR constants.

Provide explicit host choice and actionable readiness. For Apple direct-host clients choose the supplied manual-host option; automatic/cloud zones may contact another service. Browser clients must match page/signaling security.

Diagnose listener/backend, host/interface, client reachability, hosting executable's network/firewall scope, handshake/profile, then media/input. Do not assume a Kit executable rule covers standalone hosting.

Some clients select a messaging channel after video. Report service/client/channel/viewer readiness separately. Preserve renderer/stage during eligible recovery; reconnect fetches canonical state rather than stale deltas.

Use [message contract](../xr-client-message-contract/README.md) and [dispatch policy](../cloudxr-opaque-data-channel/README.md). Test the intended client rather than only localhost or a healthy server log.
