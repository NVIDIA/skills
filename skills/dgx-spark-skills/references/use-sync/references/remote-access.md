# Remote Access and Work

Use this reference for NVIDIA Sync installation, onboarding, connections, and
remote applications. Refresh the linked current guide before relying on exact UI
labels or platform requirements.

## Start or Connect

The normal flow is:

1. Install NVIDIA Sync on the user's Windows, macOS, or Ubuntu workstation by
   following [Getting Started](https://docs.nvidia.com/sync/latest/getting-started.html).
   The [DGX Spark Sync page](https://build.nvidia.com/spark/connect-to-your-spark/sync)
   is the product download entry point.
2. Add or import the remote system.
3. Connect to it in NVIDIA Sync.
4. Launch an SSH terminal, a detected IDE, the DGX Dashboard, or a Custom App.
5. Stop applications when finished and disconnect.

For a direct connection, the workstation must be able to reach the remote Linux
system over the network and the user must have SSH access. A DGX Spark or GB10
system can be discovered by its mDNS device name; use a known IP address when
mDNS is unavailable. NVIDIA Sync can instead import a working key-based alias
from the workstation's primary SSH configuration. Follow
[Direct Connections](https://docs.nvidia.com/sync/latest/direct-connections.html)
rather than inventing connection fields or SSH changes.

After a device is added, select it and connect. NVIDIA Sync always offers an SSH
terminal for the configured remote user. It can launch supported local IDEs as
remote clients, and the DGX Dashboard appears automatically on GB10
systems. Use [Applications](https://docs.nvidia.com/sync/latest/applications.html)
for the current application list and behavior.

NVIDIA Sync does not support connections through jump or bastion servers. If
the direct route is unavailable because the workstation is off-network, consider
the documented Tailscale integration rather than suggesting an unsupported Sync
topology.

## Tailscale Access

NVIDIA Sync can join a Tailscale tailnet and access an enrolled remote system
when the laptop/workstation is away from the system's local network. Sync contains the
laptop-side integration; the user does not need a separate Tailscale client
on that workstation for this workflow.

Use [Tailscale Connections](https://docs.nvidia.com/sync/latest/tailscale.html)
for the current enablement and enrollment flow. Enabling the integration opens a
browser for Tailscale sign-in. Adding a device uses a Tailscale authentication
key and may open a terminal to install and authenticate the device client.

Before the agent performs these actions, obtain consent and name the device and
tailnet effect. Treat the authentication key as a secret and have the user enter
it in the NVIDIA Sync interface; never ask them to paste it into the conversation
or expose it in a command.

NVIDIA Sync automatically chooses direct or Tailscale connectivity and can
switch when the workstation moves between networks. Removing a device from the
tailnet requires a direct connection. Disabling the Sync integration does not
remove enrolled devices from the Tailscale administration console, so explain
that residual state before the user authorizes removal or disablement.
