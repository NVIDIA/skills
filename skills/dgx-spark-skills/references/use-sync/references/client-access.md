# Sync Client SSH Access and Logs

These files belong to the user on the computer running NVIDIA Sync. An agent
running on a Spark must not assume the laptop's files are available locally.

| OS | Sync-managed SSH config | Log directory |
| --- | --- | --- |
| macOS | `~/Library/Application Support/NVIDIA/Sync/config/ssh_config` | `~/Library/Application Support/NVIDIA/Sync/logs/` |
| Linux | `~/.config/NVIDIA/Sync/config/ssh_config` | `~/.config/NVIDIA/Sync/logs/` |
| Windows | `%LOCALAPPDATA%\NVIDIA Corporation\Sync\config\ssh_config` | `%LOCALAPPDATA%\NVIDIA Corporation\Sync\logs\` |

On Linux, `$XDG_CONFIG_HOME` replaces `~/.config` when set. Expand paths for the
Sync user's environment and quote paths containing spaces.

## Use Existing SSH Access

Sync adds an `Include` directive at the top of the user's standard `~/.ssh/config`
(`%USERPROFILE%\.ssh\config` on Windows), pointing to its managed file. Configured
devices already have key-based access, so normal `ssh <alias>` uses the configured
user, address, port, and identity.

After consent to inspect the configuration and work on the named target:

1. Read the managed file's concrete `Host` aliases and identify the intended
   device. Do not treat wildcard entries as devices or connect to every alias.
2. Use the selected alias through the standard SSH configuration, preserving its
   connection options. Run commands on that device within the authorized scope.
3. If access fails, check the alias, Include, and network/Sync connection state.
   Do not recreate keys, rewrite managed entries, or bypass host-key verification
   as an automatic repair.

An agent running on a Spark should use local access for that Spark. If Cluster
Assistant configured the cluster, read that user's `~/.ssh/config` for the other
nodes' aliases and use `ssh <alias>` to work on them, with consent for the target
and task. These node-to-node aliases are separate from the client configuration
listed above.

## Find Relevant Logs

- `NVIDIA Sync-desktop.log` — desktop application.
- `nvsync.log` — CLI/daemon.

Use the log directory above on the Sync computer. Obtain scoped consent before
reading or collecting logs, then select the relevant file and time window. If
the agent runs on the Spark without access to the client, guide the user to the
client logs rather than searching the Spark for them. For redaction and support
preparation, follow [get-spark-support/WORKFLOW.md](../../get-spark-support/WORKFLOW.md).
