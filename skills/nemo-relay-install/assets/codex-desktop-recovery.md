<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Codex Desktop Recovery

This file was created before persistent NeMo Relay configuration changed Codex
Desktop.

## Setup Record

This record identifies the persistent change that the recovery steps undo:

- Created: `{{created-at}}`
- Workspace: `{{workspace-path}}`
- Planned command: `nemo-relay install codex`
- Setup thread ID: `{{thread-id-or-unknown}}`

## Why Threads May Appear Missing

Persistent NeMo Relay setup changes the active Codex provider to
`nemo-relay-openai`. Due to an open Codex Desktop provider-filtering bug, threads
associated with another provider can disappear from the sidebar after restart.
The thread data has not been deleted.

Do not inspect, copy, delete, or edit Codex session storage or SQLite state to
repair sidebar visibility.

## Restore Normal Codex Desktop Visibility

Use these steps to restore the original provider configuration:

1. Fully quit Codex Desktop.
2. Open a terminal and run:

   ```bash
   nemo-relay uninstall codex
   ```

3. Restart Codex Desktop. Threads associated with the restored provider should
   become visible again.
4. After confirming recovery, this file can be deleted.

If uninstall reports a problem, inspect the installation without deleting
session data:

```bash
nemo-relay doctor --plugin codex
```

## Continue A Thread Through Temporary Relay Wiring

Fully quit Codex Desktop before resuming the same local thread, then run:

```bash
nemo-relay codex -- resume --all
```

Select the conversation. If its thread ID is known, run:

```bash
nemo-relay codex -- resume <thread-id>
```

Avoid `resume --last` when crossing providers. A new thread created under the
Relay provider may remain hidden when Desktop returns to the normal OpenAI
provider, but it remains available through `resume --all`.

## Prompt For A New Codex Task

Paste this into a new Codex task if you want an agent to perform the recovery:

> I enabled persistent NeMo Relay integration for Codex and need to restore
> normal Codex Desktop history visibility. Run `nemo-relay uninstall codex`,
> then verify from the supported command output that the generated Relay
> provider and hook configuration was removed or restored. These supported
> Relay commands may update generated provider and hook configuration, but do
> not directly inspect, copy, delete, edit, or rewrite Codex session storage,
> private application configuration, or SQLite state. Then tell me when to
> restart Codex Desktop. If uninstall fails, run
> `nemo-relay doctor --plugin codex --json`, redact secrets from its diagnostics,
> and stop for review rather than opening Codex configuration files directly.

## References

Use these references for the upstream issue and supported Relay workflows:

- [Codex Desktop provider-filter bug](https://github.com/openai/codex/issues/24648)
- [Relay Codex integration](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/codex)
- [Relay plugin installation and uninstall](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/plugin-installation)
- [Relay transparent run](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/basic-usage#transparent-run)
