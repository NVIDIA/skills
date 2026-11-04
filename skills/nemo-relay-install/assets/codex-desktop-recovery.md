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
- Relay release: `{{relay-release}}`
- Provider identity after setup: `{{provider-identity}}`
- Setup thread ID: `{{thread-id-or-unknown}}`

## Why Threads May Appear Missing

Legacy NeMo Relay installations may select the `nemo-relay-openai` provider
alias. In affected Codex Desktop versions, switching provider identity can hide
threads associated with another provider after restart. The thread data has
not been deleted.

The currently documented `nemo-relay install codex` preserves the built-in
`openai` provider identity and retains the alias only to resume sessions created
by earlier Relay versions. Confirm the installed release's behavior in its
Codex integration guide and the setup record above before diagnosing a provider
visibility issue.

Do not inspect, copy, delete, or edit Codex session storage or SQLite state to
repair sidebar visibility.

## Restore Normal Codex Desktop Visibility

For an affected legacy alias setup, confirm that the installed release's guide
recommends uninstalling to restore the original provider configuration before
using these steps. A setup that preserves `openai` does not require uninstall
merely because this recovery file exists.

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

After confirming an affected legacy alias setup, paste this into a new Codex
task if you want an agent to perform the recovery:

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
- [Relay Codex integration](https://docs.nvidia.com/nemo/relay/latest/nemo-relay-cli/codex)
- [Relay plugin installation and uninstall](https://docs.nvidia.com/nemo/relay/latest/nemo-relay-cli/plugin-installation)
- [Relay transparent run](https://docs.nvidia.com/nemo/relay/latest/nemo-relay-cli/basic-usage#transparent-run)
