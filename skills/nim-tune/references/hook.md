# Guard hook

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

Read this when installing `nim-tune`, or when the guard denied a command. `scripts/tune_guard.py`
enforces the skill's must-never rules on every Bash call the agent makes.

## Install (Claude Code)
Add the hook to the project's `.claude/settings.json` (or `.claude/settings.local.json` for one
person). Show the user the snippet; do not edit user-level settings from the skill. Use an
absolute path or one that starts at `$CLAUDE_PROJECT_DIR` or `$HOME`, because a hook runs in the
session's current folder:

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
  "command": "python3 \"$CLAUDE_PROJECT_DIR/.agents/skills/nim-tune/scripts/tune_guard.py\" || exit 2"}]}]}}
```

Claude Code blocks a call only on exit code 2; any other failing exit code (a crash, `python3` or
the file missing) lets the command run. So the command ends in `|| exit 2`: every exit code except
a clean 0 blocks. No hook option makes a failure block by itself. The guard exits 0 to allow and 2 to deny,
with the reason on stderr, and it exits 2 itself on unreadable input or any internal error. A Bash
event without a command string is denied. An event of another tool (valid JSON with no Bash
command) is allowed. Other agents: run the same script
from their hook configuration, or in CI over the commands they ran.

## What it denies
| rule | denied | fix |
|---|---|---|
| Only the runner changes a target | `kubectl` apply, create, delete, patch, scale, ...; `helm` install, upgrade, uninstall; `docker` run, rm, stop, pull, ... and any `docker container\|image\|network\|volume\|system\|compose\|...` subcommand that is not a read (`ls`, `inspect`, `logs`, ...). Also behind a wrapper (`sudo -u x`, `timeout 60`, `env`, `nice`, `xargs`, `watch`) or inside `sh -c '...'`, `ssh HOST '...'`, `eval`, `$(...)`. Only a real dry run passes: `kubectl --dry-run=server` or `client`, `helm --dry-run`; `--dry-run=none` or `false` is a live change | Change it through `run_campaign.py --approved PLAN_ID`, or ask the user to run it |
| Only an approved plan runs | `run_campaign.py` without `--dry-run` and without `--approved` | Run `--dry-run`, show the plan, rerun with `--approved PLAN_ID` after the user's yes |
| No credential on a command line | a token shape (`nvapi-`, `hf_`, `glpat-`, bearer), `NAME_KEY=value`, `--api-key X` | Reference it by name: an exported variable, a secret, `env_from_host`, `env_file` |

The runner checks the id itself: `--approved` must equal the `plan_id` of the campaign as it is
now, so a change after the review is refused (`not_approved`). The id covers the campaign file and
every local file it deploys or sends (the base and client manifests and the dataset), the run
options (`--only`, `--max-*`, `--ready-timeout`) and the bytes of the `--engine-flags` file.

The hook reads shell syntax, not intent: a script file (`bash run.sh`) is not opened. It is a
guard rail for the agent, not a sandbox; cluster RBAC and host permissions stay the real limit.

## Try it
`python3 "<SKILL_DIR>/scripts/tune_guard.py" --check 'kubectl delete nimservice x'` prints the
decision as JSON.
