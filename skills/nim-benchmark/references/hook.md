# Guard hook

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

Read this when installing `nim-benchmark`, or when the guard denied a command. `scripts/bench_guard.py`
enforces the skill's must-never rules on every Bash call the agent makes.

## Install (Claude Code)
Add the hook to the project's `.claude/settings.json` (or `.claude/settings.local.json` for one
person). Show the user the snippet; do not edit user-level settings from the skill. Use an absolute
path or one that starts at `$CLAUDE_PROJECT_DIR` or `$HOME`:

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
  "command": "python3 \"$CLAUDE_PROJECT_DIR/.agents/skills/nim-benchmark/scripts/bench_guard.py\" || exit 2"}]}]}}
```

Claude Code blocks a call only on exit code 2. Any other failing exit code lets the command run, so
the command ends in `|| exit 2`. A hook timeout is also non-blocking, so the guard stays fast. The
guard exits 0 to allow and 2 to deny, with the reason on stderr, and it exits 2 on unreadable input.
Other agents: run the same script from their hook configuration, or in CI over the commands they ran.

## What it denies
| rule | denied | fix |
|---|---|---|
| direct tool | `aiperf`, `genai-perf`, `gym`, `ng`, `python -m aiperf`, also behind `sudo`, `env`, `xargs`, `uvx`, `uv run`, `pipx run` or `ssh` (its options parsed), inside `sh -c`, `eval`, `$(...)` or backticks, and a shell fed a line naming a tool on stdin (`echo aiperf \| sh`, `bash <<<'aiperf'`); `command -v aiperf`, `which` and `type` pass | run `scripts/nimbench.py` with the approved plan id |
| container start | `docker`, `podman` or `kubectl` commands that start one of those tools | the same |
| credential | a token shape, a key flag with a value, or `NAME_KEY=value` | pass the name of a variable, never the value |

Everything else passes, including `scripts/nimbench.py`, `pip install aiperf` and `docker logs`.

## What it cannot see
The hook reads the command line. It cannot see what a script does inside. A person can still run a
tool by another route, or start a container from a file. The plan id is the main gate: `nimbench.py`
refuses a live run without it. The hook is a second line of defense.
