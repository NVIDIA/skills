# Claude Code adapter (hooks)

The same guard/recorder logic as the Pi adapter, expressed as Claude Code
hooks. These are **templates**: copy them into your kit directory (the
driver's cwd), make them executable, and reference them from a
`.claude/settings.json` there — see `settings.template.json`.

| File | Role |
|---|---|
| `settings.template.json` | `.claude/settings.json` for the kit directory: wires guard.sh (PreToolUse, Bash matcher) + record.sh (PostToolUse), and disables unneeded plugins (measured: halves the fixed per-session startup context). |
| `hooks/guard.template.sh` | PreToolUse guard. Contract: tool call arrives as JSON on stdin; exit 2 blocks the call and stderr is shown to the agent as advice. Add one guard per discovered quirk. |
| `hooks/record.template.sh` | PostToolUse recorder → `$RD/commands.log`. Edit the command patterns and the run-dir glob for your workflow. |

The generic driver skeleton for this harness is
`../../templates/driver.template.sh` (launches `claude -p` per stage; the
same three config points as the shipped Pi pack drivers).

Framework guards that live in Pi's `guard.ts` (turn budget, loop breaker,
verify-before-commit) can be ported into `guard.sh` the same way — the study's
originals are plain-bash implementations and fit in ~30 lines each.
