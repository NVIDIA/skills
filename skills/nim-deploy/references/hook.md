# Guard hook

Read this when installing the skill in a repo, or when a command was denied. The hook is what
enforces the must-never rules; `scripts/guard.py` is the whole implementation.

## Install (Claude Code)
Add the hook to the project's `.claude/settings.json`, or to `.claude/settings.local.json` for one
person. Do not edit user-level settings from the skill: show the user the snippet. The guard does
not care where the skill lives; only the hook command must name the folder it is installed in
(`SKILL_DIR`, the folder with `SKILL.md`). Use an absolute path or one that starts at
`$CLAUDE_PROJECT_DIR` or `$HOME`: a hook runs in the session's current folder, so a bare relative
path breaks (it cannot find the guard, and every Bash call fails or, worse, runs unguarded) as
soon as the agent changes folder.

| where the skill is installed | `command` of the hook |
|---|---|
| project, `.agents/skills/nim-deploy` | `python3 "$CLAUDE_PROJECT_DIR/.agents/skills/nim-deploy/scripts/guard.py" \|\| exit 2` |
| project, `.claude/skills/nim-deploy` | `python3 "$CLAUDE_PROJECT_DIR/.claude/skills/nim-deploy/scripts/guard.py" \|\| exit 2` |
| user, `~/.claude/skills/nim-deploy` | `python3 "$HOME/.claude/skills/nim-deploy/scripts/guard.py" \|\| exit 2` (in `~/.claude/settings.json`, written by the user) |
| anywhere else | `python3 "<SKILL_DIR>/scripts/guard.py" \|\| exit 2` with the real absolute folder |

Settings snippet (replace the command with the row that matches; JSON needs the quotes escaped
as `\"`):

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
  "command": "python3 \"$CLAUDE_PROJECT_DIR/.agents/skills/nim-deploy/scripts/guard.py\" || exit 2"}]}]}}
```

Why `|| exit 2`: Claude Code blocks a tool call only when a PreToolUse hook exits with code 2.
Any other failure (python missing = 127, the file missing, an import or syntax error, a crash
before `main`) is a non-blocking error, and the command runs anyway. `|| exit 2` turns a non-zero
shell exit into a block, so the guard fails closed for those. It does not cover a hook timeout:
Claude Code treats a timeout as non-blocking too, so the guard has to stay fast (it only parses
the command). The guard exits 0 to allow and 2 to deny with the reason on stderr.
There is no `onFailure` hook option.
Other agents: run the same script from their hook configuration, or in CI over the
commands they ran.

## What it does
| class | rule |
|---|---|
| always denied | `chmod`/`chown` of the docker socket, `curl ... \| sh`, deleting a namespace, changing the kubeconfig or `KUBECONFIG`/`DOCKER_HOST`, `helm` without `-n`, any `helm --set`, a secret value on a command line, `docker system prune` |
| hidden forms denied | a program built from `$VAR` or a glob, a shell or interpreter fed `kubectl`/`helm`/`docker` text on stdin, `xargs` or `find -exec` running one of them |
| gated | `kubectl` apply, delete and other mutations; `helm` install, upgrade, uninstall; `docker` mutations; `nim stop` and `nim rm`; a script inside a run folder; the programs the run's runner starts with |

A gated command is allowed only when all of these hold: it is the only command in the Bash call (no
`&&`, `|`, `;`, redirect, `cd`, `$` expansion, `sudo`, `env`, `timeout` or `VAR=` prefix), a run
holds an unexpired approval (60 minutes) for that target, it equals one approved form exactly
(scope flags included), and every approved file still has the approved hash. Dry runs
(`kubectl --dry-run=server`, `helm --dry-run`) and read-only commands pass.

Approved forms come from `facts.next_command` and `facts.then_command` of `nimdeploy.py status`:
- `kubectl apply|delete -f RUNDIR/FILE --context CTX -n NS`;
- the helm command of the run with its absolute `-f` path, and `helm uninstall NAME -n NS
  --kube-context CTX`;
- for the docker target, the forms the run's runner declares (`runners/__init__.py`).

## Try it
`python3 "<SKILL_DIR>/scripts/guard.py" --check 'kubectl delete namespace prod'` prints
the decision as JSON. The denial text ends with a `Fix:` line.
