# Bake Checklist

Load this only when the user explicitly asks to bake the current live scene edits into source files (e.g. "bake these edits", "persist the scene changes to source"). Generic uses of "save" or "commit" (saving a file, making a git commit) do not activate this checklist.

## Lifecycle

For an interactive edit prompt, use one sim/bridge window. Do all live edits in that bridge, collect bake data, stop it once, and write source from the collected state. Do not run a fresh relaunch unless the user explicitly asks for validation/onboarding/readiness checks.

1. Launch or reuse one detached bridge:

   ```bash
   workflows/agentic/arena/run.sh ensure-bridge --env <env> --log <run>/logs/bridge.log
   BRIDGE_URL="$(workflows/agentic/arena/run.sh bridge-url --env <env>)"
   ```

2. Apply all live edits through bridge endpoints or `/script` files under `<run>/scripts`.
3. Capture the viewport under `<run>/captures` after task-relevant live edits.
4. Inspect the images. Do not bake an unseen or visually broken scene. For newly added policy/dataset cameras, use the live viewport/object state to choose the pose; verify the baked camera frames later only if running the explicit validation gate.
5. Collect bake state while the bridge is still live (`GET /object`, `POST /bake`, captures, viewport camera pose notes), then stop the bridge:

   ```bash
   workflows/agentic/arena/stop.sh --env <env>
   ```

6. Persist only the requested live changes to source from the collected state. For a live-added object that returns no `/bake` snippet, use the measured live bbox/pose and keep the source bbox bottom equal to the validated live bbox bottom; do not introduce new clearance when converting to `init_state.pos`.

## Optional Fresh-Source Validation Gate

This gate intentionally opens another sim window. Run it only when the user explicitly asks for validation, onboarding readiness, ready-to-commit checks, or a full bake gate.

```bash
local-agent/validate-bake.sh <env>
```

If the worktree already has unrelated dirty shared files from earlier work, use `I4H_BAKE_GATE_ALLOW_DIRTY=1 local-agent/validate-bake.sh <env>` only for local rehearsal. Default strict mode should stay strict for real bake review.

## Source Targets

| Change | Source |
|---|---|
| New/moved/scaled asset | `workflows/agentic/arena/arena/assets/<env>.py` |
| Robot stand pose | `workflows/agentic/arena/arena/environments/<env>_environment.py` |
| Reset randomization | `workflows/agentic/arena/arena/tasks/<env>.py` |
| Success rule | `workflows/agentic/arena/arena/tasks/<env>.py` |
| Runtime publish/consume behavior | `workflows/agentic/arena/arena/runtimes/<env>.py` |
| Policy task text | `workflows/agentic/config/environments/<env>.yaml` |
| Camera used by dataset/policy | assets + task observations + YAML mappings + modality files |

## Cheap Checks After Source Writes

These checks do not launch the sim window:

```bash
python -m py_compile <changed-python-files>
python - <<'PY'
import pathlib, yaml
for p in pathlib.Path('workflows/agentic/config/environments').glob('*.yaml'):
    yaml.safe_load(p.read_text())
PY
workflows/agentic/arena/run.sh --env <env> --dry-run
workflows/agentic/policy/run.sh --env <env> --dry-run
```

## Validation Gate Pass Criteria

Only applies when the optional fresh-source validation gate is requested:

- Static checks pass.
- Bake gate static YAML wiring passes.
- Fresh bridge relaunch reaches ready from source.
- Every expected recorded camera obs key appears in the policy observation group.
- `/cameras` renders every expected camera.
- Robot is present and upright.
- No bridge remains running after validation.

Do not explain away `RESULT: FAIL`; fix source and rerun.
