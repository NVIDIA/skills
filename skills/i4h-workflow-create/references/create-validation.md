# Create Env Validation

Load after the source files for a new env exist, before running bridge/visual validation. Keep full Isaac logs in files and summarize only status lines, artifact paths, and concise tails.

## Validation

Static checks (always):

```bash
workflows/agentic/policy/run.sh --list-envs
workflows/agentic/arena/run.sh --env <env> --dry-run
workflows/agentic/policy/run.sh --env <env> --dry-run
python -m py_compile <changed-python-files>
```

**These static checks are shallow.** `--list-envs`/`--dry-run` only resolve the env
YAML and `py_compile` only checks syntax — none of them *import* the new arena
asset/task modules or **call `get_env`**, so build-time errors (a camelCase cfg kwarg →
`TypeError`, a `func=` helper defined below its cfg class → `NameError`, a missing
event-param in a custom reset signature, or a **hallucinated kwarg** like
`IsaacLabArenaEnvironment(..., env_spacing=...)` → `unexpected keyword argument`) pass
**every** static check *and the grep gate* silently, and only surface when the env is
actually built.

**So the mandatory gate is a real bridge build plus visual validation.** Build the
env for real with `workflows/agentic/arena/run.sh ensure-bridge --env <env> --log <path>`
(detached launch + wait-for-ready; never the blocking bare `--bridge`); this calls
`get_env`, imports the new task/asset modules, and catches the build-time errors
above. Then probe geometry through the bridge, capture the viewport, and judge the
rendered image. If the current agent can inspect images, it must open/read the
capture directly and make the visual verdict itself. Use
`./local-agent/validate-env.sh <env>` only for the local-agent/coder-only path where
the agent cannot inspect images; that wrapper delegates the visual verdict to a
local VL service and requires that service to be running. **A green grep-gate +
`--dry-run` is NOT a validated env** — do not report the env created until the real
bridge build and visual inspection pass. A geometry or visual failure is real: fix
the SOURCE and re-run; never rationalize it or stop at static-only. (Same
"dry-run is shallow" rule the bake gate `validate-bake.sh` enforces.)

For the blind local-agent path, `./local-agent/validate-env.sh <env>` is the
turnkey gate: it does dry-run → bridge build → geometry → viewport capture and
delegates the image verdict to `local-agent/vlcheck.py`. If that wrapper reports
`VL-DEFERRED` because no local VLM is up, a vision-capable CLI agent should judge
the capture itself; a blind local agent should start/fix the local VLM and rerun.

The manual **probe → live-fix → bake → exit** flow below is the primary flow for
vision-capable agents and the interactive repair flow for scene geometry.

**Bridge scene-validation is a required step, not a user choice.** A forked
env's geometry is only verified in the bridge, so once the files exist run it
**automatically** — don't ask or offer a "stop at code / static-only" option (the
user can interrupt). A missing `.venv` isn't a reason to ask: run
[[i4h-workflow-setup]] first, then continue. The **only** acceptable skip is a
host that can't launch Isaac Sim (no GPU / launch fails) — then still run every
static check and **report the skip explicitly** as a blocker, never as an option
the user picked.

**Minimize bridge cold starts** (each is a ~30 s Isaac Sim launch): **batch**
all source edits (use the G1 vertical-setup numbers up front so the first build
is already at the target height) and relaunch **once**, doing every live fix in
that one session and only relaunching for source edits that change
spawn/scale/collision. A confirming relaunch just to *look* after a bake is
optional — `--dry-run` + `py_compile` are the post-bake check (at most once).

### Phase 1 — Probe

Run the steps below in order. If shell variables do not persist across tool calls, repeat the setup command before later probes.

#### Step 1 — setup

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=<env>
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/scene_edit_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts" "${RUN_DIR}/captures"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"
```

#### Step 2 — bridge session

Launch the bridge **DETACHED**, never foreground. `arena/run.sh --env <env> --bridge`
is a long-running FOREGROUND process — running it directly (or piped to `| tee`) blocks
the shell forever and the bridge dies with the call. Never do that. Use the built-in
**`ensure-bridge`** helper, which reuses an already-running bridge, otherwise launches
`--bridge` detached, and waits for ready (the bridge analogue of `policy/run.sh --ensure`):

```bash
"${REPO_ROOT}/workflows/agentic/arena/run.sh" ensure-bridge \
  --env "${ENV_ID}" --log "${RUN_DIR}/logs/bridge.log"
BRIDGE_URL="$("${REPO_ROOT}/workflows/agentic/arena/run.sh" bridge-url --env "${ENV_ID}")"
```

It returns 0 once `scene-edit bridge ready` is reached (logging to `${RUN_DIR}/logs/bridge.log`)
and exits non-zero with a log tail on failure. Run it plainly — no short `timeout`
(cold Isaac start takes minutes; override the wait with `--timeout SECONDS` if needed).
Local agent: `./local-agent/bridge.sh start "${ENV_ID}"` is the equivalent shortcut.

After it reports ready:

- `GET "${BRIDGE_URL}/objects"` — confirm every expected entity is `valid: true`.
- `GET /object?name=<key>` for table, robot, props, destinations, ground. Read `xform_ops` and `bbox`.
- **Robot upright + clear of the support surface (do not skip — this is how a
  toppling robot is caught).** A floating-base G1 can topple at spawn even when a
  viewport frame + stale `bbox` look fine. Confirm from the **live** pose, not the
  bbox: read `GET /object?name=robot` → `live.root_pose_w` **three times a second
  or two apart**.
  Upright = pelvis `x,y` steady and `z` constant; a topple = `x,y` drifting and/or
  `z` sinking turn after turn. Also confirm the robot's `x`-extent doesn't overlap
  the table's (see Footprint clearance). **Edit-mode caveat:** the keep-open idle
  holds `base_height≈0.65`, squatting the G1 to pelvis `z≈feet+0.65≈-0.14` and
  *holding* there — that steady value is the cosmetic idle squat, **not** a fall;
  only continued sinking / drift is a fall.
- `POST /capture` the **viewport** plus every task-relevant camera into
  `${RUN_DIR}/captures`, and read the JPEGs. Judge overall scene layout (heights,
  reach, placement) from the perspective **viewport** — the authoritative
  whole-scene view. Robot / POV cameras only check what the policy will see
  (manipulation-zone framing), not global layout.
- Score the scene against the checklist in Phase 2.

### Phase 2 — Live-fix

Apply fixes through the bridge ([[i4h-workflow-scene-edit]] for endpoint
patterns); write `/script` payloads under `${RUN_DIR}/scripts/`. Fix the scene
in dependency order — each asset rests on the one before it, so lock the
underlying asset before adjusting what sits on it:

1. **Support surface (table/shelf) first — set it in source, not live** (see G1
   vertical setup). Its height/scale determine where every other asset sits. Set
   `init_state.pos` / `spawn.scale` in `workflows/agentic/arena/arena/assets/<env>.py`,
   relaunch, and confirm via bbox that it rests on the ground (`z_min` ≈ ground z)
   with the tabletop at the robot's working height — before adjusting anything on it.
2. **Robot stance + reach** (see Robot Reach) — pin the reachable work-zone band
   next. For a free-standing robot (G1), first confirm it is **clear of the table
   footprint and stays upright** (re-read `live.root_pose_w` over a few steps per
   Phase 1); if it topples, fix the footprint overlap *before* tuning anything on
   the table.
3. **Props** rest on or just above the tabletop world z, within the reach band; nothing clips through.
4. **Static destinations** rest on the tabletop: bbox bottom is within 0-2 mm of tabletop z, with no visible air gap.
5. **Prop USD scales** visually match real-world dimensions (use the bbox).
6. **Cameras** see the manipulation zone with the robot in frame.
7. **Per-reset randomization** keeps props on the table and away from each other.

Steps 2–7 are live bridge edits; do not edit source during them. If a live edit
returns an error, report the request payload and error to the user. Do not
restart the bridge.

### Phase 3 — Bake

```text
POST /bake names=[<adjusted entities>]
```

Apply the returned snippets:

| Bridge result | Source |
|---|---|
| Asset xform | `workflows/agentic/arena/arena/assets/<env>.py` (`init_state.pos`, `init_state.rot`, `spawn=...scale`) |
| Robot stand | `workflows/agentic/arena/arena/environments/<env>_environment.py` (`embodiment.set_initial_pose(...)`) |
| Reset randomization range | `workflows/agentic/arena/arena/tasks/<env>.py` events cfg |
| Camera / language / dataset fields | env YAML |

Re-run static validation:

```bash
python -m py_compile <changed-python-files>
workflows/agentic/arena/run.sh --env <env> --dry-run
workflows/agentic/policy/run.sh --env <env> --dry-run
```

### Phase 4 — Exit

Stop the bridge before reporting completion or proceeding to teleop/mimic/convert/finetune.

## Visual Validation

After generating code, **always run visual validation in bridge mode** — this is required to catch geometry issues (height overlap, footprint collision, camera framing, prop placement) that static checks miss. The bridge validates:

1. **All entities valid** — `valid: true` for robot, props, destinations, cameras
2. **Robot upright** — `live.root_pose_w` shows steady `z`, no drift/sinking
3. **Footprint clearance** — robot bbox x-range doesn't overlap table bbox x-range
4. **Prop heights** — cube/tools/tray bbox bottoms sit on the tabletop; trays must be within 0-2 mm of tabletop z and must not visually float
5. **Cameras framing** — required policy cameras show the manipulation zone

Run the full flow automatically after code generation (no user choice needed):

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=<env>
RUN_DIR="${REPO_ROOT}/workflows/agentic/runs/visual_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/captures"

# Static check
"${REPO_ROOT}/workflows/agentic/arena/run.sh" --env "${ENV_ID}" --dry-run
"${REPO_ROOT}/workflows/agentic/policy/run.sh" --env "${ENV_ID}" --dry-run

# Visual validation (required) — DETACHED via ensure-bridge (never blocking `--bridge`)
"${REPO_ROOT}/workflows/agentic/arena/run.sh" ensure-bridge \
  --env "${ENV_ID}" --log "${RUN_DIR}/logs/bridge.log"
BRIDGE_URL="$("${REPO_ROOT}/workflows/agentic/arena/run.sh" bridge-url --env "${ENV_ID}")"

# ensure-bridge returns once the bridge is ready, then probe with:
curl -fsS "${BRIDGE_URL}/objects" -o "${RUN_DIR}/objects.json"          # check all entities valid
curl -fsS "${BRIDGE_URL}/object?name=robot" -o "${RUN_DIR}/robot.json"  # check live pose
curl -fsS "${BRIDGE_URL}/object?name=table" -o "${RUN_DIR}/table.json"  # check table z range
curl -fsS "${BRIDGE_URL}/object?name=tray_a" -o "${RUN_DIR}/tray_a.json" # check tray bbox bottom against table top
python -m json.tool "${RUN_DIR}/robot.json"
python -m json.tool "${RUN_DIR}/table.json"
python -m json.tool "${RUN_DIR}/tray_a.json"
python - <<PY
import json, pathlib, sys
run = pathlib.Path("${RUN_DIR}")
table = json.loads((run / "table.json").read_text())["result"]["bbox"]
tray = json.loads((run / "tray_a.json").read_text())["result"]["bbox"]
gap = tray["min"][2] - table["max"][2]
print(f"tray_a_gap_m={gap:.4f}")
if not (-0.001 <= gap <= 0.002):
    raise SystemExit("tray_a is not resting on the tabletop")
PY

curl -fsS -X POST --json '{"output_dir":"'"${RUN_DIR}"'/captures","viewport":true}' \
  "${BRIDGE_URL}/capture" -o "${RUN_DIR}/capture.json"
python -m json.tool "${RUN_DIR}/capture.json"

# Vision-capable agents: open/read the JPEGs under "${RUN_DIR}/captures/"
# directly and record the verdict yourself. Do NOT call local-agent/vlcheck.py
# unless you cannot inspect images.

# Stop Arena to stop the bridge
"${REPO_ROOT}/workflows/agentic/arena/stop.sh" --env "${ENV_ID}"
```

Coder-only local-agent fallback only:

```bash
python3 local-agent/vlcheck.py --image "${RUN_DIR}"/captures/*viewport*.jpg --prompt \
  'Validate this top-down robot tool-sort scene. Reply JSON {"pass":bool,"trays_on_table":int,"tools_on_table":int,"robot_ok":bool,"issues":[...]}. Require: exactly TWO distinct-colored destination trays sitting ON the table (not missing/sunk/floating); the task tools on the table; the humanoid upright and facing the table. A buried/kinematic tray below the tabletop reads as 0 trays.'
# pass:false → fix the source (e.g. prop z below tabletop, footprint overlap) and relaunch.
```

**The visual check is the step that catches what bbox numbers and grep gates miss**
(a tray sunk under the tabletop still has a "valid" entity and passes every static
check — only looking at the render, or asking a VL model when the agent lacks image
inspection, reveals it's gone). So a green grep gate / `--dry-run` is **not** enough;
the scene isn't validated until the bridge build is up and the capture has been
judged as pass by the current vision-capable agent or, for coder-only local-agent
runs, by the local VL fallback.
