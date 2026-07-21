---
name: i4h-workflow-validate
version: "0.6.1"
description: Validate, evaluate, or run i4h envs. Use for policy/checkpoint rollouts and scripted state-machine smoke runs.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - agentic-workflow
    - validation
    - policy-rollout
---

# i4h Workflow — Validate

## Purpose

Roll out a policy or scripted state-machine controller against an env and record verification episodes to an HDF5. Use when the user asks to validate, evaluate, run, or rollout a policy/checkpoint, or asks for surgical state-machine smoke runs.

## Base Code

These steps drive the i4h-workflows base code (the `workflows/agentic/` tree). To reuse an existing checkout, set `I4H_WORKFLOWS` to its path (no clone happens). Otherwise this resolves the current repo, or clones to `~/i4h-workflows` — pick that default without prompting. Run every command below from the resolved root:

```bash
# Resolve the i4h-workflows base code (provides workflows/agentic/).
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/agentic" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/agentic" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Basics

- **Env config (source of truth):** `workflows/agentic/config/environments/<env>.yaml` — read it for the `<env>` defaults: `policy.model_repo`/`model_revision`, `policy.task_description`, `policy.health_port`, and `arena.max_timesteps`.
- Validation runs the policy daemon and Arena together; both processes are required.
- The policy daemon is headless. Arena opens the sim window by default; add `--headless --enable_cameras --rendering_mode performance` only when the user explicitly asks for headless/no-window execution.
- In Claude Code `--print`, Codex `exec`, or any other non-interactive/fresh session, policy evaluation must use **Step 2A** as one foreground bash command. Do not start the policy and Arena in separate tool calls, do not use Claude background tasks for eval, and do not return to the user until Arena exits and the policy cleanup has run.
- In Claude Code specifically, do not use the Bash tool's background mode for `Evaluate ...` prompts, do not launch a command ending in `&`, and do not say "the eval is running in the background." The answer is not complete until the HDF5/log summary has been inspected.
- README quick-run prompts that say "with the state machine" use Arena `--state-machine` and **do not** start a policy daemon.
- Do not run the VLM annotator unless the user asks for success labels.
- `assemble_trocar` is inference-only — validate its YAML default model or a compatible N1.5 checkpoint.

## Inputs

- `ENV_ID`: env YAML id.
- `EPISODES`: `1` for sanity, more for real eval.
- `MAX_TIMESTEPS`: use the user-requested cap when the prompt gives one (for example, `300 timesteps` -> `MAX_TIMESTEPS=300`); otherwise read `arena.max_timesteps` from the env YAML for normal evaluation. Use `200` only when the user explicitly asks for a smoke, sanity, or quick check.
- `MODEL_PATH` (optional): path to a `checkpoint-NNNN/` directory containing `model-0000{N}-of-*.safetensors`, `experiment_cfg/`, and `processor/`. Omit to use YAML `policy.model_repo`.
- `USE_LATEST_CHECKPOINT=1`: set this when the prompt says "new checkpoint" or "latest checkpoint" and `MODEL_PATH` is not already known.
- `STATE_MACHINE`: true only when the prompt explicitly says state machine.

## Run

Run the steps below in order with the `bash` tool. Script paths like `policy/run.sh`, `arena/run.sh`, and `stop.sh` are commands inside bash, not tool names.

For policy/checkpoint evaluation in Claude Code `--print`, Codex, `codex exec --ephemeral`, or any other non-interactive fresh session, use **Step 2A** after setup. Background policy daemons launched by a finished shell can be cleaned up before Arena connects; the controlled shell keeps policy and Arena in one process lifetime and always stops the daemon afterward. In an interactive local-agent tmux session, the separate Step 2 / Step 3 / Step 4 flow is also acceptable.

### Step 1 — setup

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=scissor_pick_and_place
EPISODES=1
ENV_CONFIG="${REPO_ROOT}/workflows/agentic/config/environments/${ENV_ID}.yaml"
[ -f "${ENV_CONFIG}" ] || { echo "missing env config: ${ENV_CONFIG}" >&2; exit 1; }
PYTHON="${REPO_ROOT}/workflows/agentic/arena/.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON="${REPO_ROOT}/workflows/agentic/.venv/bin/python"
[ -x "${PYTHON}" ] || { echo "missing workflow python env; run i4h-workflow-setup first" >&2; exit 1; }
MAX_TIMESTEPS="${MAX_TIMESTEPS:-$("${PYTHON}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["arena"]["max_timesteps"])' "${ENV_CONFIG}")}"
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"

# For prompts such as "Run eval using new checkpoint for 300 timesteps":
#   set MAX_TIMESTEPS=300 and USE_LATEST_CHECKPOINT=1 before this block.
if [ "${USE_LATEST_CHECKPOINT:-0}" = "1" ] && [ -z "${MODEL_PATH:-}" ]; then
  MODEL_PATH="$(find "${RUNS_ROOT}" -path '*/checkpoint/checkpoint-*' -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  [ -n "${MODEL_PATH}" ] || { echo "validate: no checkpoint found under ${RUNS_ROOT}; run finetune first or set MODEL_PATH" >&2; exit 1; }
fi

RUN_DIR="${RUNS_ROOT}/eval_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"
```

### Step 2 — policy daemon

Skip this step when `STATE_MACHINE=true`. For Codex/non-interactive sessions, prefer Step 2A instead of this separate policy-daemon step.

```bash
POLICY_ARGS=(--env "${ENV_ID}" --ensure --log "${RUN_DIR}/logs/policy.log")
[ -n "${MODEL_PATH:-}" ] && POLICY_ARGS+=(--model-path "${MODEL_PATH}")
"${REPO_ROOT}/workflows/agentic/policy/run.sh" "${POLICY_ARGS[@]}"
```

Run this command exactly as a normal foreground bash command. Do not pipe it to `head`, `cat`, `tee`, or `tail`; do not add a separate stop, background launch, sleep, grep loop, curl check, or `docker ps`. `policy/run.sh --ensure` owns reuse, stop/restart, and start-and-ready behavior.

### Step 2A — controlled policy rollout

Use this instead of separate Step 2 / Step 3 / Step 4 when running in Claude Code `--print`, Codex, `codex exec --ephemeral`, or another fresh non-interactive session. Run it as a normal foreground bash command; do not put it in the background and do not answer until it prints `ARENA_STATUS`.

```bash
POLICY_ARGS=(--env "${ENV_ID}" --ensure --log "${RUN_DIR}/logs/policy.log")
[ -n "${MODEL_PATH:-}" ] && POLICY_ARGS+=(--model-path "${MODEL_PATH}")

cleanup_policy() {
  "${REPO_ROOT}/workflows/agentic/stop.sh" policy --env "${ENV_ID}" >/dev/null 2>&1 || true
}
trap cleanup_policy EXIT

"${REPO_ROOT}/workflows/agentic/policy/run.sh" "${POLICY_ARGS[@]}"
ARENA_STATUS=0
"${REPO_ROOT}/workflows/agentic/arena/run.sh" --env "${ENV_ID}" \
  --episodes "${EPISODES}" \
  --max-timesteps "${MAX_TIMESTEPS}" \
  --max-attempts 1 \
  --record-to "${RUN_DIR}/data/verify.hdf5" \
  > "${RUN_DIR}/logs/arena.log" 2>&1 || ARENA_STATUS=$?
"${REPO_ROOT}/workflows/agentic/stop.sh" policy --env "${ENV_ID}"
trap - EXIT
echo "ARENA_STATUS=${ARENA_STATUS}"
```

After this block, skip directly to Step 5.

### Step 3 — arena rollout

Skip this step if Step 2A was used.

For state-machine smoke runs, use this Arena command instead of the policy rollout command:

```bash
"${REPO_ROOT}/workflows/agentic/arena/run.sh" --env "${ENV_ID}" \
  --state-machine \
  --episodes "${EPISODES}" \
  --max-timesteps "${MAX_TIMESTEPS}" \
  --record-to "${RUN_DIR}/data/verify.hdf5" \
  > "${RUN_DIR}/logs/arena.log" 2>&1
```

For policy or checkpoint evaluation:

```bash
"${REPO_ROOT}/workflows/agentic/arena/run.sh" --env "${ENV_ID}" \
  --episodes "${EPISODES}" \
  --max-timesteps "${MAX_TIMESTEPS}" \
  --max-attempts 1 \
  --record-to "${RUN_DIR}/data/verify.hdf5" \
  > "${RUN_DIR}/logs/arena.log" 2>&1
```

### Step 4 — stop policy

Skip this step when `STATE_MACHINE=true` or when Step 2A was used.

```bash
"${REPO_ROOT}/workflows/agentic/stop.sh" policy --env "${ENV_ID}"
```

### Step 5 — summarize logs

```bash
grep -E "policy job complete|run complete|Traceback|Error|FAILED" "${RUN_DIR}/logs/arena.log" || tail -80 "${RUN_DIR}/logs/arena.log"
grep -E "policy ready|Traceback|Error|FAILED" "${RUN_DIR}/logs/policy.log" || tail -30 "${RUN_DIR}/logs/policy.log"
```

## Notes

- Launch the policy daemon with `policy/run.sh --ensure`, then launch Arena.
- In non-interactive Codex runs, keep the policy daemon and Arena in one controlled shell with Step 2A so the daemon is not cleaned up between tool calls.
- **Once Arena exits — whether episodes succeeded or failed — shut down the policy daemon.** It does not self-terminate, so leaving it running leaks GPU memory and holds its health port. Stop it with `"${REPO_ROOT}/workflows/agentic/stop.sh" policy --env "${ENV_ID}"`.
- `--record-to` must be absolute. The recorder resolves relative paths against `workflows/agentic/arena` (its CWD) and produces a nested orphan dir.
- `--max-attempts` defaults to 1 for locomanip-family envs.

## Optional Annotation

Run only on request:

```bash
"${REPO_ROOT}/workflows/agentic/annotator/run.sh" \
  --env "${ENV_ID}" \
  --output "${RUN_DIR}/annotations.jsonl" \
  offline \
  --hdf5-path "${RUN_DIR}/data/verify.hdf5"
```

## Verify

- `verify.hdf5` exists under `${RUN_DIR}/data/`.
- Arena log shows `run complete: N/M episodes succeeded`.
- Policy log contains no `Traceback`.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (`.venv` present); the `policy/run.sh` and `arena/run.sh` launches depend on it.
- An `ENV_ID` matching an env YAML id.
- A model source: either the env YAML `policy.model_repo` default, or a `MODEL_PATH` pointing at a `checkpoint-NNNN/` dir (`model-0000{N}-of-*.safetensors`, `experiment_cfg/`, `processor/`).

## Limitations

- Both the policy daemon and Arena are required; the daemon is headless and Arena opens the sim window unless the user explicitly asks for headless/no-window execution.
- `assemble_trocar` is inference-only — validate its YAML default model or a compatible N1.5 checkpoint.
- `--record-to` must be absolute; relative paths resolve against `workflows/agentic/arena` and produce a nested orphan dir.
- The VLM annotator is optional and run only on request; it is not part of the default rollout.

## Troubleshooting

- **Error:** `.venv` / import fails or `run.sh` missing - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** policy log shows `Traceback` / `Error` / `FAILED` before `policy ready` - Cause: the policy daemon failed to start (e.g. bad model source). Fix: inspect `${RUN_DIR}/logs/policy.log`; verify `ENV_ID` / `MODEL_PATH`.
- **Error:** Arena starts before the daemon is ready - Cause: launch order. Fix: launch the policy daemon first and wait for `policy ready`, then launch Arena.
- **Error:** `verify.hdf5` lands in a nested orphan dir - Cause: relative `--record-to`. Fix: pass an absolute path under `${RUN_DIR}/data/`.
- **Error:** `PermissionError` on `/data/verify.hdf5` - Cause: `RUN_DIR` was unset when Arena ran (setup was skipped or run out of order). Fix: run the setup lines first so `RUN_DIR` exists before `--record-to`.

## Final Response

Report env, model source, episodes saved vs requested, HDF5 path, log paths.
