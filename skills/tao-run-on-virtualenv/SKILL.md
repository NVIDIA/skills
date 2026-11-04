---
name: tao-run-on-virtualenv
description: Run a Python training/eval script directly in an existing local virtualenv — no docker, no container. Implements the four-verb consumer contract (submit/status/logs/cancel) over a vendored process-lifecycle runner with durable on-disk state, PID-reuse-safe identity, and process-group cleanup. Use for docker-free local execution, plain-Python model scripts, fast HPO/AutoML trial smokes, or hosts where containers are unavailable. Trigger phrases include "run in my venv", "no docker", "virtualenv execution", "local python training", "run this training script directly".
license: Apache-2.0
compatibility: Requires a local Python virtualenv (pyvenv.cfg + bin/python) with the training script's dependencies installed. Linux with procfs and pidfd signaling is required for active job cancellation; other POSIX hosts can run jobs and prove empty groups but fail closed rather than signal numeric process IDs. No nvidia-tao-sdk, no docker.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- platform
- virtualenv
- local
---

# Virtualenv — docker-free local Python execution

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

The virtualenv platform runs a Python script **natively in an existing venv** —
as an argv vector whose first element is `<venv>/bin/python`, never through a
shell, never activating anything. The vendored runner
(`references/virtualenv_runner.py`) is this platform's "native CLI" — the role
`docker`/`kubectl`/`sbatch` play elsewhere — and owns only the process
lifecycle. Job records stay with `tao_job_record.py`; specs are authored by the
agent, exactly like every other platform.

## When to use

- The workload is a **plain Python script** (its dependencies pip-installed in
  a venv), not a TAO container action.
- **No docker** on the host, or container startup cost isn't worth it (fast
  smokes, AutoML trial loops over lightweight models).
- Single node only. For TAO container actions use `tao-run-on-docker`; for
  clusters use `-slurm` / `-kubernetes`.

## Preflight

```bash
# 1. The venv is real and has an executable interpreter.
[ -f "$VENV/pyvenv.cfg" ] && [ -x "$VENV/bin/python" ] || echo "MISSING: $VENV is not a venv"
# 2. The script's top-level imports resolve inside it (catches wrong-venv early);
#    substitute the real modules your script imports.
"$VENV/bin/python" -c "import torch" || echo "MISSING: script dependency not in $VENV"
# 3. GPU visibility only if the script needs CUDA.
nvidia-smi >/dev/null 2>&1 || echo "note: no GPU visible (fine for CPU scripts)"
```

No credentials are required by the platform itself; model-specific env vars
(e.g. `HF_TOKEN`) pass through by NAME with `-e` (values never land on argv).

## Storage

Tier **A** by definition — everything is local paths. Datasets must already be
on local disk (stage with `tao-data-io` first if they live in S3). Outputs land
in the job record's `results_dir`, which IS the runner's `--job-dir`.

## Execution — the four verbs

`$BANK` = `${TAO_SKILL_BANK_PATH}`; `$RUNNER` =
`$BANK/skills/platform/tao-run-on-virtualenv/references/virtualenv_runner.py`.

### submit

1. **Author the spec** (if the script takes one) at a local path — nested
   dicts, never flat dotted keys — and **lint** the assembled command with
   `redact_secrets.py lint`.
2. **Open the record — mints the id, binds `results_dir` BEFORE launch:**
   ```bash
   JOB_ID=$("$BANK/scripts/tao_job_record.py" open --platform virtualenv \
     --image "$VENV/bin/python" --network-arch "$ARCH" --action "$ACTION" \
     --storage-tier A --results-root "$RESULTS_ROOT")
   RESULTS_DIR="$RESULTS_ROOT/$JOB_ID"
   ```
3. **Launch detached** (the runner writes a durable wrapper that gates start,
   records identity, and cleans up the process group on exit):
   ```bash
   set -a; source /path/to/.env; set +a   # omit if already exported
   python3 "$RUNNER" submit --job-dir "$RESULTS_DIR" --venv "$VENV" \
     --script train.py --job-id "$JOB_ID" --config-path "$SPEC" \
     --arg train --arg=--config={config_path} --arg=--out={results_dir} \
     --gpu-ids 0 -e HF_TOKEN
   ```
   Placeholders `{config_path}` `{results_dir}` `{job_id}` render inside
   `--arg` tokens. **A token starting with `-` must use the `--arg=TOKEN`
   form** (argparse). `--gpu-ids` sets `CUDA_VISIBLE_DEVICES`; `--gpus 0`
   hides GPUs; neither reserves anything.
4. **Record RUNNING** with the pid the runner printed:
   ```bash
   "$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state RUNNING --backend-ref "pid:<pid>"
   ```

One submit per job dir — a retry gets a NEW record (`--retry-of`), never a
re-submit into the same dir.

### status

```bash
python3 "$RUNNER" status --job-dir "$RESULTS_DIR"   # {"status": "...", ...}
```

Prints the fixed vocabulary directly: `PENDING RUNNING COMPLETE ERROR CANCELED
UNKNOWN` — no mapping table needed. Status is derived from durable files
(`exit_status.json`, launcher identity) and is safe to poll from any process,
any time, including after reboots of the polling agent. On a terminal status,
`mark` the record.

### logs

```bash
python3 "$RUNNER" logs --job-dir "$RESULTS_DIR" --tail 200
```

### cancel

```bash
python3 "$RUNNER" cancel --job-dir "$RESULTS_DIR"
"$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state CANCELED --source agent
```

Cancel marks first (a not-yet-started wrapper self-cancels at its start gate),
then uses a durable group guardian and Linux pidfds for SIGTERM→SIGKILL. The
guardian prevents PGID reuse during discovery; a pidfd pins each target across
the observation/signal boundary. `already_terminal` in the reply means the job
finished before the cancel — mark the record with the status it reports instead.

## Platform caveats

- **Linux active cancellation.** Safe active-process termination requires
  procfs, `pidfd_open`, and `pidfd_send_signal`. On another POSIX host the
  `ps`/`pgrep` fallback can prove a group is empty, but cancellation of active
  members returns `UNKNOWN` instead of risking a reused numeric PID or PGID.
- **No multi-node, no image resolution** — there is no container. The "image"
  recorded is the venv's interpreter path.
- The runner never downloads anything. Remote inputs are the agent's job to
  stage first (`tao-data-io`).
