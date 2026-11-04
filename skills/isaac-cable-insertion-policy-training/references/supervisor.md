## Reusable supervisor

The committed `scripts/supervisor.py` owns campaign process checks. Use a new JSON plan and session directory for each training campaign; evaluate existing checkpoints in their originating session; do not generate a new Python supervisor or evaluator. Invoke every command with the selected `isaaclab_training` package as the working directory and its `.venv/bin/python`. The supervisor itself uses `uv run --no-sync` for trainer, verifier, and evaluator subprocesses.

If that environment does not exist yet, the agent first follows [Setup](setup.md).
The supervisor does not install packages or change checkouts. Its `preflight`
command verifies a prepared environment before session creation, returning JSON
on success or exit code 2 with the failure. EULA acceptance is required.

### Plan

Start from `assets/session-plan.example.json`. Replace `training_root` with the user's absolute `isaaclab_training/` directory and choose the task, budget, run specs, and evaluation seeds. Follow [Omniverse EULA](eula.md) first. A new session requires the current user's acceptance receipt; the supervisor copies its timestamp into `session.json.plan` and sets `eula_accepted=true`. A plan field cannot replace the receipt. The plan is immutable for that session. Each run has a unique `id`, hypothesis, seed, `num_envs`, `max_iterations`, `num_steps_per_env`, video settings, and JSON-valued Hydra `overrides` (including range lists). The supervisor sets the reset curriculum's `num_steps_per_env` equal to PPO's rollout length and rejects a conflicting override. See [Inputs](inputs.md) for task-space and joint-space defaults. The effective `anneal_end_iter` is recorded in the normalized plan and passed to training, preserving an explicit override. For a shortened training run, set `anneal_end_iter` explicitly if the default anneal would outlast it. Leave `parent_checkpoint` unset for fresh hyperparameter comparisons. If it is set, it must be an RSL-RL checkpoint readable with PyTorch's safe `weights_only` loader; `max_iterations` then means additional updates and integer `evaluation.checkpoint_steps` are absolute RSL-RL update labels.

`decision_mode=manual` starts the first run and pauses at `decision_pending` after its evaluation. An agent inspects the report, writes a new run-spec JSON, and queues it. Recovery of a completed run also pauses at this gate. Calling `resume` while a decision is pending does not approve another run; use `queue` after reviewing the result. This supports one-group-at-a-time tuning. `decision_mode=preplanned` runs the listed candidates sequentially without an agent, useful when the comparison was designed in advance. If a run fails, inspect and log the issue before queuing a corrected attempt with a new ID. A failed attempt still consumes training time.

`video_schedule=start_end` calculates an interval of `max_iterations × num_steps_per_env − video_length`. Isaac Lab starts the first clip at env step 1 and the second at the final `video_length` env steps. The supervisor requires two decodable MP4s before calling that run verified. A training clip near the end shows the policy used during the final rollout; use checkpoint playback if an exact final-checkpoint video is needed.

Old plans containing `isaaclab_root` are rejected for new sessions. Historical source-install sessions remain readable with `status`, but must not be resumed or mutated. Create a new standalone session and explicitly supply a compatible parent checkpoint if continuation is intended.

### Progress and video checks

The supervisor uses `<training_root>/logs/rsl_rl/displayport_insertion_rizon4s`.
Custom `output_root` values and `agent.experiment_name`/`agent.run_name` overrides
are rejected so monitoring and the trainer always use the same directory.
`run` and foreground launches return a nonzero exit code when the campaign is
blocked. A successful detached launch only confirms that monitoring started;
use `status` to determine the campaign outcome.

The supervisor sets `PYTHONUNBUFFERED=1` and `HEADLESS=1`; video runs also pass
`--viz kit`. Preflight checks TensorBoard and, when any pending run requests video,
MoviePy/FFmpeg encoding and ImageIO decoding in the training package’s `.venv`. This includes
newly queued retries. Failed preflight blocks launch and is identified in the report.

Every 10 seconds the monitor checks the process and console. When console updates
are unavailable it also probes TensorBoard, at most every 30 seconds, using a
separate `.venv` process bounded to 15 seconds. It checks only the output directory
matching this attempt's unique name. A newer `model_<step>.pt` is fallback evidence
when nonempty and unchanged across two checks; this does not prove checkpoint
validity or successful completion. Old files, other attempts, file growth alone,
and repeated steps do not reset the progress timer. TensorBoard read errors are
recorded without disabling console/checkpoint monitoring.

Before an observation timeout, the monitor refreshes all evidence. The startup
deadline applies until the first observed update; thereafter the stall deadline
applies to the last advancing update. Set these deadlines to accommodate simulator
startup and the expected PPO update duration, including TensorBoard flushing.
The total training budget still applies while progress is healthy. A timeout
records `stop_origin=supervisor`, its reason, and the last progress source/time;
it does not claim a simulator crash. `status`, `session.json`, and `report.md` use
this evidence, including updates detected without console output. Final artifact
verification remains separate from progress monitoring.

### Commands

From the standalone training package root, with `SKILL` set to the absolute `isaac-cable-insertion-policy-training` directory:

```bash
python3 "$SKILL/scripts/supervisor.py" eula-status
# After the user explicitly accepts the linked agreement:
python3 "$SKILL/scripts/supervisor.py" eula-accept --user-confirmed
training_cmd .venv/bin/python "$SKILL/scripts/supervisor.py" preflight \
  --plan /absolute/path/to/session-plan.json
training_cmd .venv/bin/python "$SKILL/scripts/supervisor.py" start \
  --plan /absolute/path/to/session-plan.json
training_cmd .venv/bin/python "$SKILL/scripts/supervisor.py" status \
  --session "$SKILL/logs/<session-id>"
training_cmd .venv/bin/python "$SKILL/scripts/supervisor.py" queue \
  --session "$SKILL/logs/<session-id>" --spec /absolute/path/to/next-run.json
training_cmd .venv/bin/python "$SKILL/scripts/supervisor.py" resume \
  --session "$SKILL/logs/<session-id>"
training_cmd .venv/bin/python "$SKILL/scripts/supervisor.py" finish \
  --session "$SKILL/logs/<session-id>" --reason "No useful next hypothesis"
```

`start` creates a UTC-stamped session log before launching. The default `--launcher auto` chooses `tmux` when installed, otherwise a detached process. Use `--launcher foreground` for short tests or `--launcher systemd` on a host with working persistent user services. The supervisor itself does not require `tmux`; the same `run` subcommand executes in every mode. If the environment terminates detached descendants at tool-call exit, use `systemd` or a persistent host session rather than claiming unattended durability.

Read `report.md` first, and `session.json.state` for full status. See [Logging](logging.md)
for the compact layout and human-readable names. `events.jsonl` is the append-only history.
`status` checks PID plus `/proc` birth time. Resume reconciles a live trainer/evaluator;
committed evaluation seeds are not repeated. If a trainer disappeared without a recorded
exit code, verified artifacts may establish completion while the exit remains unavailable.
A known nonzero exit cannot be recovered as success. Uncertain runs block the series.
The supervisor checks its script hashes on resume; if the skill changed, restore the
original skill version from version control or start a new session. Scripts are not copied
into session logs. It never
silently mixes supervisor versions within a campaign.

At an idle decision point, record findings without adding another report:

```bash
training_cmd .venv/bin/python "$SKILL/scripts/supervisor.py" note \
  --session "$SKILL/logs/<session-id>" \
  --message "Lower action scale improved fixed-distribution success." --finding \
  --next-action "Confirm the selected checkpoint on independent seeds."
```

### Independent validation with fresh seeds

Use this workflow when asked to validate a trained policy on new seeds. Select the
checkpoint using existing screening/tuning results before examining independent
results. Reuse its originating session and logical run ID; do not retrain, create a
new campaign, or edit the immutable session plan.

1. Read `report.md`, `session.json`, and the per-run `evaluations.jsonl` journals.
   Confirm the run has verified artifacts and no trainer is active. If an evaluation
   is pending, recover it with its original configuration before requesting fresh seeds.
2. Honor explicit seeds and evaluation size. Otherwise choose four distinct unused
   seeds, starting at 37001 and skipping seeds already present in the session plan,
   run specs, evaluation results, requests in `events.jsonl`, or journals (including
   failed/partial evaluations). For example, after tuning on 35001–35004, use
   37001–37004 only if they have not been used. Explicit overlap with screening/tuning
   seeds cannot be labeled independent; explain the overlap and select fresh seeds
   when the user permits substitution. Repeating an existing configuration is a
   recovery/cache-reuse operation, not new validation evidence.
3. Create an evaluation JSON outside the session directory. Set `group=independent`
   and name the selected checkpoint explicitly. Match `num_envs` and `max_steps`
   to the prior comparison unless the user requests a different evaluation. Keep the
   evaluator's fixed reset/noise/success conditions unchanged. A common configuration
   is four seeds × 256 environments, or 1,024 first-episode outcomes:

```json
{"enabled": true, "group": "independent", "seeds": [37001, 37002, 37003, 37004],
 "num_envs": 256, "max_steps": 260, "checkpoint_steps": [999]}
```

4. Check the remembered EULA receipt and available GPU, then invoke the committed
   evaluator from `training_root` through the session's recorded launch environment:

```bash
python3 "$SKILL/scripts/supervisor.py" eula-status
training_cmd .venv/bin/python "$SKILL/scripts/supervisor.py" evaluate \
  --session "$SKILL/logs/<session-id>" --run-id baseline \
  --config /absolute/path/to/independent-evaluation.json
```

The run ID and checkpoint number above are examples; resolve them from the selected
session. The evaluator checks code/source/configuration provenance and rejects
independent seeds overlapping completed screening/tuning results. The broader
unused-seed check above also avoids reusing failed or earlier independent trials.
`evaluate` runs in the foreground: monitor it to completion, or use a durable host
session and verify its exit and journal. Do not call `resume` to launch validation.

5. Verify a successful result and process exit for every requested seed, no pending
   evaluation/error, and an independent entry in the existing report. Report per-seed
   and aggregate successes/episode counts, the evaluation conditions, and elapsed
   evaluation time separately from tuning results. Failed or missing seeds are not
   a complete validation. Leave campaign training decisions unchanged. Record findings
   with `note`; do not create a second report.

New evaluation seeds test the selected policy on new randomized episodes, not
repeatability across training seeds or hardware readiness. If these outcomes guide
further tuning or checkpoint selection, disclose that use and reserve another fresh
seed set for the next independent validation.

For screening other checkpoints, the same `evaluate` command accepts a configuration
with `group=screening` or `group=tuning`. Each request uses the originating run's
log/journal without changing its training plan.

Use disjoint screening/tuning and independent seeds. Recover an interrupted evaluation
by repeating the same `evaluate` command/config. Invalid checkpoint requests are
rejected before becoming pending. To abandon a failed request once its evaluator
has stopped, run `cancel-evaluation --session <absolute-session-path>`, then submit
the corrected request. Cancellation retains results, failure evidence, and consumed
budget; it does not terminate a running evaluator or replenish the budget.

Each new run fingerprints the editable training source, `pyproject.toml`, and
`uv.lock`. Evaluation checks that fingerprint and the saved `env.yaml`/`agent.yaml`
hashes before launch or cache reuse. Restore the original files if they changed;
do not replace recorded fingerprints to bypass the check. Historical runs without
these fingerprints cannot receive verified campaign evaluation. This does not pin
the checkout: updated source can be used for a new run.

Evaluation role and every per-seed
outcome remain recorded. External historical checkpoints should be recorded as imported
baselines with their original provenance; never count their inherited training as new
training in this session. Temporary input plan/spec/config files are outside the session;
their normalized contents are retained in `session.json`.

Enter all commands through the shared [process environment](environment.md),
including tmux/systemd launches; the selected environment is recorded per attempt.

The supervisor does not invoke Codex itself. At `decision_pending` or `blocked`, an active agent or scheduled task must inspect the compact status and report, diagnose errors, and queue the next run. The agent must deliver the final report to the user when the campaign completes; a file on disk does not send a message.
