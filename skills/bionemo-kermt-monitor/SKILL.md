---
name: kermt-monitor
description: Check progress for a detached KERMT run (pretrain, finetune, or any kermt_run_detached invocation). Reads run.json, queries docker for container state, tails the pretrain/finetune log, and parses progress lines (epoch, step, val loss).
license: Apache-2.0
compatibility: Requires docker and jq. Designed for Claude Code, Codex, and Nemotron.
metadata:
  owner: evax@nvidia.com
  classification: atomic-skill
  risk_tier: skill
# Line/token budget: this file is targeted at ~120 lines / ~1500 tokens — well
# within the 500-line / 5000-token cap for skill files.
---

# kermt-monitor

Companion skill for any KERMT workflow that runs detached: the three pretrain
skills (`kermt-continue-pretrain`, `kermt-pretrain-scratch`,
`kermt-add-cmim-pretrain`) plus `kermt-finetune`. `kermt-infer` and
`kermt-embed` run blocking by default and don't need this skill, but if a
user launches them detached on purpose the monitor still works (the
workflow-dispatch in step 4 handles unknown workflows by tailing the
most-recent log file in the run dir). Reads the run directory's `run.json`,
queries docker for the container's state, surfaces the latest progress,
and either tails or follows the log.

## Hardware requirements

None. This skill only reads disk + queries docker; no GPU compute.

## Inputs

One of:

- `<run-dir>` — a positional argument pointing at the directory containing
  `run.json` (e.g. `runs/continue-pretrain_2026-05-17T10-23Z`). Preferred.
- `--container <name-or-id>` — direct container reference; the skill still
  reads `run.json` from the run dir referenced inside the container's
  inspect output if available, but works degraded-mode without it.

Optional:

- `--lines N` — number of trailing log lines to print (default 50).
- `--follow` — stream `docker logs -f` until ^C. Useful for "watch the
  loss". Without it, the skill is one-shot and exits.
- `--json` — emit a structured status report instead of human-readable text.
  Useful when the parent agent wants to take downstream action.

## Workflow

Let `RUN_DIR=$1` (or whatever path the user supplies).

1. **Locate the manifest.**
   ```
   MANIFEST=$RUN_DIR/run.json
   ```
   Refuse to proceed if it doesn't exist; surface a helpful message
   pointing the user at the run-dir convention (`runs/<workflow>_<ts>/`).

2. **Parse the manifest** (Python helper):
   ```
   workflow=$(jq -r .workflow $MANIFEST)
   container_name=...   # not directly in run.json today; the skill that
                        # launched stored it in run.json under
                        # container.name during launch (see below note).
   logs_dir=$(jq -r .logs_dir $MANIFEST)
   image_tag=$(jq -r .container.image_tag $MANIFEST)
   started_at=$(jq -r .started_at $MANIFEST)
   ```

3. **Query docker for container state.**
   ```
   docker ps --filter "name=$container_name" --format \
       '{{.ID}}\t{{.Status}}\t{{.CreatedAt}}'
   ```
   If absent, fall back to `docker inspect $container_name --format
   '{{.State.Status}} (exit {{.State.ExitCode}})'` to see whether the
   container exited (ok or failed) or was removed (`--rm` after exit).

4. **Find the live log file.**
   ```
   case "$workflow" in
     continue-pretrain|pretrain-scratch)  LOG=$logs_dir/pretrain_ddp.log ;;
     finetune)                            LOG=$logs_dir/finetune.log ;;
     *)                                   LOG=$(ls -1t $logs_dir/*.log 2>/dev/null | head -n 1) ;;
   esac
   ```
   The manifest's `workflow` field disambiguates pretrain (`pretrain_ddp.log`)
   from finetune (`finetune.log`). Other workflows fall back to the
   most-recently-modified `.log` in `$logs_dir`.

5. **Show the latest progress.**
   - `tail -n $LINES $LOG` for the raw recent output.
   - Parse the last few progress lines and surface a human-friendly
     summary. The format differs per workflow:
     - Pretrain: epoch / step / val_loss
       ```
       Current epoch: 12/100  step: 4523/9000  val_loss: 0.832 (best 0.821 @ step 4100)
       ```
     - Finetune: fold / epoch / val_<metric> (e.g. val_mae for regression,
       val_auc for classification — read `args_applied.metric` from run.json)
       ```
       Fold 0  epoch 12/30  val_mae 0.187 (best 0.182 @ epoch 9)
       ```
     ```
     Wall-clock: 1h 23m since started_at; ETA ~6h remaining.
     ```

6. **Final test-metrics block (finetune, on completion).** If `workflow` is
   `finetune` AND the container has exited cleanly (`State.Status=exited`,
   `ExitCode=0`) AND `$RUN_DIR/ckpt/fold_*/test_result.csv` exists, parse it
   and emit a per-task metric table:
   ```
   Final test metrics (per task):
     Target              MAE
     HLM_clearance       0.187
     RLM_clearance       0.213
     MDR1-MDCK_efflux    0.241
     solubility_pH6.8    0.156
   ```
   The metric column matches `args_applied.metric` (mae for regression, auc
   for classification, etc.). For multi-fold or ensemble runs, average across
   folds/models and note `± std` if std > 0. Skip silently if no
   `test_result.csv` exists (run incomplete or no test split was emitted).

7. **If `--follow`, stream live logs.**
   ```
   docker logs -f $container_name
   ```
   Wraps until ^C.

8. **Stop / cleanup hints** (printed at end of one-shot mode):
   ```
   To stop:        docker stop $container_name
   To remove:     docker rm $container_name
   To re-run:    `$(jq -r .cmd_replay $MANIFEST)`
   ```

## Hard rules

- **Read-only on the user's data.** Never modify `run.json`, never touch the
  container's checkpoint dir. The monitor only inspects.
- **Don't kill the container without explicit user instruction.** If the
  user asks to stop, run `docker stop`; if they ask to abandon, leave it
  running and just exit.
- **Don't pull or modify the kermt image.** The monitor only reads.
- **JSON output mode is non-interactive.** Skip the "press ^C to exit"
  prompts and emit a single JSON document so the parent agent can pipe it.

## Note on container_name plumbing

The run.json schema as currently written does not yet include the launched
container name — `kermt_run_detached` prints it to stdout but the runner
script doesn't capture it into run.json. The monitor falls back to a
filesystem-based lookup: list `runs/<workflow>_*/` directories and match by
mtime; or accept `--container <name>` explicitly. Follow-up: have the
launching skill record container name into run.json before exiting.

## Output (text mode, default)

```
KERMT continue-pretrain · runs/continue-pretrain_2026-05-17T10-23Z
  Container : kermt-continue-pretrain-…  (Up 1 hour, status: running)
  Image     : kermt:latest@sha256:…
  Repo      : 2fe00f9 (clean)
  Started   : 2026-05-17T10:23:14Z (1h 23m ago)
  Workflow  : continue-pretrain, pretrain_mode=hybrid, world_size=2

  Latest log (last 50 lines from $LOG):
    [Epoch 12/100] step 4523/9000 loss 0.832 lr 1.2e-4
    [val] step 4100 val_loss 0.821 (new best)
    ...

  Progress: epoch 12/100, ~12% done. ETA ~6h.
  TensorBoard: tensorboard --logdir $RUN_DIR/logs/tb
  Replay command: $(jq -r .cmd_replay $RUN_DIR/run.json)
```
