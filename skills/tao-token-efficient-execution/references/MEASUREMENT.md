# Measurement protocol — one long conversation vs card execution

How to reproduce (or refresh) the token figures published in `SKILL.md` and
in the shipped pack READMEs. Both arms run the **same task, on the same host,
through the same harness, provider, and execution model**; the only variable
is execution mode. Token accounting is `scripts/analyze_usage.py` over the
Pi session JSONL each arm writes.

## Fixed for both arms

| Item | Value |
|---|---|
| Harness | Pi `>= 0.85.1` (record `pi --version`) with the kit extensions `adapters/pi/nvidia-provider.ts`, `guard.ts`, `recorder.ts` loaded in both arms |
| Provider + execution model | one `MODEL=<provider>/<model-id>[:off]` value for both arms (record it and the endpoint host); never compare arms across providers |
| Host | one GPU host; record `GPU_MODEL`, `TRAIN_IMG`, `DS_IMG` (DEFT), the AutoMLRunner wheel version (AutoML), and the bank commit |
| Workspace | a fresh copy of the same `$WS` per arm, so neither arm reuses the other's outputs |
| Accounting | `python3 scripts/analyze_usage.py <arm session dir>`; record `bill`, `no-cache bill`, and `peak context` |

Per-session turn budgets do not apply to the long-conversation arm
(`PI_KIT_TURN_BUDGET=0`); the card arm keeps the driver default.

## Workflows

### DEFT AOI loop (`tao-run-deft-aoi`)

- **Task:** the full DEFT loop with card 00's parameters — project
  `NV_PCB_Siamese`, KPI target `FAR < 0.5 %`, `--max-iterations 3`, 1 GPU,
  10 epochs, batch size 8, 1 SDG, `--min-similarity 0.9`.
- **Dataset:** the NV_PCB_Siamese workspace layout described in
  `skills/applications/tao-run-deft-aoi/SKILL.md`, with the C-RADIOv2-B
  backbone staged at `$WS/augmentation/backbone/c_radio_v2_b.safetensors`.
- **Outcome check (both arms):** `deft_context.py --state <RD>/deft_state.json`
  reports `next_stage == "complete"`, `deft_state.json` has
  `status == "complete"`, and its events end with the `loop_stop` stage
  committed by `finalize_run.py` (summary: metric met, or iteration limit
  reached). Record the stop reason, iteration count, and final FAR for each
  arm.
- **Pass threshold:** card-arm `bill` ≤ 0.40 × long-arm `bill`, and card-arm
  peak context ≤ 0.35 × long-arm peak context.

### AutoML, VCN classify (`tao-run-automl`)

- **Task:** bayesian search, exactly 4 recommendations, metric `val_loss`
  minimize, 10 epochs, batch size 8, 1 GPU / 1 concurrent job, search space
  `train.optim.lr [5e-6, 5e-4]`, `train.optim.weight_decay [1e-4, 0.05]`,
  `model.classify.train_margin_euclid [1.0, 3.0]`, no WandB, plus the zero-LR
  baseline eval (the parameters baked into the pack's cards 10 and 20).
- **Dataset:** the AOI workspace in
  `skills/applications/tao-run-automl/cards/README.md` ("Workspace layout") —
  `train/base/training_set.csv` (210 rows), `train/base/validation_set.csv`
  (249 rows), images under `kpi/images/`, `specs/baseline_spec.yaml`.
- **Outcome check (both arms):** a baseline `val_loss` is recorded, and the
  AutoMLRunner reports 4 completed recommendations each with a `val_loss`
  value (the `rec_id` / `metric` / `metric_value` keys card 30 reads). The
  card arm additionally ends with `AUTOML_DONE.marker` and no trailing FAIL
  in `progress.log`. Record the best `val_loss` for each arm.
- **Pass threshold:** card-arm `bill` ≤ 0.15 × long-arm `bill`, and card-arm
  peak context ≤ 0.35 × long-arm peak context.

The thresholds allow ≈1.5× the published card/long ratios for run-to-run
variance. A reproduction **fails** when either arm misses its outcome check
— an arm that did not finish honestly has no comparable bill — or when a
ratio exceeds its threshold. Report the per-arm numbers either way.

## Running the arms

Set up once (no secrets in `kit.env`; export the provider key in the shell):

```bash
KIT=$TAO_SKILL_BANK_PATH/skills/core/tao-token-efficient-execution
ADAPTER=$KIT/adapters/pi
. ~/.tao-kit/kit.env
. "$KIT/scripts/model_preflight.sh"; kit_prepare_model_env; kit_check_model_key "[measure]"
MEAS=$HOME/.tao-kit/measure/<workflow>/$(date +%Y%m%dT%H%M%S); mkdir -p "$MEAS/long"
```

**Card arm.** Launch the pack driver as its README describes (after the
`tao-launch-workflow` gate). Its sessions land in `~/.tao-kit/<pack>/sessions`;
point `SESSION_DIR` at `$MEAS/cards` to keep each measurement separate.

**Long-conversation arm.** One Pi session that loads only the application
skill and runs the whole workflow:

```bash
PI_KIT_WS=$WS PI_KIT_TURN_BUDGET=0 \
pi -p -na --tools read,bash,edit,write --no-context-files --no-skills --no-extensions \
  --skill "$TAO_SKILL_BANK_PATH/skills/applications/<tao-run-deft-aoi|tao-run-automl>" \
  -e "$ADAPTER/nvidia-provider.ts" -e "$ADAPTER/guard.ts" -e "$ADAPTER/recorder.ts" \
  --session-dir "$MEAS/long" --model "$MODEL" "$(cat "$MEAS/long_prompt.txt")" \
  > "$MEAS/long/stdout.log" 2>&1
```

`long_prompt.txt` states the workflow's task parameters above verbatim, the
workspace path, that the launch is pre-approved (the operator completed the
`tao-launch-workflow` gate), and that the agent must work alone and finish
with the outcome artifacts — the same information the cards carry, without
the cards. With `--no-skills`, Pi loads only the `--skill` path given.

Then, for each arm:

```bash
python3 "$KIT/scripts/analyze_usage.py" "$MEAS/long"
python3 "$KIT/scripts/analyze_usage.py" "$MEAS/cards"
```

The long arm must run in Pi: `analyze_usage.py` reads Pi session JSONL only,
so a long conversation run in another harness is not comparable through this
script.

## Reading the bill

`bill` weights tokens by Anthropic-style price ratios (output ×5, cache read
×0.1, cache write ×1.25, uncached input ×1). A provider that reports no cache
accounting — the endpoint the `nim/` provider was verified against returns
prompt/completion tokens only (recorded in `adapters/pi/nvidia-provider.ts`) —
yields `bill == no-cache bill`. That penalises the long arm more than a
caching provider would, which is why arms are only compared on the same
provider and both bills are reported.

## What to record

One row per arm: date, bank commit, Pi version, `MODEL`, endpoint host,
`GPU_MODEL`, images / wheel version, outcome-check result, stop reason or best
`val_loss`, session count, `bill`, `no-cache bill`, peak context.

## Provenance of the published figures

The figures in `SKILL.md` and the pack READMEs predate this protocol (PR #259,
older images and wheels). What the bank records about them:

- AutoML card arm: completed end-to-end on a ~35B-class execution model with
  zero guard fires (VCN classify, 4 bayesian recs).
- The raw skill did not complete honestly on any of the small models measured
  (`SKILL.md`, "Model floor"), so the published long-conversation figures were
  not produced on the ~35B card-arm model. The model and provider behind each
  long-conversation figure, the Pi version, and the per-arm outcome results
  are not recorded here.
- A HuggingFace-finetune measurement was taken on a pack that does not ship in
  this bank; it is not published because it cannot be reproduced from here.

Treat the published numbers as historical until a run under this protocol
replaces them, and record that run's table alongside the pack README it
updates.
