# AutoML card pack

Compiled stage cards for running `tao-run-automl` (AutoMLRunner wheel,
bayesian search) via the token-efficient execution kit
(`skills/core/tao-token-efficient-execution`) — fresh headless sessions, one
card per stage, progress marks in `$RD/progress.log`.

**Measured vs running the skill in one long conversation: 5.8M → 0.5M billed
tokens (-91%), peak context 198k → 42k.** Validated end-to-end on a ~35B-class
execution model with zero guard fires (VCN classify, 4 bayesian recs); the
raw-skill arm on the same model failed to complete honestly.

## Stage flow

```
00-preflight → 10-baseline-eval → 20-launch-recs (runner detached) → 30-interpret
```

- `10-baseline-eval` scores the untrained init via a zero-LR trick and gates
  on a PID file (never on pgrep of command text).
- `20-launch-recs` starts the AutoMLRunner detached with `session_id` pinned
  so interrupted runs resume the same search instead of restarting it.
- `30-interpret` harvests per-rec metrics through the wheel's own result keys
  (`rec_id`, `metric`, `metric_value`), writes `$RD/results.md`, marks DONE.

`driver.sh` owns run-dir selection (`automl2_*` under `$WS/results`), routes
on the last ok mark, and halts when the latest mark is a FAIL (no auto-retry).

## Run it

```bash
# prerequisites: kit install done, WS + VENV set in ~/.tao-kit/kit.env, API key exported.
# Standalone (no plugin): export TAO_SKILL_BANK_PATH=/path/to/tao-skill-bank first.
nohup bash "$TAO_SKILL_BANK_PATH/skills/applications/tao-run-automl/cards/driver.sh" > /dev/null &
tail -f ~/.tao-kit/automl/driver.log
```

Config (env or `~/.tao-kit/kit.env`): `WS`, `VENV` required; `MODEL`,
`TRAIN_IMG`, `SB` (skill bank the cards read references from, defaults to
this bank), `PI_KIT_TURN_BUDGET` optional.

## Version pin

- Authored against `tao-run-automl` with the 7.0.1 AutoMLRunner wheel
  (DockerSDK path: explicit `mounts` required via platform kwargs), VCN
  classify on an AOI workspace, Pi harness.
- The default `TRAIN_IMG` is resolved from `images.tao_toolkit.pyt` in
  `$SB/versions.yaml`, using the same bank that supplies recommendation templates.
  An explicit override is preserved, but must match that bank's template schema;
  the old 6.26.3 image rejects the current `train.checkpointer` field.
  Baseline eval still copies `$WS/specs/baseline_spec.yaml`; validate that
  workspace spec against the selected image during preflight.
- Historical measurements used older images/wheels. Resolving the current
  image fixes the known default mismatch; it is not an end-to-end validation
  claim for the current image and AutoMLRunner combination.
- The dataset/model specifics (VCN classify spec fields, AOI CSV layout) are
  baked into the cards — that is what makes them cheap to execute. For a
  different network or dataset, re-author the pack (kit skill → authoring
  prompt) rather than hand-editing these cards.
- Relaunching after a completed run starts a fresh run (the driver refreshes
  its launch marker on DONE). Never delete
  `~/.tao-kit/automl/.launch_marker` while a run is active.

## Baseline honesty note

The pack's "baseline" is the untrained initialization scored with the
zero-LR trick — a floor for "did the search improve anything", not a trained
reference. Improvements over it are search-quality signals, not deploy gates.
