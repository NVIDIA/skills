# DEFT AOI card pack

Compiled stage cards for running the DEFT AOI loop via the token-efficient
execution kit (`skills/core/tao-token-efficient-execution`) — fresh headless
sessions, one card per stage, state on disk through `../scripts/commit_stage.py`.

**Measured vs running the skill in one long conversation: 9.6M → 2.5M billed
tokens (-74%), peak context 242k → 59k.** On small execution models the gap is
starker: no small model completed the raw skill honestly, while cards
completed on ~35B-class models.

## Stage flow

```
00-init-baseline-train → 10-post-train → 20-evaluate → 30-post-evaluate (RCA)
      ↑                                                        │
      └── 60-merge-train ← 50-mining ← 40-routing ←────────────┘
                                        (loop until KPI gate or max iters)
```

`driver.sh` routes on the last `status=="ok"` entry in `$RD/loop_log.jsonl`,
waits on container activity, and halts on committed errors (no auto-retry).

## Run it

```bash
# prerequisites: kit install done (see the kit skill), WS set in ~/.tao-kit/kit.env,
# workspace prepared per ../SKILL.md (NV_PCB_Siamese layout), API key exported.
# Standalone (no plugin): export TAO_SKILL_BANK_PATH=/path/to/tao-skill-bank first.
nohup bash "$TAO_SKILL_BANK_PATH/skills/applications/tao-run-deft-aoi/cards/driver.sh" > /dev/null &
tail -f ~/.tao-kit/deft-aoi/driver.log
```

Config (env or `~/.tao-kit/kit.env`): `WS` required; `MODEL`, `TRAIN_IMG`,
`DS_IMG`, `MOUNTS_T`, `PI_KIT_TURN_BUDGET` optional. Audit any run with:
`bash ../scripts/deft_python.sh ../scripts/audit_deft_run.py --results-dir <RD>`.

## Version pin and known drift

- Authored and validated against the `tao-run-deft-aoi` skill at MR-123
  (scripts: `commit_stage.py`, `audit_deft_run.py`, `deft_python.sh`,
  `metric_contract.py`), mining-only routing (AnomalyGen committed with
  `--skip`), NV_PCB_Siamese workspace layout, Pi harness.
- Cards supply session-relative `--duration-sec` on commits. Card 50 uses
  `prepare_card_mining.py` and the bank's history filter to produce candidate,
  selected, and history evidence required by the current audit. It never asks
  the execution model to invent missing evidence.
- A committed `loop_stop` is not completion: the driver prepares the inference
  handoff and requires a successful completion audit before reporting DONE.
  Finalization failures halt with a nonzero status.
- These contract repairs have offline regression coverage. Re-validate the
  pack end-to-end on the selected GPU/image before an unattended full-loop run;
  the historical measurements above are not current-version validation.
- Cards are compiled artifacts: when this skill's contract changes, update the
  cards in the same MR, or re-author the pack (kit skill → authoring prompt).
- Relaunching after a completed run starts a fresh run (the driver refreshes
  its launch marker on DONE). Never delete
  `~/.tao-kit/deft-aoi/.launch_marker` while a run is active — the driver
  aborts if it vanishes mid-run.

## Execution-model guidance (measured)

Frontier models run this pack flawlessly. A ~400B MoE completed it with
correct KPIs (a few guard saves). A ~35B MoE completed the loop but produced
unreliable stage labels — fine for smoke/dev, not for unattended KPI claims.
If the execution model fails a card, the driver halts: report it, don't patch
the cards around it.
