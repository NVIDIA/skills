# DEFT AOI card pack

Compiled stage cards for running the DEFT AOI loop via the token-efficient
execution kit (`skills/tao-token-efficient-execution`) — fresh headless
sessions, one card per stage, state on disk through `../scripts/commit_stage.py`.

**Measured vs running the skill in one long conversation: 9.6M → 2.5M billed
tokens (-74%), peak context 242k → 59k.** On small execution models the gap is
starker: no small model completed the raw skill honestly, while cards
completed on ~35B-class models. Historical figures; reproduce them with the
kit's `references/MEASUREMENT.md`.

## Stage flow

```
00-init-baseline-train → 10-post-train → 20-evaluate ─┬→ 30-post-evaluate (RCA)
      ↑                                               │         │
      └── 60-merge-train ← 50-mining ← 40-routing ←───┼─────────┘
                                                      └→ driver: finalize_run.py
                                        (KPI met or max_iterations → finalize)
```

`driver.sh` routes on `scripts/deft_context.py --state $RD/deft_state.json`
(the skill's durable `next_stage`), waits on container activity, and halts when
`deft_state.json` records `status == "failed"` (no auto-retry). When the next
stage is `finalize`, the driver runs `scripts/finalize_run.py` itself with the
stop reason the committed metric implies (`metric_met` or `max_iterations`)
and reports DONE only when `deft_state.json` records `status == "complete"`.

## Run it

```bash
# prerequisites: the Pi harness (`pi`) on PATH, kit install done (see the kit skill),
# WS, MODEL and DS_IMG set in ~/.tao-kit/kit.env, workspace prepared per ../SKILL.md
# (NV_PCB_Siamese layout), API key exported.
# Standalone (no plugin): export TAO_SKILL_BANK_PATH=/path/to/nvidia-tao-toolkit first.
nohup bash "$TAO_SKILL_BANK_PATH/skills/tao-run-deft-aoi/cards/driver.sh" > /dev/null &
tail -f ~/.tao-kit/deft-aoi/driver.log
```

Launching is side-effecting: complete the `tao-launch-workflow` gate first. Sessions,
`driver.log`, and the launch marker live in `~/.tao-kit/deft-aoi/` (never in
the skill-bank checkout); run artifacts land in `$WS/results/<run>/`.

Config (env or `~/.tao-kit/kit.env`): `WS`, `MODEL` and `DS_IMG` required, with
no default; the driver stops at startup and names the one that is missing.
`TRAIN_IMG`, `MOUNTS_T`, `PI_KIT_TURN_BUDGET` optional. `MODEL` is
`<provider>/<model-id>[:thinking]` of the model that executes the cards; a
`nim/` model also needs `PI_KIT_NIM_BASE_URL`, the base URL of your own
OpenAI-compatible endpoint. `DS_IMG` is a TAO data-services image that you
supply and can pull. Cards 30 and 50 and the kit's guard are compiled for one
command form, so the image must provide `gap_analysis vcn_aoi` with pure
`key=value` overrides (no `-e <spec>`), `embedding image_embeddings` and
`tmm nearest_neighbors`; an image with another command form breaks the pack.
Ask the user for the image; never guess or substitute one, not even as an
example: show the placeholder `<data-services-image-uri>`.

`GPU_MODEL` defaults to the
first `nvidia-smi --query-gpu=name,memory.total` row; set it only when that
query is unavailable. `BACKBONE` defaults to
`$WS/augmentation/backbone/c_radio_v2_b.safetensors` (what
`scripts/stage_backbone.py --workspace $WS` writes; a legacy `model.safetensors`
there is also accepted); the driver aborts at startup if it is missing. Inspect any run with:
`bash ../scripts/deft_python.sh ../scripts/deft_context.py --state <RD>/deft_state.json`.

## Version pin and known drift

- Originally authored against an earlier revision of the `tao-run-deft-aoi` skill; rebased
  onto the post-#107 state contract: `deft_state.json` (snapshot + embedded
  `events`) is the only state file, `deft_context.py` decides the next stage,
  and `finalize_run.py` owns `loop_stop`. Scripts used: `commit_stage.py`,
  `deft_context.py`, `finalize_run.py`, `prepare_card_rca.py`,
  `prepare_card_mining.py`, `deft_python.sh`. Mining-only routing (zero rows
  routed to AnomalyGen, committed with `--skip`), NV_PCB_Siamese workspace
  layout, Pi harness.
- Cards supply session-relative `--duration-sec` on commits. Card 50 uses
  `prepare_card_mining.py` and the bank's history filter to produce candidate,
  selected, and history evidence that `commit_stage.py` validates. It never asks
  the execution model to invent missing evidence.
- Card 30 uses `prepare_card_rca.py` to write the seven-section `RCA_Report.md`
  and `rca_images/` from the gap-analysis output. The executor models are
  text-only, so spot-check verdicts are recorded as pending operator review —
  a documented harness deviation.
- Loop end is deterministic and runs in the driver (`finalize_run.py`), not in a
  card. Finalization failures halt with a nonzero status.
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
