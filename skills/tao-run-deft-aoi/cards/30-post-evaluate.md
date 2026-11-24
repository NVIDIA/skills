# CARD 30 — RCA on the evaluated iteration (feeds the NEXT iteration) — iteration $ITER

The driver routes here only when `deft_context.py` says the next stage is `rca`.
Loop end (KPI met or max_iterations) is NOT a card: the driver runs
`finalize_run.py` itself and reports DONE only when `deft_state.json` has
`status == "complete"`.

STATE GATE — the state decides, not you:
```bash
$DPY $SKILL_ROOT/scripts/deft_context.py --state $RD/deft_state.json --stage rca
```
| gate result | do |
|---|---|
| exit 0, `"iteration": "$ITER"` | steps 1→3 |
| non-zero exit (`durable next_stage is ...`) | STOP — print `STAGE_DONE 30` (the driver re-routes) |

1) Run gap analysis (pure Hydra CLI — this image REJECTS `-e <spec>` for gap_analysis):
```bash
bash -c 'set -e; OUT=$RD/$ITER/rca_results/$(date +%s); mkdir -p $OUT; WIN=$($DPY -c "import json;print(json.load(open(\"$RD/deft_state.json\"))[\"iterations\"][\"$ITER\"][\"inference_csv\"])" | xargs dirname); docker run --gpus all --rm --shm-size=8g -v $WS:$WS -v $RD:$RD -w $WS $DS_IMG gap_analysis vcn_aoi inference_results_dir=$WIN train_config=$RD/specs/$([ "$ITER" = baseline ] && echo baseline_spec.yaml || echo ${ITER}_spec.yaml) kpi_media_path=$WS/kpi/images results_dir=$OUT min_recall=1.0 top_k_per_label=50 > $OUT/rca.log 2>&1; ls $OUT'
```
Output MUST contain `kpi_gaps.parquet`, `threshold.txt`, `weak_samples_breakdown.txt` — or `unreachable_kpi.txt`.

2) Write `RCA_Report.md` + `rca_images/` from the container output — ONE command (deterministic; every number comes from the files):
```bash
bash -c 'OUT=$(ls -td $RD/$ITER/rca_results/*/ | head -1); $DPY $SKILL_ROOT/scripts/prepare_card_rca.py --results-dir $RD --workspace $WS --iter-label $ITER --rca-dir $OUT --min-recall 1.0'
```
| output | do |
|---|---|
| JSON with `"unreachable": true` | the abridged report is written; commit rca `--status error --summary "unreachable KPI at any threshold"` → `STAGE_DONE 30` → stop (operator decides: retrain or relabel) |
| JSON with `"unreachable": false` | step 3 |
| `prepare_card_rca: ...` error | commit rca `--status error --summary "<that error>"` → `STAGE_DONE 30` → stop |

Spot-check verdicts are recorded as pending operator review: this executor is text-only and cannot view images — a documented harness deviation; never invent a verdict.

3) Commit rca with the full artifact set — ONE command:
```bash
bash -c 'set -e; OUT=$(ls -td $RD/$ITER/rca_results/*/ | head -1); TH=$(cat ${OUT}threshold.txt); F=(); while IFS= read -r L; do [ -n "$L" ] && F+=(--rca-target-defect "$L"); done < ${OUT}rca_target_defects.txt; $DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage rca --rca-gaps ${OUT}kpi_gaps.parquet --rca-report ${OUT}RCA_Report.md --rca-threshold "$TH" "${F[@]}" --summary "rca: threshold=$TH; gaps written; report + spot-check images saved"'
```
If commit_stage REJECTS, its stderr names the missing artifact — fix exactly that and rerun this command. Never write state any other way.

Token backfill (`align_token_usage.py`) is NOT applicable on this harness (it reads Claude Code transcripts); skip it — documented harness deviation.

Final message exactly: `STAGE_DONE 30`
