# CARD 30 — audit-directed branch: RCA (more iterations) or LOOP END — iteration $ITER

STATE GATE — the audit decides, not you:
```bash
$DPY $SKILL_ROOT/scripts/audit_deft_run.py --results-dir $RD
```
| audit `next_action` says | do |
|---|---|
| rca (any phrasing) | BRANCH A |
| loop_stop / loop-end / complete | BRANCH B |
| `DEFT_RUN_STATUS=INVALID` | repair EXACTLY the named disk inconsistency, rerun the audit, follow it. If the repair needs a state edit, that is FORBIDDEN — commit `--status error` instead and stop |
| anything else | do what it names; it is the source of truth |

## BRANCH A — RCA (feeds the NEXT iteration; committed under $ITER)

1) Run gap analysis (pure Hydra CLI — this image REJECTS `-e <spec>` for gap_analysis):
```bash
bash -c 'set -e; OUT=$RD/$ITER/rca_results/$(date +%s); mkdir -p $OUT; WIN=$($DPY -c "import json;print(json.load(open(\"$RD/deft_state.json\"))[\"iterations\"][\"$ITER\"][\"inference_csv\"])" | xargs dirname); docker run --gpus all --rm --shm-size=8g -v $WS:$WS -v $RD:$RD -w $WS $DS_IMG gap_analysis vcn_aoi inference_results_dir=$WIN train_config=$RD/specs/$([ "$ITER" = baseline ] && echo baseline_spec.yaml || echo ${ITER}_spec.yaml) kpi_media_path=$WS/kpi/images results_dir=$OUT min_recall=1.0 top_k_per_label=50 > $OUT/rca.log 2>&1; ls $OUT'
```
- Output MUST contain `kpi_gaps.parquet`. If instead `unreachable_kpi.txt` exists: commit rca with `--status error --summary "unreachable KPI at any threshold"`, then `STAGE_DONE 30`, stop.

2) Commit rca — ONE command:
```bash
bash -c 'set -e; OUT=$(ls -td $RD/$ITER/rca_results/*/ | head -1); TH=$($DPY -c "import json;print(json.load(open(\"$RD/deft_state.json\"))[\"iterations\"][\"$ITER\"][\"threshold\"])"); $DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage rca --rca-gaps ${OUT}kpi_gaps.parquet --rca-threshold "$TH" --summary "rca: gaps parquet written (threshold=$TH)"'
```
Final message exactly: `STAGE_DONE 30`

## BRANCH B — LOOP END (terminal evaluate: KPI met or max_iterations reached)

1) `$DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage loop_stop --summary "max_iterations reached; run-best per metric contract"`
2) Token backfill (`align_token_usage.py`) is NOT applicable on this harness (it reads Claude Code transcripts); skip it — documented harness deviation, do not substitute anything.
3) End the session. The driver now owns finalization: it calls `prepare_inference_spec.py`, then `audit_deft_run.py --require-complete`, and reports DONE only if both succeed. Do not prepare the handoff, re-commit loop_stop, or claim completion here. If this session is interrupted after the commit, the next driver iteration performs the same finalization.

Final message exactly: `STAGE_DONE 30`
