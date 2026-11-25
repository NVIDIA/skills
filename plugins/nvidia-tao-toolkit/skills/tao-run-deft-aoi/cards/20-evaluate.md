# CARD 20 — evaluate with the metric contract — iteration $ITER

TAO nests inference output: the CSVs are at EXACTLY
`$RD/$ITER/inference/best_val/inference/inference.csv` and
`$RD/$ITER/inference/latest/inference/inference.csv` (note the doubled `inference/`).

STATE GATE:
```bash
bash -c 'ls $RD/$ITER/inference/best_val/inference/inference.csv $RD/$ITER/inference/latest/inference/inference.csv 2>/dev/null | wc -l'
```
| gate output | do |
|---|---|
| `2` | steps 1→2 |
| `1` | if only one checkpoint existed (best==latest), continue with that one CSV for both roles; else re-run the recorded inference chain from `tail -30 $RD/commands.log`, print `STAGE_DONE 20`, stop |
| `0` | inference containers likely still running or died: if `docker ps` shows tao-toolkit, print `STAGE_DONE 20` and stop (driver re-enters); else re-run the recorded chain, print `STAGE_DONE 20`, stop |

1) Run the bundled evaluator on BOTH CSVs (one command; emits `metric_result.json` each):
```bash
bash -c 'set -e; for N in best_val latest; do $DPY $SKILL_ROOT/scripts/analyze_kpi.py $RD/$ITER/inference/$N/inference/inference.csv --output-dir $RD/$ITER/inference/${N}_eval --score-column siamese_score; done; grep -H "\"value\"" $RD/$ITER/inference/*_eval/metric_result.json'
```

2) Pick the winner by the contract (FAR, `<`, lower wins) and commit — ONE command, one audited transaction:
```bash
bash -c 'set -e; W=$($DPY -c "import json;a=json.load(open(\"$RD/$ITER/inference/best_val_eval/metric_result.json\"));b=json.load(open(\"$RD/$ITER/inference/latest_eval/metric_result.json\"));print(\"best_val\" if a[\"value\"]<=b[\"value\"] else \"latest\")"); M=$RD/$ITER/inference/${W}_eval/metric_result.json; V=$($DPY -c "import json;print(json.load(open(\"$M\"))[\"value\"])"); TH=$($DPY -c "import json;print(json.load(open(\"$M\")).get(\"threshold\",0))"); T=$RD/$ITER/train; CK=$([ "$W" = best_val ] && ls -t $T/model_epoch_*.pth | tail -1 || ls -t $T/model_epoch_*.pth | head -1); $DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage evaluate --metric-result "$M" --best-ckpt "$CK" --inference-csv $RD/$ITER/inference/$W/inference/inference.csv --training-spec $RD/specs/$([ "$ITER" = baseline ] && echo baseline_spec.yaml || echo ${ITER}_spec.yaml) --threshold "$TH" --summary "Evaluate: FAR=$V ($W)"'
```
If commit_stage rejects, its stderr names the missing/invalid evidence — fix exactly that, rerun this command. Never write state or the log any other way, never invent a value.

Final message exactly: `STAGE_DONE 20`
