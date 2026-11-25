# CARD 40 — route weak samples + record SDG skip — iteration $ITER

STATE GATE — the state decides, not you:
```bash
$DPY $SKILL_ROOT/scripts/deft_context.py --state $RD/deft_state.json
```
| `next_stage` (for `"iteration": "$ITER"`) | do |
|---|---|
| `routing` | steps 1→3 |
| `anomalygen` (routing already committed — driver re-entry) | step 3 only |
| anything else | STOP — print `STAGE_DONE 40` (the driver re-routes) |

1) Route the PRIOR phase's rca gaps (baseline.rca feeds iter1; iter(N-1).rca feeds iterN) — ONE command; writes both parquets even when empty. SDG/AnomalyGen is disabled for this pack, so NOTHING is routed to AnomalyGen: `anomalygen_gaps.parquet` is written empty and the summary records how many rows would have gone there:
```bash
bash -c 'set -e; OUT=$RD/$ITER/routing_results/$(date +%s); mkdir -p $OUT; $DPY -c "
import json, pathlib
import pandas as pd
st=json.load(open(\"$RD/deft_state.json\")); it=\"$ITER\"; n=int(it[4:])
prev=\"baseline\" if n==1 else f\"iter{n-1}\"
gp=st[\"iterations\"][prev][\"rca_gaps_parquet\"]
df=pd.read_parquet(gp)
pool=pd.read_csv(\"$WS/augmentation/mining_pool/mining_pool.csv\")
pool_labels={str(x).upper() for x in pool[\"label\"].unique()} if \"label\" in pool else {\"PASS\"}
lab=df[\"label\"].astype(str).str.upper()
mine=df[lab.isin(pool_labels)]
sdg=df[lab.isin({\"PASS\",\"EXCESS_SOLDER\",\"MISSING\",\"BRIDGE\"})]
out=pathlib.Path(\"$OUT\".strip())
mine.to_parquet(out/\"mining_gaps.parquet\"); df.iloc[0:0].to_parquet(out/\"anomalygen_gaps.parquet\")
(out/\"routing_summary.txt\").write_text(f\"prev={prev} total={len(df)} mining={len(mine)} anomalygen=0 (SDG disabled; {len(sdg)} SDG-eligible rows not routed) dropped={len(df)-len(mine)}\n\")
print((out/\"routing_summary.txt\").read_text())"'
```
If the mining subset is empty: commit routing with `--status error --summary "no rows routed to mining (SDG disabled)"` and stop after `STAGE_DONE 40`.

2) Commit routing — ONE command:
```bash
bash -c 'set -e; OUT=$(ls -td $RD/$ITER/routing_results/*/ | head -1); $DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage routing --routing-mining ${OUT}mining_gaps.parquet --routing-anomalygen ${OUT}anomalygen_gaps.parquet --summary "$(cat ${OUT}routing_summary.txt | tr -d "\n")"'
```

3) Record the AnomalyGen skip — legal because step 1 routed zero rows to AnomalyGen (commit_stage verifies the committed `routing_anomalygen_parquet` is empty). Do NOT launch any generator, do NOT stage synthetic rows:
```bash
$DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage anomalygen --skip --summary "routing produced zero AnomalyGen rows (SDG disabled for this pack: mining-only)"
```
If commit_stage rejects the skip, report its diagnostic and stop. Never commit anomalygen without `--skip`.

Final message exactly: `STAGE_DONE 40`
