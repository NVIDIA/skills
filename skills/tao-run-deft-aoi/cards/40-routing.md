# CARD 40 — route weak samples + record SDG skip — iteration $ITER

STATE GATE:
```bash
$DPY $SKILL_ROOT/scripts/audit_deft_run.py --results-dir $RD
```
| audit output | do |
|---|---|
| `next_action` names routing for $ITER | steps 1→3 |
| `next_action` names anomalygen for $ITER (routing already committed — driver re-entry) | step 3 only |
| `DEFT_RUN_STATUS=INVALID` | repair exactly the named disk inconsistency (never edit state/log by hand); rerun the audit |
| anything else | do what `next_action` names — it is the source of truth |

1) Route the PRIOR phase's rca gaps (baseline.rca feeds iter1; iter(N-1).rca feeds iterN) — ONE command; writes both parquets even when empty:
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
ag=df[lab.isin({\"PASS\",\"EXCESS_SOLDER\",\"MISSING\",\"BRIDGE\"})]
out=pathlib.Path(\"$OUT\".strip())
mine.to_parquet(out/\"mining_gaps.parquet\"); ag.to_parquet(out/\"anomalygen_gaps.parquet\")
(out/\"routing_summary.txt\").write_text(f\"prev={prev} total={len(df)} mining={len(mine)} anomalygen={len(ag)} dropped={len(df)-len(mine.index.union(ag.index))}\n\")
print((out/\"routing_summary.txt\").read_text())"'
```
If BOTH subsets are empty: commit routing with `--status error --summary "all labels dropped"` and stop after `STAGE_DONE 40`.

2) Commit routing — ONE command:
```bash
bash -c 'set -e; OUT=$(ls -td $RD/$ITER/routing_results/*/ | head -1); $DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage routing --routing-mining ${OUT}mining_gaps.parquet --routing-anomalygen ${OUT}anomalygen_gaps.parquet --summary "$(cat ${OUT}routing_summary.txt | tr -d "\n")"'
```

3) SDG/AnomalyGen is DISABLED for this run (mining-only condition). NOTE: the skill documents the anomalygen skip for zero routed rows; skipping on a run-level condition is a documented DEVIATION of this run — the summary below states the real reason honestly. Do NOT launch any generator, do NOT stage synthetic rows:
```bash
$DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage anomalygen --skip --summary "SDG disabled for this run (mining-only condition); no synthetic rows"
```
If `--skip` is rejected as unknown, rerun with `--status ok --summary "skipped: SDG disabled (mining-only condition); zero synthetic rows"` and no artifact flags.

Final message exactly: `STAGE_DONE 40`
