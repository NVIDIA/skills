# CARD A30 — interpret: harvest recs, pick best, write results, mark DONE

STATE GATE:
```bash
bash -c 'grep -h "done ok" $RD/progress.log 2>/dev/null | tail -1; [ -f $RD/automl_result.json ] && echo RESULT_EXISTS; pgrep -f "$RD/runner/driver.py" >/dev/null && echo RUNNER_ALIVE; grep -h "runner_finished FAIL" $RD/progress.log 2>/dev/null | tail -1; echo ---GATE-END---'
```
| gate output | do |
|---|---|
| `done ok` | STOP — print `STAGE_DONE A30` |
| `RUNNER_ALIVE` | the search is NOT finished — do NOT interpret partial results. STOP — print `STAGE_DONE A30` (driver keeps waiting) |
| `RESULT_EXISTS` (runner dead) | step 1 (harvest) |
| `runner_finished FAIL`, no result | failure branch below |
| none (runner dead, no result, no FAIL) | runner was killed mid-flight: relaunch with resume — `bash -c 'cd $RD/runner && AUTOML_RESUME=1 nohup $VENV/bin/python3 $RD/runner/driver.py > $RD/runner/driver_resume.log 2>&1 & echo RESUMED'` → `STAGE_DONE A30` |

FAILURE BRANCH (`runner_finished FAIL`): `tail -40 $RD/runner/driver.log`. Known classes and exact fixes:
- `unexpected keyword argument` → card A20 step-3 note (kwarg rename / drop mounts=), fix driver.py, relaunch per the resume command above
- spec-path/`FileNotFoundError` inside rec jobs → the mounts= list is missing from runner.run — restore it exactly as card A20 step 3 defines
- anything else → `echo "interpret FAIL" >> $RD/progress.log`, print `STAGE_DONE A30`, stop (driver halts; no auto-retry)

1) Harvest: per-rec metrics, best config, results.md — ONE command (quoted heredoc: the python body is NOT shell-interpreted; wheel 7.0.1 result keys are `rec_id`/`metric`/`metric_value`/`status`/`specs`):
```bash
$VENV/bin/python3 - "$RD" <<'EOF'
import json, sys, glob
rd = sys.argv[1]
res = json.load(open(rd + "/automl_result.json"))
st = json.load(open(rd + "/state.json"))
best = res.get("best") or {}
hist = res.get("history") or []
recs = {}
try:
    from tao_automl import query_status
    ws = sorted(glob.glob(rd + "/automl_workspace/run_*"))[-1]
    for r in query_status(ws).get("recommendations", []):
        recs[r.get("rec_id")] = r
except Exception as e:
    print("query_status fallback:", e)
lines = ["# AutoML Results (bayesian, 4 recs, val_loss minimize)", "",
         f"baseline val_loss: {st.get('baseline_val_loss')}", ""]
for h in hist:
    rid = h.get("rec_id", "?")
    specs = (recs.get(rid) or {}).get("specs") or (best.get("specs") if rid == best.get("rec_id") else {})
    lines.append(f"- rec {rid}: metric={h.get('metric')} status={h.get('status','?')} params={specs}")
lines += ["", f"BEST: {json.dumps(best, default=str)[:600]}", "",
          f"vs baseline: {st.get('baseline_val_loss')} -> {best.get('metric_value')}"]
open(rd + "/results.md", "w").write("\n".join(lines) + "\n")
st["best"] = {"rec_id": best.get("rec_id"), "metric_value": best.get("metric_value"), "specs": best.get("specs")}
json.dump(st, open(rd + "/state.json", "w"), indent=1)
print("\n".join(lines[:12]))
EOF
```

2) Mark done (progress + marker) — ONE command:
```bash
bash -c 'echo "done ok" >> $RD/progress.log && touch $RD/AUTOML_DONE.marker && tail -3 $RD/progress.log'
```

Final message exactly: `STAGE_DONE A30`
