# CARD 60 — assemble+validate training CSV, commit data_merge, launch train (detached) — iteration $ITER

STATE GATE:
```bash
$DPY $SKILL_ROOT/scripts/audit_deft_run.py --results-dir $RD
```
Proceed only if `next_action` names data_merge (or merge/assemble) for $ITER; otherwise do what it names.

1) Assemble the combined CSV with monotonic growth + provenance + label normalization — ONE command:
   (iter1 = base_train with `kpi/images/` prefixed ONCE onto relative paths + mining_pool; iterN>1 = previous combined [already workspace-coordinates — do NOT re-prefix] + mining_pool. label: keep `PASS` uppercase, lowercase+strip everything else.)
```bash
bash -c 'set -e; $DPY -c "
import pandas as pd, pathlib
it=\"$ITER\"; n=int(it[4:]); rd=pathlib.Path(\"$RD\")
def norm(df):
    df[\"label\"]=[l if str(l)==\"PASS\" else str(l).lower().strip() for l in df[\"label\"]]; return df
if n==1:
    base=pd.read_csv(\"$WS/train/base/training_set.csv\")
    for c in (\"input_path\",\"golden_path\"):
        base[c]=[p if str(p).startswith((\"kpi/images/\",\"results/\",\"/\")) else \"kpi/images/\"+str(p) for p in base[c]]
    base[\"__src\"]=\"base_train\"
else:
    base=pd.read_csv(rd/f\"iter{n-1}/dataset/train_combined_iter{n-1}.csv\"); base[\"__src\"]=\"previous_iter_train\"
mp=rd/it/\"mining_filter/mining_pool.csv\"
mined=pd.read_csv(mp) if mp.exists() and mp.stat().st_size>1 else pd.DataFrame(columns=base.columns)
if len(mined): mined[\"__src\"]=\"mining_pool\"
comb=norm(pd.concat([base,mined],ignore_index=True))
prov=comb[[\"input_path\",\"golden_path\",\"label\",\"__src\"]].rename(columns={\"__src\":\"source\"})
comb=comb.drop(columns=[\"__src\"])
out=rd/it/\"dataset\"; out.mkdir(parents=True,exist_ok=True)
comb.to_csv(out/f\"train_combined_{it}.csv\",index=False)
prov.to_csv(out/f\"train_combined_{it}_provenance.csv\",index=False)
print(f\"rows={len(comb)} mined={len(mined)}\")"'
```

2) Validate (existence + leakage + REQUIRED proof JSON) — hard stop on failure:
```bash
$DPY $SKILL_ROOT/scripts/validate_training_csv.py --csv $RD/$ITER/dataset/train_combined_$ITER.csv --workspace-root $WS --validation-csv $WS/train/base/validation_set.csv --report-json $RD/$ITER/dataset/merge_validation.json
```
Known fix — FATAL "file missing" on base rows only: the `kpi/images/` prefix was applied wrong; fix the CSV (never bypass the FATAL), rerun step 1 then 2.
On leakage: commit data_merge `--status error` → `STAGE_DONE 60` → stop.

3) Commit data_merge — do NOT launch training before this commit succeeds:
```bash
$DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage data_merge --combined-csv $RD/$ITER/dataset/train_combined_$ITER.csv --provenance-csv $RD/$ITER/dataset/train_combined_${ITER}_provenance.csv --merge-validation-report $RD/$ITER/dataset/merge_validation.json --summary "validated combined training CSV"
```

4) Write the $ITER train spec + launch DETACHED — ONE command. ASYMMETRIC spec rule: ONLY `train_dataset.images_dir` moves to `/data/workspace`; the validate/test/infer dataset blocks KEEP `/data/datasets/NV_PCB_Siamese/images` (a bulk sed on images_dir breaks them — the sed below touches only the first occurrence, which is train_dataset in this spec):
```bash
bash -c 'set -e; SPEC=$RD/specs/${ITER}_spec.yaml; cp $RD/specs/baseline_spec.yaml $SPEC; PREV=$($DPY -c "import json;st=json.load(open(\"$RD/deft_state.json\"));it=\"$ITER\";n=int(it[4:]);prev=\"baseline\" if n==1 else f\"iter{n-1}\";print(st[\"iterations\"][prev][\"best_ckpt_path\"])"); PREVC=$(echo $PREV | sed "s|$RD|/results|"); RUN=$(basename $RD); sed -i -E -e "0,/^results_dir:/s|^results_dir:.*|results_dir: /results/$ITER|" -e "0,/(    images_dir:)/s|(    images_dir:).*|\1 /data/workspace|" -e "s|(      csv_path:).*training_set.csv|\1 /data/workspace/results/$RUN/$ITER/dataset/train_combined_$ITER.csv|" $SPEC; grep -q "pretrained_model_path" $SPEC && sed -i -E "s|(  pretrained_model_path:).*|\1 $PREVC|" $SPEC || sed -i "/^train:/a\\  pretrained_model_path: $PREVC" $SPEC; grep -nE "results_dir|images_dir|csv_path|pretrained_model_path" $SPEC | head -12; nohup docker run --rm --gpus all --shm-size=8g $MOUNTS $TRAIN_IMG visual_changenet train -e /results/specs/${ITER}_spec.yaml > $RD/$ITER/train/train.log 2>&1 & echo LAUNCHED'
```
Checkpoint paths in specs are CONTAINER paths (`/results/...` or `/data/workspace/...`), NEVER host `/home/...` paths.
Do NOT commit a train stage here (card 10 does that after PASS).

After `LAUNCHED` prints: final message exactly `STAGE_DONE 60` and stop.
