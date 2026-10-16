# CARD A10 — automatic baseline eval job (detached lr=0 trick)

WHY lr=0: AutoML recs start from the C-RADIOv2-B backbone + fresh heads; no full-model
checkpoint exists and `evaluate` requires one. A 1-epoch train with `train.optim.lr: 0.0`
changes no weights (AdamW decoupled weight decay also scales by lr), materializes the init
checkpoint, AND logs the baseline `val_loss` — the number the launch review needs.

STATE GATE:
```bash
bash -c 'grep -h "baseline_eval ok" $RD/progress.log 2>/dev/null | tail -1; [ -f $RD/baseline/BASELINE_JOB_DONE ] && echo MARKER_DONE; [ -f $RD/baseline/job.pid ] && kill -0 $(cat $RD/baseline/job.pid) 2>/dev/null && echo JOB_RUNNING; [ -s $RD/baseline/train.log ] && echo LOG_EXISTS; echo ---GATE-END---'
```
| gate output | do |
|---|---|
| `baseline_eval ok` | STOP — print `STAGE_DONE A10` |
| `MARKER_DONE` (no ok line) | step 3 only (harvest) |
| `JOB_RUNNING` | STOP — print `STAGE_DONE A10` (driver waits) |
| `LOG_EXISTS`, no marker, not running | job died: `tail -30 $RD/baseline/train.log`; if the error is a spec field, fix it in `$RD/baseline/baseline_lr0_spec.yaml` (known: backbone must be `/data/pretrained_models/C-RADIOv2_B.safetensors`, never `.pth`), then step 2 |
| none of the above | steps 1→2 |

1) Write the lr=0 spec from the workspace baseline spec — ONE command:
```bash
bash -c 'cp $WS/specs/baseline_spec.yaml $RD/baseline/baseline_lr0_spec.yaml && sed -i -E -e "0,/^results_dir:/s|^results_dir:.*|results_dir: /results|" -e "s|^(  num_epochs:).*|\1 1|" -e "s|^(  checkpoint_interval:).*|\1 1|" -e "s|^(  validation_interval:).*|\1 1|" -e "s|(    batch_size:).*|\1 8|" -e "s|(    lr:).*|\1 0.0|" -e "s|(    pretrained_backbone_path:).*|\1 /data/pretrained_models/C-RADIOv2_B.safetensors|" $RD/baseline/baseline_lr0_spec.yaml && grep -nE "results_dir|num_epochs|lr:|batch_size|pretrained_backbone_path" $RD/baseline/baseline_lr0_spec.yaml | head -8'
```
(`results_dir: /results` — the container mounts `$RD/baseline` at `/results`, so outputs land at `$RD/baseline/train/...` exactly where the harvest reads them. `/results/baseline` would nest an extra `baseline/`.)

2) Launch the baseline job DETACHED (train lr=0 → marker), then stop:
```bash
bash -c 'printf "#!/bin/bash\ndocker run --rm --gpus all --shm-size=8g -v $WS:/data/workspace -v $RD/baseline:/results -v $WS/kpi/images:/data/datasets/NV_PCB_Siamese/images -v $WS/train/base:/data/datasets/NV_PCB_Siamese/csv -v $WS/kpi:/data/datasets/NV_PCB_Siamese/kpi -v $WS/augmentation/backbone/model.safetensors:/data/pretrained_models/C-RADIOv2_B.safetensors $TRAIN_IMG visual_changenet train -e /results/baseline_lr0_spec.yaml > $RD/baseline/train.log 2>&1\ntouch $RD/baseline/BASELINE_JOB_DONE\n" > $RD/baseline/run_baseline.sh && chmod +x $RD/baseline/run_baseline.sh && { nohup $RD/baseline/run_baseline.sh > /dev/null 2>&1 & echo $! > $RD/baseline/job.pid; } && echo LAUNCHED'
```
After `LAUNCHED`: final message exactly `STAGE_DONE A10` and stop.

3) Harvest (only when MARKER_DONE): record baseline val_loss into state.json and mark — ONE command:
```bash
bash -c 'set -e; tail -2 $RD/baseline/train.log | grep -q "Execution status: PASS" || { echo "baseline_eval FAIL" >> $RD/progress.log; exit 0; }; $VENV/bin/python3 -c "
import json
best=None
for l in open(\"$RD/baseline/train/status.json\"):
    try: d=json.loads(l)
    except Exception: continue
    v=(d.get(\"kpi\") or {}).get(\"val_loss\")
    if v is not None: best=v if best is None else min(best,float(v))
st=json.load(open(\"$RD/state.json\")); st[\"baseline_val_loss\"]=best
json.dump(st,open(\"$RD/state.json\",\"w\"),indent=1); print(\"baseline_val_loss=\",best)"; echo "baseline_eval ok" >> $RD/progress.log'
```

Final message exactly: `STAGE_DONE A10`
