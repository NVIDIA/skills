# CARD 00 — init state + launch baseline train (detached)

STATE GATE — run exactly this, then use the table:
```bash
bash -c '[ -f $RD/deft_state.json ] && echo STATE_EXISTS || echo NO_STATE; [ -s $RD/baseline/train/train.log ] && echo TRAINLOG_EXISTS || echo NO_TRAINLOG'
```
| gate output | do |
|---|---|
| `NO_STATE` + `NO_TRAINLOG` | steps 1→4 in order |
| `STATE_EXISTS` + `TRAINLOG_EXISTS` | STOP — print `STAGE_DONE 00` (driver routes to card 10) |
| `STATE_EXISTS` + `NO_TRAINLOG` | step 4 only |

1) Scaffold (one command):
```bash
mkdir -p $RD/specs $RD/baseline/{train,inference} $RD/iter1/{routing_results,anomalygen/sdg,mining_filter,dataset/images,train,inference,rca_results} $RD/iter2/{routing_results,anomalygen/sdg,mining_filter,dataset/images,train,inference,rca_results} $RD/iter3/{routing_results,anomalygen/sdg,mining_filter,dataset/images,train,inference,rca_results}
```

2) Initialize state — NEVER hand-author deft_state.json:
```bash
$DPY $SKILL_ROOT/scripts/init_deft_state.py --results-dir $RD --workspace $WS --kpi-target "FAR < 0.5 %" --max-iterations 3 --num-gpus 1 --num-epochs 10 --num-sdg 1 --project NV_PCB_Siamese --step 0 --batch-size 8 --min-similarity 0.9 --train-container $TRAIN_IMG --ag-container $DS_IMG
```
(`--num-sdg` must be a positive int; the value is inert here — SDG is disabled and card 40 records the documented skip. If init errors on an unknown flag, drop ONLY that flag and rerun; do not switch tools.)

3) Stage the baseline spec, launch-ready (one command; the workspace copy is NOT ready —
   its checkpoint_interval exceeds the epoch budget, and its backbone points at a
   nonexistent `.pth`; ONLY the mounted `.safetensors` exists):
```bash
bash -c 'cp $WS/specs/baseline_spec.yaml $RD/specs/baseline_spec.yaml && sed -i -E -e "0,/^results_dir:/s|^results_dir:.*|results_dir: /results/baseline|" -e "s|^(  num_epochs:).*|\1 10|" -e "s|^(  checkpoint_interval:).*|\1 1|" -e "s|^(  validation_interval:).*|\1 1|" -e "s|(    batch_size:).*|\1 8|" -e "s|(    pretrained_backbone_path:).*|\1 /data/pretrained_models/C-RADIOv2_B.safetensors|" $RD/specs/baseline_spec.yaml && grep -nE "results_dir|num_epochs|checkpoint_interval|validation_interval|batch_size|pretrained_backbone_path" $RD/specs/baseline_spec.yaml | head -12'
```
Verify the grep shows: results_dir /results/baseline · num_epochs 10 · both intervals 1 · batch_size 8 · backbone `.safetensors`. If a field is wrong, fix that ONE line with sed and re-grep.

4) Launch baseline training DETACHED (the driver waits on the container, not you):
```bash
nohup docker run --rm --gpus all --shm-size=8g $MOUNTS $TRAIN_IMG visual_changenet train -e /results/specs/baseline_spec.yaml > $RD/baseline/train/train.log 2>&1 &
```

Do NOT read the skill bank or wait for training. Final message exactly: `STAGE_DONE 00`
