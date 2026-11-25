# CARD 10 — commit train + launch dual inference (detached) — iteration $ITER

STATE GATE:
```bash
tail -3 $RD/$ITER/train/train.log
```
| gate output | do |
|---|---|
| ends `Execution status: PASS` | steps 1→2 |
| ends `Execution status: FAIL` | FAILURE BRANCH below |
| file missing/empty | re-run the exact recorded train launch from `tail -30 $RD/commands.log`, then print `STAGE_DONE 10` and stop |

FAILURE BRANCH — known signatures, exact fixes (fix spec → relaunch recorded train command detached → `STAGE_DONE 10` → stop; do NOT commit). The spec file for this iteration is `$RD/specs/baseline_spec.yaml` when ITER=baseline, else `$RD/specs/${ITER}_spec.yaml`:
- `FileNotFoundError: .../C-RADIOv2_B.pth` → `bash -c 'sed -i "s|pretrained_backbone_path:.*|pretrained_backbone_path: /data/pretrained_models/C-RADIOv2_B.safetensors|" $RD/specs/$([ "$ITER" = baseline ] && echo baseline_spec.yaml || echo ${ITER}_spec.yaml)'` (a `.pth` does NOT exist anywhere; never search for one)
- `FileNotFoundError: /home/...model_epoch...` (host path in `pretrained_model_path`) → `bash -c 'sed -i -E "s|(pretrained_model_path:) *$WS|\1 /data/workspace|" $RD/specs/${ITER}_spec.yaml'`
- `checkpoint_interval` assertion → set `train.num_epochs: 10`, `train.checkpoint_interval: 1`, `train.validation_interval: 1` in that same spec file
- anything else → `$DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage train --status error --summary "train failed: <error one-liner>"` then `STAGE_DONE 10`

1) Select checkpoints (best_val = the epoch with min val_loss in status.json, matched to its checkpoint file by epoch number; latest = highest epoch), prune the rest, commit train — ONE command:
```bash
bash -c 'set -e; T=$RD/$ITER/train; SEL=$($DPY -c "
import json, glob, re, os
best_e, best_v = None, None
for line in open(\"$RD/$ITER/train/status.json\"):
    try: d = json.loads(line)
    except Exception: continue
    k = d.get(\"kpi\") or {}
    v = k.get(\"val_loss\"); e = d.get(\"epoch\", k.get(\"epoch\"))
    if v is None or e is None: continue
    if best_v is None or float(v) < best_v: best_v, best_e = float(v), int(e)
cks = sorted(glob.glob(\"$RD/$ITER/train/model_epoch_*.pth\"), key=lambda p: int(re.search(r\"epoch_(\d+)\", p).group(1)))
latest = cks[-1]
best = next((c for c in cks if int(re.search(r\"epoch_(\d+)\", c).group(1)) == best_e), latest) if best_e is not None else latest
print(best); print(latest); print(best_v if best_v is not None else \"\")
"); CK_BEST=$(echo "$SEL" | sed -n 1p); CK_LAST=$(echo "$SEL" | sed -n 2p); B=$(echo "$SEL" | sed -n 3p); for f in $T/model_epoch_*.pth; do [ "$f" = "$CK_BEST" ] || [ "$f" = "$CK_LAST" ] || rm -f "$f"; done; VL=""; [ -n "$B" ] && VL="--val-loss $B"; $DPY $SKILL_ROOT/scripts/commit_stage.py --duration-sec $(( $(date +%s) - STAGE_T0 + 1 )) --results-dir $RD --iter-label $ITER --stage train --best-ckpt "$CK_BEST" --training-spec $RD/specs/$([ "$ITER" = baseline ] && echo baseline_spec.yaml || echo ${ITER}_spec.yaml) $VL --summary "train PASS; best=$(basename $CK_BEST) latest=$(basename $CK_LAST) val_loss=${B:-unknown}"'
```
If commit_stage REJECTS the commit, read its stderr — it names the missing evidence; fix exactly that and rerun this one command. Never bypass it.

2) Write both inference specs — the spec ALREADY has an `inference:` block; rewrite its `checkpoint:` IN PLACE (NEVER append a new `inference:` block — duplicate YAML keys crash OmegaConf; NEVER add `csv_path` under `inference:` — the infer CSV is already correct at `dataset.classify.infer_dataset.csv_path`; leave `inference.results_dir` unset so outputs land at the nested path card 20 expects) — then launch ONE detached chain and stop:
```bash
bash -c 'set -e; SPEC=$RD/specs/$([ "$ITER" = baseline ] && echo baseline_spec.yaml || echo ${ITER}_spec.yaml); T=$RD/$ITER/train; CKB=$(basename $(ls $T/model_epoch_*.pth | sort -t_ -k3 -n | head -1)); CKL=$(basename $(ls $T/model_epoch_*.pth | sort -t_ -k3 -n | tail -1)); for K in best_val latest; do C=$([ $K = best_val ] && echo $CKB || echo $CKL); O=$RD/specs/${ITER}_infer_$K.yaml; cp $SPEC $O; sed -i -E "0,/^results_dir:/s|^results_dir:.*|results_dir: /results/$ITER/inference/$K|" $O; sed -i "/^inference:/,/^[a-z]/s|^  checkpoint:.*|  checkpoint: /results/$ITER/train/$C|" $O; done; nohup bash -c "docker run --rm --gpus all --shm-size=8g $MOUNTS $TRAIN_IMG visual_changenet inference -e /results/specs/${ITER}_infer_best_val.yaml > $RD/$ITER/inference/best_val.log 2>&1 && docker run --rm --gpus all --shm-size=8g $MOUNTS $TRAIN_IMG visual_changenet inference -e /results/specs/${ITER}_infer_latest.yaml > $RD/$ITER/inference/latest.log 2>&1" & echo LAUNCHED'
```
(The `sort -t_ -k3 -n` orders `model_epoch_NNN_step_*.pth` by epoch number; best_val was pruned to the min-val_loss epoch in step 1, so the lowest remaining epoch IS best_val. If only one checkpoint remains, both roles use it — harmless.)
The subtask name is `inference` — NEVER `infer` (`invalid choice: 'infer'`).

CAUTION on step 1's pruning: best_val is matched by EPOCH NUMBER from status.json — never by file mtime (`ls -t` picks newest-written, which is the LAST epoch, not the best one).

After `LAUNCHED` prints: final message exactly `STAGE_DONE 10` and stop.
