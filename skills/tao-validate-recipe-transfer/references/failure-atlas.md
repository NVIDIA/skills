# Failure atlas

Symptom → likely cause → the cheapest check that discriminates. Grouped by where the bug
lives. Use with `postmortem.md`, which orders these by likelihood per symptom class.

## Data and labels

| Symptom | Cause | Check |
|---|---|---|
| Metric ~0, loss descends | class index off-by-one (background reserved or not) | draw GT boxes from dataloader tensors, print class ids |
| Boxes shifted or mirrored | xyxy/xywh/cxcywh confusion, or normalized vs absolute | visualize from the collated tensor, not the JSON |
| Boxes systematically scaled | letterbox padding never un-mapped to original coordinates | check the inverse transform in post-processing |
| Metric drops after a format conversion | crowd/ignore flags dropped, polygons rasterized differently | diff object counts and ignore-region counts before/after |
| Small objects entirely missed | resolution / FPN levels / anchor scales inherited from source | per-size metric breakdown; `domain_gap_report.py` |
| Model suppresses obvious objects | missing annotations — unlabeled instances act as explicit negatives | review high-confidence false positives manually |
| Val >> test | near-duplicates across splits | `split_leakage_check.py` |
| Great val, bad production | non-representative test set | compare test vs production stats |
| NaN early | degenerate boxes (zero/negative area) after conversion or crop | assert w>0, h>0 in the collate function |

## Model and checkpoint

| Symptom | Cause | Check |
|---|---|---|
| Far below paper, head seems random | partial state-dict load with `strict=False` | log and assert missing/unexpected key counts |
| Param count differs from paper | wrong variant/config, wrong depth or width | rung 1 |
| A component never improves | branch detached or accidentally frozen | rung 3 |
| Loss plateaus immediately | focal prior-bias init missing; or LR ~0 after a schedule mis-scale | rung 2; print actual LR at step 0 and step 100 |
| Eval much worse than training suggests | EMA weights not used (or not saved) at eval | check which weights the eval path loads |

## Optimization

| Symptom | Cause | Check |
|---|---|---|
| Diverges after batch-size change | LR not linearly rescaled | recompute effective batch = batch × accum × GPUs |
| NaN with AMP, fine in fp32 | fp16 overflow in exp/log terms | switch to bf16 or raise loss scale |
| Unstable start | warmup shortened or removed during port | compare warmup steps against the paper's |
| Trains, then degrades | schedule far too long for a fine-tune | rung 6 with early stopping; cut schedule |
| Multi-GPU differs from single | loss reduced per-GPU then averaged again; BN batch too small | check reduction; use SyncBN |

## Evaluation

| Symptom | Cause | Check |
|---|---|---|
| Cannot match paper by 0.5–2 | test resize policy, TTA, NMS settings, or a different mAP implementation | the A3 checkpoint gate isolates this |
| mAP fine, production poor | operating point never chosen | PR curve at the deployed threshold |
| Recall capped | max-detections limit truncating dense scenes | raise the cap, re-evaluate |
| Metric varies run to run by ~0.5 | seed variance; paper reported one run | 3 seeds before investigating further |
| Suspiciously high | leakage, contamination, or a label-correlated artifact | run S6 checklist in postmortem.md adversarially |

## Deployment and export

| Symptom | Cause | Check |
|---|---|---|
| ONNX/TensorRT differs from PyTorch | preprocessing divergence in the export path | `preprocess_parity.py` |
| INT8 much worse than FP16 | unrepresentative calibration set | rebuild calibration from real production images |
| Correct on some images only | dynamic shapes handled differently post-export | test across the real aspect-ratio range |
| Off-by-one class labels in production | class name list defined in more than one place | grep for every CLASSES definition |

## Environment

| Symptom | Cause | Check |
|---|---|---|
| Custom op will not build | torch/CUDA drift since publication | container matching the repo's stated versions |
| Results differ across machines | different op implementation after a version bump, non-determinism | pin container; set seeds and deterministic flags for the comparison |
| Silent CPU fallback | op unavailable, framework falls back | assert device placement; watch throughput |
