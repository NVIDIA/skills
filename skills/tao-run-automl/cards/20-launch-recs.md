# CARD A20 — launch review (PRE-APPROVED) + detached AutoMLRunner

STATE GATE:
```bash
bash -c 'grep -hE "recs_launched ok|baseline_eval ok" $RD/progress.log 2>/dev/null | sort -u; echo ---GATE-END---'
```
| gate output | do |
|---|---|
| contains `recs_launched ok` | STOP — print `STAGE_DONE A20` |
| contains `baseline_eval ok` only | steps 1→4 |
| neither | STOP — print `STAGE_DONE A20` (card A10 must run first; driver routes) |

1) Confirm the installed runner API — the wheel owns the truth, not the docs:
```bash
$VENV/bin/python3 -c "import inspect; from tao_automl.runner import AutoMLRunner; print('INIT', inspect.signature(AutoMLRunner.__init__)); print('RUN', inspect.signature(AutoMLRunner.run))"
```
Verified on this host (7.0.1): `INIT` has `sdk=, skill_dir=, action=`; `RUN` has `train_dataset_uri=, workspace_path=, image=, automl_settings=, ..., final_eval_fn=, **platform_kwargs`. If the printed names differ, edit ONLY those kwarg names in step 3's driver.py before launching. `mounts=` is not a named kwarg — it passes through `**platform_kwargs` to DockerSDK (this is expected). `final_eval_fn` is supported but intentionally omitted in this budgeted run (winner selected by val_loss) — a documented deviation stated in the review.

2) Write the launch review — PRE-APPROVED for this budgeted run; write to disk, never wait for a human — ONE command:
```bash
bash -c 'BL=$($VENV/bin/python3 -c "import json;print(json.load(open(\"$RD/state.json\")).get(\"baseline_val_loss\",\"MISSING\"))"); printf "# AutoML Launch Review — PRE-APPROVED (budgeted run)\n- Model: tao-train-visual-changenet (classify) | Platform: local docker | Image: $TRAIN_IMG (LOCAL ONLY)\n- Shape: 1 GPU, 1 node, 1 concurrent (bayesian is sequential)\n- Data: train=$WS/train/base/training_set.csv (210 rows) val=$WS/train/base/validation_set.csv (249 rows) images=$WS/kpi/images backbone=$WS/augmentation/backbone/model.safetensors\n- Algorithm: bayesian | Budget: EXACTLY 4 recommendations | Metric: val_loss minimize | 10 epochs, batch 8, no WandB\n- Search space: train.optim.lr [5e-6,5e-4] | train.optim.weight_decay [1e-4,0.05] | model.classify.train_margin_euclid [1.0,3.0] (all automl_enabled in the packaged schema)\n- Baseline eval: val_loss=$BL (lr=0 init snapshot, see baseline/)\n- final_eval_fn: supported by wheel 7.0.1 but intentionally omitted in this budgeted run — winner selected by val_loss; documented deviation\n- Estimated runtime: 5-15 min/rec sequential -> 30-60 min total\n" > $RD/review/launch_review.md && cat $RD/review/launch_review.md'
```

3) Write the runner driver — ONE command. MACHINE FACT (cost a dead run in the study): DockerSDK does NOT route spec file paths into the container — the explicit `mounts=` list below is MANDATORY:
```bash
bash -c 'cat > $RD/runner/driver.py <<PYEOF
import json, os, traceback, uuid
from pathlib import Path
from tao_sdk.platforms.docker import DockerSDK
from tao_automl.runner import AutoMLRunner

RD = Path(os.environ["AUTOML_RD"])
WS = os.environ["WS"]
SKILL_DIR = os.environ["SB"] + "/skills/models/tao-train-visual-changenet"
IMAGE = os.environ["TRAIN_IMG"]  # local only - never pull

# Resume must point at the ORIGINAL run_<ts> workspace (runner only appends
# run_<ts> on fresh runs) and pin session_id so controller/brain state is found.
WS_ROOT = RD / "automl_workspace"
RESUME = os.environ.get("AUTOML_RESUME", "0") == "1"
_runs = sorted(WS_ROOT.glob("run_*"))
if RESUME and _runs:
    WORKSPACE = str(_runs[-1])
else:
    RESUME = False
    WORKSPACE = str(WS_ROOT)
_sid_file = RD / "automl_session_id"
SESSION_ID = _sid_file.read_text().strip() if _sid_file.exists() else uuid.uuid4().hex[:12]
_sid_file.write_text(SESSION_ID)

spec_overrides = {
    "train.num_epochs": 10,
    "train.validation_interval": 1,
    "train.checkpoint_interval": 10,
    "train.num_gpus": 1,
    "train.use_distributed_sampler": False,
    "train.sync_batchnorm": False,
    "dataset.classify.batch_size": 8,
    "dataset.classify.workers": 2,
    "dataset.classify.image_ext": ".jpg",
    "wandb.enable": False,
    "dataset.classify.train_dataset.csv_path": f"{WS}/train/base/training_set.csv",
    "dataset.classify.train_dataset.images_dir": f"{WS}/kpi/images",
    "dataset.classify.validation_dataset.csv_path": f"{WS}/train/base/validation_set.csv",
    "dataset.classify.validation_dataset.images_dir": f"{WS}/kpi/images",
    "model.backbone.pretrained_backbone_path": f"{WS}/augmentation/backbone/model.safetensors",
}
custom_param_ranges = {
    "train.optim.lr": {"valid_min": 5e-6, "valid_max": 5e-4},
    "train.optim.weight_decay": {"valid_min": 1e-4, "valid_max": 0.05},
    "model.classify.train_margin_euclid": {"valid_min": 1.0, "valid_max": 3.0},
}
try:
    sdk = DockerSDK()
    runner = AutoMLRunner(sdk=sdk, skill_dir=SKILL_DIR, action="train")
    result = runner.run(
        train_dataset_uri=f"{WS}/train/base",
        eval_dataset_uri=f"{WS}/train/base",
        image=IMAGE,
        automl_settings={"algorithm": "bayesian", "metric": "val_loss",
                         "direction": "minimize", "automl_max_recommendations": 4,
                         "session_id": SESSION_ID},
        automl_hyperparameters=["train.optim.lr", "train.optim.weight_decay",
                                "model.classify.train_margin_euclid"],
        custom_param_ranges=custom_param_ranges,
        spec_overrides=spec_overrides,
        workspace_path=WORKSPACE,
        resume=RESUME,
        gpu_count=1,
        mounts=[{"host_path": WS, "container_path": WS, "read_only": True},
                {"host_path": str(RD / "automl_workspace" / "job_results"), "container_path": "/results"}],
    )
    (RD / "automl_result.json").write_text(json.dumps(result, indent=2, default=str))
    with open(RD / "progress.log", "a") as f: f.write("runner_finished ok\n")
except Exception:
    traceback.print_exc()
    with open(RD / "progress.log", "a") as f: f.write("runner_finished FAIL\n")
    raise
PYEOF
$VENV/bin/python3 -m py_compile $RD/runner/driver.py && echo DRIVER_OK'
```
(If step 1 showed different kwarg names, sed exactly those names in `$RD/runner/driver.py`, then re-run the py_compile line. If launch later dies instantly with `unexpected keyword argument 'mounts'`, remove ONLY the `mounts=[...]` kwarg and retry once — then the job_results bind is handled by the SDK version in use.)

4) Launch DETACHED, catch instant errors only, mark, stop:
```bash
bash -c 'cd $RD/runner && nohup $VENV/bin/python3 $RD/runner/driver.py > $RD/runner/driver.log 2>&1 & echo "runner PID $!"; sleep 20; tail -8 $RD/runner/driver.log; grep -q Traceback $RD/runner/driver.log && echo INSTANT_ERROR || { echo "recs_launched ok" >> $RD/progress.log; echo MARKED; }'
```
| result | do |
|---|---|
| `MARKED` | final message exactly `STAGE_DONE A20`, stop — the runner drives all 4 recs itself (~30–60 min) |
| `INSTANT_ERROR` | read the traceback; fix per step-1/step-3 notes (kwarg rename or mounts removal); rerun step 4. Two failures → `echo "recs_launched FAIL" >> $RD/progress.log`, `STAGE_DONE A20`, stop |
