# CARD A00 — preflight: package, docker+GPU, model gate, scaffold

STATE GATE:
```bash
bash -c 'grep -h "ok$\|FAIL$" $RD/progress.log 2>/dev/null | tail -3; echo "---GATE-END---"'
```
| gate output | do |
|---|---|
| contains `preflight ok` | STOP — print `STAGE_DONE A00` |
| empty before `---GATE-END---` | steps 1→4 |
| contains any `FAIL` | STOP — print `STAGE_DONE A00` (driver halts on errors) |

1) AutoML package (lives in the card-managed venv — system python is PEP-668 externally-managed, never pip install into it):
```bash
$VENV/bin/python3 -c "import tao_automl, tao_sdk; from tao_automl.runner import AutoMLRunner; from tao_sdk.platforms.docker import DockerSDK; print('automl pkg OK')"
```
If imports fail, append `preflight FAIL` and stop. Have the operator run the application preflight with the wheel pin from `$SB/versions.yaml`; do not install a historical wheel inside this card.

2) Docker + GPU + image (LOCAL ONLY — never docker pull; the driver resolves TRAIN_IMG from $SB/versions.yaml unless explicitly overridden):
```bash
bash -c 'docker image inspect $TRAIN_IMG --format "image OK: {{.Id}}" | head -1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

3) Model gate (automl_enabled + packaged search-space schema):
```bash
bash -c 'grep -m1 "automl_enabled" $SB/skills/models/tao-train-visual-changenet/references/skill_info.yaml; $VENV/bin/python3 -c "import json;json.load(open(\"$SB/skills/models/tao-train-visual-changenet/schemas/train.schema.json\"));print(\"schema OK\")"'
```
Both must be positive (`automl_enabled: true` + `schema OK`). If either fails: `echo "preflight FAIL" >> $RD/progress.log`, print `STAGE_DONE A00`, stop.

4) Scaffold + mark:
```bash
bash -c 'mkdir -p $RD/{runner,review,baseline,automl_workspace}; touch $RD/commands.log; [ -f $RD/state.json ] || echo "{}" > $RD/state.json; echo "preflight ok" >> $RD/progress.log; tail -2 $RD/progress.log'
```

Final message exactly: `STAGE_DONE A00`
