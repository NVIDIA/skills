# Network-Enabled Bootstrap

Read this file only after Pre-Flight has resolved `network_mode=network-enabled`.
It is never part of an air-gapped run.

If `bash scripts/deft_python.sh` cannot select a dependency-complete interpreter,
create a workspace-local virtual environment and install the required helpers:

```bash
python3 -m venv <workspace>/.venv
<workspace>/.venv/bin/pip install pandas numpy matplotlib pyarrow pillow pyyaml huggingface_hub boto3
```

Rerun `bash scripts/deft_python.sh` and record its absolute interpreter path in
`init_deft_state.py --python-executable`. Networked image login/pull and model
fetches remain post-approval actions and must use the versions and asset paths
resolved by Pre-Flight.
