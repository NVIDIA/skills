# Network-Enabled Bootstrap

Read this file only after Pre-Flight has resolved `network_mode=network-enabled`.
It is never part of an air-gapped run.

If `bash scripts/deft_python.sh` cannot select a dependency-complete interpreter,
install `pyarrow` and `pyyaml` into a dedicated workspace-local environment,
then rerun Pre-Flight:

```bash
python3 -m venv <workspace>/.venv
<workspace>/.venv/bin/pip install pyarrow pyyaml
```

Record the selected absolute interpreter in
`init_deft_state.py --python-executable`. Networked image login/pull and model
fetches remain post-approval actions governed by the selected platform.
