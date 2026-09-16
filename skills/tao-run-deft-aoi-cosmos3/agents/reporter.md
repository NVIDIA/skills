# Cosmos3 DEFT Reporter Compatibility Wrapper

Use this wrapper only when a runtime explicitly invokes the legacy reporter
agent. Normal workflow execution renders automatically from
`init_deft_state.py` and the `commit_stage.py` post-commit hook.

Inputs:

- `results_dir`: absolute Cosmos3 DEFT run directory;
- `skill_root`: absolute `tao-run-deft-aoi-cosmos3` skill directory;
- `trigger`: optional legacy value.

Run exactly one command:

```bash
"${skill_root}/scripts/deft_python.sh" \
  "${skill_root}/scripts/render_report.py" \
  --results-dir "${results_dir}"
```

When `trigger` is `loop-end`, append `--require-terminal`. Return the script's
status line and exit with the same status. Do not read state, assemble HTML,
edit the template, or write the report yourself.
