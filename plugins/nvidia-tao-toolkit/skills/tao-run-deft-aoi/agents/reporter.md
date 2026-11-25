# DEFT Loop Reporter Compatibility Wrapper

Use this wrapper only when a runtime explicitly invokes the legacy reporter
agent. Normal workflow execution does not spawn a reporting agent:
`init_deft_state.py` and `commit_stage.py` already invoke the deterministic
renderer.

Inputs:

- `results_dir`: absolute DEFT run directory;
- `skill_root`: absolute `tao-run-deft-aoi` skill directory;
- `trigger`: optional legacy value; use it only to decide whether the run must
  already be terminal.

Run exactly one command:

```bash
"${skill_root}/scripts/deft_python.sh" \
  "${skill_root}/scripts/render_report.py" \
  --results-dir "${results_dir}"
```

When `trigger` is `loop-end`, append `--require-terminal`. Return the script's
single status line and exit with the same status. Do not read state, assemble
HTML, edit the template, or write the output yourself.
