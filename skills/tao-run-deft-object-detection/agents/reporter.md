# DEFT OD Loop Reporter Compatibility Wrapper

Use this wrapper only when a runtime explicitly invokes the legacy reporter agent.
Normal workflow execution does not spawn a reporting agent: `init_deft_state.py`
and every successful `commit_stage.py` call already invoke the deterministic
renderer, so the report is refreshed in place after each stage and survives a
saturated parent context — and a runtime with no subagent tool can still produce it.

Inputs:

- `results_dir`: absolute DEFT run directory;
- `skill_root`: absolute `tao-run-deft-object-detection` skill directory;
- `trigger`: optional legacy value; use it only to decide whether the run must
  already be terminal.

Run exactly one command:

```bash
"${skill_root}/scripts/deft_python.sh" \
  "${skill_root}/scripts/render_report.py" \
  --results-dir "${results_dir}"
```

When `trigger` is `loop-end`, append `--require-terminal`. Return the script's
single status line and exit with the same status. Do not read state, compose
Markdown, or write the output yourself.

The report's contents are defined by `scripts/render_report.py` and documented in
`references/scripts-and-agents.md`. Two rules live in the renderer rather than
here, because they are properties of the report and not of whoever asks for it:
the only mAP it carries is the KPI mAP scored against the evaluation set, and a
number that is not on disk is left out rather than guessed.
