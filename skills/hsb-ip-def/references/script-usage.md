# Script Usage And Preflight

Use this reference before running bundled HSB IP def scripts, or when explaining exactly how validation, generation, and comparison commands are executed.

## Preflight: bundled tool access

Before the first Generate, Validate, Compare, or "is this combo legal?" workflow in a session, verify the bundled tools are accessible. Do this before asking design questions or accepting a file path.

Steps:

1. Resolve the skill directory that contains this `SKILL.md`.
2. Confirm the bundled scripts exist and can be run:
   - `scripts/generate_def.py`
   - `scripts/validate_def.py`
   - `scripts/compare_defs.py`
3. Select a Python interpreter for this session. Prefer `/usr/bin/python3.11` when it exists; otherwise use `python3`. Record the selected interpreter as `<PY>` and use it for all bundled script commands in this skill, even if the scripts are executable.

4. Confirm the selected Python is available and meets the dependency requirements:

   ```bash
   <PY> -c "import sys, yaml; assert sys.version_info >= (3, 9), sys.version"
   ```

5. Run each script's help path through the selected Python interpreter to catch missing imports or permission problems:

   ```bash
   <PY> scripts/generate_def.py --help >/dev/null
   <PY> scripts/validate_def.py --help >/dev/null
   <PY> scripts/compare_defs.py --help >/dev/null
   ```

   Direct script execution may use an older `python3` from the shebang; do not rely on it once `<PY>` is selected.

6. If preflight fails, stop and report the specific missing tool, dependency, or permission issue. Do not continue with generation or validation until the user fixes the environment or chooses an alternate skill/tool location.

Preflight is not required for a purely textual explanation that only reads reference files, unless the answer needs script-backed validation.

## Scripts

| Script | When to run |
|---|---|
| `scripts/validate_def.py` | Every Validate workflow; also during Generate (the generator runs it internally) and Explain ("is X legal?" questions). Run as `<PY> scripts/validate_def.py <path> [--json|--text] [--ip-source <root>]` |
| `scripts/generate_def.py` | Every Generate workflow's final step. Run as `<PY> scripts/generate_def.py --archetype <slug> [--profile <yaml>] [-o <output>] [--allow-random-uuid]` |
| `scripts/compare_defs.py` | When the user asks to diff two configs or wants to know what changed. Run as `<PY> scripts/compare_defs.py <a.svh> <b.svh> [--json|--text]` |

The scripts target Python 3.9+ with PyYAML. On this system, `/usr/bin/python3.11` is the right interpreter; plain `python3` may be too old.

## `run_script()` Examples

Use the environment's script-running tool when available. Keep the selected Python interpreter consistent across preflight, generation, validation, and comparison.

```python
run_script("<PY> scripts/validate_def.py path/to/HOLOLINK_def.svh --json")
run_script("<PY> scripts/generate_def.py --profile profile.yaml -o HOLOLINK_def.svh")
run_script("<PY> scripts/compare_defs.py before.svh after.svh --text")
```

If the current agent environment exposes only a shell, run the same commands through the shell with the skill directory as the working directory.
