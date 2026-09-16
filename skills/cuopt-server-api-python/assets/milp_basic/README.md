# MILP via REST

Same problem as LP (maximize 40x + 30y, 2x+3y≤240, 4x+2y≤200) with `variable_types`: first variable integer, second continuous.

**Requires:** cuOpt server running. **Run:** `python client.py` (exits 0 if server unreachable).
**Server URL:** `--server` (default `http://localhost:8000`). Variable types: `continuous`, `integer`, `binary`.
