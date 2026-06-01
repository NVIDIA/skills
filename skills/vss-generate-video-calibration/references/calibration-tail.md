## Shared Calibration Tail (Python)

The verify → calibrate → poll → results sequence is identical across all
three input modes (videos, RTSP, sample-dataset). The mode-specific
references stop after their last upload step and reference this snippet.

Assumes `s`, `BASE_URL`, `project_id`, and `DETECTOR_TYPE` are already
bound from the preceding mode-specific Python.

```python
import time

# Step A — Verify project
s.post(f"{BASE_URL}/verify_project/{project_id}").raise_for_status()

# Step B — Start calibration (detector_type is a /calibrate argument; not consumed by /v1/config)
s.post(f"{BASE_URL}/calibrate/{project_id}",
       json={"detector_type": DETECTOR_TYPE}).raise_for_status()
print(f"[B] Calibration started (detector={DETECTOR_TYPE})")

# Step C — Poll until COMPLETED (10–60 min typical)
start, last = time.time(), ""
while time.time() - start < 3600:
    info = s.get(f"{BASE_URL}/get_project_info/{project_id}").json()
    st = info["project_info"]["project_state"]
    elapsed = int(time.time() - start)
    if st != last:
        print(f"    [{elapsed:>4}s] {st}", flush=True); last = st
    if st == "COMPLETED":
        print(f"[C] Done in {elapsed}s"); break
    if st == "ERROR":
        raise RuntimeError(f"Calibration ERROR — see GET {BASE_URL}/amc/calibrate/{project_id}/log")
    time.sleep(10)

# Step D — Results
r = s.get(f"{BASE_URL}/result/{project_id}/evaluation_statistics")
if r.status_code == 200:
    for k, v in (r.json().get("statistics") or r.json()).items():
        print(f"    {k}: {v}")

print(f"\nProject: {project_id}")
```

See [SKILL.md Shared Calibration Tail](../SKILL.md#shared-calibration-tail) for
the REST equivalents and the meaning of each project state.
