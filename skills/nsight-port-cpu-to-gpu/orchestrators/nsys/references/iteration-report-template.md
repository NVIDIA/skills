# Iteration Report Template

After each nsys round, present this report to the user.
**Do not start the next round without explicit user approval.**

Copy and fill (outer fence uses four backticks so the nested
` ```bash ` block renders correctly):

In `sequence` mode, include the active problem in the heading (e.g.
`## Porting Round {N} Complete — problem {problem_id}`) and note how
many problems remain in the sequence.

Set `{profile_dir}` to this round's resolved artifact directory.

````markdown
## Porting Round {N} Complete

### Performance summary (lead with this)
- **GPU work:** {gpu_active_pct}% of the problem window is GPU-active
  (kernels + memops).
- **Whitespace left:** {whitespace_pct}% of the window is idle /
  host-bound time.
- **Round delta:** GPU-active {prev_pct}% → {gpu_active_pct}%; problem
  wall time {prev_s}s → {now_s}s ({speedup}× vs round {N-1}).
- **Still on host (next-round candidates):** hot regions with no
  kernels yet — ...
- Source: `{profile_dir}/perf_summary.txt`

### What was done
- NVTX changes: ...
- Refactors: ...
- Functions ported this round: ...
- Device utilities: ...

### Current state
- GPU build: passes / fails
- Functions completed (all rounds): ...
- Functions pending: ...
- Last working commit: `{sha}`

### Profile artifacts
| Artifact | Path |
|----------|------|
| Report | `{profile_dir}/...nsys-rep` |
| Summary | `{profile_dir}/summary.txt` |
| Gaps | `{profile_dir}/gaps.txt` |
| Extraction meta | `{profile_dir}/extraction_meta.txt` (top K, coverage) |

Open in Nsight Systems:
```bash
nsys-ui {profile_dir}/<report>.nsys-rep
```

### Timeline summary (one compute-loop iteration)
- Measurement window (`iter:`, median range of N): ...
- Hot `function:` / `block:` regions: ...
- Unmarked gap in problem window: ...%
- GPU activity in hot regions: ... (kernels / still host)
- Expected host time (`phase:host:*`): ...

### Convergence checklist
- [ ] Hot-loop compute on GPU
- [ ] Loop data device-resident (no thrashing)
- [ ] Host-only gaps explained (MPI/I/O/allocation)

### Likely next improvements (if you continue)
1. ...
2. ...

### Continue?
Run another profiling round on this problem, or stop here?
````

Record the round in **both** lists. `progress.json:rounds[]`:

```json
{
  "round": 1,
  "problem": "case_a",
  "functions_ported": ["src/foo.f90:bar"],
  "gpu_active_pct": 61.4,
  "whitespace_pct": 38.6,
  "user_continued": null
}
```

The `problem` field is required in `sequence` mode (the active problem
id) and may be omitted in `single` mode.

…and `progress.json:profile_rounds[]` (profile
artifacts). Same `problem` rule as `rounds[]` — required in
`sequence` mode, optional in `single`. Sequence-mode example:

```json
{
  "round": 1,
  "problem": "case_a",
  "profile_dir": ".nsight-ai-port/profiles/case_a/round_1",
  "report_path": ".nsight-ai-port/profiles/case_a/round_1/gpu.nsys-rep",
  "perf_summary_path": ".nsight-ai-port/profiles/case_a/round_1/perf_summary.txt"
}
```

In `sequence` mode the profile paths are namespaced by problem id
(`.nsight-ai-port/profiles/{problem_id}/round_1/...`); do not rely on
parsing the path for the active problem — use the `problem` field.
In `single` mode paths stay under `.nsight-ai-port/profiles/round_N/`
and `problem` may be omitted.

Set `user_continued` to `true` or `false` after the user responds.
