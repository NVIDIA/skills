# Session Retrospective Template

Written to `.nsight-ai-port/skill_feedback.json:session_summary` at
wrap-up. Copy this structure; replace every placeholder.

```json
{
  "timestamp": "2025-06-08T18:30:00Z",
  "flavor": "fortran-openacc",
  "what_happened": "One paragraph: setup, flavor, key phases, targets ported, verification outcome, commit count.",
  "what_worked": [
    "State model / progress.json resumability",
    "Execution marker caught missing GPU path"
  ],
  "skill_improvements": [
    "Document libnvtx3interop discovery in nvtx.md",
    "profiling-guide: stop when CPU sampling unavailable"
  ],
  "hiccups": [
    {
      "description": "NVTX link failed: libnvToolsExt missing on CUDA 13",
      "skill_section": "flavors/fortran/nvtx.md",
      "resolution": "Used -lnvtx3interop with SDK lib dir from nvfortran -print-file-name",
      "logged_as_entry": true
    }
  ],
  "discretionary_choices": [
    {
      "choice": "NVTX names iter:timestep, function:src/kernels.f90:kernel_a",
      "why_loose": "nvtx.md examples were ambiguous before naming grammar",
      "suggested_guidance": "Mandate iter:, phase:compute:/phase:host:, function:<file>:<function>, optional block:"
    }
  ],
  "nsys_extraction": {
    "approach": "nsys stats CSV with agent-chosen top K; extraction_meta.txt; never load .nsys-rep",
    "scales_to_large_codes": "partial",
    "scaling_gaps": [
      "CPU sampling blocked — stopped and reported nsys status --environment",
      "Re-exported stats with higher K without re-profiling when needed"
    ]
  },
  "stats": {
    "commits": 12,
    "functions_ported": ["src/foo.f90:bar"],
    "profile_rounds": 2,
    "usage": {
      "run_segments": 1,
      "elapsed_wall_clock_sec": null,
      "input_tokens": null,
      "output_tokens": null,
      "cache_read_tokens": null,
      "total_tokens": null,
      "agent_turns": null,
      "cost_usd": null
    }
  }
}
```

`stats.usage` is aggregated from `usage_log` per the rules in
[skill-feedback-log.md](skill-feedback-log.md) (sum the last snapshot
of each `run_segment`; `run_segments` is the count of distinct
segments). A `null` field means the harness never exposed it — leave
it `null`; **never estimate**.

Present the same content to the user in readable markdown at wrap-up
(do not only write JSON silently).
