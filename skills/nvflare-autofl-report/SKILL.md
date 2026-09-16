---
name: nvflare-autofl-report
description: "Generate a reproducible final report, literature-outcome synthesis, JSON summary, and refreshed progress plot for a stopped or interrupted NVFLARE Auto-FL campaign."
license: Apache-2.0
compatibility: "Requires NVFLARE 2.9.0+, Python, and artifacts from an NVFLARE Auto-FL campaign."
metadata:
  version: "0.1.0"
  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"
  min-flare-version: "2.9.0"
  blast-radius: edits_files
  category: Reporting
  tags: "nvflare, federated-learning, optimization, reporting"
  languages: "python"
---

# NVFLARE Auto-FL Report

## Purpose

Turn the recorded evidence from a stopped NVFLARE Auto-FL campaign into a reproducible
Markdown report, machine-readable JSON summary, and refreshed progress plot without
changing the campaign or its results.

## Use When

Use this skill after an NVFLARE Auto-FL campaign has stopped, reached an
explicit cap, hit a hard blocker, or was manually interrupted. Use it when the
user asks for the final report, achieved improvement, literature findings,
failed ideas, reproduction details, or a refreshed progress plot.

## Do Not Use When

Do not use this skill to start or continue optimization, invent missing
results, or finalize a campaign that is still running. Use `nvflare-autofl` for
the active candidate loop. Do not stop an active campaign merely because the
user asks for a status snapshot.

## Available Scripts

| Script | Purpose | Arguments |
| --- | --- | --- |
| `scripts/generate_report.py` | Validate stopped campaign evidence and generate the report, JSON summary, and optional progress plot. | Campaign job directory; optional evidence/output paths, interruption confirmation, plot settings, and agent context. Run `python scripts/generate_report.py --help` for the complete CLI. |

Run this bundled CLI directly with Python. This skill has no NVFLARE or agent
`run_script()` helper; do not invent or call one. Resolve the script from the
directory containing this `SKILL.md`, as shown in the workflow.

## Workflow

1. Locate the job directory containing `results.tsv`, `autofl.yaml`,
   `.nvflare/autofl/campaign_state.json`, and candidate manifests.
2. Confirm execution has stopped. Prefer campaign state with
   `final_response_allowed=true`. When a process was abruptly interrupted and
   state is stale, independently confirm no campaign or job process remains,
   then use `--confirm-interrupted`. Before finalizing, confirm that campaign
   state, `results.tsv`, and available candidate manifests contain no pending
   candidate. Finalize or abandon pending work through `nvflare-autofl` first.
3. Generate the deterministic report artifacts:

   ```bash
   python "$REPORTER" <job-dir>
   ```

   Resolve `REPORTER` once to the absolute path of
   `scripts/generate_report.py` relative to this `SKILL.md`. Do not assume a
   provider-specific skill installation directory or `$CODEX_HOME`.

4. Read both **autofl_final_report.md** and `autofl_report_summary.json`. Check
   warnings about metric use, executed budget changes, campaign-state/ledger
   disagreement, missing provenance, and incomplete interruption state.
5. Give the user the baseline, best score, delta, strongest candidate lineage,
   concise "what helped" and "what did not help" findings, literature ideas
   that helped or failed, selection rationale, reliability caveats, and
   absolute artifact paths.

The helper attempts to refresh `progress.png` by reusing the product Auto-FL
plotter. Plotting is optional evidence: if plotting dependencies are missing or
the artifact is not a valid PNG, the helper preserves the artifact, records a
warning and `artifacts.progress_plot_available=false`, and still writes the
Markdown and JSON reports without embedding the broken image. It does not
modify source, candidate manifests, `results.tsv`, or campaign state.
All relative helper path options, including `--plotter`, resolve from the
campaign job directory rather than the shell's current working directory.
The helper holds the campaign lifecycle lock from evidence loading through
artifact writes. It refuses a busy campaign and rejects writable output paths
that alias campaign evidence, `job.py`, trust-contract source paths, or another
output, including filesystem aliases. Outputs must not match the trust
contract's allowed source-creation patterns.
A read-only campaign archive is reportable only when its persisted
`campaign.lock` already exists and writable output paths are supplied; the
persisted lock target alone does not imply live contention.

## Interrupted Campaigns

The report helper must refuse state with `final_response_allowed=false` unless
the human has said the campaign was stopped/interrupted and the agent confirms
that execution is no longer active. Only then run:

```bash
python "$REPORTER" <job-dir> --confirm-interrupted
```

This records a reporting-time interruption assertion; it does not rewrite the
campaign state or pretend the runner finalized cleanly. It bypasses only stale
stop state. A `candidate` ledger row or pending-candidate state always blocks
finalization. Any available candidate manifest whose status is not a recognized
terminal value (`keep`, `discard`, `crash`, or `abandoned`) also blocks. Missing,
unknown, or unreadable status is unfinished evidence because completion cannot
be established. The agent must finalize or abandon that candidate first.

## Troubleshooting

| Error or symptom | Cause | Solution |
| --- | --- | --- |
| Reporting is refused because `final_response_allowed=false`. | The campaign may still be active, or persisted state may be stale after an interruption. | Confirm no campaign or job process remains. If a human confirmed the interruption, rerun with `--confirm-interrupted`; otherwise continue or stop the campaign through `nvflare-autofl`. |
| Reporting is refused for a pending candidate or unknown manifest status. | The recorded evidence cannot establish that candidate execution finished. | Finalize or abandon the candidate through `nvflare-autofl`, then regenerate the report. |
| The campaign lock is busy. | Another campaign or reporting process holds the lifecycle lock. | Identify and wait for the active process. Never bypass or delete a live lock. |
| The JSON says `progress_plot_available=false`. | Plotting dependencies are unavailable or the generated artifact is not a valid PNG. | Use the completed Markdown and JSON reports, review their warning, and install the campaign's plotting dependencies before retrying if a plot is required. |
| An output path is rejected. | The path aliases protected campaign evidence, source, or another output. | Choose distinct writable output paths outside protected campaign inputs and rerun. |

## Report Contract

The final report must include:

- campaign termination reason, objective, metric source, direction,
  environment, cap, abandoned-candidate count, and declared fixed budget;
- baseline, best retained result, score delta, runtime, failures, and status
  counts;
- selected-candidate rationale, strict retained improvements, representative
  non-improvements, grouped failures, and outcomes by recorded algorithm
  family;
- running-best trajectory selected by first, final, and largest objective improvements,
  plus a refreshed `progress.png` when plotting is available, with explicit
  plot availability in the JSON summary otherwise;
- best-candidate manifest, patch hash, base-candidate lineage, inherited code
  changes, artifacts, and exact baseline/best commands;
- every recorded literature checkpoint, its event ID and source markers,
  explicitly linked candidates, and whether measured evidence helped, matched,
  failed, or did not confirm the idea;
- discarded/crashed ideas and deterministic comparability warnings;
- optional agent model, reasoning effort, cost, or tooling notes when supplied;
- absolute paths to `autofl.yaml`, `results.tsv`, campaign state,
  `progress.png`, `autofl_report_summary.json`, and
  **autofl_final_report.md**.

The report must distinguish imported/declared budget from executed command
arguments. It must warn when the best candidate changed training compute or
comparison population, when authoritative state disagrees with ledger-derived
accounting, or when repeated selection used a test-like metric. Product Auto-FL
campaigns retain the imported `min` or `max` direction; the report rejects
legacy minimization evidence without direction provenance. It must not add PR-specific sections such as "Product Findings"
unless the user explicitly requests them.
`best` means a scored retained baseline or `keep` row; an unretained scored
`discard` may appear only as `best_observed`. Candidate and crash rows never
become retained best results, milestones, or literature improvements.
Baseline identity is determined strictly by `status=baseline`, matching the
campaign guard. The report preserves per-run metric name, extraction source,
artifact, candidate kind, algorithm family, and literature event linkage.
It does not infer algorithm families or mechanisms from candidate names.

Read [report-contract.md](references/report-contract.md) when interpreting
lineage, literature outcomes, budget warnings, or interrupted state.

## Limitations

- The report is only as complete as the persisted ledger, campaign state, and
  candidate manifests; it does not validate or reconstruct unrecorded claims.
- Results from a single campaign do not establish robustness or
  generalization.
- The helper does not start, stop, resume, or resubmit campaign jobs.
- Progress plotting remains optional, so Markdown and JSON may be produced
  without an embeddable PNG.
- Copied campaign archives may retain only partial provenance when recorded
  absolute manifest paths are no longer available.

## Requirements

- Treat `results.tsv` as recorded evidence; never repair scores by guessing.
- Work without Git. Do not commit or push unless the user separately asks.
- Preserve the campaign and job sources exactly as found.
- Never bypass campaign-lock contention or output-path collision checks.
- Use candidate manifests when available, but still report partial provenance
  when copied artifacts make old absolute manifest paths unavailable.
- Keep conclusions proportional to the evidence. A single run is a candidate,
  not a robustness claim.
- For POC/production, report standard NVFLARE job IDs and downloaded artifacts
  already present in the ledger; do not resubmit jobs during reporting.

