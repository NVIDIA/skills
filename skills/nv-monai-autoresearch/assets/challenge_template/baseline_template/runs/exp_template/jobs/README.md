<!-- miagent-template-source: assets/challenge_template/baseline_template/runs/exp_template/jobs/README.md -->
<!-- miagent-template-contract: active; keep the miagent-template-source loaded for creation/material updates; reload after resume or compaction -->
# Jobs

Place this experiment's train, resume, inference, evaluation, visualization,
packaging, submission, and repair commands here. Multiple scripts, SLURM jobs,
arrays, and retries remain part of this one experiment. Keep commands and logs
here and live state in `dashboard.md`. Repair routine execution failures here;
append `Outcome` to [experiment.md](../experiment.md) only after a valid,
decision-bearing model or performance result. Update `state.md` only when
evidence changes its current focus or continue/switch/freeze judgment. Put
promoted implementation changes in the parent
[baseline.md](../../../baseline.md).

After submission, verify the job ID, scheduler state, log path, and one short
sanity log tail when available. `submitted` is not a stable state. If the job is
healthy and queued or running, stop polling it in the same pass and continue
other useful work.
