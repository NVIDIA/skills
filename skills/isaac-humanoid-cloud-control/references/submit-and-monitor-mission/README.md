# Verify an existing humanoid mission

Resolve `submit-and-monitor-mission` from the pinned Mission Control checkout.
Read its runtime contract at `skills.submit-and-monitor-mission.skill_md` in the
manifest. No separate agent-catalog installation or filesystem search is needed.
The older name `mission-lifecycle-verification` is not a required dependency.

This is an instruction-only handoff. The parent has already validated the G1
destination and submitted through `scripts/submit_navigation_mission.py`.
Pass that existing mission ID, the selected robot identity, cloud host/ports,
initial pose evidence, and the recorded Compose project/runtime log locations.
Apply the upstream polling, pose-progress, log-correlation, terminal-state, and
reporting sections. Do not repeat payload creation or submission, run
`submit_mission.sh`, or start overlapping retries. For an ambiguous submission
response, resolve the existing mission ID before taking further action.

Adapt upstream loopback examples to the selected cloud host and use the G1
launch/Nav2 logs instead of `nova-carter-sil`. Resolve Dispatch containers from
the recorded Compose project rather than assuming the current directory.
Retain the parent's strict completion checks, including pose motion and the
visible browser stream. Report failures or missing evidence accurately.

If the manifest predates this dependency, rerun:

```bash
python3 "$SKILL_DIR/scripts/doctor.py" dependencies
```

Use the normal explicit dependency preparation path if the pinned checkout is
missing. The skill already exists at the declared Mission Control commit; no
upstream pin change or new repository is required.
