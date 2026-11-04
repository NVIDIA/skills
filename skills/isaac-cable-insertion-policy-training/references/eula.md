## Omniverse EULA acceptance

When no location has been selected, the read-only `eula-status` command may run
from the current directory so remembered acceptance can inform the setup question.
For the remaining commands, first create/enter the selected standalone training package root as in
[Setup](setup.md). Use system `python3`; uv, the
`.venv`, and Isaac Sim are not required to record consent. Require acceptance
before installing or launching Isaac Sim. Reuse a valid receipt or the user's
prior explicit acceptance instead of asking again during setup and training.

Before the first Isaac Sim command in a new skill invocation, run `python3 "$SKILL/scripts/supervisor.py" eula-status` from the selected standalone training package root. The status command uses only the standard library and does not import Isaac Sim. It checks a private, Git-ignored record at `logs/.eula-acceptance-uid<uid>.json` for the current Linux user and Isaac Sim 6.1. A plan field or inherited environment variable alone is not evidence of the user's acceptance.

If there is no valid record, show the user the [NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html) and ask this complete question: **"Do you accept the NVIDIA Omniverse License Agreement for Isaac Sim 6.1 and authorize this skill to remember your answer and set `OMNI_KIT_ACCEPT_EULA=YES` for installation, verification, training, and evaluation?"** Require an explicit Yes. If the user declines or does not answer, stop before importing or launching Isaac Sim. If the user already gave an explicit Yes earlier in the current conversation, that answer remains valid; do not ask again.

After explicit acceptance, run `python3 "$SKILL/scripts/supervisor.py" eula-accept --user-confirmed` from the standalone training package root.
Use `--user-confirmed` only after the user's actual reply; the CLI flag itself is not consent.
A user working directly in a terminal can instead run `eula-accept` without the flag and type `YES` at the prompt.
`eula-status` then reports the acceptance time.
The receipt is scoped to the current OS user, agreement URL, and Isaac Sim 6.1; a changed scope requires a new answer.
The receipt is ignored by Git and has owner-only permissions.

The EULA URL uses `latest`, so this check cannot discover a terms revision at the same URL without external review. When maintainers learn that the agreement changed, update `EULA_SCOPE` in `scripts/supervisor.py` to invalidate existing receipts and ask again.

The supervisor refuses preflight, session creation, or resume without the receipt. It copies the acceptance time into each new session plan, marks `eula_accepted=true`, and passes `OMNI_KIT_ACCEPT_EULA=Yes` to its authorized preflight, trainer, and evaluator subprocesses. During agent-managed setup, scope the same variable to authorized installation/verification commands after acceptance. Do not add the variable to `.bashrc` or set it globally. `python3 "$SKILL/scripts/supervisor.py" eula-revoke` forgets the receipt for future launches; it does not cancel an already-running Isaac Sim process. Isaac Sim also has its own first-import prompt and remembers an answer given there, but this skill uses its receipt so detached runs never depend on an unseen prompt.

Isaac Sim 6.1 documents both the first-import prompt and `OMNI_KIT_ACCEPT_EULA=YES` (also `Y` or `1`, case-insensitive): [Python environment installation](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html#running-isaac-sim).
