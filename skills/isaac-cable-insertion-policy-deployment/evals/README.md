# Evaluation inputs

These prompts are evaluation definitions. They do not bundle trained checkpoints
or assume access to the original author's machine. An empty `files` list means
no input artifacts are included in the repository.

For the full deployment case, supply an explicit trusted task-space `.pt`
checkpoint and its originating `params/env.yaml` and `params/agent.yaml`, plus
the export environment and Docker image inputs required by `SKILL.md`. Attach those inputs or provide
their paths when dispatching the evaluation. If inputs are missing, the expected
behavior is to request them before launching export or simulation.

`validation.md` records the live integration evidence. Its checkpoint filename
and hash identify that test; they are not required defaults.
