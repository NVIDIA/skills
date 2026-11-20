---
name: kumo-model-persistence
description: Publish and reload Kumo models through the Snowflake registry, reconstruct compatible scoring inputs, and retain or retrieve run diagnostics and output files through a Snowflake stage. Use for saved-model scoring and artifact retention without retraining by default.
---

# Persist and reuse a model

Preserve the known task and explicit run/version. A registry model stores a portable model
bundle, not source tables, the DiskGraph index or the scoring population. A local retained
bundle is a filesystem artifact inside the notebook environment, not another connector.

Read [publication and restore](references/reuse.md) for fit-time publication, version recording,
trusted-bundle loading and reconstruction. Read [artifact retention](references/retention.md)
when results must survive the notebook session. Use
[custom populations](../kumo-custom-populations/SKILL.md) for exact scoring rows.

Resolve only missing information: model registry or retained bundle, exact version, source
and graph reconstruction, population/anchors, and any requested output or retention destination.
Do not reopen the trained target or choose the latest version. Loading does not authorize
training, replacing a registry version, or overwriting a destination. No hosted model/job ID
lookup is provided by this local API.

Report model name/version, reconstructed snapshot identity, population, prediction row count
and exact durable destinations. A returned registry record proves publication completed; a
checked stage transfer proves retention. Neither proves later scoring on compatible data.
