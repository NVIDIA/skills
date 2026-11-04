<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Guide-Based First-Value Checks

The selected guide is the source of truth for its current procedure and proof.
Read the entire relevant path before adapting it.

## Source And Compatibility

Use a guide compatible with the installed stable release. Keep its commands,
configuration, and referenced sample files together. If the requested guide is
only in a pending PR, identify that revision and any unpublished prerequisites.
Do not treat source-built test results as package-publication evidence.

If the guide uses a different version, check compatibility rather than silently
changing the installed package. Derive any necessary compatibility restriction
from current documentation, with a source; do not maintain a second version
matrix in the skill.

## Adapt Only The Environment

Resolve the guide's working directory, explicit configuration selection,
provider requirements, package manager, permissions, and output paths in the
user's environment. Explain persistent host changes and service restarts when
the selected guide needs them.

Preserve existing configuration and artifacts. Prefer a fresh trial location
when old files could satisfy verification accidentally. Do not delete evidence
or rewrite unrelated settings merely to reproduce a guide's directory layout.
Protect captured data and summarize results without dumping payloads.

## Check The Guide's Actual Proof

Follow its expected output and artifact checks after the documented flush or
shutdown sequence. Establish that artifacts belong to the current invocation.
Do not create expected agent-produced artifacts yourself to make checks pass.

Use the guide's event categories, relationships, schema, and format. Do not
apply another path's tool/model requirements or demand an additional exporter.
Distinguish local deterministic callbacks from live provider validation.

When a check fails, diagnose the smallest failed boundary using the guide and
current diagnostic documentation. A successful model reply or a doctor check
alone does not establish event capture. Report what succeeded, what failed,
and the remaining uncertainty.

If a guide example conflicts with the installed release's documented public
API or a reproducible runtime diagnostic, check that release's API before
adapting it. Make only the smallest verified correction within the trial's
scope, explain the departure from the guide, and retain the initial failure in
the result. A corrected run must not be reported as a successful verbatim run
of the original guide.
