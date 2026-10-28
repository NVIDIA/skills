<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Repository routes

These are routing targets, not guarantees of public availability or compatibility.
Read the entrypoint at the selected revision before relying on it. If a candidate
path is absent, inspect the repository's skill index and report the gap.

For each actual handoff, capture the expert's declared version separately from
its Git revision and the component/API versions being discussed. Use the version
rules in the router. The [release record](releases.md) records inspected revisions
and version metadata; these routes do not establish runtime compatibility.

| Need | Public source | Expert or documented entrypoint |
| --- | --- | --- |
| Access existing storage, configure client authentication, integrate USD resolution | [ovstorage](https://github.com/nvidia-omniverse/ovstorage) | [ovstorage expert](https://github.com/nvidia-omniverse/ovstorage/blob/v0.4.0/skills/ovstorage-expert/SKILL.md) |
| Implement content service contracts or adapters; local-process or cluster services | [usd-content-workspace](https://github.com/nvidia-omniverse/usd-content-workspace/tree/release/v0.1.0) | [workspace expert on release/v0.1.0](https://github.com/nvidia-omniverse/usd-content-workspace/blob/release/v0.1.0/skills/usd-content-workspace-expert/SKILL.md) |
| Publish and consume exact package versions and dependencies | [ovpackage](https://github.com/nvidia-omniverse/ovpackage) | Verified task-skill fallback: [CLI](https://github.com/nvidia-omniverse/ovpackage/blob/0.2.1/.agents/skills/ovpackage-cli/SKILL.md) or [API](https://github.com/nvidia-omniverse/ovpackage/blob/0.2.1/.agents/skills/ovpackage-api/SKILL.md); no dedicated expert found at this revision |
| Concurrent USD edits, session integration and recovery | [ovlive](https://github.com/nvidia-omniverse/ovlive) | Documentation fallback: [client](https://github.com/nvidia-omniverse/ovlive/blob/v0.1.0/client/README.md), [support boundary](https://github.com/nvidia-omniverse/ovlive/blob/v0.1.0/client/docs/ovlive-support-boundary.md), [service](https://github.com/nvidia-omniverse/ovlive/blob/v0.1.0/service/README.md). No SKILL.md or AGENTS.md found at this revision; do not invent an expert handoff. |
| Local/shared source-content or derived-data caching | [usd-caches](https://github.com/nvidia-omniverse/usd-caches) | [cache expert](https://github.com/nvidia-omniverse/usd-caches/blob/v0.1.0/skills/usd-caches-expert/SKILL.md) |

Use `release/v0.1.0` for usd-content-workspace, including local checkout selection.
Resolve and record the branch's current commit for each handoff; branch links can
advance. Keep this branch selection when refreshing the release record.

For prebuilt deployments, follow [NGC artifacts and owning documentation](prebuilt-artifacts.md).
The workspace expert owns service deployment and release-inventory interpretation;
the cache expert owns Hub, OVCC and OVDC placement and integration. The ovlive
client and its separately versioned Live Edit Service require compatible hosts;
a source tag alone does not demonstrate a runnable integration.

Handoff example: “We have one developer on Linux, existing filesystem content,
and want a plain local service process. Preserve the filesystem as authority.
Using this source revision, identify the smallest runnable option, actual APIs,
prerequisites, limitations and a disposable save/readback check. Cite source files;
do not deploy anything.”

A multi-region case should additionally give worker locations, source location,
read reuse, derived-data generation and allowed retention. Request a topology and
its evidence; do not guess cache counts, bucket replication or automatic failover.
