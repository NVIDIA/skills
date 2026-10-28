<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Discover the content workflow

Start with what the user needs to create and who or what consumes it next. Then
identify the authoritative sources, required handoff, and execution locations.
Keep user requirements, proposed designs, and verified component behavior separate.
These examples help discover requirements; they do not assert ready-made connectors,
format conversion, domain correctness, or deployed integrations.

## Recognize the industry and the content task

| Context and example request | Content path to explore | Key qualification |
| --- | --- | --- |
| Manufacturing and factory planning: “Use our engineering data to compare production layouts.” | Parts and assemblies → task-suitable assets → equipment/cell/factory scenes → planning or simulation | Which dimensions, placements, engineering metadata and revisions must survive the handoff? |
| Automotive: “Share vehicle variants between design, visualization and simulation.” | Design assemblies → geometry, materials and variant products → configured vehicle or road scene | Which configuration and source revision does each consumer accept? |
| Architecture, engineering and construction: “Coordinate building and site content from several tools.” | CAD/BIM and survey sources → discipline contributions → coordinated building/site → review or simulation | How are coordinates, units, source authority and discipline updates reconciled? |
| Operational digital twins: “Keep a deployed scene aligned with changes in our facility.” | Accepted scene baseline + asset changes and observations → reviewed update → consuming application | Which changes update content, which are live state, and what must remain reproducible? |
| Media, entertainment and general 3D creation: “Let artists and agents build scenes together.” | Models, textures, materials, rigs and animation → asset products → layout, lighting and shots → rendering | Which role owns each output and when may downstream work adopt it? |
| Robotics and physical AI: “Turn robot and environment assets into repeatable training runs.” | Robot/environment sources → simulation-ready representations → task scenes and configurations → training/evaluation outputs | Which content, physics assumptions, software/configuration and inputs identify a run? |
| Fleet simulation and autonomy: “Run many vehicle or robot scenarios across a compute farm.” | Shared assets and environments + scenario variants → pinned job inputs → distributed runs and results | Which workers reuse which versions, and how are results linked to scenario inputs? |
| Other fields, including logistics and immersive environments | Source content → owned contributions → accepted product → receiving task | Ask what the recipient must do; do not force an industry-specific pipeline. |

Industry is context, not a deployment prescription. One project can combine several
rows. Asset creation is often a dependency graph: a material, model, rig, layout
and simulation configuration may advance independently. Do not force every project
through a linear modeling-to-rendering sequence.

Distinguish working edits, review previews, published versions and accepted
configurations. A new version is not automatically approved for every consumer.
Creators can continue working while downstream people and jobs consume an
identified baseline; adopting a change is a separate decision.

For conversion and handoff, ask what must survive: source and instance identities,
units, coordinates, relationships, metadata, materials or behavior. Preserve the
editable source and record intentional losses. Verify the required open, edit,
save, export or simulation operation in the receiving tool.

## Choose a workflow path, then the component expert

Use the public links in [repository routes](repositories.md). Load only the expert
needed for the current decision; ownership below is a routing guide.

| User intent | Establish first | Route and first useful proof |
| --- | --- | --- |
| Use existing infrastructure with USD | Current file/object stores or asset systems, identity, native formats, protocols, and read/write needs | `ovstorage` for client access/resolution; `usd-content-workspace` for service contracts/adapters. Open one representative asset and its dependencies through the intended application. |
| Create or refine content with people and agents | Native source versus derived USD, contribution ownership, required consumer semantics | Coordinate the creation/conversion specialist, then access and publication experts. Verify what the receiving task needs; opening a USD file alone does not establish simulation readiness. |
| Version USD content and publish accepted work | Editable sources, complete output products, dependency versions, approval boundary and retention | `ovpackage` for publications and dependencies; `ovstorage` for file identity and conditional writes. Reopen an exact accepted scene after an upstream asset changes. |
| Collaborate on an asset or scene | Explain live sessions and reviewed contributions, then qualify tools, ownership and desired interaction | `ovlive` for concurrent USD sessions and interactive feedback; access/publication experts for handoffs. Test ownership, conflict behavior and recovery in the actual host applications. |
| Deliver USD to scale-out compute | Source and worker locations, job input versions, repeated reads, network limits and allowed caching | `usd-caches` with access/package experts. Compare cold and repeated runs while verifying identical input versions and complete dependencies on fresh workers. |
| Feed results back into content refinement | Baseline scene, input versions, transformation/settings, produced outputs and acceptance decision | Publication/access experts with the domain specialist. Trace a result to its inputs and adopt a changed asset explicitly without altering the old baseline. |

Treat PLM, BIM, asset-management systems, shares and object stores as possible
sources, not prevalidated integrations. A new service may expose compatible
contracts while those systems remain authoritative. Qualify native application
access separately from USD reference resolution.

For distributed jobs, distinguish scene/asset inputs from code, configuration and
random seeds needed for repeatability. Record the inputs the job actually used,
give outputs explicit ownership, and link results to that record. Content Services
does not establish scheduler support, simulation fidelity or reproducibility of
an entire training system by itself.

## Explain collaboration options before qualifying

A vague collaboration request is an opportunity to explain what the team could
do. Offer [ovlive](https://github.com/nvidia-omniverse/ovlive) by name and explain
its role in concurrent USD sessions before narrowing the workflow. Cover these
complementary options in plain language:

| Option | What it enables and why it helps |
| --- | --- |
| Live USD collaboration with `ovlive` | People and agents participate in a shared editing session through compatible integrations. Shared visibility lets a person inspect an agent's contributions as work progresses, gives collaborators common scene context, and can reduce repeated export/reload handoffs and feedback delays. Useful for interactive scene assembly, joint iteration and review while editing. |
| Independent contributions with reviewed publication | People and agents own separate assets, layers or other outputs and publish identified versions. This supports asynchronous work and gives downstream consumers a stable baseline they can adopt explicitly. |
| A combination | Use a live session to iterate together, then review and publish the accepted result for other teams or simulation jobs. Continued live work does not silently change a job's pinned inputs. |

For example, propose a person adjusting a factory layout while an agent updates
an assigned material contribution, with both inspecting the shared USD scene.
Explain the intended benefit: the person can guide and correct the work during
creation. Treat this as a workflow to qualify in their tools, not a verified
integration or a claim that every agent harness can join without integration.

After explaining the options, ask which applications and agent interfaces they
use, and whether they want live work, reviewed handoffs or both. Route session
integration, supported operations, overlapping edits, permissions and recovery to
the ovlive expert at the selected revision. Do not promise automatic conflict-free
editing, arbitrary native-file collaboration, approval, or version history from a
live session alone. Keep the benefits clear even when implementation support still
needs verification; disclose unavailable source evidence without inventing details.

## Ask questions that change the next decision

- Establish the current problem: synchronization delays, stale reads, concurrent
  writes, reproducibility, or storage growth. “Consistency” needs an observable
  expectation such as rejecting an outdated save or reopening an exact revision.
- Separate the application-to-service connection from the service-to-storage
  connection. A Linux service using a NAS mount does not establish how a Windows
  or macOS application opens its native files.
- For each first-test application, establish native source formats, USD exports,
  runtime/OS support, and actual open/save integration. A USD resolver does not
  make every native file operation service-aware.
- “Several agents on one asset” leaves ownership unresolved. Distinguish separate
  contributions/layers, serialized edits to one native file, and simultaneous
  USD session edits. Explain ovlive and its benefits when offering these choices;
  live collaboration can be useful even when contributors own different parts.
- Ask whether the user accepts a managed store or needs ordinary folders that
  applications edit directly. Do not initialize a managed backend over existing
  source folders or promise transparent compatibility between those models.
- Clarify local-copy policy: "Is managed local caching okay, provided your central
  system remains authoritative, or do you need content never to persist on the
  device? Are you mainly trying to avoid manual downloads and unmanaged copies?"
  Record the answer before choosing cache placement or remote authoring. If caching
  is acceptable, establish relevant retention, cleanup and lost-device protection
  requirements with the cache expert; do not assume those controls are implemented.
  Distinguish disposable read caches from writable working files, unsaved edits,
  autosaves and offline replicas. Direct network access alone does not establish
  that applications leave no local data. Avoid treating a ban on unmanaged copies
  as a ban on all caching, or treating permission to cache as permission for offline
  authoring or unrestricted duplication.

## Return a capability map with evidence

For each requested capability, state the owning repository, implementation at
its selected revision, application/backend prerequisites, and one acceptance
check. Label it as implemented, example-only, a build recipe, or unknown; record
runtime evidence separately. Avoid presenting the whole portfolio as features
already installed in a proposed single-host service.

Version history, immutable asset releases, deduplication, conflict detection,
Live collaboration and backup solve different problems. Ask the storage expert
about file versions and conditional writes, the package expert about exact
published dependencies, and the Live expert about concurrent USD sessions.
Do not infer deduplication or transactions merely from package versioning.
Whole-file deduplication reuses identical files; it does not imply block deltas
for a changed large native document. Measure actual retained bytes.

When source data, converters or assets change, identify affected consumers before
adoption. Preserve the previous exact inputs and verify the proposed update with
the receiving application. Package dependencies do not automatically establish
the closure of every USD reference, texture or other resource a consumer needs.

For NAS-backed services, ask the workspace expert to qualify where content,
metadata databases, manifests, journals and locks live. Check persistence,
rename/locking and failure behavior on the actual mount. Neither SMB nor NFS
alone supplies version history, application conflict resolution or recovery.
Do not promise one protocol is faster without a representative measurement.

For cache advice, distinguish workstation caches, shared source-content caches,
and caches of derived computation. Verify backend, client routing and host support. A local cache
is not automatically a LAN cache, and a remote content cache is not necessarily
an accelerator for an ordinary NAS mount. Distinguish examples tested against
mocks from real cache-binary and application checks.

## Small proof before expansion

Choose one representative content handoff and receiving task. Agree on output
ownership and success criteria before writing. For collaboration, use two
contributors and test a stale save; for versioning, test exact old-version reads
and publication; for compute, use a fresh worker and verify dependency closure.
Add save/readback, storage-growth and interruption/restart checks where relevant.
For a custom service, separate API conformance from authenticated application
tests, permissions, recovery and performance. Skipped tests and pod readiness are
not workflow passes. A small successful proof does not qualify production scale.

When asked where to start with source, give the owning repository and the actual
entrypoint at the selected revision. Follow with the smallest example and the
implementation framework; explicitly label any recipe that requires new code.
Verify the public source path and accessibility before recommending it. Do not
turn a workstation-specific local path into release guidance.
