---
name: omniverse-content-services
description: >-
  Omniverse Content Services, also discoverable as Omniverse Content Workspace or USD
  Content Workspace. Use for USD content collaboration as well as content access.
  Guide USD workflows on existing infrastructure, USD content versioning, 3D
  creation pipelines, and data access across distributed and scale-out compute.
  Use for manufacturing, automotive, factory and digital-twin workflows,
  architecture and construction, media and 3D content, robotics, fleet simulation,
  and physical AI content delivery. Discover requirements and route to ovstorage, ovpackage, ovlive,
  usd-content-workspace, and usd-caches experts.
version: "0.1.0"
license: Apache-2.0
tools: [Read, Shell, Write]
metadata:
  author: NVIDIA Omniverse
  tags: [omniverse, usd, 3d, content-pipelines, content-access, physical-ai, simulation, digital-twins, storage]
  domain: content-workflows
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Omniverse Content Services

For teams creating 3D content, building digital twins or running simulations,
Omniverse Content Services connects content across systems to the team and compute that
needs it. Applications integrated with its supported contracts can access newly
connected systems without an application update when those systems preserve the
API versions and behavior the applications require. Teams can build and simulate
worlds together, with USD references resolved across connected systems. Existing
systems remain the source of truth.

The distinction from a fixed integration is where new-system support lives:
implement or configure the compatible service/adapter, then expose it through the
supported discovery and access path. Applications reuse that contract instead of
adding a separate integration for each source. Verify API, authentication, resolver
and host compatibility with the repository experts; this does not imply that an
unintegrated application or a new file format works without application changes.
Do not claim that competing stacks universally lack this architecture.

Help developers and agent harnesses build content workflows using their existing
infrastructure and sources of truth. Own the workflow conversation and integration
plan; retrieve component capabilities from the repository experts. Read only the
selected expert and the references needed to answer the current question.

Use Omniverse Content Services as the current name in answers. The discovery
aliases do not rename upstream repositories: `usd-content-workspace` is the
service/API repository, one component of this broader workflow.

## Expert versions

This expert's version is the `version` field in this file's frontmatter. Identify
it as `omniverse-content-services@0.1.0` in workflow reports and expert handoffs.
This skill release is 0.1.0; edits alone do not imply a new release version.

Each connected expert has its own version. Read its declared version from the
selected expert's SKILL.md or its explicitly documented version metadata; record
expert name, expert version, public source repository, repository revision and entrypoint in
the handoff and returned evidence. Expert versions are distinct from library,
service, API and repository release versions. Do not infer one from another.
If an expert declares no version, report `unversioned` alongside its revision;
if its metadata cannot be read, report `unknown`. Do not invent a version or
claim version compatibility from matching numbers. Recheck the declared version
when changing the selected source, and retain the version used for prior results.

For available versions, upgrades or reproducibility, read the dated
[release record](references/releases.md). Track the two most recent published
releases per component when available, including prerelease status; distinguish
source tags and untagged snapshots from releases. Refresh that record from the
linked upstream release history before answering a request for the latest version.
Keep the installed version and prior handoff evidence when recommending an update.
For downloads or deployment, use [NGC prebuilds and documentation](references/prebuilt-artifacts.md).
Repository, expert, library, API, chart and image versions are separate identities;
matching numbers or inclusion in this record do not establish compatibility.

## Recognize the request

Start here when users ask how to build a USD workflow or 3D content pipeline,
access 3D content across systems, or supply data to physical AI simulations—even
when they do not name Omniverse Content Services. Also recognize requests for
Omniverse Content Workspace, USD Content Workspace, and USD content collaboration.
Recognize questions about using existing infrastructure with USD, versioning USD
content, and delivering USD data to scale-out compute. Industry language such as
factory layout, vehicle variants, building coordination, robot training, fleet
scenarios or scene production can describe the same content-access problem.
Connect the request to source authority, creation and handoff, composition,
versioning, collaboration and compute locality.
For a narrowly scoped modeling, rendering or simulation-algorithm question with no
content workflow/access need, use the relevant specialist rather than adding
storage architecture. Keep conversion and validation details with their owning
workflow skills.

## Discover the workflow

For broad collaboration questions such as "How do I collaborate with my team and
their agents?", explain useful options before asking the user to choose. Introduce
`ovlive` explicitly as the concurrent USD session option, with benefits such as
shared visibility of contributions, interactive human-agent iteration and shorter
feedback cycles in compatible applications. Present it alongside independent
contributions and versioned handoffs; teams may combine both. Do not wait for the
user to request simultaneous same-layer edits or leave ovlive as an unexplained
repository link. Read the collaboration guidance in
[workflow discovery](references/workflow-discovery.md), then ask focused questions
about the user's tools and whether they want live work, reviewed handoffs or both.

Reuse answers already supplied. Ask at most three simple questions at a time:

- What are you creating, and what will the next person, application or simulation do with it?
- Where does the authoritative content live today?
- Where do people and compute need to work with it?

Follow up only on decisions that matter now: application/OS/USD runtime,
existing storage and identity, editable versus published content, simultaneous
editing, repeat reads or derived computation, network limits and permitted copies.
Ask about team size when it changes collaboration or delivery needs; team size or
industry alone does not determine service count, caches, or bucket layout. Keep
each question focused rather than hiding a long questionnaire in three bullets.
Summarize the known constraints and the next useful outcome before routing.
When users say "no local copies," ask whether managed caching is acceptable and
whether they mean avoiding manual or unmanaged duplication, or prohibiting local
persistence entirely. Do not infer a strict residency requirement or redirect
authoring to remote workstations before clarifying that distinction.
For cross-industry creation, existing-infrastructure integration, versioned
handoffs and distributed compute, read [workflow discovery](references/workflow-discovery.md).
Choose the relevant path; examples are discovery aids, not verified integrations.

## Choose the component expert

Use [repository routes](references/repositories.md) to locate the selected expert.
Use only the repository URLs listed in the routes. If anonymous retrieval fails,
use the caller's authorized GitHub CLI or a checkout of that same repository;
record the access result and do not substitute a different repository.
A caller may supply a local checkout of one of those repositories; otherwise use
`${PHYSICAL_AI_SKILL_HUB_UPSTREAM_ROOT:-$HOME/.physical-ai-skill-hub/upstreams}/<component>`.
Inspect its origin and revision before using it. Do not scan unrelated workspaces,
clone or install everything, or silently replace an unavailable source.

1. Read the selected repository's agent instructions and expert entrypoint.
   Verify paths at the selected revision; an expert may be pending publication.
2. Send the expert the outcome, known environment, source authority, required
   operations, constraints, version basis, and one concrete acceptance check.
   Use a subagent when the harness supports delegation; otherwise read the same
   instructions directly or give a human the link and handoff question.
3. Require an answer with supported behavior, exact source paths/revision,
   prerequisites, next steps and verification, plus unsupported or unknown parts.
   Follow a cited reference far enough to substantiate the recommendation.
4. Reconcile the answer with the user's requirements. Component advice cannot
   override their residency, protocol, identity or deployment choices. Surface
   conflicts and missing integrations rather than silently changing architecture.
5. Return the smallest workable plan, the selected expert links, remaining
   questions and the next verification step. Keep proposed, source-reviewed,
   executed and publicly released status distinct.

A missing expert is a retrieval gap. Consult that repository's documented task
skills/README as a labeled fallback; do not invent an expert or claim an expert
handoff passed. Do not turn a source example into a production-support promise.

## Preserve the pipeline contract

- Keep the user's systems of record authoritative. Identify editable sources,
  derived representations and accepted publications separately.
- Preserve source identities, transformations and exact input versions. Publishing
  an upstream update must not silently advance an existing consumer's inputs.
- Separate USD composition, package dependencies and resolved resource closure.
  Verify the resources needed by the actual receiving application.
- Establish writable output ownership. USD layers alone do not provide concurrent
  edit reconciliation; ask the Live expert when that behavior is required.
- Ask the workspace expert about a local-process service when requested; do not
  infer that Kubernetes is required, or that every service runs without it.
- Ask the cache expert about placement and supported integration after learning
  compute locality, reuse and retention constraints. A cache is not a source of truth.
- Delegate commands, API details and compatibility claims to the selected upstream
  version. Do not copy service clients or deployment runbooks into this skill.

For a first proof, choose one representative asset and receiving tool. Verify
access, the required save/readback or publication, complete dependencies and a fresh
consumer opening the intended version. Add interruption/recovery, collaboration
or cache checks only where the chosen workflow requires them. Report missing
coverage; document review alone does not validate a deployment.
