<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Omniverse Content Services

Version: **0.1.0**

A workflow expert for USD content collaboration, 3D content pipelines, and
physical AI simulation data access. It helps identify source systems, authoring
and compute locations, collaboration needs, and permitted caching, then routes
implementation questions to the appropriate repository expert. Existing systems
remain the source of truth.

Use it for manufacturing and factory planning, automotive configurations,
architecture and construction, digital twins, media and 3D asset creation,
robotics, and fleet simulation. It connects these tasks to common questions:
how to use existing infrastructure with USD, create and publish content versions,
coordinate contributions, and deliver consistent inputs to scale-out compute.

## Use the expert

Load [SKILL.md](SKILL.md) in your agent harness, or ask:

> Use Omniverse Content Services to design a content pipeline from our existing
> storage to our authoring applications and simulation workers.

Other starting questions:

- How can our factory team use CAD and existing storage in a USD workflow?
- How do we version a vehicle or building scene without silently changing downstream inputs?
- How can robot or fleet simulation workers reuse the right USD assets across sites?

The expert also recognizes Omniverse Content Workspace, USD Content Workspace,
and USD content collaboration as discovery phrases.

## Public components

| Repository | Role |
| --- | --- |
| [ovstorage](https://github.com/nvidia-omniverse/ovstorage) | Content access, authentication, and USD resolution |
| [usd-caches](https://github.com/nvidia-omniverse/usd-caches) | Source-content and derived-data caching |
| [usd-content-workspace](https://github.com/nvidia-omniverse/usd-content-workspace/tree/release/v0.1.0) | Service contracts, adapters, and deployment workflows from `release/v0.1.0` |
| [ovlive](https://github.com/nvidia-omniverse/ovlive) | Concurrent USD collaboration |
| [ovpackage](https://github.com/nvidia-omniverse/ovpackage) | Versioned publication and package dependencies |

See [repository routes](references/repositories.md) for expert discovery and
[workflow discovery](references/workflow-discovery.md) for qualification questions
and first-proof criteria. Repository routes are not compatibility guarantees:
verify availability, source revisions, and application support before execution.

The dated [release record](references/releases.md) tracks the latest two published
releases where available, source-only tags, and each connected expert's version
status. See [NGC prebuilds and documentation](references/prebuilt-artifacts.md) for
service images, Helm charts, cache packages and their owning guides. This workflow
expert remains **0.1.0**; component versions advance independently.

The expert distinguishes proposed designs, source-reviewed capabilities, and
runtime validation. It does not replace the component experts or claim that an
untested deployment is ready for production.
