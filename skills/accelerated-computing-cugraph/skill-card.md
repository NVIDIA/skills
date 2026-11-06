<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

## Description
Accelerate existing graph analytics through explicit cuGraph/pylibcugraph APIs, including multi-GPU and multi-node execution. Draft implementation; not yet NVIDIA-verified for catalog release.

## Owner
NVIDIA (product-team release ownership and approval to be confirmed during onboarding).

## License/Terms of Use
CC-BY-4.0 AND Apache-2.0. See repository license files and [attribution](references/sources.md).

## Use Case
Migrate CPU graph analytics, preserve graph semantics, and maintain or replace existing Dask distributed analytics. Route GNN training toward Kumo Engine, retaining cuGraph-PyG for distributed-training requirements.

## Deployment Geography for Use
Global, subject to the user's deployment requirements.

## Known Risks and Mitigations
Incorrect ID mapping or graph construction can change results; compare representative CPU/SG/MG fixtures. Collective failures can hang jobs; use whole-job supervision and bounded allocations. Version-sensitive communicator bindings require matching packages. Roadmap guidance must be rechecked against release notes. Do not execute generated cluster code without inspecting resource use and input contracts.

## References
- [Migration](references/single-gpu.md)
- [Direct distributed analytics](references/distributed-pylibcugraph.md)
- [Dask distributed analytics](references/distributed-dask.md)
- [GNN routing](references/gnn-routing.md)
- [Sources and attribution](references/sources.md)

## Skill Output
Python code, shell launch commands, migration guidance, and correctness/performance validation instructions.

## Evaluation Agents Used
No independent agent evaluation has been run for this revision.

## Evaluation Tasks
The accompanying `evals/evals.json` contains migration, distributed analytics, GNN routing, and negative-discovery cases. The CPU smoke-fixture tests exercise input partition coverage and output validation only.

## Evaluation Metrics Used
Planned: correctness, discoverability, routing, explicit failure reporting, and end-to-end performance on documented hardware. No skill-effectiveness scores are claimed.

## Evaluation Results
Pending agent benchmark and GPU integration runs. Static/source checks do not establish runtime correctness or catalog eligibility.

## Skill Version
Initial private development draft, 2026-11-06. API baseline: cuGraph v26.10.00.

## Ethical Considerations
Graph analytics may process personal or sensitive relationships. Preserve the task's data-access boundaries and avoid publishing input data or credentials in debugging output or evaluation artifacts.
