---
name: omniverse-usd-performance-tuning
description: "Diagnose and optimize USD stages for slow loading, high memory, low FPS or validation; preserve asset boundaries and reuse shared definitions."
license: Apache-2.0
tools:
  - Read
  - Shell
  - Write
compatibility: >
  Requires a verified standalone Usd Optimize/OpenUSD runtime for operations, usd-validation-nvidia for supported checks, and a writable output workspace. Remote assets require a working resolver. Render profiling is an external handoff.
metadata:
  version: "0.5.0"
  author: NVIDIA Omniverse
  tags:
    - triage
    - performance
    - usd
    - profiling
  domain: ai-ml
  languages:
    - python
---
# Omniverse USD Performance Tuning

## Purpose

Improve loading, memory use or rendering for the requested workload. Preserve the prim paths,
appearance, placement and behavior required by downstream applications. Follow the component/subcomponent model in NVIDIA's
[Factory-Level USD Structuring guide](https://docs.omniverse.nvidia.com/vfi/latest/guide/factory-level-structuring.html).
Apply the relevant parts of that model; a static asset does not need new animation or physics.

## Inputs

Required: the source USD stage. When no workload is specified, use factory layout
planning and visualization and record that assumption in the request/report.
Explicit workload and preservation requirements take precedence.
Optional: an output workspace, preservation requirements, physical error budget,
existing library indexes and recorded runtime evidence. Use values from the current
user request; reuse recorded values only while they remain applicable. Follow
[setup](references/setup.md#workspace-and-artifacts) when no output path is supplied.

## Prerequisites

Verify the selected runtime and required capabilities using [setup](references/setup.md).
Keep its preflight record with the run; installation and remote access are conditional
steps described there.

## Instructions

1. **Inspect.** Establish the workload, preservation requirements and output workspace.
   Set up [progress and dashboard access](references/verification.md#progress-and-steering) before assessment.
   Verify [setup](references/setup.md) and inspect the selected source configuration.
2. **Judge boundaries.** Review equipment, service-system sections and architectural
   elements in the selected configuration, including animation and interiors. Record
   owners, editable interfaces, equipment categories and optional families using [boundary guidance](references/structure.md).
3. **Classify and author.** Classify accepted assets children before parents.
   Capture or reuse the original [profile](references/verification.md)
   before authoring. Prepare reusable definitions, materials and occurrence opinions
   using [asset authoring](references/asset-authoring.md).
4. **Optimize unique definitions.** Run supported native [operations](references/operations.md)
   concurrently on disjoint scopes with the recorded physical budget.
5. **Publish semantic libraries.** Use one writer per category layer and remap consumer references.
6. **Check composition.** Check authored ownership, references, bindings, placement, animation
   and interface accessibility through the [written-output checks](references/verification.md#written-output-checks).
7. **Profile and report.** Compare the final consumer with the original baseline under
   comparable conditions and report measured benefits, completion and operations by asset definition.

```mermaid
sequenceDiagram
    participant A as Agent
    participant T as USD tools
    participant W as Definition workers
    participant L as Library publisher
    participant D as Read-only progress dashboard
    Note over A,L: One compact progress state also renders text progress
    T-->>D: Recorded phase, assessment rounds, counters and decision reasons
    A->>T: Inspect original selected configuration
    T->>T: Collect cached child-to-parent summaries and repetition nominations
    T->>T: Load supplied indexes once and batch authored product clues
    T-->>A: Product review groups, member context, animation and reuse evidence
    loop Until boundaries and interiors are reviewed
        A->>A: Select asset boundaries, then identify required internal prims and properties
        A->>T: Submit boundary decisions with reviewed scope, required edits, category and optional family
        T->>T: Reuse unchanged source evidence, recompute boundary-dependent results
        opt Geometry ownership requires reconstruction
            T->>T: Regroup or native split, remap paths and reassess affected regions
        end
        T-->>A: Remaining candidates and member exceptions
    end
    T->>T: Classify selected child assets before parent assets, confirm reuse
    T->>T: Capture or reuse original baseline profile
    T->>T: Confirm accepted product identities from source and supplied indexes; reconcile complete parent-child groups
    T->>T: Skip worker inputs for confirmed groups; author remaining materials, editable prims and occurrence overrides
    T->>T: Determine merge groups on prepared interiors; preserve selected child assets and required prims
    T->>W: Read-only input USD layers for existing-library matching
    par Independent preparation jobs
        W->>W: Confirm remaining product, exact source or authorized normalized reuse; preserve occurrence placement
        W->>W: Consolidate accepted product or compatible exact/normalized leaf inputs; retain every occurrence
    end
    W-->>T: Confirmed library hits and definitions requiring optimization
    T->>W: Owned geometry prim paths in read-only input layers
    par Independent definition scopes
        W->>W: Remove small meshes when authorized; apply asset representation: reserve merge inputs or fit analytic prims; reduce, merge (native clustering if required), prune, flatten, prune
    end
    W->>L: Geometry outputs and assembly authoring outcomes
    L->>L: Share eligible static whole definitions within current category publication
    L->>L: Share remaining whole meshes; write USD reuse metadata and compact index
    L->>T: Output USD stage with rewritten references and dependencies
    T->>T: Check authored composition, ownership, animation and interfaces
    T->>T: Profile final output USD stage and report results by asset definition
    T-->>A: Measured changes, coverage and unresolved work
    T-->>D: Final verdict and report/library paths
```

Use the provided tools for traversal, native operations, authoring and checks;
the agent supplies engineering judgments. Preserve current variant selections,
authored animation, required materials and interfaces. Trust supported USD Optimize
operations with valid settings and scopes. Deep geometric comparisons and whole-scene
edit trials belong in development tests, not ordinary execution.

Complete when every unique definition has an executed or evidenced unchanged outcome,
every asset has justified ownership and structure, and the written consumer passes
composition checks. Retain native failures, unresolved judgments and missing required
capabilities as incomplete work. Repeat only affected work when evidence changes.

## Scope and authorization

- Diagnosis and validation requests remain read-only. An optimization request authorizes
  work in a separate output workspace within its stated preservation limits.
- Never overwrite source layers or inherited libraries without explicit authorization.
- Reuse authorization already given. Ask only for a missing decision that changes identity,
  required appearance or a loss budget; explain the affected component and consequence.
- Report only observed changes. Measure rendering before claiming higher FPS or lower GPU use.

## Load only the relevant guidance

| Request | Read |
|---|---|
| Validation only | [Setup](references/setup.md), then the validation section of [operations](references/operations.md) |
| Approved operation chain | [Setup](references/setup.md), [operations](references/operations.md), [verification](references/verification.md) |
| Existing or persistent asset library | Add the [library index](references/library-reuse.md) requirements to the assessment |
| Missing standalone runtime | [Installation](references/install-standalone.md) |
| Remote stage | [Remote access](references/remote-assets.md) before the first open |
| Report from completed evidence | [Verification](references/verification.md) |

Use [upstream package guidance](references/upstreams/usd-optimize.md) for selected operation
mechanics and proxy workflows. Keep large logs on disk as specified in setup.

## Limitations

Convert CAD input with the source converter's workflow and NVIDIA's
[CAD conversion guidance](https://docs.omniverse.nvidia.com/vfi/latest/guide/cad-conversion.html).
Render profiling requires NVIDIA/omniperf or the user's available profiler; USD open
and traversal timings alone do not establish rendering improvements.

## Troubleshooting

For unavailable imports, operations or remote assets, follow [setup](references/setup.md).
For incomplete execution or failed composition checks, follow
[verification](references/verification.md#written-output-checks); retain the failed
artifacts and report the missing capability or failed check.

## Examples

After runtime preflight, use the verified Python interpreter from setup to inspect a stage:

```text
python scripts/successor_assessment.py inspect --source /data/factory.usdc --workspace /data/factory-review
```

Review the generated candidates using [boundary guidance](references/structure.md),
fill the generated decision template, then submit it:

```text
python scripts/successor_assessment.py decide --workspace /data/factory-review --decisions /data/factory-review/decisions.json
```

These commands perform assessment; continue through the instructions above only when
review is complete. Run from the skill directory; generated data stays in the workspace.

## Output Format

Maintain [live progress and editable steering](references/verification.md#progress-and-steering) throughout execution.

For end-to-end optimization, provide the written USD assembly, validated report JSON,
and generated Markdown and HTML reports. State the runtime, operations actually executed,
preservation and coverage results, measured changes, source-write status, and artifact paths.
Diagnosis-only work may return a findings summary stating that no optimized stage was written.
