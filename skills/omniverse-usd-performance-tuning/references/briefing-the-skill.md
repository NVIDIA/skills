---
agent_context: usd-performance-workflow
agent_routes:
  - omniverse-usd-performance-tuning
agent_next:
  - workflow.md
freshness: 2026-08-13
version: "0.4.1"
---

# Briefing this skill

How to write the request. For humans preparing a job, and for agents checking
whether a brief has enough in it to proceed.

The skill measures the asset and presents what it finds. It cannot know what the
result is *for*, and that is what decides the strategy. Three converted CAD stages
of comparable size need three different dominant operations — the right one for a
rack destroys a production line. So the brief supplies intent and the skill
supplies evidence; a brief that states nothing leaves the skill guessing at gates
that exist precisely because guessing is wrong.

None of this requires knowing operation names. Describe the deliverable.

## The four things a brief should answer

Each maps to a gate that stalls or mis-fires without it.

| Say | Because |
|---|---|
| **What the asset is for** | `merge` destroys per-part addressability and is intent-gated. "Nobody selects individual parts, there is no physics or articulation to preserve" is what unlocks it. Without that the skill must assume you need every part selectable. |
| **Whether the structure may change** | Phase 2e asks whether to restructure into shared prototypes. Answer it — "instancing is not what I want here", or "share whatever repeats" — and the gate resolves. Unattended runs otherwise decide from measured reuse; interactive runs stop to ask. |
| **A quality budget, not a triangle budget** | See below. This is the one that silently costs you. |
| **What must not change** | Extent, named units, materials, load time. "Nothing may visibly disappear and the line must not shrink" is a real constraint the skill can check. |

## Say the quality budget. Do not name a triangle count

> Preserve surface detail down to about half a millimetre, and let the triangle
> count land wherever that puts it.

Naming a polygon target is the most natural thing to ask for and the most
expensive. A stated rate is the documented condition that switches
`decimateMeshes` out of error-budget mode: `maxMeanError` goes to `0.0` and the
only stop condition left is the quota, so the decimator removes whatever it must
to reach the number.

Measured on one CAD station, two runs of the same brief differing only in this
clause:

| Brief said | Surface lost |
|---|---|
| "roughly 11 million triangles" | **3.84%**, with 2 m regions losing up to 45% |
| "half a millimetre of detail" | **0.00%** |

If you have a hard polygon ceiling, say it as a ceiling *and* give a tolerance —
"stay under 12M triangles but never coarser than 0.5 mm" — so the quality floor
survives.

Likewise, avoid asking for a **mesh count**. The cheapest way to satisfy one is to
fuse dissimilar materials into a single mesh split by `GeomSubset`, and each
subset is a draw call. One run took that route to 6,674 subsets and measured 12
FPS against 50 for the same geometry at 298 draws. Ask for the outcome — "it has
to open fast with fifty files loaded" — and let the counts fall out.

## Three asset shapes, three different right answers

Measured on three converted stages of similar size. Reuse is "share of points
recoverable by instancing at that depth".

| | Production line | Rack of repeated units | Building / MEP |
|---|---|---|---|
| Reuse near the top (depth 2-4) | 1.7-10.3% | **72-83%** | already instanced |
| Reuse deeper (depth 5-6) | 32-39% | 66% / 34% | — |
| Materials | ~300 | ~400 | **130,000+, 99.8% redundant** |
| **Dominant lever** | merge to named units | instance repeated subtrees | deduplicate materials |

The shapes are recognisable without measuring. A line has unique stations with
repeating fasteners inside them, so reuse is deep and thin. A rack repeats whole
assemblies near the top. A BIM export usually arrives instanced already, and its
cost is material explosion rather than geometry.

If you do not know which you have, say so and ask the skill to report the
duplication profile before deciding. It measures this in Phase 2 anyway.

### A. Production line — merge-dominant

Measured: 85,871 meshes to 2,820, 617 of 617 named units kept, 0.00% surface
lost, cold open 3.256 s to 0.124 s.

> This is one station of a battery assembly line. I need it to open quickly and
> stay responsive with about fifty other files like it loaded at once. Nobody
> selects individual bolts, and there is no physics or articulation to preserve.
>
> Keep the assembly structure down to the level of named units — the stations,
> equipment groups and named machines that have real names in the tree. Below that
> the names are CAD hashes and carry no meaning, so merge everything beneath each
> named unit into that unit. I want to still be able to find a given station's
> robots in the tree when you are done.
>
> Do not work to a triangle budget. Work to a quality budget: preserve surface
> detail down to about half a millimetre, and let the triangle count land wherever
> that puts it.
>
> Nothing may visibly disappear, and the line must not shrink.

The named-unit sentence is doing more work than it looks. It gives you a navigable
tree *and* spatial locality for free, because a named unit is a machine and a
machine is spatially compact.

### B. Rack of repeated units — instancing-dominant

Not yet run end to end; the duplication profile is measured, the brief is not.

> This is a rack assembly. Whole units repeat throughout it, so share whatever
> repeats rather than merging it — I would rather have one copy of each part
> referenced many times than fused geometry. Keep it lossless if you can; do not
> reduce triangles unless you tell me what it buys.
>
> It needs to open quickly and stay light in memory. Nothing may visibly change.

Merging first would be the expensive mistake here: it bakes each copy's placement
into its points, and the duplicates stop being duplicates. On this shape that
forfeits most of the available win.

### C. Building / MEP — material-dominant

Not yet run end to end; the material profile is measured, the brief is not.

> This is a building services model exported from BIM. It already uses instancing,
> so leave the geometry structure alone unless you find something clearly wrong
> with it. What I want is the material count brought down — I expect most of them
> are duplicates of a handful of real materials.
>
> Do not change which surface has which appearance.

## What the skill will still ask you

Even a complete brief leaves decisions the skill will surface. That is intended —
they are the ones with a real trade behind them:

- **Restructure or optimize as-is**, when reuse sits in the ambiguous band.
- **A tolerance**, when a bounded-loss operation would run above its conservative
  band, or on a target carrying functional precision (articulated, physics,
  metrology).
- **Destructive operations**, when the brief has not waived what they cost.

Answering these in the brief up front is faster than being asked, but leaving them
is safe.

## Related

- `workflow.md` — the phase flow the brief drives, and the operation ordering
  invariants.
- `usd-optimize-run-operations/references/operation-safety.md` — apply-authority
  classes, and why a stated triangle target is a ceiling rather than a mode switch.
- `usd-optimize-run-operations/references/units-and-tolerances.md` — how a stated
  millimetre tolerance converts to stage units.
