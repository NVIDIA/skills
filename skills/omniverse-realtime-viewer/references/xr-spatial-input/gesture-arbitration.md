# Gesture routing, capture and rearm

Use this optional editor mapping when the product calls for left-hand teleport,
right-hand selection ray and fist manipulation. It avoids a palm-up menu that
can conflict with the headset shell. Recognition comes from
[hand recognizers](hand-recognizers.md); aiming/filtering comes from
[gesture math](gesture-filtering.md). Other mappings must declare their own
priorities rather than silently combining these gestures.

## Separate state

Keep per-side observed pinch/fist/validity and suppression latches, plus:

- Selection: none, scene prim, or panel handle (one authoritative target).
- Pointer capture: source, hand, stable node, activation policy and generation.
- Manipulation capture: source, hand, target, begin poses and filter state.
- Teleport: armed, preview, grace, release-pending, or wait-both-open; gesture ID,
  aim/twist baseline and correlated query/preview state.

Do not encode all of these as one `selected` boolean. Evaluate one immutable
snapshot, choose an owner, then emit semantic edges once. Process both hands even
when only one owns the tool. Unknown/absent input is not a release or open hand.

## Priority on each accepted update

1. Focus/session/stage/source invalidation cancels affected captures once.
   Physical-controller takeover cancels and suppresses hand tools until rearm.
2. An active teleport plus a fresh right pinch cancels teleport, consumes that
   pinch and any competing left selection edge, clears its query/target and
   enters wait-both-open. It must not also activate a panel or select a prim.
3. An uncancelled teleport in preview, grace or release-pending retains exclusive
   tool ownership. Process only its update/release/recovery and suppress new
   pointer/manipulation captures until it finishes. A fresh right fist while
   teleporting cannot start moving a previously selected panel.
4. Existing pointer or manipulation capture owns its full cycle, including a
   release outside the hit surface. Another held fist cannot steal it.
5. A selected panel handle plus an eligible fresh fist captures panel movement
   before scene manipulation; preserve initial panel/hand offset. A selected
   scene target may use the configured right-fist movement policy.
6. Outside those captures, right pinch enables the selection ray. A fresh left
   pinch while that ray is active selects/activates its current hit: visible
   UI first, then scene. Consume the left cycle so it cannot start teleport.
7. A fresh left pinch with no selection ray/capture begins teleport. Hold owns
   it until valid release or cancellation; brief aim miss does not transfer it
   to another tool. Right-hand absence alone does not cancel this left hold.

For a simultaneous pair of fresh pinches, evaluate right-ray eligibility before
the left edge, so the pair selects instead of starting then cancelling teleport.
Define this tie rule in tests. A left pinch selects a bar through the active
selection ray; it is not an undocumented second standalone left-hand ray.
Controller selection follows its profile mapping through the same UI/scene
router rather than adopting the hand-button names.

## Teleport transitions

| State/event | Effect |
| --- | --- |
| Armed + eligible left begin | Capture gesture/bases; clear old target; preview |
| Preview + valid owner sample | Update aim/facing; issue or poll bounded current query |
| Preview + transient owner joint loss | Grace; retain visual/owner, invalidate commit permission |
| Grace + valid recovery before deadline | Resume with new query; do not reuse stale commit permission |
| Grace expiry or lifecycle loss | Cancel once; suppress until valid rearm |
| Preview + observed left release | Release-pending; use [landing acceptance](teleport-acceptance.md) |
| Preview/grace/release-pending + cancel | Invalidate pending completions; wait-both-open |
| Wait-both-open + both valid and open in a current snapshot | Arm, without starting; require a subsequent fresh left edge |

For this baseline, open means neither pinch nor grab latch active with required
feature joints valid. Configure tracking grace in elapsed time plus a maximum
sample count, and apply it to both active-invalid and inactive-hand loss if that
is the chosen product policy. Focus/session loss is immediate, not tracking grace.

## Selection and capture closure

Selecting prim B replaces A in one action and clears panel selection/drag.
Selecting a bar replaces scene selection and stores that stable surface target.
Pointer activation follows node policy: rows act on press once; buttons on
release over the captured node; bars select without implicitly moving.

After a manipulation ends, an already-closed non-owner must open before it can
start. Reacquisition of a missing closed hand cannot synthesize a fresh fist.
Clear filters and begin offsets when capture identity changes. Test both hand
orders, same-update pinch ties, scene/UI selection replacement, cancel during
pending query, one-hand loss and open→fresh-left-pinch rearm.
