# Reconstructable pinch and fist recognizers

This is a configurable application baseline, not an OVXR gesture API or a
universal definition of a fist. It reproduces the intended index-to-thumb-base
grab gesture while tolerating other fingers being inferred as partly open.
Read [hand validity](hands.md) first; use raw measured geometry for recognition
and [filtered poses](gesture-filtering.md) only for aiming/movement.

## Features from one coherent hand

Require finite, position-VALID joints for each feature. Do not replace missing
joints with zero or borrow the other hand. Let S be the polyline length:

```text
wrist → middle_metacarpal → middle_proximal → middle_intermediate
      → middle_distal → middle_tip
```

Reject S below an explicit physical epsilon. S and all distances must be in the
same space/units; ratios are invariant to positive uniform stage/user scale.

```text
pinchRatio = distance(thumb_tip, index_tip) / S
grabRatio = min(distance(index_tip, thumb_metacarpal),
                distance(index_tip, thumb_proximal)) / S
```

Use the available valid thumb-base positions; neither valid means grab unknown.
Optional diagnostic finger curl is chord/path from metacarpal to tip along each
finger's measured chain. A low ratio means curled. Requiring all four curl
ratios made inferred-finger noise reject otherwise natural grabs; do not add
that gate silently to the index-to-base baseline.

## Hysteresis and tuning

Example initial thresholds, to tune against real hands and the requested
ergonomics rather than treat as SDK constants:

| Feature | Begin at or below | Keep active at or below |
| --- | --- | --- |
| Left teleport/selection pinch ratio | 0.06 | 0.09 |
| Right ray/cancel pinch ratio | 0.08 | 0.12 |
| Index-to-thumb-base grab ratio | 0.22 | 0.32 |

For an inactive recognizer use begin; for an active one use the more permissive
hold threshold. Only valid accepted observations advance the latch. Unknown
is not open and must not generate release or satisfy rearm. Track validity,
active latch, suppressed-until-open and observation time independently per side.

An optional right-pinch contact cap uses measured thumb/index tip radii R:
`thresholdDistance = min(S*ratio, R*gain)` when R/S is plausible (for example
0.01–0.20); otherwise use S*ratio. Example begin/hold gains are 1.0/1.35.
Do not use max: it would broaden a deliberately strict pinch. The left baseline
uses the ratio alone so large or noisy radius estimates cannot cause teleport.

## Resolve overlapping recognition before emitting tool edges

A closed hand can put thumb/index tips close together. Compute candidate
features first; then let an eligible existing capture own its cycle. For a
fresh, selected-target manipulation, give the grab candidate priority over
pinch and suppress that side's pinch until it opens. Do not emit both selection
and movement from one observation. Right-pinch cancellation of an existing
teleport remains a separate high-priority semantic rule, not a new selection.
Use [gesture arbitration](gesture-arbitration.md) for the full state transitions.

Tracking grace may retain an existing visual/owner briefly, but cannot start a
new action or fabricate a release. On expiry cancel exactly once. A returning
hand already closed must open before beginning another capture. Physical
controller ownership cancels/rearms hand tools rather than leaving hidden holds.

## Deterministic traces

Generate valid joint chains and vary only the intended features: open→near
threshold→pinch→small jitter→open; curled index touching thumb base while other
fingers remain extended; tip pinch with index away from thumb base; missing
thumb base; inferred-but-VALID positions; degenerate S; scale changes; left/right
independent loss; cancellation followed by one-hand-open then both-open.
Assert one begin/end per valid cycle, no release from missing data, no pinch
leak from a grab and no restart until fresh rearm. Add physical-headset tuning
separately; synthetic ratios do not prove anatomy or runtime joint quality.
