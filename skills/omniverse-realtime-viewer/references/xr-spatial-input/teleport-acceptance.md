# Teleport trajectory, landing and facing commit

This is a complete conservative application policy to implement, not an OVXR
teleport API. Use [navigation](navigation.md) for managed-anchor conversion,
[gesture math](gesture-filtering.md) for independent aim/twist, and
[arbitration](gesture-arbitration.md) for cancellation and rearm.

## Bounded trajectory and scene query

Keep one persistent, stage-generation-scoped query structure. With stage-world
origin o, normalized aim d, stage up u, and effective stage units per physical
metre k, sample a ballistic trajectory:

```text
p(t) = o + k * (speed_m_s*t*d - 0.5*gravity_m_s2*t*t*u)
```

Choose bounded duration/segment count and maximum physical reach; expose those
as product tuning. Example seed values are speed 6 m/s, gravity 9.81 m/s² and
duration 2 s, not headset constants. Cast each segment only over its own length
and select the first collision in trajectory order, not the nearest point to
the hand among all results. A fixed segment budget trades precision for cost;
test small ledges at the longest supported range before setting it.

For the current managed-anchor path, convert the measured ray once into the
BVH's declared stage-world space. Do not apply another inverse navigation
transform. Convert physical sizes through k once, including marker radius,
slope probes and clearance. Pending navigation is not the active query anchor.

## Landing predicate

Require finite hit/normal, allowed geometry, declared normal orientation and
`dot(normalizedNormal,u) >= cos(maxSlope)`. Do not flip every hit normal upward:
that could turn a ceiling underside into a landing. In two-sided or winding-
ambiguous geometry, resolve sidedness by the query contract or reject it.

Separate surface walkability from body clearance. For clearance-aware teleport,
query a configurable physical capsule/volume above the destination with the
installed query capability. If only rays are available, a finite ring of probes
is an explicitly approximate policy, not proof of capsule clearance; label or
disable destinations accordingly. No hit must not fall back to an invented
infinite floor unless the user selected that behavior. Marker lift is visual
only and must not raise the user's landed floor point.

## Freshness, forgiving preview and release

Tag each query with `(stageGeneration, inputEpoch, anchorRevision, gestureId,
sampleSequence, sampleTime, querySequence)`. Preserve submission/sample time
when an async result arrives. Distinguish not-requested, pending, miss, error
and hit. None of the first four authorizes commit from an older cached hit.

While held, a finite tracking/query grace can keep the last arc visible with
an invalid/stale style, without keeping commit permission. Smooth measured aim,
not the validity outcome; wrist motion must still steer throughout the hold.
On tracking recovery require a new eligible result before restoring the marker.

On observed release enter release-pending with a fixed deadline. Retain an
already-issued matching query or request one bounded final query. Accept only
the current gesture/anchor/stage with valid release tracking, age below the
configured limit and a walkable target. For this conservative baseline, it must
match the last submitted valid preview's prim and position within a named
physical tolerance; a changed destination cancels instead of teleporting to an
unseen location. Record the preview's draw submission, not a claim of physical
pixel observation. A product may deliberately present a new final preview
before committing, but must implement and test that extra state explicitly.

After success or cancellation consume the release once, clear target/query
state and reject late results. Right-pinch cancellation always wins over a
result delivered in that same update. Do not block the render owner waiting
for a scene query or make release depend on whether an arc is rendered in USD
versus the compositor.

## Facing and anchor transaction

One intuitive policy captures head-floor heading H0 at teleport begin and sets
desired heading `H0 + relativeWristYaw`. Keep arrow and accepted yaw paired with
the same preview. Define clockwise sign against the physical client; do not
assume a local joint quaternion's roll convention.

At commit, find current world head h, current head-floor heading H, measured
head-to-floor height L in stage units, current anchor A and accepted hit p.
Require a located floor/reference and valid projected heading; LOCAL origin or
a near-vertical head forward vector is not sufficient evidence. Then, for row
vectors, with feet q=h-u*L and delta=desiredHeading-H:

```text
A_next = A * T(-q) * R_about_stage_up(delta) * T(p)
```

This rotates around the current floor point and relocates it to p, preserving
uniform scale and physical head height. Validate the full candidate before one
managed-anchor setter. Update accepted navigation only after success and mark
it applied only after the matching frame latch. Clear the gesture on failure;
do not apply yaw and translation as separately successful teleports.

Test first collision, slope/ceiling rejection, clearance limits, rotated/scaled
anchors, non-neutral wrist start, delayed release queries, miss after a hit,
cancel/result races, and head motion between initial preview and commit.
