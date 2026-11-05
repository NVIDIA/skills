# Independent aiming, twist and filtered manipulation

This is application math, not an OVXR gesture API. Consume one validated joint
snapshot in a declared common space. Raw geometry owns pinch/fist recognition;
filtered tool poses own presentation/manipulation. Filtering recognizer inputs
can delay release and accidentally change pinch thresholds.

## Teleport swing/twist decomposition

OpenXR hand joints do not include an elbow. A practical axial proxy is the
wrist-to-knuckle midpoint; label it as a proxy rather than measured forearm
orientation. For valid wrist W, index metacarpal I, little metacarpal L:

```text
a = normalize((I + L)/2 - W)
s = normalize((I - L) - a * dot(I - L, a))
```

At pinch begin retain a0, s0, the measured initial aim basis, and initial
origin offset o0 = aimOrigin0 - W0. Build initial aim from measured anatomy,
not a guessed joint quaternion axis. One usable left-hand policy takes heading
from wrist-to-thumb-tip projected onto the stage floor and elevation from the
opposite palm normal; validate sign against observed hands and stage up.

Transport the initial basis and origin offset with the shortest swing from a0
to current a, ignoring twist. For any vector v, c=dot(a0,a), k=cross(a0,a):

```text
swing(v) = v + cross(k,v) + cross(k,cross(k,v))/(1+c)
aimDirection = normalize(swing(initialAimDirection))
aimOrigin = currentWrist + swing(o0)
twist = atan2(dot(a,cross(swing(s0),s)), dot(swing(s0),s))
destinationYaw = gain * clamp(twist, -maxTwist, maxTwist)
```

Transport all aim-basis rows consistently, preserving a proper rotation. Pure
axial twist changes facing without moving the thumb-origin offset or steering
the target. Swing and wrist translation remain live throughout the hold; do not
freeze the initial axis. Near c=-1 there is no unique shortest swing: hold only
within declared tracking grace, then cancel instead of dividing by near zero.

For a small-twist mapping, maxTwist=30 degrees and gain=12 gives a full turn from
+30 degrees **relative to pinch start**. Starting at -30 and ending at zero is
a full turn, not half a turn. This gain is product policy, not a universal
ergonomic default. Render the arrow from the yaw used at commit; do not smooth
a wrapped angle across the ±pi discontinuity.

## Filter independent signals

Filter wrist position in physical metres and axial direction as a normalized
vector with separate timestamped One Euro streams. Transport aim from that
filtered axis. Compute facing residual from coherent measured bases, then use
the tool's scalar/angular filter if needed. Independently filtering joints with
unequal lag can break their rigid-palm relationship and mix twist into aiming.

For each vector stream retain previous raw x, filtered x and signed filtered
velocity v. Given valid interval dt:

```text
alpha(f,dt) = 1 / (1 + 1/(2*pi*f*dt))
rawVelocity = (x - previousRawX)/dt
v = lerp(previousV, rawVelocity, alpha(derivativeCutoff,dt))
cutoff = minCutoff + beta * length(v)
filteredX = lerp(previousFilteredX, x, alpha(cutoff,dt))
```

Filter signed velocity **before** taking its length; otherwise alternating
jitter becomes false sustained speed and opens the cutoff. Normalize filtered
directions. For orientations use shortest-arc quaternion interpolation,
hemisphere correction and a signed rotation-vector derivative, not Euler angles.

Example fast teleport prefilters use minCutoff=6 Hz with position beta=8 and
direction beta=2, derivativeCutoff=1 Hz. These assume metres/unit directions,
not arbitrary stage units. Tune with measured total lag when downstream tool
filtering is also enabled. Duplicate/reversed times must not advance the filter.
On long gaps (for example 100 ms), source/anchor discontinuities, capture end,
or context change, reset to the next valid sample.

## Accelerated, stable panel movement

Selection is not capture. Select the permanent grab bar; a fresh fist captures
its surface and owning hand. Preserve the current panel pose on that first
sample, record hand H0 and panel P0 in LOCAL metres, and latch ownership.
Keep orientation unchanged unless rotation is explicitly part of the tool.

For delta=H-H0, choose precision distance d0, full-gain distance d1>d0 and maximum
gain G>=1. Let t=clamp((length(delta)-d0)/(d1-d0),0,1),
g=1+(G-1)*t*t*(3-2*t), target=P0+g*delta. This increases reach smoothly and
reversibly without accumulating frame-rate-dependent offsets. Filter the
amplified target in LOCAL metres, initializing at P0. Returning to H0 returns
the target to P0. Do not filter pinch/fist state or repaint text for pose changes.

Reset on release, tracking expiry, selected target replacement or lifecycle
change. Another already-closed fist cannot steal capture or start immediately
on the owner's release: it must open and close again. Scene selection clears
panel selection and drag. Preserve the hand-to-panel offset instead of snapping
the panel origin onto the fist.

## Deterministic tests before headset tuning

Generate rigid joint traces, not only yaw values. Assert pure twist preserves
aim direction/origin; pure swing changes aim without spurious yaw; combined
swing/twist remains independent; and a second swing still responds. Test
non-neutral capture poses, centimetre stages, user scale, rotated anchors, Y/Z
up, degenerate/antipodal bases, timestamp gaps, jitter/step/ramp response, grace
expiry, cancel/both-open rearm, and no stale-target release commit. For panels
verify no initial jump, reversible gain, owner latching, selection replacement,
no pose-only raster uploads, and reset after loss.
