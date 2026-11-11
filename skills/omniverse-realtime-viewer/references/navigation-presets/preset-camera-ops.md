<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Preset Camera Operations

These operations extend the shared camera state and matrix/commit rules in
`camera-controls`. Use radians, render-pixel coordinates after letterbox
mapping, `camera.world_up`, `MAX_ELEVATION`, and `MIN_DISTANCE` from that
reference. Run `sanitize_camera` before input and matrix generation.

The snippets use application helpers, not SDK APIs: `eye(cam)` returns the
current world-space eye, `eye_offset(cam)` is `(eye - target) / distance`, and
`camera_basis(cam)` returns `(right, up, forward)` using the row-major camera
basis in `camera-controls`. `orbit` updates the shared angles and clamps
elevation. Commit the resulting pose through the runtime camera adapter.

## Cursor-locked pan and zoom

On a plane through the target perpendicular to the view, one pixel maps to:

```python
def units_per_pixel(cam, fov_y_rad, render_h):
    return 2.0 * cam.distance * math.tan(fov_y_rad * 0.5) / max(1, render_h)

def pan_locked(cam, dx, dy, fov_y_rad, render_h):
    upp = units_per_pixel(cam, fov_y_rad, render_h)
    right, up = camera_basis(cam)[:2]
    cam.target = cam.target - right * (dx * upp) + up * (dy * upp)
```

Do not multiply the horizontal factor by aspect ratio; square pixels use the
same world-units-per-pixel scale in both directions.

For zoom toward a cursor, compute the target-plane point `p` under the cursor,
then scale both the target offset and camera distance by the same factor:

```python
def zoom_at_cursor(cam, notches, px, py, fov_y_rad, render_w, render_h,
                   per_notch=0.90):
    n = max(-6.0, min(6.0, float(notches)))
    new_distance = max(MIN_DISTANCE, cam.distance * (per_notch ** n))
    f = new_distance / cam.distance
    p = point_at_target_plane(cam, px, py, fov_y_rad, render_w, render_h)
    cam.target = p + (cam.target - p) * f
    cam.distance = new_distance
```

`point_at_target_plane` intersects the cursor ray with the plane through the
target. The shared camera basis supplies `right`, `up`, and `forward`; screen
Y must be inverted when creating the ray. The construction preserves the
cursor point because the new eye is `p + f * (old_eye - p)`.

For a perspective camera, implement the target-plane helper as follows:

```python
def point_at_target_plane(cam, px, py, fov_y_rad, render_w, render_h):
    right, up, forward = camera_basis(cam)
    half_h = cam.distance * math.tan(fov_y_rad * 0.5)
    x = (2.0 * px / render_w - 1.0) * half_h * render_w / render_h
    y = (1.0 - 2.0 * py / render_h) * half_h
    return cam.target + right * x + up * y
```

Require positive render dimensions. These formulas assume perspective
projection; orthographic navigation uses aperture-based scaling instead.

## Eye-fixed look and focus

For first-person look, capture the eye before changing azimuth/elevation, then
re-pivot the target to that same eye:

```python
def look(cam, dx, dy, sens_rad_per_px=0.0045):
    e = eye(cam)
    orbit(cam, dx, dy, sens_rad_per_px)
    cam.target = e - cam.distance * eye_offset(cam)
```

Double-click focus can set a new target to a picked point while retaining the
eye. Prefer native pick depth when the delivery path provides it; otherwise
use the target-plane approximation. Do not intersect a sphere centered on the
target—the eye is already on that sphere and the near root is zero.

## Axis views and fly

Blender-style views set azimuth/elevation directly, using the existing
elevation clamp: front `(0, 0)`, back `(pi, 0)`, right `(pi/2, 0)`, left
`(-pi/2, 0)`, and top/bottom `(0, +/-MAX_ELEVATION)`. These angle labels
assume the controller's matching front-axis convention; map named views to
the active stage up-axis and application front direction. Step-orbit keys use `pi/12`.
If axis snapping is added, latch it once per gesture after a clear movement
threshold and release the latch with the gesture.

Fly movement is frame-rate independent and gravity-aligned:

```python
def fly_move(cam, held, speed, dt):
    right, up, fwd = camera_basis(cam)
    w_up = cam.world_up
    fwd_h = fwd - np.dot(fwd, w_up) * w_up
    right_h = right - np.dot(right, w_up) * w_up
    if np.linalg.norm(fwd_h) > 1e-3: fwd_h /= np.linalg.norm(fwd_h)
    if np.linalg.norm(right_h) > 1e-3: right_h /= np.linalg.norm(right_h)
    move = (fwd_h * (('w' in held) - ('s' in held))
            + right_h * (('d' in held) - ('a' in held))
            + w_up * (('e' in held) - ('q' in held)))
    length = np.linalg.norm(move)
    if length < 1e-6: return False
    cam.target += move / length * (speed * dt)
    return True

def fly_speed(state, notches):
    n = max(-6.0, min(6.0, float(notches)))
    state.speed = min(1e6, max(1e-3, state.speed * (1.25 ** n)))
```

Initialize speed relative to the fitted scene, for example `0.1 * distance`,
and apply any boost as a documented multiplier. Clear the held-key set on
focus loss, disconnect, viewport deactivation, and preset switch.

Tests should assert cursor fixed points and eye preservation to numerical
tolerance, exact `speed * dt` displacement for a single axis, and zero
world-up component for planar fly motion.
