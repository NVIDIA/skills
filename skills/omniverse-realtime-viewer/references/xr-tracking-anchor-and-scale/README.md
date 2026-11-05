# Viewer spawn, navigation and user scale

Read [library tracking-anchor-and-scale](../dependencies/ovxr.md) for API, units, basis conversion, fallback and latching. Do not copy numeric enums or ctypes layouts into the viewer.

Choose spawn from desktop camera, scene origin or explicit prim. Save spawn separately from navigation. Application Reset restores spawn; SDK clear resets its managed anchor to identity.

[Navigation](../xr-spatial-input/navigation.md) owns head-pivot movement/yaw/scale and teleport. Define user-scale control direction, preserve source content and convert input once. Keep pending versus applied revisions and reference-space type; LOCAL does not imply physical floor.

Acceptance includes non-origin/parented spawn, Y/Z-up, nondefault units, scale above/below one and reconnect. Require unchanged source content, aligned hands/controller rays and new client pixels. Mathematical fixtures alone do not prove navigation.
