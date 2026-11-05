<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Skia And Yoga For XR UI

Use public packages pinned in an isolated Node package:

```json
{
  "private": true,
  "type": "module",
  "dependencies": {
    "@napi-rs/canvas": "1.0.8",
    "yoga-layout": "3.2.1"
  }
}
```

Commit `package.json` and `package-lock.json` beside the renderer. Verify current
public availability before adopting these example versions. Install with
`npm ci`. Use `npm ci --offline` only after the exact locked artifacts exist in
the npm cache. Do not introduce internal NVIDIA APIs or unpinned framework
packages for this path.

Do not silently replace Skia with regex-parsed SVG or a low-resolution
`fillRect` renderer. Supervise one persistent child process, bound response
size and stderr, enforce a render timeout, and terminate then kill on timeout.
