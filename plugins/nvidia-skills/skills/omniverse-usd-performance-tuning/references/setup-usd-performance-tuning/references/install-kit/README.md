<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Install Kit (pointer)

Kit is not an optimization or validation runtime for this skill — it survives
only as the opt-in render-profiling adjunct (Kit -> omniperf). This package
does not maintain Kit install instructions; follow the external docs:

- Kit SDK / app installs: https://docs.omniverse.nvidia.com/kit/docs/
- Render profiling skills (FPS / VRAM / frame time): NVIDIA/omniperf

If a Kit profiling session is used, follow
`references/runtime-artifact-token-budget.md` before reading Kit logs or
Tracy output.

The optimize+validate path needs no Kit: see
`install-usd-optimize-standalone` and
`install-usd-validation-nvidia-standalone`.
