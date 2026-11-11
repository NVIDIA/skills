<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Language Trial Routing

Use this path when application code owns its tool or model calls. Prefer a
maintained framework integration when it already owns that boundary.

Read the current guide for the established language:

- [Python](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/python)
- [Node.js](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/nodejs)
- [Rust](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/rust)

Resolve the guide compatible with the installed stable package. Derive runtime
requirements, dependencies, activation APIs, sample program, run location, and
verification from it. Do not copy those recipes or version requirements here.

Preserve the existing manifest and package manager; create a new project only
for a standalone trial. Use the guide's referenced sample files from the same
revision. When adapting an existing application, explain the smallest change
without replacing unrelated business logic.

Follow the guide's distinction between local callbacks and a live provider,
including its cleanup and event-delivery requirements. Do not demand inference
credentials for a local example or claim that local callbacks validate real
provider traffic.

Use [Guide-Based First-Value Checks](first-value.md). After the proof, additional
application coverage belongs to `nemo-relay-instrument-calls`.
