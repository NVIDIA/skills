<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Buffer sharing and synchronization contract

The verified sample routes exchange regular filesystem artifacts. A producer
finishes before its consumer starts; the consumer reopens the canonical path
and reauthenticates its size and SHA-256. The producer owns the artifact until
successful process completion, and cleanup waits until the consumer finishes.

For every plan, result, and producer-to-consumer boundary, emit this object
directly in the plan or evidence. Do not import or generate it with a runtime
helper:

```json
{
  "schema_version": "1.0",
  "transport": "filesystem-artifact",
  "memory_domain": "host-file",
  "copy_required": true,
  "zero_copy_verified": false,
  "allocation_owner": "producer",
  "consumer_access": "reopen-after-producer-completion",
  "synchronization": {
    "mode": "process-completion-and-artifact-reauthentication"
  },
  "verified_sharing_handles": [],
  "verified_sync_primitives": [],
  "not_established": [
    "nvsci-buf", "nvsci-sync", "dma-buf", "nvbufsurface", "egl-image",
    "cuda-external-memory", "cuda-ipc-memory", "cuda-event",
    "cuda-external-semaphore"
  ]
}
```

`not_established` is an evidence verdict, not an unsupported-product verdict.
CPU-buffer or device-memory sample modes, API availability, header presence,
and successful encode/decode do not establish external interoperability. A
request for a listed non-file primitive remains unresolved unless an
authenticated operation proves its handle, format, ownership, lifetime,
signal, wait, and reuse rules. Never silently fall back or claim zero copy.

The traditional codec APIs expose CUDA ordering at specific boundaries:

- `CUVIDPROCPARAMS::output_stream` orders mapping and decoder-output
  post-processing performed by `cuvidMapVideoFrame()`. It is not an input
  dependency.
- `NvEncSetIOCudaStreams()` binds CUDA streams used to order encoder input and
  output processing. This alone does not prove output completion, external
  sharing, or zero copy.

`cuvidParseVideoData()` supplies CPU-addressable compressed bytes and
`cuvidDecodePicture()` receives `CUVIDPICPARAMS`; neither accepts a CUDA stream
or event to wait upon before input consumption. Preserve the documented input
and metadata lifetimes. Report these APIs as capabilities until the exact
in-process operation is authenticated; do not alter the filesystem contract.
