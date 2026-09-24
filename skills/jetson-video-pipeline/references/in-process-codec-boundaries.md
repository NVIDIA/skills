<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# In-process decoder and encoder boundaries

Use this profile when a custom in-process graph connects the traditional CUVID
decoder path or a CUDA producer to NVENC. It defines codec-facing ports only.
An intermediate node owns its transform, allocation, copy, format conversion,
completion signal, and input-release decision. Authenticate all selected APIs
against installed target headers and samples.

This is not a single-allocation zero-copy claim. Decoder mapping can perform a
driver-managed conversion or copy, and an intermediate producer can write a
distinct encoder-input allocation.

## Required request and capacity

Record codec/framing, coded and output dimensions, format, planes, pitch,
alignment, allocation owner, GPU/context, bounded frame or EOS policy, timeout,
pool and queue depths, backpressure policy, and release conditions. Do not
silently default memory ownership or capacities. Derive decoder surface and
mapping capacity from maximum simultaneous ownership. If a callback maps
before it can acquire downstream capacity, include the mapping that can wait;
otherwise acquire capacity before mapping when the authenticated callback
design permits it. Validate the choice under pressure rather than relying on a
fixed formula.

## Decoder input and mapped output

Compressed input is CPU-addressable host memory consumed through
`cuvidParseVideoData()` and `cuvidDecodePicture()`. Every in-process plan must
state explicitly that `cuvidDecodePicture()` has no decoder-input CUDA-stream
or event wait parameter. Bytes and slice metadata must be valid for their
required lifetime. Reconcile submitted, decoded, displayed, and independently
decoded counts.

A mapped output uses a `CUdeviceptr` plus pitch in CUDA device memory. Supply
`CUVIDPROCPARAMS::output_stream` for mapping/post-processing order, and retain
the picture index, pointer, pitch, layout, timestamp, stream, and exactly-one
unmap obligation. Call `cuvidUnmapVideoFrame()` only after authenticated
downstream completion. Treat `CUDA_ERROR_MAP_FAILED` as capacity or lifetime
failure, not permission to unmap an in-use frame. A helper that copies a mapped
frame into another allocation and unmaps early does not establish this retained
mapped-surface boundary.

## CUDA producer to NVENC

Use a bounded, reusable CUDA-device-pointer input pool. Register allocations
persistently with `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR` when supported,
then map them according to the installed API lifecycle. Record a CUDA event
after the producer's final write and make the NVENC-bound input stream wait on
that event before encode submission.

Follow the installed `NvEncSetIOCudaStreams()` ABI exactly: pass pointers to
live `CUstream` handle variables as `NV_ENC_CUSTREAM_PTR`, not raw handle
values, and keep those variables alive for the session. Zero-initialize and
version every NVENC structure. Establish the stream binding only after
successful encoder initialization and before submitting pictures; authenticate
that lifecycle against the installed headers and samples. For CUDA-pointer
image registration, declare `NV_ENC_INPUT_IMAGE`, use the format returned by
`NvEncMapInputResource()`, and populate required picture fields including
`NV_ENC_PIC_STRUCT_FRAME`.

Do not call CUDA or codec APIs from a CUDA host callback unless their installed
contracts explicitly permit it. A callback may issue a bounded nonblocking
notification; defer other work to a permitted worker with the correct CUDA
context current. Propagate failures to the owning slot.

Conservatively retain an encoder input allocation through successful encoded
output completion and NVENC input unmap. Earlier reuse requires authenticated
input-consumed evidence. Unlock bitstream output only after its positive
payload is copied or written; unregister resources before freeing the pool.

## Per-slot proof

Associate compressed-input identity, decoder picture index and mapping,
downstream frame, CUDA event, NVENC registered/mapped input, output buffer,
timestamps, and state transitions. Require bounded backpressure, event-before-
wait ordering, no early reuse, exact-once cleanup on success and failure, EOS
drain, and independent decode of the exact fresh bitstream. Profiling must
separate application synchronization/copies from library and driver activity.
Missing evidence leaves the edge unresolved.
