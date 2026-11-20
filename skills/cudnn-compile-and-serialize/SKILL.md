---
name: cudnn-compile-and-serialize
description: Explain or implement cuDNN Frontend C++ parallel plan compilation, graph serialization, cached plan loading, and deviceless ahead-of-time compilation. Adapt the C++ samples into application code with explicit plan selection, tensor bindings, and error handling.
---

# Parallel compilation and serialization

Use this skill to turn a graph build or startup-cost requirement into C++ code.
For explanation requests, show the relevant phases and APIs. For implementation
requests, edit the target code and verify it; do not stop with an outline.

## Read the matching source

Use the user's cuDNN Frontend checkout and selected revision. Source paths are
relative to that checkout, not to this installed skill. If no checkout is
available, read the files from
[NVIDIA/cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend).
The links below use `develop` for navigation; replace it with the selected tag
or commit before reading. Resolve a branch or tag to one commit and use that
commit for all source reads. Do not assume the installed skill is inside the
source repo. If the source cannot be read, report that limit before giving
revision-specific code or compatibility claims.

Read these files at that revision before coding:

- [Parallel compilation](https://github.com/NVIDIA/cudnn-frontend/blob/develop/samples/cpp/misc/parallel_compilation.cpp)
- [Serialization](https://github.com/NVIDIA/cudnn-frontend/blob/develop/samples/cpp/misc/serialization.cpp)
- [Deviceless compilation](https://github.com/NVIDIA/cudnn-frontend/blob/develop/samples/cpp/misc/deviceless_aot_compilation.cpp)
- [Deviceless workflow](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/utilities/deviceless-ahead-of-time-compilation.md)
- [Graph APIs](https://github.com/NVIDIA/cudnn-frontend/blob/develop/include/cudnn_frontend/graph_interface.h) and
  [plan implementation](https://github.com/NVIDIA/cudnn-frontend/blob/develop/include/cudnn_frontend/plans.h)

Check the target repo's contributor instructions before editing code. Use
current signatures and version guards; samples illustrate a workflow, not a
production error policy.

Establish the graph operations, dimensions, strides, datatypes, tensor UIDs,
target GPU, compiled/runtime cuDNN versions, and workspace limit from the code.
Determine whether the user needs faster startup, a plan cache, CPU-side AOT,
or several of these. Ask only for missing requirements that affect the code.

## Explain the phases

| Phase | Result | Device requirement |
| --- | --- | --- |
| Define and validate a graph | Operation/tensor contract | Host-side definition |
| Query heuristics and check support | Candidate execution plans | Handle or supported device-properties path |
| Build plans | Executable candidate(s); may compile kernels | Handle or supported deviceless path |
| Select a plan and serialize | Bytes for the selected built plan | Supported serialization path |
| Deserialize | A restored executable plan | Handle, or device properties for the handle-less overload |
| Bind tensors and execute | GPU outputs | Compatible GPU, handle, buffers, workspace |

Parallel compilation reduces host-side build time. It does not run the graph
in parallel on the GPU. Serialization stores plan state and tensor metadata;
it does not store the application's input/output buffers or their addresses.
Deviceless AOT still needs a GPU to capture device properties and to execute.

## Adapt the graph preparation

Keep the application's graph construction and tensor contract. The handle-based
sequence is:

```text
validate()
build_operation_graph(handle)
create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})
apply requested plan filters
check_support()
build_plans(...) or build_plan_at_index(index)
```

Preserve required numerical, behavior-note, and workspace filters. The parallel
sample selects `BehaviorNote_t::RUNTIME_COMPILATION` to focus its benchmark;
do not impose that filter on unrelated applications.

`BuildPlanPolicy_t::HEURISTICS_CHOICE` builds a usable heuristic choice.
`BuildPlanPolicy_t::ALL` attempts all candidates. Neither means measured
autotuning. If latency ranking is required, benchmark successful plans with
valid buffers and GPU timing, then select the winner explicitly.

Replace sample `REQUIRE` calls with the application's status/exception policy.
Preserve each `error_t` and its message. A failed candidate may be skipped if
another valid candidate remains; a failure of every candidate must be reported.

## Compile in parallel without sharing mutable graph state

The current sample launches threads calling `build_plan_at_index(i)` on one
graph. Inspect the implementation before adopting this pattern: each build can
write the same candidate field in the graph's plan state. Do not treat the
sample as a thread-safety guarantee.

The `do_multithreaded_builds` argument also exists in the API, but the current
plan implementation rejects `true`. Do not enable it merely because it compiles.
Recheck both facts when working on a newer revision.

For application code:

1. Use bounded concurrency and an independently constructed `Graph` for each
   worker/job. Rebuild its descriptors; do not assume a copied graph has
   independent backend state. Parallelize separate graph builds, or prepare
   separate graph instances when comparing plans for the same operation.
2. Give handle-based workers their own RAII handles and correct CUDA device
   context. For deviceless jobs, share only a fully built, read-only
   `DeviceProperties` descriptor; each worker still owns its graph and results.
3. Query and filter candidates before building them. Indices belong to one
   graph's candidate list. Do not equate indices across separately queried
   graphs; match engine/configuration identity when assigning a specific plan.
4. Return a status and plan/artifact identity from each worker. Join all workers
   and collect exceptions on every exit path. Use `std::thread` or an existing
   executor; do not add a scheduler dependency for this task.
5. Select the successful result after joining. Do not serialize an arbitrary
   last-finishing plan. Keep its graph, UIDs, workspace size, and bytes together.

If only one graph instance is available, serialize its mutations; do not claim
parallel compilation speedup from a lock around every build. Distinct worker
graphs avoid that shared mutation. Do not concurrently execute or serialize a
graph while it is being built. Keep execution handles and workspace separate
when GPU execution itself is concurrent.

## Serialize and restore an executable plan

Assign stable, explicit UIDs to nonvirtual tensors before serialization when
the consumer is another process. Keep a binding manifest with UID, dimensions,
strides, datatype, and pass-by-value requirements. Do not retain tensor-object
addresses as a cross-process interface.

The following is an integration fragment. `graph` is a prepared, supported
`fe::graph::Graph`, `handle` is valid, and `selected_index` identifies a
successful candidate. Return each failed status to the caller:

```cpp
auto status = graph.build_plan_at_index(selected_index);
if (status.is_bad()) return status;

std::vector<uint8_t> blob;
status = graph.serialize(blob);
if (status.is_bad()) return status;

fe::graph::Graph restored;
status = restored.deserialize(handle, blob);
if (status.is_bad()) return status;

int64_t workspace_bytes = 0;
status = restored.get_workspace_size(workspace_bytes);
if (status.is_bad()) return status;
```

Calling `build_plan_at_index(selected_index)` after selection sets the active
candidate in the current implementation. `serialize(blob)` stores that selected
built plan, not the whole candidate set. The restored plan is at index zero;
do not carry its source index into the restored graph. The serialization sample
also checks that the selected plan's behavior notes survive this round trip.

Allocate fresh device buffers and the queried workspace. Rebuild a
`std::unordered_map<int64_t, void*>` from the saved UID contract, then call
`restored.execute(handle, variant_pack, workspace_ptr)`. Preserve host scalar
lifetime for pass-by-value tensors. Derive output sizes from the graph's output
shape; the convolution sample's `n*k*h*w` allocation is specific to its padding.
Keep buffers and workspace alive until GPU work completes.

Use `std::vector<uint8_t>` for the plan blob. It is a binary UBJSON payload in
the current frontend. A `std::string` or `std::vector<char>` can select a
structural JSON overload instead. Structural graph serialization alone does
not supply a built executable plan. Check `serialize_structure`,
`enforce_precompiled`, and `run_warmup` in the selected headers before using
those options; do not promise zero compilation or zero device work on reload.
Serialization is unavailable with `CUDNN_FRONTEND_SKIP_JSON_LIB` enabled.

## Separate deviceless AOT into three stages

1. **Capture on the target GPU.** Construct `fe::DeviceProperties`, call
   `set_device_id(device_id).build()`, then `serialize(device_bytes)`. This
   stage requires a compatible device; a CPU worker cannot capture it.
2. **Build on a CPU node.** Deserialize those bytes into a new
   `DeviceProperties` object. Set it on the graph with `set_device_properties`,
   define the graph, then call
   `graph.build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})` without a handle.
   Serialize the selected plan. The worker still needs the supported cuDNN
   libraries and compilation dependencies; deviceless does not mean dependency-free.
3. **Load and execute on a compatible GPU.** Use
   `restored.deserialize(handle, blob)`, bind fresh pointers by UID, query
   workspace, and execute. Validate the target against the captured properties.

For handle-less plan loading, first call
`restored.set_device_properties(device_properties)`, then
`restored.deserialize(blob)`. A cuDNN handle is still required by `execute`.
Parallel handle-less deserialization can share read-only device properties,
with a separate restored graph per worker.

The deviceless sample gates device-property serialization at cuDNN 9.11.
The handle-less deserialization API has a 9.8 floor in current headers, while
runtime coverage follows the sample's 9.11 policy. Check both compiled headers
and loaded backend; do not describe the full sample as verified on 9.8–9.10.

## Add cache behavior only when requested

Store bytes in binary mode and check file I/O. For persistent caches, write a
temporary artifact and publish it atomically after success. Include the graph
contract, frontend/backend versions, target device properties, CUDA/compiler
environment, and plan filters in cache compatibility metadata. Do not claim
portability across arbitrary GPU architectures or software versions.

Handle absent, malformed, truncated, or incompatible artifacts explicitly.
Rebuild only if the caller permits it; otherwise return an actionable error.
Keep cache loading distinct from graph compilation so a hit can be measured
without silently taking the build path. Avoid a cache service or format layer
when a caller-owned byte vector or file satisfies the request.

## Verify the converted code

- Compile against the selected headers and run the smallest supported graph.
  Check candidate failures, zero supported plans, status propagation, and worker
  completion. For parallel code, repeat with one and several workers and check
  that results remain attached to the correct job and plan.
- Compare original and restored execution on identical inputs with fresh
  output/workspace buffers and an independent numerical reference. Check the
  UID contract, workspace requirement, and selected behavior notes. Test cache
  misses and invalid blobs when implementing persistence.
- For AOT, run the middle stage with no GPU accessible; then load and execute
  on the intended GPU. Running all stages on a GPU host does not verify the
  deviceless deployment claim.
- Measure cold build, parallel build, serialization, reload, and steady-state
  execution separately. Synchronize GPU work for execution timing; record
  software/compiler cache state and worker count for compilation comparisons.
- Use `build/bin/samples --list-tests` and the exact sample names. The cases
  include `Deviceless compilation`, `CSBR Graph with serialization`, and
  `SDPA Graph with serialization`. `Parallel build` has an unconditional
  `SKIP`; its benchmark flags do not enable it. Use a local opt-in harness or
  narrowly enable it for the requested measurement; do not report a skip as a pass.

Run focused pre-commit checks on changed files and trim redundant comments in
the code you edit. When a conversion changes numerical APIs, kernels, datatypes,
or supported GPU targets, verify replay and batch invariance in both supported
directions and update the selected source revision's
[determinism support page](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/determinism.md).
Report commands, actual results, version/device limits, and unavailable coverage.
