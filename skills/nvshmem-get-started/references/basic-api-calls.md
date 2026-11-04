# Basic NVSHMEM API Calls

Use this reference after the conceptual mental model is understood or when a user directly asks which C/C++ NVSHMEM primitives to use. Present only the API groups relevant to the user's goal, and treat signatures containing `TYPENAME`, `TYPE`, `SIZE`, or `OP` as naming patterns rather than literal function names.s

## Contents

- [Setup, Exit, and Query](#setup-exit-and-query)
- [Symmetric Memory](#symmetric-memory)
- [Remote Memory Access](#remote-memory-access)
- [Invocation Contexts](#invocation-contexts)
- [Blocking and Nonblocking RMA](#blocking-and-nonblocking-rma)
- [Signaling and Atomics](#signaling-and-atomics)
- [Ordering and Completion](#ordering-and-completion)
- [Point-to-Point Synchronization](#point-to-point-synchronization)
- [Teams and Collectives](#teams-and-collectives)
- [Collective Kernel Launch](#collective-kernel-launch)

## Setup, Exit, and Query

Start with the core and extension headers and the job-lifecycle calls:

```cpp
#include <nvshmem.h>
#include <nvshmemx.h>

void  nvshmem_init(void);
void  nvshmem_finalize(void);
int   nvshmem_my_pe(void);
int   nvshmem_n_pes(void);
void *nvshmem_ptr(const void *dest, int pe);
void  nvshmem_info_get_version(int *major, int *minor);
```

- Use `nvshmem_init` and `nvshmem_finalize` collectively to establish and release the PE job.
- Use `nvshmem_my_pe` and `nvshmem_n_pes` to query the world-team rank and size.
- Use `nvshmem_ptr` only as an optional direct-access optimization. It returns a directly usable pointer when the peer object is accessible and null otherwise; the result is not a symmetric address for use in other NVSHMEM calls.

For an application that already has an MPI communicator, initialize MPI first and use attribute-based initialization instead of `nvshmem_init`:

```cpp
nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
MPI_Comm comm = MPI_COMM_WORLD;
attr.mpi_comm = &comm;
nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
```

Teach both supported device-selection patterns:

- If a launcher-provided node-local rank is available before initialization, call `cudaSetDevice(local_rank)` before the initialization routine.
- Otherwise use two-stage initialization: call the initialization routine, query a node-local PE with `nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)`, call `cudaSetDevice`, and only then call an operation such as symmetric allocation that completes device initialization.

Do not call general NVSHMEM operations between two-stage initialization and device selection; only use the documented setup and PE/team query routines needed to select the device.

## Symmetric Memory

Introduce the collectively managed symmetric heap:

```cpp
void *nvshmem_malloc(size_t size);
void *nvshmem_calloc(size_t count, size_t size);
void *nvshmem_align(size_t alignment, size_t size);
void  nvshmem_free(void *ptr);
```

Have every PE call allocation and deallocation in the same order with identical allocation arguments. Use the returned local symmetric address together with a destination PE to name the corresponding remote object. Require the remote operand of RMA, AMO, signal, wait, and collective operations to be symmetric or provided through another explicitly documented symmetric-memory mechanism; ordinary `cudaMalloc` memory is not automatically a remote symmetric object.

Mention `NVSHMEM_SYMMETRIC_SIZE` only for configurations that use a statically sized symmetric heap, and retrieve the exact release-specific behavior before recommending a value.

## Remote Memory Access

Explain the three common naming families: `TYPENAME` for typed elements such as `int` or `float`, `SIZE` for fixed-width transfers, and `mem` for byte counts.

```cpp
void nvshmem_TYPENAME_put(TYPE *dest, const TYPE *source,
                          size_t nelems, int pe);
void nvshmem_TYPENAME_get(TYPE *dest, const TYPE *source,
                          size_t nelems, int pe);
void nvshmem_putSIZE(void *dest, const void *source,
                     size_t nelems, int pe);
void nvshmem_putmem(void *dest, const void *source,
                    size_t nbytes, int pe);
void nvshmem_getmem(void *dest, const void *source,
                    size_t nbytes, int pe);

void nvshmem_TYPENAME_p(TYPE *dest, TYPE value, int pe);
TYPE nvshmem_TYPENAME_g(const TYPE *source, int pe);

void nvshmem_TYPENAME_iput(TYPE *dest, const TYPE *source,
                           ptrdiff_t dst_stride, ptrdiff_t src_stride,
                           size_t nelems, int pe);
void nvshmem_TYPENAME_iget(TYPE *dest, const TYPE *source,
                           ptrdiff_t dst_stride, ptrdiff_t src_stride,
                           size_t nelems, int pe);
```

Use put to copy local data to a remote symmetric destination, get to copy from a remote symmetric source into local storage, `p`/`g` for a scalar, and `iput`/`iget` for strided data.

## Invocation Contexts

Distinguish where an operation is issued:

| Form | Invocation context | Meaning |
| --- | --- | --- |
| `nvshmem_*` host form | CPU code | Issue the documented host operation directly. |
| `nvshmem_*` device form | CUDA kernel | Let a GPU thread initiate the operation. |
| `nvshmemx_*_on_stream` | CPU code with a CUDA stream | Enqueue a supported operation in stream order; it is asynchronous with respect to the host. |
| `nvshmemx_*_warp` / `nvshmemx_*_block` | CUDA kernel | Have all threads in the warp or block cooperate on one operation. |

Require every participating thread in a cooperative warp/block call to use identical arguments. Recommend cooperative forms for sufficiently large kernel-initiated transfers, but verify availability for the chosen operation and release.

## Blocking and Nonblocking RMA

- A blocking put makes its local source reusable when the call returns but does not by itself synchronize a remote consumer with the update.
- A blocking get or fetching operation returns its requested result to the caller, subject to NVSHMEM ordering rules.
- An `_nbi` operation initiates work and requires a later completion operation before relying on completion or reusing affected local buffers.

```cpp
nvshmem_putmem_nbi(dest, source, nbytes, pe);
/* issue more independent work */
nvshmem_quiet();
```

Use `_nbi` variants to pipeline independent transfers, then complete the batch with quiet in the same issuing context.

## Signaling and Atomics

Use put-with-signal for a producer/consumer transfer whose data delivery is paired with an atomic update to a symmetric `uint64_t` signal object:

```cpp
void nvshmem_TYPENAME_put_signal(
    TYPE *dest, const TYPE *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

__device__ uint64_t nvshmem_signal_wait_until(
    uint64_t *sig_addr, int cmp, uint64_t cmp_value);
```

Use `NVSHMEM_SIGNAL_SET` to replace the signal value or `NVSHMEM_SIGNAL_ADD` to add to it. The signal establishes delivery for the data coupled to that put; do not assume it completes unrelated earlier operations without the required ordering operation.

Use AMOs for remote counters, work queues, ownership, or dynamic scheduling on symmetric objects:

```cpp
TYPE nvshmem_TYPENAME_atomic_fetch_add(TYPE *dest, TYPE value, int pe);
void nvshmem_TYPENAME_atomic_add(TYPE *dest, TYPE value, int pe);
TYPE nvshmem_TYPENAME_atomic_compare_swap(
    TYPE *dest, TYPE cond, TYPE value, int pe);
TYPE nvshmem_TYPENAME_atomic_fetch_inc(TYPE *dest, int pe);
```

Explain fetching versus non-fetching forms and verify datatype, invocation-context, and transport support for the target release.

## Ordering and Completion

Keep the guarantees separate:

| Call | Guarantee |
| --- | --- |
| `nvshmem_fence()` | Order earlier operations before later operations to the same destination PE; do not wait for completion. |
| `nvshmem_quiet()` | Complete previously issued operations on symmetric objects from the calling PE in the matching host or device context; do not synchronize with another PE. |
| `nvshmemx_quiet_on_stream(stream)` | Enqueue completion of prior device-side operations in stream order. |

Use fence when only per-destination ordering is needed. Use quiet when completion or safe local-buffer reuse is required. Follow either with the appropriate signal/wait or collective operation when a consumer must coordinate with the producer.

## Point-to-Point Synchronization

Use wait/test operations on a local symmetric object that another PE updates:

```cpp
void nvshmem_TYPENAME_wait_until(
    TYPE *ivar, int cmp, TYPE cmp_value);
int nvshmem_TYPENAME_test(
    TYPE *ivar, int cmp, TYPE cmp_value);
```

Use comparisons such as `NVSHMEM_CMP_EQ`, `NVSHMEM_CMP_NE`, `NVSHMEM_CMP_GT`, `NVSHMEM_CMP_GE`, `NVSHMEM_CMP_LT`, and `NVSHMEM_CMP_LE`. Explain that wait blocks while test polls, and mention the documented `_all`, `_any`, `_some`, and vector families only when the user needs array-wide conditions. Pair signal objects with `nvshmem_signal_wait_until` or its documented on-stream form.

## Teams and Collectives

Describe `nvshmem_team_t` as a PE subset and collective-ordering domain. Use `NVSHMEM_TEAM_WORLD` for all PEs and `NVSHMEMX_TEAM_NODE` for the predefined node-local team. Query, create, translate, and destroy application teams with:

```cpp
int nvshmem_team_my_pe(nvshmem_team_t team);
int nvshmem_team_n_pes(nvshmem_team_t team);
int nvshmem_team_translate_pe(
    nvshmem_team_t src_team, int src_pe, nvshmem_team_t dest_team);
int nvshmem_team_split_strided(/* parent and split arguments */);
int nvshmem_team_split_2d(/* parent and split arguments */);
void nvshmem_team_destroy(nvshmem_team_t team);
```

Treat team creation as collective on the parent team and require team collectives to occur in the same program order on every member.

Introduce synchronization and representative data collectives:

```cpp
void nvshmem_barrier(nvshmem_team_t team);
void nvshmem_barrier_all(void);
int  nvshmem_sync(nvshmem_team_t team);
void nvshmem_sync_all(void);

int nvshmem_TYPENAME_broadcast(
    nvshmem_team_t team, TYPE *dest, const TYPE *source,
    size_t nelems, int root);
int nvshmem_TYPENAME_fcollect(
    nvshmem_team_t team, TYPE *dest, const TYPE *source,
    size_t nelems);
int nvshmem_TYPENAME_alltoall(
    nvshmem_team_t team, TYPE *dest, const TYPE *source,
    size_t nelems);
int nvshmem_TYPENAME_OP_reduce(
    nvshmem_team_t team, TYPE *dest, const TYPE *source,
    size_t nelems);
```

Explain that barrier is a rendezvous that also completes prior remote updates from participants, whereas sync is a rendezvous that does not complete prior NVSHMEM RMA or AMO updates. Present `fcollect` as an all-gather-like operation and `OP_reduce` for operations such as sum, product, minimum, maximum, and bitwise reductions.

Although these calls exist, explicitly prefer NCCL when an application is dominated by standard bulk-collective calls, especially host-issued all-reduce, all-gather, reduce-scatter, or broadcast operations.

## Collective Kernel Launch

Require collective launch when a CUDA kernel itself invokes NVSHMEM synchronization or collective APIs such as wait, barrier, or a collective operation:

```cpp
void *args[] = {&arg0, &arg1};
nvshmemx_collective_launch(
    (const void *)kernel, grid, block, args,
    dynamic_shared_bytes, stream);
```

Explain that this launch is collective across PEs and uses cooperative launch to provide forward progress. Check the supported grid size with `nvshmemx_collective_launch_query_gridsize`. Ordinary kernel launch remains valid for a kernel that only issues non-synchronizing NVSHMEM operations, such as the scalar put in the ring-shift example below.

Use the official [API index](https://docs.nvidia.com/nvshmem/api/latest/api.html), [setup and query reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html), [memory management reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html), [RMA reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html), [signaling reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html), [point-to-point synchronization reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/sync.html), [memory ordering reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html), [collectives reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html), [teams reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/teams.html), and [kernel launch reference](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html) when exact signatures or version-specific availability matter.
