# NVSHMEM Mental Model

Use this reference when teaching the conceptual mental-model path. Establish the concepts before introducing API names, then connect them in the ring example. Keep concrete API-family lookup in the separate reference selected directly by `SKILL.md`.

## Contents

- [Mental Model](#mental-model)
- [Illustrative Ring Shift](#illustrative-ring-shift)

## Mental Model

Explain these concepts without introducing API names:

- **Processing elements (PEs):** An NVSHMEM job runs one SPMD program in multiple processes. Each process is a PE with a unique ID and typically controls one GPU.
- **Symmetric memory:** Every PE collectively creates matching objects in its GPU-resident symmetric heap. A remote object is identified by the local symmetric address of the corresponding object plus the destination PE, not by exchanging raw pointer values.
- **One-sided communication:** The initiating PE can put, get, update, or signal remote symmetric memory without a matching receive call at the destination.
- **Execution contexts:** Communication can originate in CPU code, inside CUDA kernels, or as work ordered on a CUDA stream. These contexts compose differently with surrounding CUDA work and have distinct completion domains.
- **Teams:** The world team contains all PEs; predefined or application-defined teams represent subsets for topology-aware queries, synchronization, and collectives.
- **Ordering, completion, and synchronization:** Issuing an operation, completing it, making its result observable, and coordinating with another PE are different events. Choose the mechanism that provides the guarantee the consumer needs.
- **Lifecycle:** Initialize collectively, select the local CUDA device, create symmetric data, communicate and coordinate, destroy symmetric data collectively, and finalize.

Use a simple address picture when helpful:

```text
remote object = (local symmetric address, destination PE)
```

Emphasize that a symmetric pointer is meaningful as an NVSHMEM operand on the PE that obtained it. Do not send its numeric value to another PE and reuse it there as a local pointer.

## Illustrative Ring Shift

Use this compact adaptation of NVIDIA's `simple_shift` example to connect the concepts and calls. State that it is a teaching example, not a compilation or launch recipe: it intentionally omits error checking and environment setup.

```cpp
#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void shift_to_next_pe(int *destination) {
    int me = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int next = (me + 1) % npes;

    nvshmem_int_p(destination, me, next);
}

int main(void) {
    cudaStream_t stream;
    int received;

    nvshmem_init();
    cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE));
    cudaStreamCreate(&stream);

    int *destination = (int *)nvshmem_malloc(sizeof(int));
    shift_to_next_pe<<<1, 1, 0, stream>>>(destination);
    nvshmemx_barrier_all_on_stream(stream);

    cudaMemcpyAsync(&received, destination, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("PE %d received %d\n", nvshmem_my_pe(), received);

    nvshmem_free(destination);
    nvshmem_finalize();
}
```

Map the example back to the model:

- Setup creates the PE world, selects a GPU by node-local PE rank, and creates a CUDA stream.
- Symmetric allocation gives every PE a corresponding `destination` integer.
- The kernel identifies its PE and writes that ID to the next PE's corresponding object with a one-sided scalar put.
- The on-stream barrier completes the ring updates and synchronizes PEs before the stream copies the local result to the CPU.
- Collective deallocation and finalization end the symmetric-memory and runtime lifecycles.

Link to the authoritative [Example NVSHMEM Program](https://docs.nvidia.com/nvshmem/api/latest/using.html#example-nvshmem-program). Route to `$nvshmem-docs` before adapting the example to a version-specific API, build, or launch environment.
