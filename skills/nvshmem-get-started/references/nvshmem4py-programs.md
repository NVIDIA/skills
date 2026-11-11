# NVSHMEM4Py Programs

Use this reference for a first NVSHMEM4Py program or run. It provides the stable workflow; obtain current API names, signatures, supported array frameworks, package compatibility, and exact commands from `$nvshmem-docs` before presenting runnable code.

## Program Model

- An NVSHMEM4Py job is an SPMD Python program: every PE runs the same script and normally controls one GPU.
- The Python bindings use the NVSHMEM model: communication targets are symmetric GPU-resident objects plus a destination PE. Do not treat an arbitrary CuPy array or PyTorch tensor as remotely addressable symmetric memory.
- A remote operation, its completion, and synchronization with the remote consumer are separate requirements. Choose the ordering, completion, signal/wait, or collective mechanism that the Python API documents for the operation.
- Call initialization and finalization collectively. Create and release symmetric arrays collectively in matching order on every PE.

## First-Program Lifecycle

1. Choose one PE per GPU for the first run and determine the node-local rank.
2. Select the local CUDA device before creating GPU arrays or initializing the Python binding, as required by the chosen initialization path.
3. Initialize NVSHMEM4Py with either its standalone path or its MPI-interoperable path. Use MPI only when the script and launch setup require it.
4. Allocate symmetric arrays through NVSHMEM4Py's documented allocation interface; initialize their local contents.
5. Use the documented Python remote-memory, collective, or synchronization operation that matches the communication pattern.
6. Synchronize or complete work in the documented CUDA-stream context before consuming results on the host or reusing buffers.
7. Free all symmetric arrays collectively, then finalize.

For a first validation, use two PEs on one node and an operation with easy-to-check per-PE output. Scale to more GPUs or nodes only after that run succeeds.

## Simple Two-PE Program

Use this MPI-initialized all-reduce as a first example after `$nvshmem-docs` confirms that the APIs apply to the user's version. It assigns each PE a local value of one plus its node-local rank, sums the symmetric source arrays across `TEAM_WORLD`, and prints the result. With two PEs on one node, each PE should print a destination array containing `3`.

```python
from cuda.core.experimental import Device, system
from mpi4py import MPI
import nvshmem.core as nvshmem

local_rank = MPI.COMM_WORLD.Get_rank() % system.num_devices
device = Device(local_rank)
device.set_current()
stream = device.create_stream()

nvshmem.init(device=device, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

source = nvshmem.array((2, 2), dtype="float32")
destination = nvshmem.array((2, 2), dtype="float32")
source[:] = local_rank + 1
destination[:] = 0

nvshmem.reduce(nvshmem.Teams.TEAM_WORLD, destination, source, "sum", stream=stream)
stream.sync()
print(f"PE {nvshmem.my_pe()} received:\n{destination}")

nvshmem.free_array(source)
nvshmem.free_array(destination)
nvshmem.finalize()
```

Treat the example as an API-validated starting point, not a universal template. It requires an MPI-capable environment with `mpi4py`; use `$nvshmem-docs` to select the matching launcher and initialization method. When reviewing a user-provided script, compare it against this lifecycle instead of requiring the same collective or array shape.

## Running a Python Job

Gather the script path, Python executable or activated environment, launcher, PE and node counts, and whether the run is local or scheduled. Confirm that all participating nodes can access:

- the same Python environment and NVSHMEM4Py package;
- the script and its dependencies;
- CUDA and NVSHMEM runtime libraries; and
- the launcher bootstrap expected by the NVSHMEM installation.

NVSHMEM4Py uses the NVSHMEM launch methods. Common command shapes are:

```bash
nvshmrun -n <number-of-pes> <python> <script.py> [arguments]
```

```bash
srun -n <number-of-pes> <python> <script.py> [arguments]
```

```bash
mpirun -n <number-of-pes> <python> <script.py> [arguments]
```

Treat these as shapes, not release- or site-specific prescriptions. Verify bootstrap settings, launcher flags, and environment propagation with `$nvshmem-docs` for the selected NVSHMEM version and cluster.

## Documentation Handoff

Invoke `$nvshmem-docs` for any of the following before making a concrete recommendation:

- a first runnable Python example or exact `nvshmem.core` call;
- NVSHMEM4Py package installation or Python/CUDA requirements;
- NVSHMEM and NVSHMEM4Py version compatibility;
- MPI, CuPy, PyTorch, CUDA stream, or device-API interoperability;
- multi-node bootstrap, transport, scheduler, or launcher configuration; or
- an import error, a hang, or a failed Python job.

Use the latest documentation when no version is supplied. When an NVSHMEM or NVSHMEM4Py version is known, pass it to `$nvshmem-docs`; do not infer compatibility from a package name.

## Official Starting Points

- [Installing NVSHMEM Language Bindings for Python](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem4py-install-proc.html)
- [NVSHMEM4Py API and compatibility overview](https://docs.nvidia.com/nvshmem/api/latest/api/language_bindings/python/overview.html)
- [NVSHMEM launch methods](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html#launching-nvshmem-programs)
