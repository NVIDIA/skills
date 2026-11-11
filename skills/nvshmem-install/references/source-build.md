# NVSHMEM Source-Build Decisions

Read the versioned requirements, installation procedure, release notes, and launcher documentation through `$nvshmem-docs` before using any identifier below. Confirm that every option exists and has the documented meaning for the resolved release.

## Source Acquisition

- Obtain released source builds from the official NVIDIA NVSHMEM GitHub releases page: `https://github.com/NVIDIA/nvshmem/releases`.
- Resolve the requested version to a published GitHub release tag, then use that release's published source archive or tag. Verify that the archive/tag version matches the resolved NVSHMEM version before configuring the build.
- Record the GitHub release URL and tag as source provenance. Use a direct archive URL only when it is exposed by that release page; do not construct one from a naming pattern.

## Configuration Order

1. Resolve and obtain the GitHub release source as described above.
2. Select a user-owned, shared, or system install prefix.
3. Select only the CUDA architectures present in the deployment target unless the user needs a portable multi-architecture build.
4. Inspect every available RDMA NIC or fabric provider and select the smallest matching remote transport set. When no remote transport is selected, set every remote-transport support option off.
5. Select the bootstrap required by the launcher or parent programming model.
6. Decide whether to build C/C++ tests, examples, packages, Hydra, and NVSHMEM4Py.
7. Locate every enabled dependency and verify its documented version and build configuration.
8. Generate one explicit CMake configure command followed by `cmake --build` and install commands.

## Transport Selection

| Target or intent | Candidate option family | Guidance |
| --- | --- | --- |
| No selected remote fabric | None | Set every remote-transport support option off when the user limits the installation to local GPUs, no matching NIC/provider exists, or the user accepts a diagnosed capability loss |
| Mellanox InfiniBand/RoCE | `NVSHMEM_IBRC_SUPPORT` | Enable by default when the probe identifies a Mellanox RDMA NIC and its documented driver, peer-memory, and atomic requirements pass |
| Mellanox InfiniBand/RoCE with DevX | `NVSHMEM_IBDEVX_SUPPORT` | Consider when the probe and versioned requirements establish the required DevX and `libmlx5` support |
| GPU-initiated InfiniBand | `NVSHMEM_IBGDA_SUPPORT` | Enable by default with Mellanox InfiniBand/RoCE when every documented NIC, driver, peer-memory/DMA-BUF, and mode requirement passes |
| GPUNetIO/GDAKI | `NVSHMEM_GPUNETIO_SUPPORT` | Enable by default with Mellanox InfiniBand/RoCE when its documented prerequisites pass; assess DOCA separately for advanced features and ordering semantics |
| Site UCX stack | `NVSHMEM_UCX_SUPPORT` | Use when the cluster supplies a compatible UCX build and this transport is intended |
| Amazon EFA or Slingshot/libfabric | `NVSHMEM_LIBFABRIC_SUPPORT` | Select the documented EFA or CXI provider matching the detected fabric and account for release-specific runtime limitations |

### Default Remote-Transport Rule

1. Read the probe's RDMA devices, link-layer/device details, and libfabric providers together with the versioned requirements.
2. If a Mellanox InfiniBand or RoCE NIC is present, consider IBRC, IBDEVX, IBGDA, and GPUNetIO as the default candidate set. Assess each transport's documented prerequisites independently and enable only candidates whose requirements receive a `pass` verdict; a user may explicitly opt out of a passing candidate.
3. If an EFA or Slingshot/CXI provider is present, select the matching libfabric transport and provider settings.
4. Select UCX only when the user or site requires the UCX transport; do not select it merely because UCX is installed alongside a more specific matching transport.
5. If the matching transport has a failed documented prerequisite, disable that transport, report the capability loss, and ask whether to proceed without the NIC-matched baseline remote path when applicable. If a prerequisite is unknown, resolve it before generating an installation command.

Do not omit remote support solely because a same-node smoke test is convenient. A NIC-equipped target should receive a remote-capable build by default.

Apply a strict verdict gate to every candidate transport: `pass` permits default enablement, `unknown` blocks generation of an exact CMake configuration until resolved, and `fail` disables the transport. Do not treat user opt-in as satisfying an unknown or failed mandatory prerequisite, and do not silently omit an unknown candidate.

- Do not enable transports unrelated to the detected fabric. On Mellanox InfiniBand/RoCE, the default remote set is IBRC, IBDEVX, IBGDA, and GPUNetIO.
- Use `NVSHMEM_USE_GDRCOPY` only after checking whether the selected transport and required atomic behavior need it.
- Satisfy the union of prerequisites for every enabled transport. Do not disable GDRCopy, peer-memory/DMA-BUF, verbs, or another documented dependency while retaining a transport that requires it.
- Treat optional dependency absence as a reason to disable an optional enhancement, not to bypass its requirement. When it prevents the NIC-matched baseline remote transport, ask before accepting a build with no remote transport support.
- Explain the performance/capability reason for each selected transport, including why a Mellanox InfiniBand/RoCE target enables IBDEVX, IBGDA, and GPUNetIO by default.
- Check release notes for transport-specific limitations and required runtime variables.

## Bootstrap Selection

| Launch/application model | Build/runtime family | Guidance |
| --- | --- | --- |
| Standalone with packaged Hydra | PMI-1/Hydra | Use the bundled or documented Hydra workflow when no site launcher is required |
| Slurm with PMI-2 | `NVSHMEM_DEFAULT_PMI2` and runtime PMI selection | Match the site's Slurm PMI support and release-specific launch flags |
| Slurm or Open MPI with PMIx | `NVSHMEM_PMIX_SUPPORT`, `NVSHMEM_DEFAULT_PMIX` when appropriate | Locate the exact PMIx implementation used by the launcher and avoid mixing incompatible installations |
| Existing MPI application | `NVSHMEM_MPI_SUPPORT` | Build the MPI bootstrap/integration and initialize NVSHMEM with the application's communicator |
| Existing OpenSHMEM application | `NVSHMEM_SHMEM_SUPPORT` | Build against the site's OpenSHMEM implementation and use its launcher |
| Application-managed rendezvous | UID-based initialization | Use only when the application distributes NVSHMEM unique IDs correctly; this is an application initialization choice, not a generic launcher replacement |

- Prefer the site's established launcher on a managed cluster.
- Inspect the launcher, PMI/PMIx libraries, and application initialization model together. A launcher name alone is insufficient.
- Do not set a compile-time default that conflicts with the intended runtime selection.
- On heterogeneous clusters, validate the same bootstrap library and plugin visibility on every node.

## Build Shape

Consider these current option families, but verify each through `$nvshmem-docs` for the requested release:

- Install prefix: `NVSHMEM_PREFIX` and the corresponding native CMake install-prefix behavior.
- CUDA targets: `CMAKE_CUDA_ARCHITECTURES` or the documented NVSHMEM compatibility variable.
- CUDA compiler: use the resolved `CUDA_HOME/bin/nvcc` in separate validation blocks unless the probe established that `nvcc` is already on `PATH`.
- Tests and examples: `NVSHMEM_BUILD_TESTS`, `NVSHMEM_BUILD_EXAMPLES`.
- Python: `NVSHMEM_BUILD_PYTHON_LIB` and any release-specific Python/device-library options.
- Packages: `NVSHMEM_BUILD_PACKAGES` and the requested DEB, RPM, or archive format options.
- Hydra: the release's Hydra build/install option or helper procedure.

Prefer this command structure after resolving all values:

```bash
cmake -S <source> -B <build> \
  -D<option>=<value> \
  ...
cmake --build <build> --parallel <jobs>
cmake --install <build>
```

Do not leave ambiguous option placeholders inside an “exact commands” block. Put unresolved values in a short prerequisites list and wait for them.

For interactive shells or tmux windows, prefer ordered commands without persistent `set -euo pipefail` changes, and stop after the first failure to avoid closing the user's shell or pane.

## Build Review

Before requesting execution approval, show:

- Source release and provenance.
- Install and build directories.
- CUDA architecture set.
- Every enabled and explicitly disabled transport.
- Selected bootstrap and launcher.
- Dependency prefixes.
- Whether tests, examples, Python bindings, Hydra, and packages are built.
- Estimated network downloads and locations written.
- Post-install environment and exactly two baseline smoke tests: a same-node initialization/hello test, then a two-node initialization/hello test using the default remote transport. The two-node test must run its executable and use its working directory, inputs, and logs from a shared filesystem path visible to both allocated nodes; do not use `/tmp`. These tests may be recommended by default, but running them still requires approval for the applicable command block. Do not add alternate-transport, alternate-HCA, IBDEVX, IBGDA, GPUNetIO, performance, or broader functional tests unless the user explicitly requests them. Use `"$PREFIX/bin/nvshmem-info" -n -b` for the user-facing post-install information check; do not show bare `nvshmem-info` because its successful default invocation is silent. Each separately presented validation block must explicitly define every install, source, build, CUDA, path, library-path, and launcher value it consumes rather than relying on variables from an earlier block. If a two-node test cannot run, state that limitation explicitly.
