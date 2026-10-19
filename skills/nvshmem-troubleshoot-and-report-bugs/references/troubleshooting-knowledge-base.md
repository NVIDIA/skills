# NVSHMEM Troubleshooting Knowledge Base

This reference is a symptom index, not a substitute for live documentation. It was seeded from official documentation displaying NVSHMEM 3.7.0 in July 2026. Before asserting that an entry applies to a user's release, invoke `$nvshmem-docs` with the complete installed version and verify the current FAQ, environment-variable reference, limitations, known issues, and later fixed issues.

## Topic Index

| Topic | Start here when evidence mentions |
| --- | --- |
| [Launch and Bootstrap](#launch-and-bootstrap) | Bootstrap plugins, dynamic loading, MPI, PMI, or PMIx initialization |
| [Transport and Network Memory](#transport-and-network-memory) | GPUDirect RDMA, InfiniBand devices, CUDA VMM, memory registration, or CQ errors |
| [GPU Mapping and Topology](#gpu-mapping-and-topology) | PE-to-GPU placement, P2P access, HCA affinity, sockets, or remote paths |
| [Language Bindings and Device Code](#language-bindings-and-device-code) | NVSHMEM4Py initialization succeeds but a custom CUDA kernel faults or cannot use device APIs |
| [Synchronization and Correctness](#synchronization-and-correctness) | Hangs, concurrent collectives, ordering, visibility, invalid pointers, or unsupported operations |
| [Teams and Runtime Resources](#teams-and-runtime-resources) | Team creation, symmetric-heap allocation, internal teams, multi-CTA collectives, or runtime resource limits |
| [Platform Services](#platform-services) | IMEX, fabric memory, device nodes, permissions, or container device access |
| [Build and Package Integration](#build-and-package-integration) | CMake package discovery, device linking, CUDA architecture coverage, or development packages |
| [Versioned Issues and Fixes](#versioned-issues-and-fixes) | Symptoms that may match a known fix in a later NVSHMEM release |
| [Pre-Start and External Failures](#pre-start-and-external-failures) | Shell, executable, loader, scheduler, container, or launcher failures before NVSHMEM starts |

See [How to Use an Entry](#how-to-use-an-entry) before confirming a match and [Primary Sources](#primary-sources) for the authoritative documentation set.

## How to Use an Entry

Require both a compatible symptom and the entry's stated conditions for a confirmed match. Otherwise use it only as a hypothesis. Always identify the first causal line; later `non-zero status`, `aborting`, launcher cleanup, and peer termination messages are usually consequences.

For each match, report:

- **Signature**: exact or semantic evidence in the supplied logs.
- **Likely cause and owner**: NVSHMEM, application, dependency, or platform.
- **Action**: the smallest supported correction or discriminating check.
- **Verification**: the expected evidence after the action.
- **Version applicability**: verified live, not inferred from this snapshot.

## Launch and Bootstrap

### Bootstrap Plugin Cannot Be Loaded

- **Signatures:** `Bootstrap unable to load`, a missing `nvshmem_bootstrap_*.so`, `bootstrap_loader_init returned error`, or loader text such as `cannot open shared object file` during NVSHMEM initialization.
- **Likely cause and owner:** The selected NVSHMEM bootstrap plugin is absent from the runtime library search path, was not installed, or was built for a different MPI/OpenSHMEM/PMI implementation. This is normally an installation or runtime-environment issue.
- **Action:** Confirm the selected bootstrap, the exact plugin filename, whether the file exists under the same NVSHMEM installation as `libnvshmem_host`, the effective dynamic-loader search path on every node, and the dependency libraries reported by the loader. Use `$nvshmem-docs` before naming a bootstrap-plugin environment variable because available selectors vary by release.
- **Verification:** Every PE loads the intended plugin and INFO logs progress past bootstrap initialization.
- **Version applicability:** The plugin architecture and option names are version-sensitive; verify them for the installed release.
- **Source:** [General FAQ: bootstrap module loading](https://docs.nvidia.com/nvshmem/api/faq.html#general-faqs)

### PMIx and Open MPI Symbol Mismatch

- **Signatures:** `pmix_mca_base_component_repository_open`, an MCA component that cannot load, `missing symbol`, or text saying a component was compiled for a different OpenPMIx version.
- **Likely cause and owner:** NVSHMEM, Open MPI, and the runtime launcher are loading incompatible PMIx implementations. This is a bootstrap/dependency compatibility problem rather than an application-kernel defect.
- **Action:** Record the Open MPI, PMIx, launcher, and NVSHMEM build/runtime versions and library paths. Verify that Open MPI and NVSHMEM use a compatible external PMIx installation. Do not recommend rebuilding until `$nvshmem-docs` confirms the procedure for the installed version.
- **Verification:** Loader warnings disappear and all PEs finish bootstrap initialization.
- **Version applicability:** Known FAQ guidance is historical; verify current compatibility and any later PMIx fixes.
- **Source:** [Running NVSHMEM Programs FAQ](https://docs.nvidia.com/nvshmem/api/faq.html#running-nvshmem-programs-faqs)

### All Processes Report PE 0

- **Signatures:** A launcher requests multiple processes, but INFO output from several operating-system PIDs labels every process as `PE 0`, each process reports `Running NVSHMEM with 1 PEs`, or `nvshmem_n_pes()` returns 1 in every process. Do not confuse a logger's fixed severity or device field with the PE identifier.
- **Likely cause and owner:** The launcher started several singleton NVSHMEM jobs because the executable's initialization path and the selected launcher/bootstrap do not share rank and job membership. A shipped test can also require a test-harness launcher mode that a general application does not. Ownership is normally the application/launcher/bootstrap integration, not GPU mapping.
- **Action:** Identify the initialization API used by the executable and the process manager actually launching it. For MPI initialization, verify that MPI is initialized first and that the same communicator is passed to `nvshmemx_init_attr`; for PMI, PMIx, UID, or direct launch, select only the bootstrap documented for the installed version. Compare with an official minimal example under the same launcher. Do not copy a test-only environment switch into an application unless the matching test source or documentation requires it.
- **Verification:** One rerun reports unique PE IDs from 0 through `n-1`, and every process reports the requested total PE count before initialization advances.
- **Version applicability:** Bootstrap defaults, selectors, and test-harness switches are release-specific. No universal setting or fixed release is established for this symptom.
- **Sources:** [Running NVSHMEM programs](https://docs.nvidia.com/nvshmem/api/using.html#running-nvshmem-programs), [Library setup and attribute-based initialization](https://docs.nvidia.com/nvshmem/api/gen/api/setup.html)

## Transport and Network Memory

### GPUDirect RDMA Memory Registration Failure

- **Signatures:** `mem registration failed`, `transport get memhandle failed`, or failure registering GPU memory with an InfiniBand transport.
- **Likely cause and owner:** The platform cannot register CUDA memory for RDMA because the supported DMA-BUF, `nvidia-peermem`, or legacy peer-memory path is absent, incompatible, inaccessible, or not active. This is usually a driver/kernel/platform configuration issue.
- **Action:** Use the version-matched installation requirements to identify supported GPUDirect RDMA paths. Collect read-only evidence for driver, kernel, CUDA, NIC, RDMA device, and peer-memory state. Ask the administrator to correct missing driver or kernel support; do not load modules or change the system.
- **Verification:** The selected remote transport registers the symmetric heap and initialization proceeds.
- **Version applicability:** Do not assume the FAQ's legacy `nv_peer_mem` example is the preferred path on current systems.
- **Sources:** [General FAQ: memory registration](https://docs.nvidia.com/nvshmem/api/faq.html#general-faqs), [Using NVSHMEM prerequisites](https://docs.nvidia.com/nvshmem/api/using.html)

### InfiniBand Device Discovery Failure

- **Signatures:** `get_device_list failed`, no devices returned by `ibverbs`, or INFO logs showing an IB transport was selected but no usable HCA was found.
- **Likely cause and owner:** InfiniBand libraries are present, but no active or visible device is available inside the host, container, allocation, or device cgroup. This is normally a platform or container configuration issue.
- **Action:** Confirm that the job is expected to use IB, inspect read-only RDMA device and link state, compare host and container visibility, and verify any HCA include/exclude or PE-mapping settings for the installed version.
- **Verification:** The intended HCA appears on every participating node and the transport selects it consistently.
- **Source:** [General FAQ: InfiniBand device discovery](https://docs.nvidia.com/nvshmem/api/faq.html#general-faqs)

### RoCE GID or Address-Handle Initialization Failure

- **Signatures:** A RoCE run fails during IBGDA setup with `ibv_create_ah` and `EINVAL`, reports a GID or `sysfs` query failure, finds an HCA but no usable active port/GID, or works on the host while failing in a container with the same NIC.
- **Likely cause and owner:** First suspect an inactive or invisible RoCE port, an inaccessible container `sysfs` view, or an address-family/range/GID selection mismatch. A public NVSHMEM defect also showed that an uninitialized address-handle field could produce the same `ibv_create_ah` `EINVAL` signature. Configuration/visibility is owned by the network or container platform; the exact defect signature is owned by NVSHMEM.
- **Action:** From the exact job or container, collect read-only HCA, port, link-layer, active-state, GID-table, and INFO transport-selection evidence, then compare it with the host. Use `$nvshmem-docs` before setting `NVSHMEM_IB_GID_INDEX`, `NVSHMEM_IB_ADDR_FAMILY`, `NVSHMEM_IB_ADDR_RANGE`, or HCA mappings. If an active visible GID is selected but the exact `ibv_create_ah` `EINVAL` remains, compare the installed source/version with public issue 21 and test a version-verified candidate fix without changing the fabric.
- **Verification:** INFO logs select the intended active RoCE HCA, port, and GID on every PE, and a minimal two-PE initialization succeeds. For the defect path, the same configuration succeeds only with a build known to contain the fix.
- **Version applicability:** Dynamic RoCE GID discovery was added in 3.1.7, and 3.3.9 improved GID discovery when containerized `sysfs` queries fail. Public issue 21 is auditable defect evidence, but the official release notes do not name its fixed release; do not assert that 3.5.19 or another release contains the fix without live verification. The 3.5.19 release notes separately document a CX-4 Ethernet/RoCE limitation; do not extrapolate that limitation to another release or NIC.
- **Sources:** [RoCE environment variables](https://docs.nvidia.com/nvshmem/api/gen/env.html#nvshmem-ib-gid-index), [NVSHMEM 3.1.7 release notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3107.html), [NVSHMEM 3.3.9 release notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3309.html), [public issue 21](https://github.com/NVIDIA/nvshmem/issues/21)

### CUDA VMM Interoperability Failure

- **Signatures:** libfabric memory registration returns `Cannot allocate memory`; NVSHMEM then reports a memhandle or proxy-channel allocation failure; or CUDA-aware MPI/UCX crashes around `process_vm_readv`/CMA while using NVSHMEM symmetric memory.
- **Likely cause and owner:** The selected network or MPI stack cannot interoperate with CUDA VMM-backed NVSHMEM memory for that software combination.
- **Action:** Confirm whether VMM is enabled, which transport or MPI path accesses the buffer, and whether the version-matched FAQ or release limitations prescribe `NVSHMEM_DISABLE_CUDA_VMM=1`. Treat that setting as a documented compatibility workaround, not a universal fix.
- **Verification:** Repeating the minimal case with the documented setting avoids registration or CMA failure without changing other variables.
- **Version applicability:** CUDA VMM requirements and transport support have changed across releases; live verification is mandatory.
- **Sources:** [General and MPI interoperability FAQs](https://docs.nvidia.com/nvshmem/api/faq.html), [NVSHMEM release limitations](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/index.html)

### InfiniBand Completion Protection Error

- **Signatures:** `ibv_poll_cq failed` with completion status 10 or 4, followed by IBRC progress or quiet failure.
- **Likely cause and owner:** In the FAQ examples, status 10 indicates a remote protection error, commonly an RMA/AMO target outside the remote symmetric heap. Status 4 indicates a local protection error, commonly a local buffer that is neither symmetric nor registered. Passing a pointer returned by `nvshmem_ptr` or `nvshmemx_mc_ptr` back into an NVSHMEM RMA/AMO can also violate the expected address form. This is usually application buffer misuse.
- **Action:** Validate the first failing operation, source and target allocation provenance, bounds, PE, lifetime, and registration. Decode any other completion status using the verbs provider documentation rather than extrapolating these two cases.
- **Verification:** A minimal operation using valid symmetric/registered buffers completes without a CQ protection error.
- **Version applicability:** Verify pointer and registration rules against the installed release.
- **Source:** [General FAQ: `ibv_poll_cq` failures](https://docs.nvidia.com/nvshmem/api/faq.html#general-faqs)

## GPU Mapping and Topology

### Incorrect PE-to-GPU Mapping

- **Signatures:** multiple PEs unexpectedly select GPU 0, INFO logs say `More than 1 PE per GPU detected. This is an MPG run`, separate PEs create streams for the same device ordinal, PE count exceeds intended GPU count, or NVLS setup fails with `cuMulticastAddDevice failed`, `Subscribing multicast group failed`, and `NVLS resource setup failed`. A container can show several GPUs to `nvidia-smi` while each launched process still resolves to the same process-local ordinal.
- **Likely cause and owner:** The launcher, scheduler, container, or application establishes an unintended PE-to-GPU mapping, selects a device too late, or exposes a different/restricted CUDA device list to each PE. NVLS failures are downstream evidence when logs already prove duplicate GPU selection; without that evidence, investigate other NVLS/platform causes. Ownership is normally launcher, container, scheduler, or application configuration.
- **Action:** Record rank-to-host-to-visible-device mapping from inside each launched PE, including `CUDA_VISIBLE_DEVICES`, CUDA device count, selected ordinal, and GPU UUID/PCI bus ID. Confirm device selection occurs before NVSHMEM allocations, synchronization, communication, collective launch, or device-side NVSHMEM calls. Compare one run outside the container when permitted. Verify the installed release's multiprocess-per-GPU support and restrictions before recommending sharing or disabling NVLS.
- **Verification:** Each PE reports the intended stable physical GPU mapping, the unexpected MPG message disappears for a one-PE-per-GPU job, and initialization progresses past NVLS setup without changing unrelated transport settings.
- **Version applicability:** Multiprocess GPU support and restrictions are version-dependent.
- **Sources:** [General FAQ: PE/GPU mapping](https://docs.nvidia.com/nvshmem/api/faq.html#general-faqs), [Using NVSHMEM and multiprocess GPU support](https://docs.nvidia.com/nvshmem/api/using.html#multiprocess-gpu-support)

### GPU, HCA, and Remote-Path Topology

- **Signatures:** warning that an IB HCA and GPU are not connected to the same PCIe switch; failure only across sockets or nodes; no P2P access between a GPU pair; or selected NICs are disabled/non-IB.
- **Likely cause and owner:** The chosen GPU/HCA path is unavailable or suboptimal. Cross-socket communication requires an available supported remote path when GPUs are not mutually P2P-accessible.
- **Action:** Collect `nvidia-smi topo -m` or equivalent topology, HCA/link state, PE/GPU mapping, and INFO transport selection. Verify version-specific HCA affinity variables through `$nvshmem-docs`. Treat the PCIe-switch message as a performance warning unless another error establishes functional failure.
- **Verification:** A topology-supported GPU/HCA pairing initializes and remote communication completes; performance-only cases are handed to a performance skill.
- **Source:** [General and GPU interconnection FAQs](https://docs.nvidia.com/nvshmem/api/faq.html)

## Language Bindings and Device Code

### NVSHMEM4Py Host Initialization Leaves a Loaded CUDA Module Uninitialized

- **Signatures:** `nvshmem.core.init()` succeeds, PE queries and host/on-stream operations work, but a custom CUDA module that calls NVSHMEM device APIs later raises an illegal-memory-access or invalid-device-state error. Changing UID versus MPI bootstrap or merely passing a device does not fix the custom kernel.
- **Likely cause and owner:** NVSHMEM4Py initializes through the NVSHMEM host library, while dynamically loaded cubin/device code has separate NVSHMEM device state. The application loaded the CUDA module but did not initialize that module for NVSHMEM device API use. This is normally application integration, not a bootstrap failure.
- **Action:** Confirm `nvshmem.core.init_status()` is fully initialized and the intended CUDA device is current. After loading the cubin and after NVSHMEM device initialization completes, call the installed release's documented CUDA-module initialization API (currently `nvshmemx_cumodule_init`) and check its return value before launching the kernel. For module-independent library loading, use the corresponding API only when the installed release documents it; do not infer it from a newer release.
- **Verification:** A minimal kernel in the same loaded module can query its PE or perform one in-bounds symmetric operation, and the original kernel no longer faults without changing bootstrap method.
- **Version applicability:** Official NVSHMEM4Py bindings and `nvshmemx_culibrary_init` were added in 3.3.9. The current 3.7.0 setup API documents host-library initialization and `nvshmemx_cumodule_init`; exact Python binding and code-loading APIs must be checked for the installed complete version.
- **Sources:** [NVSHMEM4Py initialization](https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/initialization.html), [host-library and CUDA-module initialization](https://docs.nvidia.com/nvshmem/api/gen/api/setup.html#nvshmemx-cumodule-init), [NVSHMEM 3.3.9 release notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3309.html)

## Synchronization and Correctness

### Stream or Collective Ordering Hang

- **Signatures:** deterministic or intermittent hang around an `_on_stream` operation, collective, wait, or synchronization point, without an earlier transport error.
- **Likely cause and owner:** Common application causes include blocking stream-0 CUDA calls in an iterative phase, circular stream dependencies, or more than one collective over the same PE set in flight concurrently.
- **Action:** Identify the last operation reached by every PE and all streams participating in it. Check for `cudaDeviceSynchronize`, blocking `cudaMemcpy`, stream priorities, concurrent barriers/collectives over the same team, divergent PE control flow, and ordinary kernel launches containing NVSHMEM synchronization. Compare same-node P2P, cross-socket, and multi-node behavior to localize the path.
- **Verification:** Serialize the suspected collective or remove one dependency in a minimal test and confirm progress. Do not treat changed timing alone as proof.
- **Sources:** [Debugging FAQ](https://docs.nvidia.com/nvshmem/api/faq.html#debugging-faqs), [CUDA and NVSHMEM interactions](https://docs.nvidia.com/nvshmem/api/cuda-interactions.html)

### Completion or Receive-Visibility Error

- **Signatures:** a receive buffer remains stale after a flag changes, data is consumed after a barrier while `_on_stream` work is still pending, or correctness changes when adding synchronization.
- **Likely cause and owner:** The application confuses issue order, local completion, remote visibility, and host/device operation scope. A plain load loop on a remotely updated flag does not perform the NVSHMEM consistency action supplied by wait/test APIs.
- **Action:** Reconstruct the producer operation, completion primitive, stream, signal update, consumer wait/test, and buffer use. Verify required `quiet`, stream synchronization, barrier, wait, or test semantics with `$nvshmem-docs`; do not substitute one primitive based only on current implementation behavior.
- **Verification:** A minimal producer/consumer sequence using documented completion and visibility primitives produces stable data without timing-dependent workarounds.
- **Sources:** [NVSHMEM API usage and debugging FAQs](https://docs.nvidia.com/nvshmem/api/faq.html), [Memory ordering](https://docs.nvidia.com/nvshmem/api/gen/api/ordering.html)

### Invalid Memory or Unsupported Remote Operation

- **Signatures:** remote protection errors, invalid-address failures, wrong data only over IB, or use of strided `iput`/`iget` across a remote network transport.
- **Likely cause and owner:** The remote target is not a valid symmetric object, the local source does not satisfy registration rules, the pointer is a direct peer pointer rather than the symmetric address expected by the RMA API, or the selected operation is unsupported over the remote transport.
- **Action:** Record allocation API, PE, pointer origin, size, lifetime, registration, selected transport, and exact RMA/AMO. Check the installed release's unsupported-operations table. Do not generalize P2P-local buffer allowances to remote network paths.
- **Verification:** The equivalent supported operation with valid symmetric/registered operands succeeds over the same path.
- **Sources:** [Miscellaneous and GPU interconnection FAQs](https://docs.nvidia.com/nvshmem/api/faq.html), [Unsupported operations](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/unsupported-operations.html)

## Teams and Runtime Resources

### Symmetric Heap Allocation Exhaustion

- **Signatures:** `Not enough space for allocating memory`, `allocate_physical_memory_to_heap failed`, `nvshmem_malloc`/`nvshmem_calloc` returns a null pointer, or an allocation failure is followed by barrier errors, illegal memory access, mutex cleanup, and launcher abort messages.
- **Likely cause and owner:** The requested allocation or aggregate live symmetric memory exceeds the effective static heap, dynamic per-GPU limit, available GPU/fabric memory, or a platform allocation limit. Different allocation arguments or call order across PEs are an application correctness error. A library abort after an otherwise valid out-of-memory request conflicts with the documented null-return contract and is a possible NVSHMEM error-path defect.
- **Action:** Treat the first allocation error as causal. Record the allocation API, size, call order, live-allocation peak, and return value on every PE; verify that every PE calls collective allocation routines with identical arguments. Use `$nvshmem-docs` to check `NVSHMEM_SYMMETRIC_SIZE`, `NVSHMEM_MAX_MEMORY_PER_GPU`, CUDA VMM mode, and platform memory constraints for the complete version. If a null pointer is returned, prevent all later buffer use and take a coordinated error path. If the process aborts before the application can inspect the return, preserve the logs and test a smaller allocation before filing a defect.
- **Verification:** The smallest case returns a non-null symmetric pointer on every PE after reducing the live allocation or applying a version-supported heap limit, and no downstream barrier or illegal-access error appears.
- **Version applicability:** Heap policy, defaults, and limits vary by release. Current 3.7.0 documentation specifies a null return when allocation does not succeed; reproduce process termination on the exact installed version before classifying that behavior as a defect.
- **Sources:** [Memory management API](https://docs.nvidia.com/nvshmem/api/gen/api/memory.html), [runtime memory settings](https://docs.nvidia.com/nvshmem/api/gen/env.html#nvshmem-symmetric-size)

### NVSHMEM_MAX_TEAMS Exhaustion

- **Signatures:** `No more teams available`, `Unable to allocate enough duplicate teams`, an error advising `NVSHMEM_MAX_TEAMS`, or team creation returning failure after earlier teams succeeded.
- **Likely cause and owner:** The configured simultaneous-team limit is too small for user-visible teams plus internal teams reserved by NVSHMEM. Multi-CTA collectives and NVLS can substantially increase internal team use.
- **Action:** Count simultaneously live application teams and identify enabled collective/NVLS behavior. Use `$nvshmem-docs` to obtain the default, internal-team accounting, and supported setting for the installed version. Increase `NVSHMEM_MAX_TEAMS` only enough for the verified requirement and preserve the original reproducer for comparison.
- **Verification:** INFO logs show the new limit, team creation completes, and the original workload no longer reports exhaustion.
- **Version applicability:** In documentation displaying 3.7.0, the default is 128 and an NVLS-enabled user team can require up to 48 internal teams. Do not apply those numbers to another release without live verification.
- **Source:** [`NVSHMEM_MAX_TEAMS` environment variable](https://docs.nvidia.com/nvshmem/api/gen/env.html#nvshmem-max-teams)

## Platform Services

### IMEX Channel Missing or Inaccessible

- **Signatures:** on a multi-node NVLink/IMEX platform, CUDA or fabric-memory setup reports insufficient permission, a required IMEX channel device is missing, or access differs by user/container/node. A generic CUDA permission error without this platform context is not a confirmed match.
- **Likely cause and owner:** The system administrator has not created the IMEX channel device after boot, the job cannot access it through filesystem or device-cgroup permissions, users are assigned conflicting channels, or nodes in the IMEX domain are configured inconsistently. This is a platform administration issue, not an NVSHMEM application defect.
- **Action:** Ask the user to collect read-only output showing the `nvidia-caps-imex-channels` entry in `/proc/devices`, the character devices and permissions under `/dev/nvidia-caps-imex-channels/`, container device visibility, and the accessible channel on every node. Ask the administrator to verify IMEX service health and assign exactly the intended channel. Do not run `mknod`, change permissions, load drivers, or invoke `sudo`.
- **Verification:** The same user and container can access the intended channel consistently on all nodes and the fabric-memory initialization step succeeds.
- **Version applicability:** Verify that the target platform and CUDA/NVSHMEM combination actually uses IMEX before applying this entry.
- **Source:** [NVIDIA IMEX Channels guide](https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/imexchannels.html)

## Build and Package Integration

### CMake Cannot Find or Correctly Link the NVSHMEM Package

- **Signatures:** `find_package(NVSHMEM REQUIRED)` cannot find a package configuration despite an apparent installation, `nvshmem.h` or CUDA/CCCL headers are missing under a host compiler, host/device NVSHMEM symbols are unresolved, or manual include/library flags select files from different prefixes.
- **Likely cause and owner:** The development package or CMake configuration files are missing, CMake is resolving a different NVSHMEM prefix than the runtime, or the project bypasses the package's imported host/device targets and therefore loses transitive include/link requirements. This is normally package/application build integration.
- **Action:** Confirm the selected installation contains its CMake configuration and development headers, inspect CMake's resolved package directory and cache, and remove mixed-prefix hints. For documentation version 3.7.0, use config-mode `find_package(NVSHMEM REQUIRED)` and the documented `nvshmem::nvshmem_host` and `nvshmem::nvshmem_device` targets as needed instead of reconstructing their flags. Reconfigure in a fresh build directory. Route missing package content or older-layout questions to `$nvshmem-install`.
- **Verification:** CMake resolves NVSHMEM from the intended prefix, the imported targets exist, and a minimal host-only or host-plus-device program compiles and links without manually adding internal CUDA/CCCL include paths.
- **Version applicability:** The current imported-target names are documented for 3.7.0; older package layouts and target metadata can differ. Verify the installed release before prescribing names or transitive dependencies.
- **Sources:** [Integrating NVSHMEM into CMake projects](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html#integrating-nvshmem-into-cmake-projects), [building NVSHMEM applications and libraries](https://docs.nvidia.com/nvshmem/api/using.html#building-nvshmem-applications-libraries)

### CUDA 13 SM75 Package or Device-Link Failure

- **Signatures:** With an NVSHMEM 3.3.20 CUDA 13 package and an SM75 target, device linking reports undefined NVSHMEM device symbols such as `nvshmemi_device_state_d` or `nvshmemi_transfer_rma_*`, or package inspection shows no usable SM75 device code.
- **Likely cause and owner:** NVSHMEM 3.3.20 removed SM70 support in a way that also caused supported SM75 builds with CUDA 13 to fail. This is an NVSHMEM package/device-library defect, not an application kernel bug.
- **Action:** Confirm the complete NVSHMEM version, CUDA major version, target architecture, and that compile and link steps use one NVSHMEM prefix. If all conditions match, test a compatible NVSHMEM 3.3.24 or later package whose official release notes retain the needed CUDA/GPU compatibility; otherwise route a source-build architecture selection to `$nvshmem-install`. Do not generalize this fix to a non-SM75 link error.
- **Verification:** The same minimal SM75 device-link step succeeds with the candidate package and a device-side NVSHMEM smoke test runs on the target GPU.
- **Version applicability:** Official release notes identify 3.3.20 as the failing CUDA 13/SM75 release and 3.3.24 as restoring support. They do not establish that unrelated versions or architectures have the same cause.
- **Source:** [NVSHMEM 3.3.24 release notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3324.html)

## Versioned Issues and Fixes

### High-Signal Versioned Fix Leads

Use this table only to select release-note sections to verify. Require compatible logs and configuration; a similar symptom does not establish identity with the fixed bug.

| Fixed in | Candidate symptom or condition | Official release note |
| --- | --- | --- |
| 3.7.0 | Initialization hangs when PEs observe different NCCL availability | [NVSHMEM 3.7.0](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3700.html) |
| 3.7.0 | `nvshmem_ptr` segfault before symmetric-heap initialization | [NVSHMEM 3.7.0](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3700.html) |
| 3.7.0 | IBGDA RC multi-port endpoint or completion-queue indexing failure | [NVSHMEM 3.7.0](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3700.html) |
| 3.7.0 | All-to-all block-scoped warp quiet problem | [NVSHMEM 3.7.0](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3700.html) |
| 3.6.5 | Incorrect `NVSHMEM_TEAM_SHARED` on imbalanced P2P groups in MNNVL | [NVSHMEM 3.6.5](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3605.html) |
| 3.6.5 | PMIx bootstrap mishandles empty environment variables | [NVSHMEM 3.6.5](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3605.html) |
| 3.6.5 | libfabric build/runtime compatibility failure | [NVSHMEM 3.6.5](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3605.html) |
| 3.5.21 | ABI break related to the internal team structure | [NVSHMEM 3.5.21](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3521.html) |
| 3.3.24 | CUDA 13 device link fails for SM75 with 3.3.20 packages | [NVSHMEM 3.3.24](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3324.html) |

Before recommending an upgrade, establish the installed full version, verify the fix text live, and check the candidate release's known issues and compatibility against the user's CUDA, driver, GPU, CPU, NCCL, bootstrap, and transport context. A release note that only states a fix does not prove that every older release is affected. Describe the installed release as predating a candidate fix unless official evidence names it as affected or a controlled upgrade test confirms the match.

## Pre-Start and External Failures

### Clearly External Pre-Start Failure

- **Signatures:** shell syntax error, executable not found, permission denied before execution, missing application shared library reported by the dynamic loader, scheduler rejection, container startup failure, or launcher host/allocation validation failure before any PE enters NVSHMEM.
- **Likely cause and owner:** Shell, packaging, filesystem, scheduler, container, or launcher configuration. NVSHMEM debug settings cannot produce logs because NVSHMEM did not start.
- **Action:** Identify the component that emitted the error and provide its smallest direct check: resolve the executable and library path, validate allocation and host selection, or reproduce container startup independently. If the missing library is part of the NVSHMEM installation, route remediation to `$nvshmem-install`.
- **Verification:** The process reaches NVSHMEM initialization and begins emitting NVSHMEM output. If it then fails, return to the INFO logging gate.
- **Version applicability:** None until NVSHMEM starts, except package/layout details verified through `$nvshmem-docs` or `$nvshmem-install`.

## Primary Sources

- [Troubleshooting and FAQs](https://docs.nvidia.com/nvshmem/api/faq.html)
- [Environment Variables](https://docs.nvidia.com/nvshmem/api/gen/env.html)
- [Library Setup, Exit, and Query](https://docs.nvidia.com/nvshmem/api/gen/api/setup.html)
- [Memory Management](https://docs.nvidia.com/nvshmem/api/gen/api/memory.html)
- [NVSHMEM4Py Initialization](https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/initialization.html)
- [Current Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/index.html)
- [Prior Releases](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/index.html)
- [NVSHMEM Support](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-support.html)
- [NVIDIA IMEX Channels](https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/imexchannels.html)
