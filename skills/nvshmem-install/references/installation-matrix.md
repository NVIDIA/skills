# NVSHMEM Installation Matrix

Use this reference to select a method. Retrieve all exact package names, versions, URLs, image tags, and compatibility requirements through `$nvshmem-docs` before producing commands.

## Method Selection

| Method | Prefer when | Avoid or reconsider when |
| --- | --- | --- |
| Site-provided module | A managed cluster already supplies a compatible NVSHMEM and CUDA, or NVIDIA HPC SDK stack | The required version or features are unavailable, or the module is not visible on compute nodes |
| APT/network repository | The target is a supported Debian-family distribution and system packages are acceptable | The user lacks package-management authority, needs an isolated prefix, or requires a custom build |
| Local DEB repository | The target is Debian-family, artifacts are already available, or the environment is offline | Repository key/package provenance cannot be verified |
| DNF/YUM network repository | The target is a supported RPM-family distribution and system packages are acceptable | The user lacks package-management authority, needs an isolated prefix, or requires a custom build |
| Local RPM repository | The target is RPM-family, artifacts are already available, or the environment is offline | Repository/package provenance cannot be verified |
| OS-agnostic archive | A supported binary exists and a user-owned or shared prefix is preferred | The archive is incompatible with the target CUDA, CPU architecture, libc, or desired features |
| PyPI | The user needs NVSHMEM4Py in an isolated Python environment | No released object matches the Python, CUDA, CPU, or base NVSHMEM combination |
| Conda | The user already manages the workload with Conda and a compatible package is published | Channel/package compatibility cannot be established |
| HPC SDK container | The user wants a reproducible environment without installing NVSHMEM on the host | The image's bundled NVSHMEM version is unknown or does not satisfy a pinned request |
| CMake source build | The user needs custom transports, bootstrap support, architectures, packages, patches, or install prefix | A supported released object already meets the requirements and simplicity is preferred |

When no method is requested, recommend in this order:

1. Use PyPI or Conda for a Python-only request when the live documentation confirms a compatible released object.
2. On a managed cluster, use a compatible site-provided module when available.
3. Use a network package repository for C/C++ on a supported distribution only when the user has affirmed administrator rights and authority over the target scope.
4. Use an HPC SDK development container when host isolation or reproducibility is the priority and cluster policy permits it.
5. Use an archive for an unprivileged prefix when a compatible archive exists and every participating node can access it.
6. Build from source only when released objects do not satisfy the target or requested feature set.

If administrator status is unknown, ask once in an interactive workflow. If it remains unknown or the workflow is non-interactive, treat it as unprivileged for planning: consider only a compatible site module, an approved container, or a user-owned prefix whose write authority is established, and do not select a system-package method. For an unprivileged cluster user, do not present APT, DEB, DNF, YUM, or RPM as the recommended path; provide those details only as an administrator handoff when needed.

For a CMake source build, use the selected release's source archive or tag from the official NVIDIA NVSHMEM GitHub releases page (`https://github.com/NVIDIA/nvshmem/releases`).

## Binary Packages and Archives

- Distinguish runtime libraries from development headers and static libraries. Install only the pieces required by the workload.
- Prefer the network repository for an unpinned latest installation when the official guide recommends it. Prefer a local repository or downloaded archive for offline or controlled deployments.
- If a repository candidate is newer than the latest matching NVSHMEM documentation, report `Documentation Version` and `Candidate Artifact Version` separately. Default to the newest fully documented release and ask before choosing the newer candidate. If the user does not answer or the workflow is non-interactive, keep the fully documented release selected and mention the newer candidate only as a caveat.
- Pin every relevant package when the user requests a specific NVSHMEM or CUDA combination. Do not mix an unpinned dependency with a pinned NVSHMEM package.
- Copy direct artifact URLs only from retrieved official pages. Do not synthesize repository paths or filenames from apparent naming conventions.
- Verify CPU architecture naming expected by the selected repository or archive; do not translate architecture labels from memory.
- For a shared cluster prefix, verify that all compute nodes see the same files and compatible host drivers. Do not modify a modulefile unless separately requested and approved.
- Include persistent environment instructions only when required, and label shell-local exports separately from installation commands.

## Python

- Prefer a new virtual environment or an existing user-selected environment. Never alter the system Python by default.
- For a Python-only NVSHMEM4Py request, reject `binary archive` and `system repository/package` as installation methods. Use PyPI or Conda for released bindings; use a source build only when explicitly requested. A native NVSHMEM archive or system NVSHMEM package is not an NVSHMEM4Py installation artifact.
- Treat NVSHMEM4Py's PyPI/Conda version as independent from the NVSHMEM runtime/library version. Report both when available; do not apply the newer-runtime-candidate rule merely because the binding package has a numerically newer or differently formatted version.
- Verify the Python interpreter, CUDA family, CPU architecture, base NVSHMEM compatibility, and wheel/Conda availability through the versioned NVSHMEM4Py documentation.
- Determine from current package metadata whether the Python object supplies or depends on NVSHMEM native libraries. Do not assume that a successful wheel download guarantees the native runtime can load.
- When both APIs are requested, select compatible C/C++ and Python artifacts from the same supported release combination.
- Use a source build only when no compatible released object exists or the Python bindings must match a custom NVSHMEM build.

## HPC SDK Containers

- Use the official NGC catalog to select a current or pinned tag. Never treat an HPC SDK tag as an NVSHMEM version.
- Prefer a development image for compiling applications. Use a runtime image only for an already-built workload whose runtime dependencies match it.
- Inspect the selected image or running container to establish the actual bundled NVSHMEM version before declaring success.
- Verify host driver compatibility, GPU passthrough, CPU architecture, and Docker or Apptainer/Singularity availability without privilege escalation.
- For multi-node use, account for the site's scheduler integration, network device exposure, host libraries, mounts, and container policy. Do not invent generic fabric passthrough flags.
- Pulling or building an image is an installation action. Present the exact command and request explicit approval before running it.

## Validation Ladder

Tailor commands to the selected method and launcher:

1. **Artifact check:** query the package/environment or inspect the prefix; report the actual installed version.
2. **Consumer check:** configure or compile a minimal C/C++ consumer, or import NVSHMEM4Py and load its native bindings.
3. **Same-node runtime check:** launch two PEs on two reachable GPUs and complete initialization/finalization.
4. **Multi-node check:** when relevant, launch one PE on each of two nodes and exercise the intended transport.

Use a temporary build directory for consumer checks. Validation must not install new dependencies or write to a permanent prefix. State the expected exit status and output for each test.

For shell preflight commands, avoid improvised version comparisons whose ordering is easy to reverse. Prefer a package-native version comparator; otherwise report the observed and required versions separately for explicit evaluation. Check every generated validation command for shell correctness before returning it.
