# NVSHMEM Command Execution and Validation

Read this reference before presenting installation commands, requesting execution approval, or validating an installation.

## Present Commands and Request Approval

Return the recommendation before any mutation:

1. Show the resolved NVSHMEM and documentation versions. Also show a distinct package, Python, HPC SDK, image, or bundled-library version when it differs.
2. Summarize the requirement verdicts and selected method.
3. Show one ordered command block with comments marking downloads, repository changes, builds, install-prefix writes, and environment-only steps. Keep interactive blocks free of persistent shell-option changes. Before invoking an executable from a binary archive, set `LD_LIBRARY_PATH` explicitly to include the resolved `PREFIX/lib` while preserving any existing value; do not assume archive executables have an RPATH.
4. State which commands require network access, elevated privileges, scheduler resources, or writes outside a user-owned prefix.
5. Show validation commands and their expected success signal. Each separate validation block must define every consumed value, including `PREFIX`, `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH`, and launcher variables. Do not rely on variables from an earlier block.
6. Unless the user asked for command-only guidance, explicitly ask in an interactive workflow whether to run the exact installation block. In a non-interactive workflow, default to command-only guidance and stop.

If the user requests command-only guidance or does not approve execution, finish with the commands and do not ask again. If execution is approved, run only validation commands included in that approved block unless the user asks otherwise.

## Validation

Always provide validation commands:

1. Confirm the installed artifact and actual NVSHMEM version. For C/C++, set <code>LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"</code>, then use `"$PREFIX/bin/nvshmem-info" -n -b`; a bare `nvshmem-info` can succeed silently. Apply the same explicit runtime-library path before any archive-installed binary.
2. Compile or configure a minimal C/C++ consumer against installed headers and libraries. Determine the target GPUs' CUDA architecture from the GPU-enabled target allocation and compile for that architecture; do not let `nvcc` fall back to its default architecture. For direct `nvcc` validation, pass both `-rdc=true` and `-arch=sm_<CC>` (for example, H100 has `<CC>` = `90`, so use `-arch=sm_90`). For CMake validation, set `CMAKE_CUDA_ARCHITECTURES=<CC>` and enable separable CUDA compilation for a target that uses NVSHMEM device APIs. Use NVSHMEM's imported CMake targets when available rather than reconstructing transitive device-link flags. If the probe found `CUDA_HOME` but not `nvcc` on `PATH`, invoke `"$CUDA_HOME/bin/nvcc"` directly or explicitly set `PATH`.
3. For Python, import `nvshmem`, report its version, and confirm native dependencies load.
4. Resolve and show the selected launcher and bootstrap before the smoke tests for every installation method, including PyPI and Conda. The baseline suite has two smoke-test slots: first, a two-PE same-node initialization/hello test with the selected compatible launcher; second, when a remote transport is enabled and two nodes are available, a two-node test with the selected bootstrap and default remote transport. Never silently omit a slot. If the same-node test cannot run because no compatible launcher, two GPU-visible PEs, scheduler access, or allocation is available, show its status as `blocked / unvalidated` with the specific missing prerequisite; do not substitute a Python import, native-library load, or single-PE run, and do not claim multi-PE runtime validation succeeded. If remote execution is intended but the two-node test cannot run, show it as `blocked / unvalidated` and state that the remote transport is unvalidated; show it as `not applicable` only when the resolved target intentionally has no remote transport. The two-node test's executable, input files, logs, and any launcher-visible working directory must be under a path mounted by both allocated nodes; never use `/tmp` for this test. Resolve and show that shared absolute path in the validation block.
5. Do not add alternate-transport, alternate-HCA, IBGDA, GPUNetIO, performance, or broader functional tests unless explicitly requested. Running baseline smoke tests still needs approval for the applicable command block.
6. For a container, inspect the bundled NVSHMEM version; do not infer it from the image tag.

## Output Contract

Use applicable explicit headings: `Target Summary`, `Resolved Version`, `Requirements Check`, `Recommended Method`, `Commands`, `Administrator Handoff`, `Source Configuration`, `Validation`, and `Caveats and Handoff`. Keep commands copyable, distinguish observations from assumptions, and include recorded administrator and target-scope authority in `Target Summary`.
