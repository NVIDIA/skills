---
name: cuopt-developer
version: "26.06.00"
description: Contribute to NVIDIA cuOpt codebase including C++/CUDA, Python, server, docs, and CI. Use when the user wants to modify solver internals, add features, submit PRs, or understand the codebase architecture.
---

# cuOpt Developer Skill

Contribute to the NVIDIA cuOpt codebase. This skill is for modifying cuOpt itself, not for using it.

**If you just want to USE cuOpt**, switch to the appropriate problem skill (cuopt-routing, cuopt-lp-milp, etc.)

---

## Developer Behavior Rules

These rules are specific to development tasks. They differ from user rules.

### 1. Ask Before Assuming

Clarify before implementing:
- What component? (C++/CUDA, Python, server, docs, CI)
- What's the goal? (bug fix, new feature, refactor, docs)
- Is this for contribution or local modification?

### 2. Verify Understanding

Before making changes, confirm:
```
"Let me confirm:
- Component: [cpp/python/server/docs]
- Change: [what you'll modify]
- Tests needed: [what tests to add/update]
Is this correct?"
```

### 3. Follow Codebase Patterns

- Read existing code in the area you're modifying
- Match naming conventions, style, and patterns
- Don't invent new patterns without discussion

### 4. Ask Before Running — Modified for Dev

**OK to run without asking** (expected for dev work):
- `./build.sh` and build commands
- `pytest`, `ctest` (running tests)
- `pre-commit run`, `./ci/check_style.sh` (formatting)
- `git status`, `git diff`, `git log` (read-only git)

**Set up pre-commit hooks** (once per clone):
- `pre-commit install` — hooks then run automatically on every `git commit`. If a hook fails, the commit is blocked until you fix the issue.

**Still ask before**:
- `git commit`, `git push` (write operations)
- Package installs (`pip`, `conda`, `apt`)
- Any destructive or irreversible commands

### 5. No Privileged Operations

Same as user rules — never without explicit request:
- No `sudo`
- No system file changes
- No writes outside workspace

---

## Before You Start: Required Questions

**Ask these if not already clear:**

1. **What are you trying to change?**
   - Solver algorithm/performance?
   - Python API?
   - Server endpoints?
   - Documentation?
   - CI/build system?

2. **Do you have the development environment set up?**
   - Built the project successfully?
   - Ran tests?

3. **Is this for contribution or local modification?**
   - If contributing: will need to follow DCO signoff

4. **Which branch should this target?**
   - During development phase: `main`
   - During burn down: `release/YY.MM` (e.g., `release/26.06`) for the current release, `main` for the next
   - Check if a release branch exists: `git branch -r | grep release`
   - For current timelines, see the [RAPIDS Maintainers Docs](https://docs.rapids.ai/maintainers/)

## Project Architecture

```
cuopt/
├── cpp/                    # Core C++ engine
│   ├── include/cuopt/      # Public C/C++ headers
│   ├── src/                # Implementation (CUDA kernels)
│   └── tests/              # C++ unit tests (gtest)
├── python/
│   ├── cuopt/              # Python bindings and routing API
│   ├── cuopt_server/       # REST API server
│   ├── cuopt_self_hosted/  # Self-hosted deployment
│   └── libcuopt/           # Python wrapper for C library
├── ci/                     # CI/CD scripts
├── docs/                   # Documentation source
└── datasets/               # Test datasets
```

## Supported APIs

| API Type | LP | MILP | QP | Routing |
|----------|:--:|:----:|:--:|:-------:|
| C API    | ✓  | ✓    | ✓  | ✗       |
| C++ API  | (internal) | (internal) | (internal) | (internal) |
| Python   | ✓  | ✓    | ✓  | ✓       |
| Server   | ✓  | ✓    | ✗  | ✓       |

## Safety Rules (Non-Negotiable)

### Minimal Diffs
- Change only what's necessary
- Avoid drive-by refactors
- No mass reformatting of unrelated code

### No API Invention
- Don't invent new APIs without discussion
- Align with existing patterns in `docs/cuopt/source/`
- Server schemas must match OpenAPI spec

### Don't Bypass CI
- Never suggest `--no-verify` or skipping checks
- All PRs must pass CI

### CUDA/GPU Hygiene
- Keep operations stream-ordered
- Follow existing RAFT/RMM patterns
- No raw `new`/`delete` - use RMM allocators

## Build & Test

### PARALLEL_LEVEL

`PARALLEL_LEVEL` controls the number of parallel compile jobs. It defaults to `$(nproc)` (all cores), which can cause OOM on machines with limited RAM — CUDA compilation is memory-intensive. Set it based on your system's available RAM (roughly 4-8 GB per job):

```bash
export PARALLEL_LEVEL=8   # adjust based on available RAM
```

### Build Everything

```bash
./build.sh
```

### Build Specific Components

```bash
./build.sh --help      # Lists build options
./build.sh libcuopt    # C++ library
./build.sh libmps_parser libcuopt --skip-routing-build --skip-tests-build --skip-c-python-adapters --cache-tool=ccache  # native LP/MIP-focused build without routing/tests/adapters
./build.sh cuopt       # Python package
./build.sh cuopt_server # Server
./build.sh docs        # Documentation
```

### Run Tests

```bash
# C++ tests
ctest --test-dir cpp/build

# Python tests
pytest -v python/cuopt/cuopt/tests

# Server tests
pytest -v python/cuopt_server/tests
```

<!-- skill-evolution:start — always fetch datasets before running tests -->
#### Download test datasets before running tests

cuOpt tests depend on MPS/data files that are not checked into the repo. A
missing dataset surfaces as a `MPS_PARSER_ERROR ... Error opening MPS file`
test failure at 0ms — it is not a build or logic failure.

Before running any C++ or Python tests, follow the dataset download and
`RAPIDS_DATASET_ROOT_DIR` export steps in the repo's `CONTRIBUTING.md`
("Building for development" section) — that is the canonical list and mapping.

If a test fails with a missing-file error, run the matching download step from
`CONTRIBUTING.md` and re-run the test. Do not report missing-dataset failures
back to the user as the task outcome.
<!-- skill-evolution:end -->

## Python Bindings

cuOpt uses Cython to bridge Python and C++. See [resources/python_bindings.md](resources/python_bindings.md) for the full architecture, parameter flow walkthrough, key files, and Cython patterns.

## Before You Commit

### 1. Install Pre-commit Hooks

Run once per clone to have style checks run automatically on every `git commit`:

```bash
pre-commit install
```

If a hook fails, the commit is blocked — fix the issues and commit again. To check all files manually (e.g., before pushing), run `pre-commit run --all-files --show-diff-on-failure`.

### 2. Make Meaningful Commits

Group related changes into logical commits rather than committing all files at once. Each commit should represent one coherent change (e.g., separate the C++ change from the Python binding update from the test addition). This makes `git log` and `git bisect` useful for debugging later.

### 3. Sign Your Commits (DCO Required)

```bash
git commit -s -m "Your message"
```

### 4. Use Forks for Pull Requests

Never push branches directly to the main cuOpt repository. Use the fork workflow:

```bash
# 1. Clone the main repo
git clone git@github.com:NVIDIA/cuopt.git
cd cuopt

# 2. Add your fork as a remote
git remote add fork git@github.com:<your-username>/cuopt.git

# 3. Create a branch from the appropriate base (see branching strategy below)
git checkout -b my-feature-branch

# 4. Make changes, commit, then push to your fork
git push fork my-feature-branch

# 5. Create PR from your fork → upstream base branch
```

This applies to both human contributors and AI agents. Agents must never push to the upstream repo directly — provide the push command for the user to review and execute from their fork.

### Pull Requests Created by Agents

When an AI agent creates a pull request, it **must be a draft PR** (`gh pr create --draft`). This gives the developer time to review and iterate on the changes before any reviewers get pinged. The developer will mark it as ready for review when satisfied.

### PR Descriptions

Keep PR summaries **short and informative**. State what changed and why in a few bullet points. Avoid verbose explanations, full file listings, or restating the diff. Reviewers read the code — the summary should give them context, not a transcript.

## Coding Conventions

### C++ Naming

| Element | Convention | Example |
|---------|------------|---------|
| Variables | `snake_case` | `num_locations` |
| Functions | `snake_case` | `solve_problem()` |
| Classes | `snake_case` | `data_model` |
| Test cases | `PascalCase` | `SolverTest` |
| Device data | `d_` prefix | `d_locations_` |
| Host data | `h_` prefix | `h_data_` |
| Template params | `_t` suffix | `value_t` |
| Private members | `_` suffix | `n_locations_` |

### File Extensions

| Extension | Usage |
|-----------|-------|
| `.hpp` | C++ headers |
| `.cpp` | C++ source |
| `.cu` | CUDA source (nvcc required) |
| `.cuh` | CUDA headers with device code |

### Include Order

1. Local headers
2. RAPIDS headers
3. Related libraries
4. Dependencies
5. STL

### Python Style

- Follow PEP 8
- Use type hints
- Tests use pytest

## Error Handling

### Runtime Assertions

```cpp
CUOPT_EXPECTS(condition, "Error message");
CUOPT_FAIL("Unreachable code reached");
```

### CUDA Error Checking

```cpp
RAFT_CUDA_TRY(cudaMemcpy(...));
```

## Memory Management

```cpp
// ❌ WRONG
int* data = new int[100];

// ✅ CORRECT - use RMM
rmm::device_uvector<int> data(100, stream);
```

- All operations should accept `cuda_stream_view`
- Views (`*_view` suffix) are non-owning

Read existing code in `cpp/src/` for real examples of RMM allocation, stream-ordering, RAFT utilities, and kernel launch patterns.

## Test Impact Check

**Before any behavioral change, ask:**

1. What scenarios must be covered?
2. What's the expected behavior contract?
3. Where should tests live?
   - C++ gtests: `cpp/tests/`
   - Python pytest: `python/.../tests/`

**Add at least one regression test for new behavior.**

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Main build script | `build.sh` |
| Dependencies | `dependencies.yaml` |
| C++ formatting | `.clang-format` |
| Conda environments | `conda/environments/` |
| Test data | `datasets/` |
| CI scripts | `ci/` |

## Common Tasks

### Adding a Solver Parameter

1. Add to settings struct in `cpp/include/cuopt/` and wire into `set_parameter_from_string()` in `cpp/src/`
2. Expose in Python — if using the string-based interface, the parameter is auto-discovered (no `.pyx` change needed). Add a convenience method in `SolverSettings` if warranted. See [resources/python_bindings.md](resources/python_bindings.md) for the full checklist.
3. Add to server schema (`docs/cuopt/source/cuopt_spec.yaml`) if applicable
4. Add tests at C++ and Python levels
5. Rebuild: `./build.sh libcuopt && ./build.sh cuopt`
6. Update documentation

### Adding a Dependency

All dependencies are managed through `dependencies.yaml` — never edit `conda/environments/*.yaml` or `pyproject.toml` files directly. The file uses [RAPIDS dependency-file-generator](https://github.com/rapidsai/dependency-file-generator) format:

1. Find the appropriate group in `dependencies.yaml` (e.g., `build_cpp`, `run_common`, `test_python_common`)
2. Add the package under the correct `output_types` (`conda`, `requirements`, `pyproject`, or a combination)
3. Run `pre-commit run --all-files` — the RAPIDS dependency file generator hook regenerates downstream files automatically
4. Verify: check that `conda/environments/` and relevant `pyproject.toml` files were updated

### Adding a Server Endpoint

1. Add route in `python/cuopt_server/cuopt_server/webserver.py`
2. Update OpenAPI spec `docs/cuopt/source/cuopt_spec.yaml`
3. Add tests in `python/cuopt_server/tests/`
4. Update documentation

### Modifying CUDA Kernels

1. Edit kernel in `cpp/src/`
2. Follow stream-ordering patterns
3. Run C++ tests: `ctest --test-dir cpp/build`
4. Run benchmarks to check performance

## Common Pitfalls

| Problem | Solution |
|---------|----------|
| Cython changes not reflected | Rerun: `./build.sh cuopt` |
| Missing `nvcc` | Set `$CUDACXX` or add CUDA to `$PATH` |
| OOM during build | Lower `PARALLEL_LEVEL` (e.g., `export PARALLEL_LEVEL=8`) |
| CUDA out of memory | Reduce problem size |
| Build fails with CUDA errors on older driver | Conda installs `cuda-nvcc` for the latest supported CUDA (e.g., 13.1), but your GPU driver may not support it. Check with `nvidia-smi` — the top-right shows max CUDA version. Override with: `conda install cuda-nvcc=12.9` (or whichever version your driver supports). See [CUDA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/) |
| Slow debug library loading | Device symbols cause delay |

## CI Gotchas

| Failure | Cause | Fix |
|---------|-------|-----|
| Style check | Formatting drift | Run `pre-commit run --all-files` and commit fixes |
| DCO sign-off | Missing `-s` flag | `git commit --amend -s` (or rebase to fix older commits) |
| Dependency mismatch | Edited `pyproject.toml` or `conda/environments/` directly | Edit `dependencies.yaml` instead, let pre-commit regenerate |
| Skill validation | Missing frontmatter or version mismatch | Run `./ci/utils/validate_skills.sh` locally to diagnose |

For CI scripts and pipeline details, see [ci/README.md](../../ci/README.md).

## Canonical Documentation

- **Contributing/build/test**: [CONTRIBUTING.md](../../CONTRIBUTING.md)
- **CI scripts**: [ci/README.md](../../ci/README.md)
- **Release scripts**: [ci/release/README.md](../../ci/release/README.md)
- **Docs build**: [docs/cuopt/README.md](../../docs/cuopt/README.md)
- **Python binding architecture**: [resources/python_bindings.md](resources/python_bindings.md)

## Third-Party Code

**Always ask before including external code.** When copying or adapting external code, you must attribute it properly, verify license compatibility, and flag it in the PR. See the [Third-Party Code section in CONTRIBUTING.md](../../CONTRIBUTING.md#third-party-code) for the full process.

## Security Rules

- **No shell commands by default** - provide instructions, only run if asked
- **No package installs by default** - ask before pip/conda/apt
- **No privileged changes** - never use sudo without explicit request
- **Workspace-only file changes** - ask for permission for writes outside repo
