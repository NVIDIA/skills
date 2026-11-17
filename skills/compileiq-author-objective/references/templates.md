# Objective Function Templates

Paste-ready skeletons for the five compile paths CompileIQ supports today.
Each template is ≤80 lines and points at the in-tree canonical example for
deeper context.

> **All templates must be customized** for your kernel, problem size, and
> correctness reference. They are scaffolds, not finished objectives.

---

## 1. NVBench (CUDA source compiled via NVCC)

Canonical reference: [`examples/compilers/nvbench_example/optimize_reduction.py`](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/nvbench_example/optimize_reduction.py).

```python
import os, re, shutil, subprocess, tempfile
from pathlib import Path

from compileiq.ciq import Search, SearchConfiguration
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import INVALID_SCORE, ProblemType
from compileiq.utils.helpers import save_compiler_config

SCRIPT_DIR = Path(__file__).parent.resolve()
SOURCE_CU  = SCRIPT_DIR / "kernel.cu"
NVBENCH    = Path(os.environ["NVBENCH_PATH"])
ARCH       = "sm_100"


def build_and_run(acf_path: str | None, tmpdir: str) -> float | None:
    """Compile kernel.cu with optional ACF; run via NVBench; return P75 latency (s)."""
    exe = Path(tmpdir) / "bench"
    flags = [f"-arch={ARCH}", "-O3", "-std=c++17",
             f"-I{NVBENCH}/include", f"-L{NVBENCH}/lib",
             f"-Xlinker=-rpath,{NVBENCH}/lib"]
    if acf_path:
        # nvcc --apply-controls applies the ACF to every stage it lists, but only
        # for Blackwell or later targets (sm_100+). For earlier targets nvcc
        # rejects the flag, so fall back to ptxas-only injection.
        if int(ARCH.removeprefix("sm_").rstrip("af")) >= 100:
            flags += ["--apply-controls", acf_path]
        else:
            flags += ["-Xptxas", f"--apply-controls={acf_path}"]
    cmd = ["nvcc", *flags, str(SOURCE_CU),
           f"{NVBENCH}/lib/objects-Release/nvbench.main/main.cu.o",
           "-lnvbench", "-lcudart_static", "-lcuda",
           "-o", str(exe)]
    try:
        subprocess.run(cmd, check=True, timeout=120, capture_output=True)
        result_json = Path(tmpdir) / "r.json"
        subprocess.run(
            [str(exe), "-d", "0", "--stopping-criterion", "entropy", "--jsonbin", str(result_json)],
            check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # parse the JSON for your benchmark's P75 latency
        import json
        return json.loads(result_json.read_text())["benchmarks"][0]["devices"][0]["summaries"]["p75"]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def objective(config: str) -> float:
    with tempfile.TemporaryDirectory(prefix="ciq_") as tmpdir:
        acf_path = None
        if isinstance(config, str):                        # optimization run
            acf_path = str(Path(tmpdir) / "controls.acf")
            save_compiler_config(acf_path, config)
        score = build_and_run(acf_path, tmpdir)            # baseline if acf_path is None
        return score if score is not None else INVALID_SCORE


if __name__ == "__main__":
    tuner = Search(
        objective_function=objective,
        search_space=PtxasSearchSpace(version="13.3"),
        search_config=SearchConfiguration(problem_type=ProblemType.MIN, generations=10, pool_size=15),
        dump_results=SCRIPT_DIR / "results.csv",
    )
    results = tuner.start(num_workers=1)
    best = results.get_best_result()
    print(f"best P75: {best['score_1']*1000:.4f} ms")
    save_compiler_config(str(SCRIPT_DIR / "best.acf"), best["params"])
```

---

## 2. Triton (mixed user + compiler search space)

Canonical reference: [`examples/compilers/triton_example/mixed_triton.py`](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/triton_example/mixed_triton.py).

```python
import os, re, shutil, subprocess, tempfile
import torch, triton, triton.language as tl

import compileiq.search_spaces.base as ss
from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import BASELINE_CONFIG, INVALID_SCORE, SearchConfiguration, WorkerTypes
from compileiq.utils.gpu import gpu_benchmark_mode
from compileiq.utils.helpers import save_compiler_config

DEVICE = triton.runtime.driver.active.get_active_torch_device()
USER_CONFIGS = [
    {"block_m": 32, "block_n": 64, "block_k": 32, "stages": 3, "warps": 4},
    {"block_m": 64, "block_n": 128, "block_k": 32, "stages": 4, "warps": 4},
    # ... add the configs your kernel supports
]


@triton.jit
def my_kernel(...):  # your Triton kernel
    ...


DEFAULT_USER_CONFIG = USER_CONFIGS[0]   # what you would run without tuning


def run(a, b, acf_path, cfg):
    M, K = a.shape
    _, N = b.shape
    assert M % cfg["block_m"] == 0
    assert N % cfg["block_n"] == 0
    assert K % cfg["block_k"] == 0
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = ((M // cfg["block_m"]) * (N // cfg["block_n"]),)
    my_kernel[grid](
        a, b, c, M, N, K, ...,
        BLOCK_M=cfg["block_m"],
        BLOCK_N=cfg["block_n"],
        BLOCK_K=cfg["block_k"],
        num_stages=cfg["stages"],
        num_warps=cfg["warps"],
        ptx_options=f"--apply-controls={acf_path}" if acf_path else None,
    )
    return c


def measure(cfg, acf_path):
    a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16)
    ref = torch.matmul(a, b)
    try:
        out = run(a, b, acf_path, cfg)
    except Exception:
        return INVALID_SCORE
    if not torch.allclose(out, ref, atol=1e-2, rtol=0):
        return INVALID_SCORE
    return triton.testing.do_bench(
        lambda: run(a, b, acf_path, cfg), warmup=100, rep=1000, return_mode="mean"
    )


def objective(mixed_config) -> float:
    ptxas_path = shutil.which("ptxas")
    os.environ["TRITON_PTXAS_PATH"] = ptxas_path
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_path
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    # Baseline run: CompileIQ passes BASELINE_CONFIG ({}) even for a mixed
    # (list) search space, so check before unpacking. Measure the default
    # user config with no ACF.
    if mixed_config == BASELINE_CONFIG:
        return measure(DEFAULT_USER_CONFIG, acf_path=None)

    user_space, ptxas_config = mixed_config
    cfg = USER_CONFIGS[user_space["config_idx"]]
    with tempfile.NamedTemporaryFile(suffix=".acf", delete=True) as f:
        save_compiler_config(f.name, ptxas_config)
        return measure(cfg, acf_path=f.name)


if __name__ == "__main__":
    search_space = [
        {"config_idx": ss.range(0, len(USER_CONFIGS) - 1)},
        PtxasSearchSpace(version="13.3"),
    ]
    tuner = Search(
        objective_function=objective,
        search_space=search_space,
        search_config=SearchConfiguration(problem_type="min", generations=10, pool_size=32),
        dump_results="results.csv",
    )
    with gpu_benchmark_mode(clock_mhz=1965, raise_on_failure=False):
        results = tuner.start(task_timeout=20)
    best = results.get_best_result()
    user_space, ptxas_config = best["params"]
    print(f"best config index: {user_space['config_idx']}")
    save_compiler_config("best.acf", ptxas_config)
```

---

## 3. Helion

Helion is Meta's CUDA template language. Use Helion's official ACF API plus
`HELION_SKIP_CACHE=1` to force recompile per evaluation. See the canonical
softmax ACF walk-through at https://helionlang.com/examples/acfs/softmax_acf.html.

```python
import os, tempfile
import torch
import helion

from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import INVALID_SCORE, SearchConfiguration
from compileiq.utils.helpers import save_compiler_config

os.environ["HELION_SKIP_CACHE"] = "1"


@helion.kernel
def my_helion_kernel(x: torch.Tensor) -> torch.Tensor:
    ...   # your kernel


def reference(x: torch.Tensor) -> torch.Tensor:
    ...   # known-good PyTorch / NumPy reference


def objective(config: str) -> float:
    if isinstance(config, dict) and not config:
        acf_path = None
    else:
        with tempfile.NamedTemporaryFile(suffix=".acf", delete=False) as f:
            save_compiler_config(f.name, config)
            acf_path = f.name

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    ref = reference(x)

    try:
        # Pseudo: pass acf_path through Helion's ACF API.
        # Consult the Helion docs (link above) for the exact API in your Helion version.
        out = my_helion_kernel.with_acf(acf_path)(x) if acf_path else my_helion_kernel(x)
        if not torch.allclose(out, ref, atol=1e-2, rtol=0):
            return INVALID_SCORE
        # Use your preferred GPU timer (cudaEvent, torch.cuda.Event, do_bench, …)
        return measure(lambda: my_helion_kernel.with_acf(acf_path)(x) if acf_path else my_helion_kernel(x))
    except Exception:
        return INVALID_SCORE


if __name__ == "__main__":
    tuner = Search(
        objective_function=objective,
        search_space=PtxasSearchSpace(version="13.3", variant="att"),  # attention-friendly default
        search_config=SearchConfiguration(problem_type="min", generations=10, pool_size=15),
        dump_results="results.csv",
    )
    results = tuner.start(num_workers=1)
    best = results.get_best_result()
    save_compiler_config("best.acf", best["params"])
```

---

## 4. Raw PTX (you already have a `.ptx` file)

Use when you have a pre-generated PTX you want to tune at the PTXAS stage,
independent of any framework. Canonical reference:
[`examples/compilers/ptxas_example/ptx_spill.py`](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/ptxas_example/ptx_spill.py).

```python
import subprocess, tempfile
from pathlib import Path

from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import INVALID_SCORE, SearchConfiguration
from compileiq.utils.helpers import save_compiler_config

PTX_FILE = Path("kernel.ptx")
ARCH     = "sm_100"


def objective(config: str) -> float:
    with tempfile.TemporaryDirectory(prefix="ciq_") as tmpdir:
        acf_path = Path(tmpdir) / "controls.acf"
        cubin    = Path(tmpdir) / "kernel.cubin"
        save_compiler_config(str(acf_path), config)
        try:
            subprocess.run(
                ["ptxas", "-arch", ARCH, "--apply-controls", str(acf_path),
                 "-o", str(cubin), str(PTX_FILE)],
                check=True, capture_output=True, timeout=60,
            )
            # Replace with a real measurement: e.g. cuobjdump for register/spill counts,
            # or a tiny CUDA driver-API harness that times the cubin.
            return measure_cubin(cubin)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return INVALID_SCORE


def measure_cubin(cubin):
    ...   # your driver-API timer
```

---

## 5. cuTeDSL (incl. FA4 with TVM-FFI compilation)

cuTeDSL uses `cute.compile()` with an options string. Inject
`--ptxas-options '--apply-controls <path>'` through that string. If you can't
reach the `cute.compile()` call site (e.g., FA4's TVM-FFI path), monkey-patch
`CompileCallable.__call__`.

```python
import os, tempfile
import torch
from cutlass.base_dsl.compiler import CompileCallable

from compileiq.ciq import Search
from compileiq.search_spaces.compilers import PtxasSearchSpace
from compileiq.types import INVALID_SCORE, SearchConfiguration
from compileiq.utils.helpers import save_compiler_config

_original_call = CompileCallable.__call__


def _patched_call(self, *args, **kwargs):
    options = kwargs.get("options", "") or ""
    acf_path = os.environ.get("COMPILEIQ_ACF_PATH")
    if acf_path:
        ptxas_opts = f"--apply-controls {acf_path}"
        kwargs["options"] = f"{options} --ptxas-options '{ptxas_opts}'".strip()
    return _original_call(self, *args, **kwargs)


CompileCallable.__call__ = _patched_call


def objective(config: str) -> float:
    if isinstance(config, dict) and not config:
        os.environ.pop("COMPILEIQ_ACF_PATH", None)
        return measure_fa4()                # baseline
    with tempfile.NamedTemporaryFile(suffix=".acf", delete=False) as f:
        save_compiler_config(f.name, config)
        os.environ["COMPILEIQ_ACF_PATH"] = f.name
    try:
        return measure_fa4()
    except Exception:
        return INVALID_SCORE


def measure_fa4():
    # your FA4 / cuTeDSL kernel invocation + timer
    ...


if __name__ == "__main__":
    tuner = Search(
        objective_function=objective,
        search_space=PtxasSearchSpace(version="13.3", variant="att"),  # attention
        search_config=SearchConfiguration(problem_type="min", generations=10, pool_size=15),
        dump_results="results.csv",
    )
    results = tuner.start(num_workers=1)
    best = results.get_best_result()
    save_compiler_config("best.acf", best["params"])
```

---

## Picking a template

| You have | Start from |
|---|---|
| A CUDA `.cu` file you build with `nvcc` | NVBench template |
| A Triton kernel with autotuner configs | Triton template |
| A Helion kernel | Helion template |
| A pre-generated `.ptx` | Raw PTX template |
| FA4 or a cuTeDSL pipeline | cuTeDSL template |

After customizing, run the **O0/O3 canary** (see the parent SKILL.md) before
launching the full search.
