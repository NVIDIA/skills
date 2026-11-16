# NVTX Instrumentation — Fortran

Language docs: `flavors/fortran/`. Nsys Round 0 uses this for marker
placement on **Fortran that actually runs** (`config.json:from_lang`).
Use NVTX3 from Fortran via `ISO_C_BINDING`.

NVTX is independent of the offload framework — these markers wrap
host call ranges. GPU kernels (OpenACC, OpenMP target, or later
CUDA/Kokkos after a language transition) still appear on the nsys
timeline from the runtime.

## Linking (discover — never hardcode paths)

CUDA 13+ / recent HPC SDK builds often **omit** `libnvToolsExt`.
Prefer **`libnvtx3interop`**; keep `-lnvToolsExt` as legacy fallback.

Discover the library directory at setup time. **Always append
`|| true`** to `-print-file-name` — it may exit non-zero under
`set -e` when the library is off the default search path.

```bash
COMPILER="${FC:-nvfortran}"

# Prefer NVTX3 interop (CUDA 13+)
NVTX_LIB=$("$COMPILER" -print-file-name=libnvtx3interop.so 2>/dev/null || true)
if [ -z "$NVTX_LIB" ] || [ ! -f "$NVTX_LIB" ]; then
  # -print-file-name may return empty or a bare filename — search SDK tree
  SDK_BIN=$(dirname "$(command -v "$COMPILER")")
  NVTX_LIB=$(find "$SDK_BIN"/.. -name libnvtx3interop.so -type f 2>/dev/null | head -1)
fi
if [ -n "$NVTX_LIB" ] && [ -f "$NVTX_LIB" ]; then
  NVTX_LINK=-lnvtx3interop
else
  NVTX_LIB=$("$COMPILER" -print-file-name=libnvToolsExt.so 2>/dev/null || true)
  if [ -z "$NVTX_LIB" ] || [ ! -f "$NVTX_LIB" ]; then
    SDK_BIN=$(dirname "$(command -v "$COMPILER")")
    NVTX_LIB=$(find "$SDK_BIN"/.. -name libnvToolsExt.so -type f 2>/dev/null | head -1)
  fi
  NVTX_LINK=-lnvToolsExt
fi

# Fail loudly if neither library was found — never emit bogus
# -L/dirname args (dirname of an empty string is ".").
if [ -z "$NVTX_LIB" ] || [ ! -f "$NVTX_LIB" ]; then
  echo "NVTX library not found via $COMPILER -print-file-name or SDK search" >&2
  exit 1
fi
NVTX_LIBDIR=$(dirname "$NVTX_LIB")
# Add to link line: -L"$NVTX_LIBDIR" $NVTX_LINK -Wl,-rpath,"$NVTX_LIBDIR"
```

Record `$NVTX_LINK` and `$NVTX_LIBDIR` in
`.nsight-ai-port/build_notes.txt`. **Never** hardcode machine-specific
absolute SDK paths in a performance build script.

Run `scripts/profile-check.sh {to_lang}-{framework}` during setup —
it prints NVTX link hints.

The wrapper below builds and links on the **CPU** path — no GPU
offload flags required. NVTX ranges are host-side.

## Thin Wrapper Module

Create a project-local module (e.g. `nvtx_markers_mod`):

```fortran
module nvtx_markers_mod
  use iso_c_binding, only: c_char, c_int, c_null_char
  implicit none
  private
  public :: nvtx_range_push, nvtx_range_pop

  interface
    integer(c_int) function nvtxRangePushA(name) bind(C, name="nvtxRangePushA")
      import :: c_char, c_int
      character(kind=c_char) :: name(*)
    end function nvtxRangePushA

    integer(c_int) function nvtxRangePop() bind(C, name="nvtxRangePop")
      import :: c_int
    end function nvtxRangePop
  end interface

contains

  subroutine nvtx_range_push(label)
    character(len=*), intent(in) :: label
    integer(c_int) :: ignored
    ignored = nvtxRangePushA(trim(label)//c_null_char)
  end subroutine nvtx_range_push

  subroutine nvtx_range_pop()
    integer(c_int) :: ignored
    ignored = nvtxRangePop()
  end subroutine nvtx_range_pop

end module nvtx_markers_mod
```

## Mandatory Naming Grammar

Use **only** these prefixes. Names are stable before any porting plan
exists — do not use `port:` (that implied post-plan tagging).

| Prefix | Use | Example |
|--------|-----|---------|
| `iter:` | The repeating unit of work — **the** measurement window | `iter:timestep` |
| `phase:compute:` | Major section that looks like compute | `phase:compute:dynamics` |
| `phase:host:` | Likely host-only (MPI, I/O, init) — a hint, not a plan veto | `phase:host:mpi_halo`, `phase:host:io` |
| `function:` | A specific source function (`file:function`) | `function:src/kernels.f90:kernel_a` |
| `block:` | Optional sub-region inside a function (e.g. one loop nest) | `block:kernel_a_k_loop` |

Rules:

1. **`function:`** — tag every function you suspect is hot **before**
   planning. Use `file:function` matching source layout (same strings
   as `porting_plan.json` will use later).
2. **`block:`** — optional; use when a function is large and you need
   time breakdown for one loop nest or section. Nest inside the
   matching `function:` range.
3. **`iter:`** — one push/pop per iteration of the outermost repeating
   unit of work. This is the range the profiler uses as its measurement
   window, so emit exactly **one** `iter:` name per program. If the
   program has no repeating unit (a single-shot pipeline), omit `iter:`
   entirely — the window then falls back to the `phase:compute:` span.
4. **`phase:compute:` vs `phase:host:`** — every phase MUST declare its
   class. `host` typically means MPI, I/O,
   allocation, init, teardown. The class still drives the default
   profiling window, so an unclassified or misclassified phase can
   corrupt the measurement: a host phase treated as compute can
   swallow the entire window and report 0% GPU activity.

Example:

```fortran
do iter = 1, niter
  call nvtx_range_push("iter:timestep")
  call nvtx_range_push("phase:compute:dynamics")
  call nvtx_range_push("function:src/driver.f90:timestep_driver")
  call kernel_a(...)
  call nvtx_range_pop()  ! function
  call nvtx_range_pop()  ! phase:compute
  call nvtx_range_push("phase:host:halo_exchange")
  call exchange_halos(...)      ! excluded from the window automatically
  call nvtx_range_pop()  ! phase:host
  call nvtx_range_pop()  ! iter
end do
```

Inside `kernel_a`:

```fortran
call nvtx_range_push("function:src/kernels.f90:kernel_a")
call nvtx_range_push("block:ij_loop")
! ... loops ...
call nvtx_range_pop()
call nvtx_range_pop()
```

Before **planning from an NVTX trace**, every translation target you
propose must have a matching `function:<file>:<function>` range in
the profile window (or add/refine tags and re-profile first).

## Git Hygiene

Commit NVTX instrumentation separately from GPU translation:

```bash
git commit -m "nvtx: mark problem compute path for profiling"
```

Do not strip all markers until this problem's porting effort is done.
