# Long-Tail Gotchas

Reference material for `../SKILL.md`. The skill's
own Gotchas section keeps the top tier — failure modes that block
a build outright or silently produce wrong artifacts. Everything
else lives here: invariants, deploy patterns, performance hints.

## Source-of-truth invariants

- **The copied Makefiles are tooling.** For Branch B / C trees,
  `$KS/Makefile` and `$KS/kernel/Makefile` are free-floating
  copies from `$BSP_SRC`. For Branch A (mono-repo), the same
  files are tracked by the mono-repo's pristine commit — the
  `[ -f ]` guard in the "Common setup" step means the cp is a no-op there. If a
  future BSP release ships different Makefiles, `jetson-init-image`
  re-extracts the BSP and the "Common setup" step picks up the new
  copies on next build.
- **`kernel_src_build_env.sh` is the source of truth.** Don't
  hardcode `kernel-noble`, the OOT module list, or
  `KERNEL_DEF_CONFIG` anywhere — read them from the sourced env
  file. If you find yourself writing `KERNEL_SRC_DIR=kernel-noble`
  in the skill, you've duplicated NVIDIA's source of truth.

## DT-mode build mechanics

- **DT mode uses out-of-tree (`O=$KOUT`); other modes use in-tree.**
  DT-only is a documented optimization that needs only `prepare`
  artifacts (cheap to redo in `$KOUT`); kernel + OOT pollute the
  source trees with `.o`, `.config`, etc., which kbuild's own
  `.gitignore` covers. The source-repo dirty check in the "Detect dirty source repos" step
  uses `git status -uno` (ignore untracked) so build artifacts
  don't appear as dirty.
- **`KERNEL_HEADERS` vs `KERNEL_OUTPUT` in DT mode.** The BSP top-
  level Makefile's `nvidia-dtbs` target binds `srctree=$(KERNEL_HEADERS)`
  and `objtree=$(KERNEL_OUTPUT)` and then includes
  `$(srctree)/scripts/Makefile.compiler`. That file is a kernel
  *source* file — it does NOT get materialized into `$KOUT` by
  `make prepare`. So `KERNEL_HEADERS` must point at the kernel
  source dir (`$KS/kernel/$KERNEL_SRC_DIR`), not `$KOUT`. Setting
  both to `$KOUT` fails immediately with
  `No rule to make target '$KOUT/scripts/Makefile.compiler'`.
  Resist the urge to "simplify" by collapsing the two variables —
  they have different semantics (srctree vs objtree).

## Mode prerequisites + performance

- **OOT mode prereq: previously-built kernel headers.** `make
  modules` reads `$KS/kernel/$KERNEL_SRC_DIR/include/generated/`.
  Without a prior `make -C kernel`, the build dies on missing
  headers. Refuse with "run `/jetson-build-source kernel` (or
  `full`) first" instead of producing a confusing error.
- **`make -C kernel` is much cheaper on incremental than clean.**
  A single-file edit triggers one `CC`, one `LD`, a vmlinux
  relink, and an `Image` repackage. Don't `rm -rf` the kernel
  build state between runs — clean builds cost an order of
  magnitude more than incremental, for no gain.
- **DTSI edits in unrelated chip families still rebuild every
  DTB.** `nvidia-dtbs` is monolithic — touching
  `hardware/nvidia/t23x/` rebuilds Thor DTBs too. the "Write the build manifest" step's
  manifest filter (skip byte-identical against pristine) catches
  the no-op writes; the build cost itself is unavoidable until
  NVIDIA's DT Makefile gains per-chip targets.

## Build outputs + manifest

- **`obj-y` edits produce no `.ko`.** Files compiled into
  `vmlinux` (e.g. `drivers/thermal/thermal_core.c`) ship inside
  the kernel `Image`, not as separate modules. The manifest's
  trace logic routes these to `kernel/Image` — not to any
  `lib/modules/.../<file>.ko`.
- **A full kernel build produces many binary artifacts; the
  manifest is usually tiny.** A typical edit touches 1–3 source
  files; the manifest's entries map to the `.ko` / `Image` those
  files specifically affect. The rest of the rebuild is scaffold.
  If you ever find yourself emitting a manifest comparable in size
  to the full build's output set, the trace-to-dirty policy has
  been bypassed — fix the policy, don't loosen it.
- **Install consolidation is automatic only in `full` mode.** Most
  customizations don't need the install legs — the manifest
  entries in the "Write the build manifest" step are enough for `jetson-promote-image` to do
  per-file copies. Invoke with the `full` argument when you
  specifically want the `Source/.build/install-stage/` tree for
  manual deploy or layout diff against the shipped
  `rootfs/lib/modules/`.

## Manual deploy patterns (bypass Deploy)

These let you exercise a single edit on a running DUT in seconds
instead of going through promote → flash → validate. Pre-reqs:
DUT's `uname -r` matches the build's vermagic (e.g. `6.8.12-tegra`);
module signing not enforced
(`cat /sys/module/module/parameters/sig_enforce` → `N`).

| Artifact | DUT path | Reload |
|---|---|---|
| OOT `.ko` | `/lib/modules/<ver>/updates/<path>/<name>.ko` | `sudo depmod -a && sudo modprobe -r <mod>; sudo modprobe <mod>` (or reboot) |
| In-tree `.ko` | `/lib/modules/<ver>/kernel/<path>/<name>.ko` | same |
| `obj-y` change (in `Image`) | replace `/boot/Image` (back up first) | `sudo reboot` |
| DTB | flash the kernel-dtb partition (`flash.sh -k kernel-dtb …`) | reboot after flash |

Use `dev_info(&pdev->dev, "...")` for module probe paths;
`printk()` for `obj-y` paths. Check with `dmesg | grep <marker>`
after reboot.
