## Command appendix

The commands the verbs above expect the agent to issue or have the user
issue, grouped by class so the agent reaches for the right family
without searching prose. Each command appears with the verb it belongs
to and the H2 anchor in this file or in
[CAPABILITIES.md](../CAPABILITIES.md) that explains the expected output.

| Class | Command | Owning verb / anchor | Reads as healthy when … |
| --- | --- | --- | --- |
| Detect install | `pkg-config --modversion doca-common` | [`## test`](../TASKS.md#test) step 1 / [CAPABILITIES.md ## Version compatibility](../CAPABILITIES.md#version-compatibility) | Returns a single version string. Empty / "not found" ⇒ install or `PKG_CONFIG_PATH` is wrong. |
| Detect install | `pkg-config --list-all \| grep -i doca` | [`## configure`](../TASKS.md#configure) step 2 | Lists at least one `doca-*` module. Empty ⇒ install profile probably wrong. |
| Detect install | `doca_caps --version` | [`## test`](../TASKS.md#test) step 2 | Matches `pkg-config --modversion doca-common`. Mismatch ⇒ partial upgrade. |
| Detect install | `cat /opt/mellanox/doca/applications/VERSION` | [CAPABILITIES.md ## Version compatibility](../CAPABILITIES.md#version-compatibility) | Fallback when `pkg-config` is broken. Returns the same string. |
| Wire build | `pkg-config --cflags doca-<library>` | [`## configure`](../TASKS.md#configure) step 2 | Returns `-I/opt/mellanox/doca/infrastructure/include` (or trace-flavor equivalent). |
| Wire build | `pkg-config --libs doca-<library>` | [`## configure`](../TASKS.md#configure) step 2 / [`## run`](../TASKS.md#run) checklist | Returns `-L/opt/mellanox/doca/lib/<arch>-linux-gnu -ldoca_<library>` plus deps. |
| Wire runtime | `mount \| grep huge` | [`## configure`](../TASKS.md#configure) step 7 / [CAPABILITIES.md ## Observability](../CAPABILITIES.md#observability) | Shows a `hugetlbfs` mount. Empty ⇒ no hugetlbfs mount (pages may still be reserved; the `/proc/meminfo` row is the reservation check). |
| Wire runtime | `cat /proc/meminfo \| grep -i huge` | [`## configure`](../TASKS.md#configure) step 7 | Shows `HugePages_Total` > 0 and `HugePages_Free` > 0. |
| Wire runtime | `devlink dev show` | [`## configure`](../TASKS.md#configure) step 8 | Lists the expected BlueField / ConnectX PCIe BDF. |
| Wire runtime | `cat /sys/class/net/*/phys_port_name` | [`## configure`](../TASKS.md#configure) step 8 | Shows the representor port names for the device. |
| Wire runtime | `lsmod \| grep mlx5` | [CAPABILITIES.md ## Error taxonomy](../CAPABILITIES.md#error-taxonomy) | Shows `mlx5_core`, `mlx5_ib` loaded. |
| Detect mode | `mlxconfig -d <pcie> q INTERNAL_CPU_MODEL` | [CAPABILITIES.md ## Capabilities and modes](../CAPABILITIES.md#capabilities-and-modes) | Matches the mode the program expects (`EMBEDDED_CPU(1)` for switch mode). |
| Reach an install | `docker pull <ngc-doca-image>:<tag-from-catalog-tags-page>` | [`## no-install`](../TASKS.md#no-install) Path 0 | Pull succeeds; tag exists on NGC. |
| Reach an install | `docker run -it --mount type=bind,src="$HOME/dev",dst=/work <ngc-doca-image>:<tag-from-catalog-tags-page> bash` | [`## no-install`](../TASKS.md#no-install) Path 0 | Shell starts; `/opt/mellanox/doca` is populated inside. |

The appendix lists the *families* of commands; the verbs above describe
the *when* and *why*. New commands must come from a public NVIDIA source
(documented in [CAPABILITIES.md ## Capabilities and modes](../CAPABILITIES.md#capabilities-and-modes)
or `doca-public-knowledge-map/SKILL.md`);
agent-invented flags fail the bundle's anti-hallucination contract
declared in AGENTS.md `## Ground rules` (bundle `AGENTS.md#ground-rules-every-agent-must-follow`).

## Deferred task verbs

- **`install`.** Installing DOCA itself on a fresh host or BlueField is a knowledge-map question — the canonical Installation Guide and the package profile choice are documented there. Defer to `doca-public-knowledge-map ## Layout of an installed DOCA package` (`doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package`) and the Installation Guide URL it routes to. The fastest way to reach an install on a non-Linux host (or any host without DOCA) is [`## no-install`](../TASKS.md#no-install) Path 0 (NGC container).
- **Build / first-app derivation / running a program.** These are *programming verbs*, not env verbs, and live in `doca-programming-guide/SKILL.md`. After this skill's `## configure` and `## test` have produced a known-good environment, hand off there.
- **Library API specifics.** Constructing a Flow pipe, RDMA queue setup, etc. — outside the scope of this skill. After this skill's verbs have produced a known-good environment, hand off to `doca-programming-guide/SKILL.md` for the cross-library patterns and then to the matching library skill (e.g. `doca-flow/SKILL.md`) for API-level guidance.
- **`deploy` / `rollback` at fleet scale.** Provisioning multiple BlueFields, coordinated firmware/BFB updates across a fleet, or staged rollback is **fleet-orchestration scope**, externally owned and out of this bundle's SDK scope. Do **not** hand-roll per-host `bfb-install` loops or static-pod rollouts at that scale. Route to the orchestration entry-point in `doca-public-knowledge-map ## Deploying DOCA services at scale` (`doca-public-knowledge-map/references/services.md#deploying-doca-services-at-scale--orchestration-entry-point-personascale-routing`), which points at DOCA Platform Framework (DPF), NVIDIA Network Operator, and the Kubernetes Launch Kit with authoritative links. Single-host / handful-of-DPUs PoC deployment stays in-bundle (`doca-container-deployment/SKILL.md` / `doca-bare-metal-deployment/SKILL.md`).
