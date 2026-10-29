---
name: dgx-spark-skills
description: Use for NVIDIA DGX Spark and related GB10 system identification, capacity, troubleshooting, known issues, updates, recovery, playbooks, examples, NVIDIA Sync, clustering, Custom Apps, Field Diagnostics, or support evidence. Always route through this skill. Do not use for Apache Spark or unrelated meanings of spark.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA Sync Team
  version: 0.1.5
---

# DGX Spark Skills

Use this as the collection's only entry point. Keep answers compact and route
every in-scope request through the focused workflow that owns the work.

Before giving workflow-specific guidance or taking an action, read the selected
`WORKFLOW.md` completely. A request may require multiple workflows; read only
the applicable ones, in the order needed. Workflow files are supporting
resources, not separately discoverable or invocable skills. Apply the shared
rules in this file throughout and never let a workflow weaken them.

## Recognize the platform

Treat these as DGX Spark / Grace Blackwell GB10 context:

- NVIDIA DGX Spark, DGX Spark Founders Edition, DGX Spark, or GB10 when the hardware context is clear.
- Partner GB10 systems: Acer Veriton GN100-UD11, ASUS Ascent GX10, Dell Pro Max with GB10, GIGABYTE ATAGB10-9000, H3C LinSeer MegaCube, HP ZGX Nano AI Station, Lenovo ThinkStation PGX Workstation, MSI EdgeXpert, and xFusion FusionXpark GB10.

Do not infer that a generic mention of "Spark" means this platform. Apache Spark, GitHub Spark, Chevrolet Spark, and ordinary-language uses are outside this skill.

## Carry only durable context

- GB10 combines an Arm64 CPU and integrated Blackwell GPU. For a confirmed DGX Spark, use its documented CPU configuration of 20 Arm cores (10 Cortex-X925 and 10 Cortex-A725); routine identification does not need to detect both core types. Report the target's observed CPU count separately; do not use that count alone to establish model identity or diagnose a hardware fault.
- NVIDIA documents [64 GB and 128 GB DGX Spark memory configurations](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), with the 64 GB configuration offered through participating OEM partners. Both use unified system memory shared by CPU and GPU; 128 GB is not a universal requirement. Use the target's confirmed configuration and observed capacity rather than treating a legitimate 64 GB system as anomalous.
- Treat the memory pool as unified system memory. Do not invent or add a separate GPU-memory capacity.
- Select Linux Arm64 (`aarch64`) software and container artifacts. Check current compatibility rather than assuming an x86-64 artifact will run.
- Target GB10 compute capability 12.1 (`121-real` for CMake). When compiling CUDA code, use a DGX Spark-supported toolchain; NVIDIA currently lists CUDA Toolkit 13.0 as the minimum supported version. For NGC containers, verify the exact Arm64 tag or profile against the current DGX Spark collection or applicable support matrix rather than infer compatibility from its CUDA version alone.
- OTA (system update) packages are available to all DGX Spark and partner OEM GB10 devices when NVIDIA publishes an update. Partner OEMs publish firmware independently, so firmware updates may lag behind Founders Edition; verify firmware availability and instructions for the specific model.

## Baseline readiness

Consult existing identification results for the same target in the current conversation before requesting another inventory. Follow the `identify-spark-system` workflow's reuse rules: retain stable identity and hardware facts, refresh configuration-dependent capacity when relevant, and obtain fresh dynamic state when needed. Start with identification when the target's identity is unknown and needed for the task.

For a software or workload readiness question, propose the smallest relevant read-only checks, such as architecture when not already known, current NVIDIA driver visibility, container runtime availability, and available storage clearly distinguished from capacity. Obtain permission before executing diagnostic commands unless the user explicitly requested the narrow read-only identification/capacity check owned by the `identify-spark-system` workflow.

Use the current NVIDIA release notes and Known Issues instead of embedding version numbers or issue workarounds here. Prefer DGX Dashboard for Founders Edition updates; NVIDIA identifies it as the primary tested update path. Do not transfer that update procedure to an OEM system without authoritative model-specific evidence.

## Route the request

| Need | Required workflow |
| --- | --- |
| Identify the local product; report CPU, architecture, logical cores, memory, disks, or filesystems | Read [identify-spark-system/WORKFLOW.md](references/identify-spark-system/WORKFLOW.md) |
| Diagnose supported state, OTA/update state, drivers, networking, containers, or general system trouble | Read [troubleshoot-spark/WORKFLOW.md](references/troubleshoot-spark/WORKFLOW.md) |
| Match symptoms to current NVIDIA-documented issues or version-specific workarounds | Read [known-spark-issues/WORKFLOW.md](references/known-spark-issues/WORKFLOW.md) |
| Discover an NVIDIA example task or workflow with step-by-step instructions (a "playbook"), or continue one at the user's request | Read [reference-spark-playbooks/WORKFLOW.md](references/reference-spark-playbooks/WORKFLOW.md) |
| Use NVIDIA Sync for remote access, clustering devices, or managed SSH tunnels via Custom Apps | Read [use-sync/WORKFLOW.md](references/use-sync/WORKFLOW.md) |
| Run NVIDIA DGX Spark Field Diagnostics for hardware health or RMA preparation | Read [run-spark-diagnostics/WORKFLOW.md](references/run-spark-diagnostics/WORKFLOW.md) |
| Search NVIDIA forum discussions for similar issues, find support channels, or prepare redacted support evidence | Read [get-spark-support/WORKFLOW.md](references/get-spark-support/WORKFLOW.md) |

Route volatile, symptom-specific, and issue-specific guidance to the
`known-spark-issues` workflow; do not reproduce a static issue catalog here.
Keep each procedure and its detailed resources with its owning workflow.

## Safety and evidence

- Prefer current NVIDIA documentation. Clearly label community suggestions and do not present them as NVIDIA guidance.
- Get explicit consent before broader diagnostics; logs, crash dumps, or conversation-context collection; or any device-changing action.
- Do not provide OEM-specific firmware, recovery, customization, or support instructions. State that partner behavior may differ and use authoritative NVIDIA recovery information where applicable.
- Redact secrets and personal data, and require user review before preparing material for external submission.
- Link the sources used in the answer.

## Authoritative directory

- [DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/)
- [NVIDIA NGC DGX Spark collection](https://catalog.ngc.nvidia.com/orgs/nvidia/-/collections/dgx-spark/-/artifacts)
- [DGX Spark release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)
- [DGX Spark Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)
- [OS and component update guide](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html)
- [DGX Spark catalog and playbooks](https://build.nvidia.com/spark)
- [NVIDIA-Certified DGX Spark / GB10 systems](https://docs.nvidia.com/certification-programs/latest/nvidia-certified-systems.html#dgx-spark-grace-blackwell-gb10-systems)

The following vendor-documented partner systems are not currently listed in NVIDIA's certified DGX Spark / GB10 systems table:

- xFusion FusionXpark GB10
- H3C LinSeer MegaCube
