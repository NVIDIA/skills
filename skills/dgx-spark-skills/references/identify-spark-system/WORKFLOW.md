# Identify Spark System

Use this for local system identity or hardware-capacity questions. First consult any prior result for the same target in the current conversation using the reuse rules below. When identification or a capacity refresh is needed, run the bundled helper on that target:

```bash
python3 -B references/identify-spark-system/scripts/identify-spark-system.py
```

An explicit request to identify the system or report its capacity authorizes this narrow, read-only native check; do not add a separate confirmation. If the user asked only a general product question, answer it without running local commands. Never claim that the agent's host is the user's target device when they differ.

The helper reads `/opt/nvidia/dgx-oobe-customizations/customize.json`, standard read-only Linux interfaces, and read-only `util-linux` inventory commands. It writes nothing and emits stable JSON.

## Reuse and refresh

Reuse observations within the current conversation only when they are clearly associated with the same target. Retain the target, observation time when known, literal identity fields, classification, capacities, and warnings. Do not rerun merely because another Spark skill is invoked, and describe reused values as previously observed rather than freshly checked.

| Information | Refresh policy |
| --- | --- |
| Stable identity and hardware: product identity, architecture, configured logical cores | Reuse unless the target changes, a relevant change is reported, or the user requests a fresh check. |
| Configuration-dependent capacity: total system memory, attached disks, disk and filesystem capacities | Reuse for general context; refresh after relevant reconfiguration or when the task depends on current capacity. |
| Dynamic state: free/used disk space, available memory, utilization, current mounts, driver/runtime readiness | Obtain fresh observations when needed through the focused workflow that owns the task and its consent rules. The identification helper does not establish this readiness or collect utilization/free-space values. |

Disk capacity is distinct from free space; configured logical cores do not establish CPU availability to a process or container. Do not rerun the full inventory solely to answer a dynamic-state question.

If the target association is uncertain, establish the target before reusing or collecting data. Retry missing or failed fields when they become necessary for the task; preserve their warnings until refreshed. This reuse policy relies on conversation context and does not create a persistent cache.

## Interpret the result

1. Preserve the literal `company`, `deviceType`, and `infoURL` values read from `customize.json`, along with the reported identity read status.
2. Accept automatic NVIDIA classification only when both literal fields in the NVIDIA matcher in [spark-systems.yaml](references/spark-systems.yaml) exactly match the JSON values. Aliases support discovery only; never use them to identify the local system.
3. Report one of:
   - `nvidia-dgx-spark-founders-edition` for the exact NVIDIA payload.
   - `unknown` for missing, unreadable, malformed, incomplete, unsupported, unmatched, or ambiguous identity.
4. Do not infer an OEM from the vendor name, product name, chassis, CPU, GB10 presence, marketing aliases, or similarity to another system. State which literal fields were observed and what evidence is missing.
   An unknown local identity result does not invalidate a legitimate partner product.
   OEM catalog entries are product references only. Ask the user to confirm the
   exact model from their device label or purchase documentation and consult its
   authoritative vendor guidance. Label this as user-confirmed OEM identity;
   preserve the helper's automatic `unknown` result and raw observations.
5. Present `system_memory.capacity_bytes` as unified CPU/GPU system-memory capacity only for a recognized GB10 classification. For an unknown system, call it total system-memory capacity with GB10 semantics unverified. Never add a separate GPU-memory capacity.
6. Report the observed CPU architecture and logical-core count. Use the confirmed product identity to explain its documented CPU configuration below; the helper does not detect CPU models.
7. Describe block-device and mounted-filesystem values as capacities, not utilization or free space. The helper intentionally does not collect used-space percentages.

If a required command is unavailable, retain the partial result and its warnings rather than filling missing observations from nominal product specifications.

### Identity file unavailable

When `customize.json` cannot be read, explicitly state that automatic model
identification could not be completed from the current execution environment.
Preserve the helper's status and `unknown` classification. Warn that filesystem
sandboxing, container isolation, or file permissions can prevent access even
when the file exists on the Spark host. These are possible explanations, not
established causes; a missing-file result inside a restricted environment does not prove
host-level absence, an unsupported device, or a hardware fault. If JSON was read
but is malformed, report that separately rather than describing it as missing.

Give a concrete next step: offer to run the helper directly on the target host
through an authorized session, or ask the user to confirm the exact model from
its device label or purchase documentation. Record user-confirmed identity
separately from automatic classification. Do not bypass sandbox restrictions,
request privileged container access, or broaden filesystem access solely for
identification. Do not fabricate replacement identity JSON or recommend
reinstallation merely because this file is unavailable.

Continue with available inventory and generally applicable Spark guidance;
identify container-visible capacities as such rather than asserting host totals.
Confirm model applicability before recommending model-dependent firmware or
recovery procedures. Do not leave the response at `unknown` without explaining
the limitation and the next step.

### CPU configuration

For a confirmed DGX Spark, report NVIDIA's [documented configuration](https://docs.nvidia.com/dgx/dgx-spark/hardware.html): **20 Arm cores (10 Cortex-X925 + 10 Cortex-A725)**. No CPU-model detection or validation is needed. Keep this specification separate from the helper's observed logical-core count; a different visible count alone does not establish a hardware fault.

An exact NVIDIA identity match or user-confirmed DGX Spark model supports this answer. Preserve automatic `unknown` when relying on user confirmation. If identity remains unconfirmed, establish the model before assigning this specification.

### Memory configuration

DGX Spark configurations include 64 GB and 128 GB of unified memory; see
[NVIDIA's specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).
An identity match does not establish the installed-memory configuration. Preserve
the helper's classification and raw fields; do not infer a particular memory SKU
or OEM from capacity alone, or diagnose missing memory merely because a confirmed
64 GB system differs from a 128 GB specification.

The helper reports Linux-visible total memory from `MemTotal`, not free or
available memory. For example, 63,084,990,464 bytes is approximately 63.08 GB
(decimal) or 58.75 GiB (binary). Distinguish this observation from nominal installed
capacity. Memory reservations can reduce the visible total; current application
usage does not reduce `MemTotal`. Do not assert the cause of a difference without
evidence. If the confirmed installed configuration and observed total appear
inconsistent, explain the uncertainty and request consent for any further checks.
For workload sizing, use the actual target's capacity and obtain fresh available
memory when needed; do not carry over 128 GB workload-fit assumptions to 64 GB.

## Boundaries

The native identity/capacity helper is the only no-second-confirmation diagnostic in this workflow. Obtain explicit consent before any additional diagnostic execution, log or crash-dump collection, conversation-context collection, or device-changing action. This workflow does not modify firmware, packages, storage, filesystems, or settings and does not provide OEM-specific recovery or support procedures.
