# Configuration Generation and Repair

For understood scalar changes to a valid template, use the entrypoint's small-edit
path: copy, edit requested values/dependent references and native-validate the
output. An explicit generator or renderer request takes precedence over this
shortcut. Polling, engine, memory/ownership and reorder changes can require
coordinated configuration and application changes even when one scalar changes.

For a new supported role/topology profile, read `docs/config-generation.md` and
the selected subcommand's help, then use the existing `raw-pair` or `socket-pair`
generator. Validate every emitted role. For custom/application templates, use
documented `render` support when appropriate or explicitly requested; preserve
application-owned sections. JSON Pointer overrides replace existing paths;
unresolved placeholders or unknown paths are errors. Check actual help rather
than invent advanced-mode flags. Do not add a parallel generator.

For a repair, diagnose native validation/startup errors and operational
constraints, then make the smallest supported repair; a validation failure does
not automatically require regeneration. Reuse supplied runtime and topology.
Hardware-free validation needs no NIC enumeration or dependency audit when the
compatible validator is supplied. Resolve only missing CPU/GPU affinity,
endpoint, memory, frame/buffer, queue and hugepage facts. Spark/GB10 UMA uses
`host_pinned`; IGX dGPU acquisition uses `device` packet/payload/output buffers.

For raw ibverbs MTU constraints or TX QP failures, read
[initialization limits](ibverbs-init-limits.md), even when semantic validation
succeeds. For mode changes, use [polling](polling-modes.md) or
[reorder selection](reorder-selection.md) only when relevant.

Raw ibverbs static RX flow IDs must be positive and unique. Preserve valid IDs;
allocate unused positive IDs for zero-valued flows, retaining queue-action
destinations. Update dependent reorder and understood application references
together. If duplicates or unknown consumers make the mapping ambiguous,
report it before changing it.

Write fresh outputs and preserve unrelated values/application sections. Show
the actual command/status and changed fields or flow-ID mapping. Use the supplied
native validator; if missing, identify the prerequisite and mark output
unvalidated. YAML parsing/stubs do not substitute for native semantic validation,
and validation does not prove hardware capability, delivery or payload integrity.
