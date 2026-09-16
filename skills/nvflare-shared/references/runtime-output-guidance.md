# Runtime And Output Location Guidance

Use this reference when choosing generated source layout, export locations,
simulation workspaces, and validation output locations.

## Runtime Root

Generated source (`client.py`, `job.py`, `model.py`, `aggregators.py`) may sit
beside the training source. Simulator workspaces, exports, models, and logs go
to a host-provided runtime directory or one simple temporary directory (for
example `tempfile.mkdtemp()`). Report the exact paths in the final answer.

Filesystem hardening — per-user permission modes, ownership or symlink audits,
run manifests, and randomized directory hierarchies — is owned by the agent
host, not enforced by the skill. Do not construct these rituals.

## Generated Source Layout

When the current project or job source root is writable, use it as the default
generated FLARE source location. Add or update standard source files such as
`client.py`, `job.py`, `model.py`, `requirements.txt`, and small config files
beside the existing training files unless the user explicitly asks for another
target directory.

Do not create an extra wrapper folder such as `nvflare_jobs/<job_name>/` by
default. Preserve original training files such as `train.py` as references
instead of renaming or overwriting them unless the user explicitly asks for an
in-place rewrite.

An existing source file is not a collision when it is imported by or part of the
detected training program, regardless of its filename. Preserve and reuse
relevant source files by import, mechanical copy/rename into the generated
layout, thin wrapper, or targeted edit; add only missing FLARE entry points such
as `client.py`, `job.py`, and optional `aggregators.py` beside them.

## Read-Only Source Roots

If the source root is not writable — it may be owned by another user or mounted
read-only — read code and data from it but write generated source to a writable
directory: a user-provided path when available, otherwise the runtime directory.
Point the job at the original data through a configurable `train_args` value
rather than a path hardcoded in the generated code — see the "Data Location"
rule in `conversion-common.md` — and report both the read-only source root and
the exact generated job source location.

## Runtime And Export Outputs

Keep generated source and runtime artifacts separate. The source root may hold
generated source files, but simulation workspaces, exports, generated model
artifacts, caches, and validation results go to the host-provided runtime
directory (or one temporary directory), not the source root by default. If the
user provides another path, use it.

Defaults inside the runtime directory:

- exported job config: `<runtime-dir>/job_config/`;
- simulation workspace: `<runtime-dir>/workspace/`;
- generated validation outputs or evaluation records: `<runtime-dir>/results/`;
- diagnostic logs: `<runtime-dir>/logs/`;
- generated reports: `<runtime-dir>/reports/`.

If the user asks to keep runtime artifacts in the project workspace, do so and
report the exact chosen location. Otherwise keep generated runtime output
separate from source edits so code review and agent context are not polluted by
simulator or artifact files. Report the exact runtime, export, and result
paths.

## Export Directory

For generated `job.py` files that support NVFLARE export, use the exported job
config location selected above unless the user provides an export directory. The
exported job root is the exact directory later passed to
`nvflare job submit -j`. Report that exact directory in the final answer.
