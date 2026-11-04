<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# PVA environment report

Schema authority: skill workflow

Generated at UTC: `<time>`

Verdict: `<READY | READY_WITH_LIMITATIONS | NOT_READY>`

## Request

- Requested capabilities: `<list>`
- Required build mode: `L4T`
- EMU policy: `<optional | required>`
- PVA generations: `<generations>`
- Project type: `<type>`
- Project root: `<absolute path or not applicable>`
- Optional features: `<features>`

## Environment

- Host operating system: `<value>`
- Host architecture: `<value>`
- PVA SDK root: `<path>`
- PVA SDK version: `<version>`
- PVA Solutions version: `<version or not applicable>`
- PVA Solutions source: `<repository/archive location and commit when available, or not applicable>`
- Selected target compiler: `<NVVPU | ASIP/Chess | none>`
- EMU status: `<AVAILABLE | UNAVAILABLE | UNPROVED>`
- EMU build mode: `<EMU | not proved>`
- EMU configure options: `<proved SDK, generation, compiler, and other CMake options, or not proved>`
- EMU probe evidence: `<configure/build/run commands, exit codes, and saved log paths, or why not run>`
- Packaged SDK documentation version: `<VERSION file value>`
- Generated HTML references: `<count>`
- Generated Programmer's Guide chapters: `<count>`
- Hardware target requested: `<yes | no>`

## SDK references

- `sdk_reference_root`: `<absolute WORK_DIR/references/sdk path>`
- `sdk_reference_lock_sha256`: `<SHA-256 of reference-lock.json after successful verification, or not verified>`

The `public-reference-corpus` check below records the verification command and
result for this directory and the selected SDK. Consumers must use this path.

## Checks

| ID     | Area     | Required     | Result                                       | Evidence                    | SDK documentation root, source path, and SHA-256 | Remediation        |
| ---    | ---      | ---:         | ---                                          | ---                         | ---                                      | ---                |
| `public-reference-corpus` | SDK references | yes | `<PASS / LIMITATION / FAIL>` | `<verify command and exit status, or why not run>` | `<SDK root, manifest, and lock paths>` | `<run pva-setup for missing or failed references, then rerun Doctor; or none>` |
| `pva-solutions-source` | PVA Solutions source | `<yes when selected / no otherwise>` | `<PASS / LIMITATION / FAIL / NOT_REQUESTED>` | `<absolute project path, required build files, version, and source metadata>` | `<local project metadata>` | `<invoke pva-setup for missing source, or none>` |
| `project-lfs-assets-<mode>` | Project assets | `<yes / no>` | `<PASS / LIMITATION / FAIL / NOT_AVAILABLE / NOT_REQUESTED>` | `<build mode, file-presence and pointer checks, affected paths, or why not needed>` | `<local build rules and LFS metadata>` | `<restore affected assets or none>` |
| `<ID>` | `<area>` | `<yes / no>` | `<PASS / LIMITATION / FAIL / NOT_AVAILABLE / NOT_REQUESTED>` | `<redacted command result>` | `<URL>`                                  | `<action or none>` |

Use one project-asset row per requested build mode when compilation is in scope.

Compute the verdict from rows with Required = yes: any `FAIL` means `NOT_READY`;
otherwise any `LIMITATION` means `READY_WITH_LIMITATIONS`; otherwise `READY`.
`NOT_AVAILABLE` and `NOT_REQUESTED` belong only to rows with Required = no.

## Failed checks

- `<check ID and reason, or none>`

## Limitations

- `<check ID and reason, or none>`

## Secret handling

- Secret values printed: `no`
- Credentials recorded: `no`

## Next action

`<one precise action or none>`
