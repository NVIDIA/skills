---
name: nvat-troubleshooting
description: Use when NVAT installation, linking, evidence collection, CoRIM appraisal, EAR verification, or relying-party policy fails, or when investigating runtime versus reference measurements, measurement indices, and OID meanings.
license: Apache-2.0
---

# Troubleshoot NVAT

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Identify the failed stage and explain the failure from its evidence. Prefer the
CoRIM workflow for new deployments; diagnose the user's actual command when
they are using a legacy interface.

For Blackwell, FSP responders currently use CoRIM, while GSP responders use
TCG RIM. This may change depending on the deployment model; confirm support
for the installed SDK release. Establish the responder before selecting
format-specific diagnostics.

Identify the evidence schema before diagnosing a verifier failure. Hopper
legacy/TCG collection uses `collect-evidence`; its JSON wrapper contains
`result_code`, `result_message`, and `evidences`, not CMW. See
[legacy input triage](references/diagnostics.md#legacy-input-triage) when a
legacy collection file is supplied to `attest-v2`.

## Collect the failure context

Get the exact command or API call, NVAT/CLI version, OS and architecture,
container/CVM context, exit code, and error output. Inspect the relevant
subcommand help before suggesting flags. Preserve the failing input and
original diagnostics: recollecting evidence can change the observation.

Use existing collection/appraisal output and logs first. If a replay is
appropriate, reproduce with the same evidence input, policy, reference-manifest
versions, and trust settings.
Where supported, select `--format json --log-level trace` and capture stdout
and stderr separately. Read
[diagnostic collection](references/diagnostics.md#capture-a-file-appraisal)
before constructing the command. Treat logs and evidence as potentially
sensitive; omit service keys, private keys, and unnecessary identifiers from
anything shared outside the user's environment.

## Locate the failed stage

| Symptom | Investigate |
| --- | --- |
| Missing binary or v2 command | Installation, PATH, release capability |
| Missing native library or symbol | Loader search path and CLI/library version compatibility |
| GPU/NVML/permissions error | Device access in the actual collection environment |
| CMW parse or evidence authentication error | Input format, evidence profile, signature, chain, nonce |
| Manifest retrieval or certificate error | Locator, fetched artifact, HTTPS trust, signing trust, OCSP, clock |
| Measurement mismatch or negative EAR | Matched environment, reference candidate, measurement key, dependencies |
| Policy rejection | Selected policy, expected purpose/profile, required claims and freshness |
| Signed EAR verification failure | Trusted issuer/JWKS, key ID, signature, time claims, expected nonce |

Consult the relevant official references in
[diagnostics](references/diagnostics.md#official-references). Keep device identity,
CoRIM/CoEV signing, and HTTPS trust configuration distinct.

For FSP locator, firmware-hint, or missing-CoEV errors, use
[FSP prerequisite failures](references/diagnostics.md#blackwell-fsp-prerequisite-failures).
Successful collection alone does not establish that these appraisal inputs
are available.

## Explain measurement failures

Read [measurement comparison](references/measurements.md) for runtime/reference
hex reporting, index provenance, and OID lookup. For TCG results, use the
existing mismatch measurement and opaque-data records described there.
An EAR's environment summary does not contain every raw measurement. Trace
mismatch records can help, but may omit successful comparisons or the original
SPDM index. Request the
missing artifact instead of reconstructing it from an OID or array position.

## Resolve and verify

Recommend the smallest change the diagnosis supports, then rerun the affected
check when authorized and feasible. Keep evidence/manifest signatures,
revocation, and the intended policy enabled for deployed verification. Changing
a rejection into acceptance by disabling checks is not a repair.

Report the failed stage, observed facts, likely cause with supporting evidence,
proposed fix, and verification result. If blocked, name the exact missing input.
Distinguish an operational failure, completed negative appraisal, and policy
rejection. Do not claim the device passed merely because an SDK call returned
`NVAT_RC_OK`, a JWT decoded, or a command emitted JSON.

## Prepare a bug report

When the user wants to report a bug, follow
[bug-report guidance](references/bug-reports.md). Collect the relevant version
and environment details, reproduction steps, and failure output. Prepare
reviewed copies of the files needed to explain the failure and package only
those files. Keep the archive local until the user chooses to share it.
