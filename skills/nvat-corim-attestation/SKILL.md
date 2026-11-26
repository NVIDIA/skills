---
name: nvat-corim-attestation
description: Use when collecting CMW evidence or performing GPU attestation with the NVIDIA Attestation SDK CoRIM verifier, selecting an appraisal policy, or interpreting EAR results from nvattest.
license: Apache-2.0
---

# Attest with NVAT and CoRIM

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Use the CoRIM workflow: collect evidence into a Conceptual Message Wrapper
(CMW), appraise it against applicable CoRIM reference manifests, and interpret
the Entity Attestation Result (EAR) with the selected relying-party policy.
These interfaces are experimental and depend on the installed release.

For Blackwell, FSP responders currently use CoRIM, while GSP responders use
TCG RIM. This may change depending on the deployment model; confirm support
for the installed SDK release. If only the GPU family is known, establish
the responder before selecting this workflow.

This workflow requires CoRIM-compatible CMW input. For Hopper legacy/TCG
collection, use `nvattest --format json collect-evidence --device gpu
--gpu-evidence-source nvml` after checking the installed help and collection
prerequisites. Its `result_code`/`result_message`/`evidences` wrapper is not CMW.
Use the documented legacy `attest` workflow for that evidence; do not route
it into `attest-v2` merely because both formats are JSON.

## Establish capabilities and inputs

Check `nvattest version`, `nvattest collect-evidence-v2 --help`, and
`nvattest attest-v2 --help`. Match the release to the official
[command reference](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-cli/command-reference.html).
If v2 commands are absent, find a compatible public release or source build.
Legacy `attest` does not perform the same CoRIM workflow.

Determine the collection environment, evidence source, expected challenge
nonce, deployment purpose, and whether the result crosses a trust boundary.
Choose a policy from the deployment's acceptance requirements. Ask for its
intended purpose and approved policy when those are unknown. The current
`cc-bounce-buffer` policy accepts purposes `CC SPT` and `CC MPT`;
`cc-tdisp` accepts `TDISP`. Both also require the supported EAR/GPU profiles,
affirming statuses, at least one submodule, and evidence nonce checks.
Confirm these requirements in the installed release. `nvattest show-policy`
lists policies or prints their source; it does not detect the deployment or
choose its policy. Use a deployment-approved Rego file when a built-in policy
does not express the required checks. Do not weaken requirements to obtain
acceptance.

## Run appraisal

Read [the workflow](references/corim-workflow.md) for executable examples,
result semantics, signing requirements, and API documentation.

For live collection, verify supported hardware, driver/NVML access, and CVM
requirements. If the CMW already exists, appraise that file without
recollecting evidence. File input can still require RIM/CoEV retrieval and
OCSP access.
Describe a workflow as offline only after confirming that its external
dependencies are satisfied locally.

Select JSON explicitly with `--format json` when saving machine-readable
output. Preserve stdout, stderr, and the process exit code separately.
Check collection success before using the resulting file. Pass a relying-party
challenge nonce when the workflow requires freshness; a newly generated nonce
is not automatically bound to a remote relying party's request.

For Blackwell FSP, check the [FSP prerequisites](references/corim-workflow.md#blackwell-fsp-prerequisites)
before appraisal. Successful CMW collection does not supply all locator or
CoEV inputs. Establish the actual firmware hint or applicable backup inputs;
do not invent firmware values or reference locations.

Keep evidence and manifest signature verification, certificate-chain
validation, and revocation checks enabled for deployment. Use the built-in
NVIDIA production roots and default HTTPS trust for the standard deployment.
Add a custom root or TLS setting only when the deployment requires one and
its trust source is established. Device identity, CoRIM/CoEV signing, and
HTTPS trust use separate settings; change only the relevant one.

## Interpret and report

Distinguish three outcomes: the SDK produced a result, the EAR appraised the
evidence as `affirming` or `contraindicated`, and the selected policy accepted
or rejected it. An SDK success code alone does not establish device trust.
Report aggregate and per-submodule status, policy outcome, exit code, and
the relevant error details. Distinguish missing inputs and verifier errors
from a completed negative appraisal.

An unsigned EAR is useful for local development inspection. A remote relying
party needs an authenticated result, normally a signed EAR JWT verified
against its configured trusted issuer/JWKS and expected nonce. Decoding a JWT
or reading JSON does not authenticate it. Explain which signing/trust setup
is missing. Do not generate production keys or change trust configuration
implicitly.
