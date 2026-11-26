<!-- SPDX-License-Identifier: Apache-2.0 -->

# Diagnostic collection and result triage

## Capture a file appraisal

Check the installed version's help first. The following Bash example assumes
an existing CMW and the deployment's approved Rego policy at `./policy.rego`:

```bash
(
if nvattest --format json --log-level trace attest-v2 \
  --evidence-source file \
  --evidence-file ./evidence.cmw.json \
  --relying-party-policy ./policy.rego \
  > diagnostic.ear.json 2> diagnostic.trace.log; then
  appraisal_rc=0
else
  appraisal_rc=$?
fi
printf 'attest-v2 exit code: %s\n' "$appraisal_rc"
exit "$appraisal_rc"
)
```

The subshell returns the original command status. With `set -e`, an automated
caller stops on rejection or error after the report is printed. An interactive
parent shell remains open. Handle the nonzero status explicitly if the caller
needs to continue collecting diagnostics.

Reproduce the invocation with its original policy selector, signing options,
and trust configuration. Replace the example filenames with the actual inputs
and fresh output paths. Do not overwrite the original evidence or diagnostics.
Review captures before sharing them.

The CLI writes result data to stdout and its SDK/CLI logs to stderr. Trace
messages can contain a timestamp/source prefix followed by JSON. A log file
is not a single JSON document. Extract the JSON payload of an identified
record before parsing it; do not assume `jq diagnostic.trace.log` will work.

File-based appraisal still needs references and certificate-validation inputs.
A later fetch can return a different manifest. Preserve the artifacts actually
used, their locators, versions, and available digests when investigating a
field mismatch. A diagnostic replay is a new run. It cannot prove the original
run's exact inputs or establish fresh device state.

## Legacy input triage

First distinguish the legacy collection wrapper from CMW and EAR. Hopper
legacy/TCG collection uses `collect-evidence --device gpu
--gpu-evidence-source nvml --format json` in a supported NVML/CC environment.
Its output contains `result_code`, `result_message`, and an `evidences` array.
Check the process exit code and wrapper result before reusing the evidence.

That wrapper is not input to `attest-v2`. For legacy saved-file appraisal,
the GPU evidence reader expects the `.evidences` array extracted after
successful collection, with the matching original nonce. Consult the installed
`attest --help` for `--gpu-evidence-source file` and `--gpu-evidence-file`.
Extracting the array or renaming the file does not convert it to CMW.

If CoRIM is required, establish whether the target release and evidence
profile support the required collection/conversion path. A schema mismatch
does not show that the GPU has bad measurements or that signatures should
be disabled. Apply the error/output semantics for the command actually used;
legacy collection's `result_code` is not an EAR status or a v2 policy exit code.

## Blackwell FSP prerequisite failures

Establish the responder and release first. Current Blackwell FSP CoRIM
appraisal may need lookup metadata or backup inputs that a successful CMW
collection did not provide. The current GPU/SPDM CMW builders omit firmware
hints; repeating the same collection does not add them.

When no locator is found during evidence processing, the verifier tries FSP
locator synthesis using the certificate-derived hardware model, the SPDM
block 52 device identifier/apSKU, and the evidence item's `hints.fwversion`.
Inspect the specific failure before changing an input:

| Observation | Next step |
| --- | --- |
| `malformed hints record` | On this synthesis path, invalid JSON, a non-object payload, or a non-string `fwversion` stops processing before backup locators. Correct the hint in a copy, preserving original evidence and other metadata. |
| `firmware-version hint does not match expected format` | Obtain the actual firmware for this device/capture. The string must contain five dotted hexadecimal byte pairs. An absent hint leaves an empty string; a missing or invalid-format string can fall back to a configured backup locator. |
| Device identifier empty, missing `APSKU`, truncated apSKU, or invalid hardware model | Check the original SPDM block 52 and certificate-derived model against the release's support. Report the missing/unsupported input; do not fabricate a measurement or model. |
| `required CoEV is absent` | For a synthesized locator with no used backup CoEV, the RIM response must include CoEV. Obtain applicable service collateral or an established backup mapping. A retrieved CoRIM alone is insufficient on this path. |

The item's `hints` must be an `application/json` CMW record whose second
element is unpadded base64url of the JSON object bytes. The decoded object's
`fwversion` is a string. A wrong media type or invalid CMW record can fail
before FSP synthesis. Hints are untrusted lookup metadata and do not prove
the measured firmware's identity.

Check the installed `attest-v2 --help` before suggesting fallback flags.
`--backup-rim-locator <URI>` uses a known-correct supplied locator only if no
locator remains after evidence processing and synthesis; it does not override
one. `--backup-spdm-concise-evidence <path>` supplies applicable CBOR mapping
only when primary evidence yields no ECTs. This implementation accepts tagged
SPDM TOC (`#6.570`) or ConciseEvidence (`#6.571`); a TOC can also supply locators.
Validate its provenance and applicability to the actual SPDM measurements.
It does not replace the CoRIM or override usable primary evidence.

Preserve the failed CMW and identify the actual missing input before deciding
whether new collection is needed. Retry appraisal with corrected lookup
metadata or applicable fallback inputs when the original evidence remains
suitable for the request. Keep signature, chain, revocation, and policy checks
enabled. These prerequisite errors do not establish a runtime/reference
measurement mismatch.

## Read the EAR

For the current CLI JSON shape, inspect:

- `ear_status` and each `submods.<label>.ear_status`;
- `submods.<label>.ear_nvidia_error_details`;
- `submods.<label>.ear_verifier_claims.ear_nvidia_evidence`;
- `submods.<label>.ear_verifier_claims.ear_nvidia_rims`;
- `submods.<label>.ear_verifier_claims.ear_nvidia_evidence_rim_cmp`.

The EAR is the top-level object; there is no required outer `ear` member.
Check the installed release's shape before using these paths. Missing optional
fields can mean the relevant stage did not produce a value.

| Observation | Interpretation |
| --- | --- |
| `NVAT_RC_OK` with a contraindicated EAR | The SDK produced a negative appraisal result |
| Error 30, `NVAT_RC_EVIDENCE_MEASUREMENT_MISMATCH` | Reference/evidence comparison failed; related environments can include dependency failures |
| Error 109, `NVAT_RC_RIM_LOCATOR_UNAVAILABLE` | No usable RIM locator after discovery and configured fallbacks |
| Error 104, `NVAT_RC_RIM_NOT_FOUND` | A locator did not yield the requested RIM |
| Error 107, `NVAT_RC_RIM_MEASUREMENT_NOT_FOUND` | The fetched RIM lacked required reference measurements |
| `attest-v2` exit 1 | Selected policy rejected the result |
| `attest-v2` exit 2 | Verifier/policy error or no usable result |
| EAR-mode `verify-token` exit 2 | Authenticated EAR rejected by policy |

Confirm numeric errors against the selected revision's public header. Do not
apply `attest-v2`'s exit table to another command.

For code 30, inspect `related_envs` and the comparison's `mismatched_env`.
Follow `failed_dependencies` in detailed diagnostics before claiming that
each listed component has its own bad measurement. An uncorroborated
attester claim is not a trusted fact solely because it appears in an EAR.

For loading errors, inspect the selected binary and libraries without changing
system-wide loader paths first. On Linux, use the installed package metadata
and `readelf -d` where available to inspect dependencies. Check Rust binding
build settings and native library revision together for Rust linking failures.

For time or certificate failures, inspect clock synchronization, chain status,
the correct trust store, and responder reachability. Keep cryptographic checks
enabled while fixing the cause. Local test fixtures with disabled checks must
remain labeled as unauthenticated development results.

## Official references

- [CLI command reference](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-cli/command-reference.html)
- [CLI installation](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-cli/introduction.html)
- [CoRIM verifier and result semantics](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/corim_verifier.html)
- [CoRIM configuration](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/corim_configuration.html)
- [EAR format](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/ear_format.html)
- [C public header](https://github.com/NVIDIA/attestation-sdk/blob/main/nv-attestation-sdk-cpp/include/nvat.h.in)
- [Rust introduction](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-rust/introduction.html)

Use version-matched docs/source. `/latest/` and public `main` can differ from
the installed release.
