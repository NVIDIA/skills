<!-- SPDX-License-Identifier: Apache-2.0 -->

# CoRIM workflow

CoRIM support requires NVIDIA Attestation SDK 2.0.0 or later. Earlier releases
do not support this workflow. Check the installed subcommand's help because
available options vary by release.

## Collect CMW evidence

For a supported live NVML collection environment:

```bash
nvattest --format json collect-evidence-v2 \
  --evidence-source nvml > evidence.cmw.json 2> collection.log
```

Check the exit code immediately and validate that the file contains CMW JSON
before proceeding. Live NVML collection generates a nonce when none is supplied.
For a relying-party challenge, add `--nonce` with that challenge in hex and
retain its association with the request.

For an existing SPDM transcript, `collect-evidence-v2 --evidence-source
spdm-files` requires `--spdm-transcript-file` and `--cert-chain-file`.
Supply the expected nonce to enable the corresponding freshness check.
Check the release's Corelib support and hardware restrictions before using
`--evidence-source corelib` for non-CC collection.

## Blackwell FSP prerequisites

In the current implementation, the GPU-evidence and SPDM-transcript CMW
builders do not attach firmware hints. Collection can therefore succeed while
FSP appraisal still lacks required inputs. Check whether the installed release
supplies these fields. This implementation has no firmware-hint CLI option.

When evidence processing supplies no RIM locator, the FSP verifier attempts to
construct one using:

- The hardware model established from the device certificate chain.
- The device identifier measurement from SPDM block 52, including its `APSKU`
  tag and the following two-byte little-endian apSKU value.
- The evidence item's `hints.fwversion`, containing the actual firmware version
  as five dotted hexadecimal byte pairs. Its accepted pattern is
  `^[0-9A-Fa-f]{2}(\.[0-9A-Fa-f]{2}){4}$`.

Obtain the firmware value for the same device and evidence capture from the
collection environment. If it is unavailable, report that missing input or
establish an applicable backup locator. Do not substitute a sample value or
construct a guessed RIM URL.

For a producer-supplied hint, the selected evidence item's `hints` member must
be a CMW record: `["application/json", "<encoded-payload>"]`. The payload is
a UTF-8 JSON object with a string `fwversion` member. Encode those JSON bytes
as unpadded base64url; do not put the object directly in `hints` or encode only
the firmware string. When adding lookup metadata locally, work on a copy of
the CMW, select the actual evidence label, and preserve other hint members,
the transcript, certificates, and all nonces. Hints are untrusted lookup
metadata; they do not establish the firmware's authenticity or repair stale
evidence. Appraisal must still authenticate and compare the evidence.

Check `attest-v2 --help` for these conditional fallback inputs:

| Input | When it applies |
| --- | --- |
| `--backup-rim-locator <URI>` | A known-correct, deployment-supplied locator, used only if evidence processing and FSP synthesis leave no locator. Repeatable; it does not override an existing or synthesized locator. |
| `--backup-spdm-concise-evidence <path>` | Applicable caller-supplied CBOR mapping, used only when primary evidence yields no mapped evidence claims (ECTs). This implementation accepts tagged SPDM TOC (`#6.570`) or ConciseEvidence (`#6.571`) and maps the actual SPDM measurements. A TOC can also supply locators. |

These inputs are optional fallbacks, not routine requirements. Establish their
source and applicability to the device/firmware before using them. A backup
CoEV supplies measurement mapping, not a replacement CoRIM. Keep signature,
chain, and revocation checks enabled.

A missing hint can cause synthesis to fail and allow a configured backup
locator to be used. Malformed hint JSON, a non-object payload, or a non-string
`fwversion` instead produces `malformed hints record` before backup locators
are tried on this path. Correct the metadata encoding/type first.

For a synthesized locator, the verifier fetches CoRIM and CoEV from the RIM
service. If the response lacks CoEV and no backup CoEV was used, appraisal
reports `required CoEV is absent`. Obtain the applicable service collateral
or a supported backup CoEV; a firmware hint alone does not supply that mapping.
This requirement is specific to the synthesized-locator path. Do not treat
every absent optional CoEV as fatal.

Proceed once the required inputs are available. Report unresolved prerequisites
separately from a completed measurement mismatch or policy rejection.

## Appraise an existing CMW

After checking any applicable FSP prerequisites above, this example appraises
an existing CMW using the `cc-bounce-buffer` policy.
It produces an unsigned EAR because no signing key is supplied.

```bash
(
if nvattest --format json attest-v2 \
  --evidence-source file \
  --evidence-file ./evidence.cmw.json \
  --relying-party-policy builtin:cc-bounce-buffer \
  > result.ear.json 2> appraisal.log; then
  appraisal_rc=0
else
  appraisal_rc=$?
fi
printf 'attest-v2 exit code: %s\n' "$appraisal_rc"
exit "$appraisal_rc"
)
```

The subshell reports and returns the original command status without closing
an interactive parent shell. In a script using `set -e`, rejection or error
stops the script after the status is reported. If the caller needs to continue
inspecting artifacts, handle the subshell's nonzero status explicitly.
Match the policy to the deployment. `cc-tdisp` and custom Rego files
serve different requirements; inspect the selected policy before using it.

The current JSON output is the EAR object with an additional `ear_jwt` member
when supplied by the CLI, not an object nested under `ear`. Check the actual
output shape in the selected version before extracting fields.

| Interface | Result semantics |
| --- | --- |
| `attest-v2` exit 0 | Selected policy accepted the result |
| `attest-v2` exit 1 | Selected policy rejected the result |
| `attest-v2` exit 2 | Verifier/policy error or no usable result |
| SDK verification returns `NVAT_RC_OK` | A result was produced; inspect the EAR and apply policy |

Do not reuse this exit-code table for `verify-token`. Its EAR policy rejection
uses exit 2; see the command reference for that command's complete table.

Inspect `ear_status` and `submods.<label>.ear_status`. Labels identify evidence
items, not device types. Use the profiles and claims to identify device type.
Error details are under `submods.<label>.ear_nvidia_error_details`.
`ear_nvidia_evidence_rim_cmp` lists matched, mismatched, and unmatched
environments; it is not a dump of every raw measurement.

For example, the SDK can return `NVAT_RC_OK` with a `contraindicated` EAR and
error 30 (`NVAT_RC_EVIDENCE_MEASUREMENT_MISMATCH`). This is a completed negative
appraisal. Related environments can include dependency failures. A listed
component therefore does not necessarily have an independent digest mismatch.

## Sign and transport a result

For a deployed signer, add the existing protected key file and configured
issuer/key identifier to the appraisal command:

- `--ear-signing-key-file /etc/nvat/ear-signing-key.pem`
- `--ear-signing-issuer https://verifier.example.com`
- `--ear-signing-kid ear-signing-key-1`

These are example deployment values. The issuer must serve the corresponding
public keys at `/.well-known/jwks.json`. Share public keys, not the signing key.
Check for a nonempty `ear_jwt` before transporting the compact JWT.

At the relying party, verify the JWT with `verify-token --token-type ear`,
its configured trusted verifier URL via `--nras-url`, the expected challenge
via `--nonce`, and an appropriate `--relying-party-policy`. Obtain the trusted
URL from relying-party configuration, never from an unauthenticated token.
The `--nonce` check binds the collection nonce; check evidence-item freshness
and the policy's nonce requirements as well. A replayed file does not become
fresh simply because it was appraised again.

In the current file-input implementation, `attest-v2 --evidence-source file`
reads the CMW unchanged; adding `--nonce` to that command does not replace or
validate the file's challenge. Check the CMW collection and evidence-item
nonces against the original request through the supported verification flow.
The top-level EAR collection nonce alone does not prove that the same nonce
was authenticated in every evidence item.

## Official references

- [CLI command reference](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-cli/command-reference.html)
- [CoRIM verifier and C examples](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/corim_verifier.html)
- [CoRIM configuration and trust stores](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/corim_configuration.html)
- [EAR format](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/ear_format.html)
- [Rust CoRIM workflow](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-rust/corim_verifier.html)
- [Public releases](https://github.com/NVIDIA/attestation-sdk/releases)

For API work, use matching C headers or Rust bindings and these examples.
Check ownership, resource cleanup, and error returns at that revision. Do not
guess API functions from CLI option names.
