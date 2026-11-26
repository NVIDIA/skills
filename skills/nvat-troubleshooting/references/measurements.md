<!-- SPDX-License-Identifier: Apache-2.0 -->

# Runtime versus reference measurements

## Read TCG mismatch records

For legacy TCG appraisal, inspect the returned device claims before requesting
additional captures. CLI `attest --format json` puts those claims in its
`claims` array. Authenticated detached-EAT verification can also return the
claims. Reading an unverified file is useful for diagnosis, but does not
authenticate its contents.

GPU claims can contain `x-nvidia-mismatch-measurement-records`. Each record
includes `index`, `goldenSize`, `goldenValue`, `runtimeSize`, `runtimeValue`,
and `measurementSource` (`Driver` or `Firmware`). The value strings are already
hex in the current SDK implementation. Preserve leading zeros and report the
supplied sizes. `runtimeValue: "NA"` with runtime size zero means that the
expected runtime measurement was missing; it is not an all-zero digest.

The current SDK stores SPDM block index minus one in these TCG measurement
records. For example, record index 6 corresponds to SPDM block index 7.
Label both values and the convention. Verify that convention for the producing
SDK/service version; do not apply it to a CoRIM environment's `class.index`.

The verifier compares runtime measurements against all allowed golden
alternatives, but a mismatch record currently includes only the first golden
alternative. Retrieve the exact RIM if all alternatives are needed. These are
mismatch records, not a complete list of matched or unappraised values.

Also inspect `x-nvidia-mismatch-opaque-data-records` when present. Its numeric
MIN_SVN values describe a minimum threshold, not digest equality. Missing
runtime fields require a different explanation from an SVN below the minimum.

The synthetic legacy fixture in `evals/fixtures/tcg-mismatch.json` exercises
these distinctions. It is a claims fragment, not a signed attestation result.
For the implementation, see the matching revision of
[`claims.h`](https://github.com/NVIDIA/attestation-sdk/blob/main/nv-attestation-sdk-cpp/include/nv_attestation/claims.h),
[`GPU verification`](https://github.com/NVIDIA/attestation-sdk/blob/main/nv-attestation-sdk-cpp/src/gpu/verify.cpp),
and [`GPU evidence parsing`](https://github.com/NVIDIA/attestation-sdk/blob/main/nv-attestation-sdk-cpp/src/gpu/evidence.cpp).

## Read CoRIM comparison diagnostics

An EAR's `ear_nvidia_evidence_rim_cmp` lists environments that matched,
mismatched, or were unmatched. It does not provide a complete measurement dump.
From the EAR alone, report those environments and error details, then identify
the missing inputs needed for a measurement-level explanation.

The verifier's current trace output includes `CoRIM mismatch:` records with
serialized environment outcomes. Read `attempts[].reason.mkey_mismatches` for
the measurement key, mismatch fields, and available reference/evidence values.
`key_not_in_evidence` means absence, not a zero measurement.
`reason.failed_dependencies` identifies failures propagated from other
environments. Trace diagnostics describe candidates and failures; do not
assume they contain all successful evidence measurements.

For a full dump, obtain the original CMW/evidence, any SPDM transcript and
certificate chain, the exact CoRIM/CoEV artifacts used, and the applicable
profile/decoder. Decode with tools supporting those formats at that version.
If no suitable export or decoder is available, say which required values are
inaccessible.

## Correlate before comparing

Record each value with its evidence item/device, environment identity,
measurement key, algorithm, and reference candidate. Environments can carry
class ID, model, layer, class index, instance, and group. Preserve those fields
to avoid merging different devices or firmware components.

Distinguish:

- SPDM measurement-block index;
- environment `class.index`, including values derived from DICE TCB information;
- measurement-map key (`mkey`), which may be an OID, UUID, integer, text, or absent;
- position in an array, which has no implied hardware-index meaning.

Only report an SPDM index when the original record or an explicit
`spdm-indirect` mapping establishes it. Never substitute `class.index`, an OID
suffix, or an array offset. For multiple reference candidates, show each
relevant candidate and its outcome instead of selecting a convenient golden
value. A candidate's reference measurement is authoritative only within the
applicable authenticated manifest and appraisal context.

## Convert values to hex

Use the serializer/format definition for that field. In the current internal
comparison diagnostic format, a digest serializes as
`<algorithm>;<standard-base64-bytes>`; decode the suffix to bytes before
formatting hex. This is different from base64url used by some EAR fields.
Preserve leading zeros, byte length, and the original value and algorithm.
Do not hash the encoded string, reverse bytes, or interpret all fields as hex.
Binary version fields can already be hex; flags, SVN, and ranges need their
own typed representation. Confirm this schema in the matching public revision
before decoding another release's records.

Explain the verifier's actual comparison: digest equality by algorithm,
masked raw values, exact/minimum SVN, integer ranges, and constrained flags
have different semantics. An unknown, absent, malformed, or unavailable value
must be labeled explicitly. Reporting only that values differ can misdescribe
a valid masked or range comparison.

Use a report with these fields when available:

| Field | Content |
| --- | --- |
| Evidence item and environment | Device/label, class ID, model, instance, layer |
| Measurement | Original key, SPDM index and its source, separate class index |
| Reference | Manifest locator/digest and candidate identity |
| Values | Algorithm, evidence hex, reference hex or typed constraint |
| Outcome | Direct mismatch, absent value, candidate mismatch, or dependency failure |
| Mapping | OID name and mapping source/version, or unknown |

## Resolve OIDs

First identify the OID's role: an environment class identifier, measurement
key, certificate extension, or hash algorithm. These are different namespaces
and do not imply the same component name.

Use the relevant published profile, manifest metadata, or a local mapping
from the user. For a local mapping, record its source and version. Keep the
numeric OID alongside the friendly name. If no authoritative mapping exists,
report the name as unknown and ask for the profile or mapping. A local mapping
supplies display names only; it does not change the evidence, selected
references, or appraisal verdict.

For exact implementation details, inspect the matching public revision of
[`corim_verify.cpp`](https://github.com/NVIDIA/attestation-sdk/blob/main/nv-attestation-sdk-cpp/src/corim_verify.cpp)
and [`corim.cpp`](https://github.com/NVIDIA/attestation-sdk/blob/main/nv-attestation-sdk-cpp/src/corim.cpp).
The [CoRIM verifier guide](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/latest/sdk-c/corim_verifier.html)
describes DICE and SPDM conversion. The digest fixture in this skill's evals
is synthetic and tests formatting; it is not signed hardware evidence.
