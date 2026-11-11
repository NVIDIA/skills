# Public sources and version boundaries

This skill targets the released RFD4-Proteina NIM. The following public NVIDIA
sources were inspected on October 20, 2026:

| Source | Contract used by this skill |
| --- | --- |
| [Product page and API schema](https://build.nvidia.com/ipd/rfd4-proteina) | Hosted URL, Bearer authentication, hold header, client timeout, request/response schema. |
| [Machine-readable product page](https://build.nvidia.com/ipd/rfd4-proteina.md) | Public model description and hosted Python/curl examples. |
| [Deploy tab](https://build.nvidia.com/ipd/rfd4-proteina/deploy) | Public container, NGC credentials, port 8000, readiness, task discovery, and complete PD-1 binder request. |
| [Model card](https://build.nvidia.com/ipd/rfd4-proteina/modelcard) | Release date, model version, tasks, scientific scope, terms, and model-level evaluation disclosures. |
| [NGC container listing](https://catalog.ngc.nvidia.com/orgs/nim/teams/ipd/containers/rfd4-proteina) | Public release location linked from the model card; consult for available tags and digests. |

The public model card gives an October 20, 2026 release and version 1.0.0.
The API schema embedded in the product page also reports version 1.0.0.
The hosted snippets use:

- `https://health.api.nvidia.com/v1/biology/ipd/rfd4-proteina/generative`
- `Authorization: Bearer <NVIDIA_API_KEY>`
- `NVCF-POLL-SECONDS: 300` and a 600-second client timeout.

The Deploy tab uses `nvcr.io/nim/ipd/rfd4-proteina:latest` and the local route
`http://localhost:8000/biology/ipd/rfd4-proteina/generative`. A container tag is
not an immutable version identifier; record its resolved digest and the returned
model/runtime metadata when comparing runs.

## Bundled data provenance

Only the complete PD-1 request published on the Deploy tab is bundled.
[assets/source-manifest.json](../assets/source-manifest.json) records its source,
retrieval date, and SHA-256. The JSON is formatted consistently; the parsed
request, including the mmCIF content, matches the published example.

Search and replicate examples in this skill are explicitly described as
adaptations. They are not additional NVIDIA validation results. Unit-test
response fixtures are synthetic and test the client rather than model quality.
Upstream model and data terms remain separate from the toolkit's license.

## Validation and signing

[evals/validation-results.json](../evals/validation-results.json) records checks
of this public-service revision. [BENCHMARK.md](../BENCHMARK.md) tracks its live
agent evaluation status. Neither a published model card nor passing offline
client tests establishes an agent benchmark or a completed hosted design.

Earlier evaluation/signing artifacts do not attest to this revised endpoint,
authentication, dataset, and documentation. A new evaluation and service-account
signature are required before catalog onboarding can pass. No old signature is
retained as evidence for the modified bundle.
