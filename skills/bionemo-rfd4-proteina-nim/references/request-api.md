# Public API contract and transport

Sources: [RFD4-Proteina product page and API schema](https://build.nvidia.com/ipd/rfd4-proteina)
and [local deployment example](https://build.nvidia.com/ipd/rfd4-proteina/deploy),
checked October 20, 2026. The published OpenAPI document identifies version 1.0.0.

## Endpoints and authentication

Hosted inference:

```text
POST https://health.api.nvidia.com/v1/biology/ipd/rfd4-proteina/generative
Authorization: Bearer <NVIDIA_API_KEY>
Content-Type: application/json
NVCF-POLL-SECONDS: 300
```

Get a key using the product page and set `NVIDIA_API_KEY` in the environment.
The bundled client also accepts `NGC_API_KEY` when `NVIDIA_API_KEY` is absent or
empty. It never writes the Authorization header to its transport metadata or
accepts the key as a CLI argument. Missing or malformed credentials fail before
any network request. Offline modes do not read credentials.

Local inference:

```text
POST http://localhost:8000/biology/ipd/rfd4-proteina/generative
Content-Type: application/json
```

No `/v1` prefix or API key is sent for local inference. `--base-url`, then
`RFD4_NIM_URL`, override the local host and port. The base URL must not contain
credentials, a query, fragment, or route. Hosted mode uses its fixed URL and
rejects `--base-url` so a hosted credential cannot be sent to an alternate host.

Submission uploads the complete request, including inline structures and
conditions, to the chosen service. Follow the product page's data-use terms;
use only input data permitted at that destination. Selecting `local` is not
proof of locality if its base URL points to another machine.

The helper makes one POST, refuses redirects, and does not retry automatically.
Its default 600-second timeout is per socket operation, not a completion promise.
The hold header asks the hosted service to wait up to 300 seconds before
returning a pending response. For self-hosted capability discovery use
`/v1/tasks`, `/openapi.json`, and `/v1/health/ready`.

## Request envelope

Fields are `task`, `input`, `sampling`, `partial`, `search`, `metrics`, and
`output`; `task` and `input` are required. Unknown top-level fields are rejected.
See [parameters](parameters.md) for the seven tasks and design controls.

The public [PD-1 example](../assets/requests/16_ppi_binder.json) embeds a complete
annotated mmCIF. To use a user-authored annotated input, construct a request
locally from that file, preserving all annotations and molecular context:

```python
import json
from pathlib import Path

source = Path("target_annotated.cif")
payload = {
    "task": "generation",
    "input": {
        "format": "annotated_cif",
        "file": {"filename": source.name, "content": source.read_text()},
    },
    "sampling": {"recycle_strategy": "thinking_medium", "num_replicates": 1},
    "metrics": {"tags": ["design"]},
    "output": {"structure_format": "mmcif", "include_sequence": True},
}
Path("request.json").write_text(json.dumps(payload, indent=2) + "\n")
```

This does not create conditioning in an ordinary mmCIF. The input must actually
encode what is fixed and what is designable. Template regions can contain
intentional NaN coordinates; generated output must have finite coordinates.

For an existing JSON condition spec, replace the input object:

```python
spec = json.loads(Path("condition_spec.json").read_text())
structures = [Path("target.cif")]  # Every structure referenced by the user's spec.
payload["input"] = {
    "format": "condition_spec",
    "spec": spec,
    "files": [{"filename": p.name, "content": p.read_text()} for p in structures],
}
Path("request.json").write_text(json.dumps(payload, indent=2) + "\n")
```

`spec` is an object, not serialized YAML or a full training configuration.
If supplied as YAML, parse it with a safe YAML parser first. Its `input` and
other structure references must resolve to attached bare filenames, not server
paths. Filenames are at most 255 characters. The helper performs basic
attachment checks; the serving implementation validates the actual conditions.

The API accepts mmCIF, not PDB. Use a chemistry-aware conversion and preserve
bonds, charges, and conditions if conversion is needed. Renaming a PDB file
is insufficient. Public RFdiffusion fields such as `contigs`, `input_pdb`,
`hotspot_res`, `diffusion_steps`, and `num_designs` do not belong in this envelope.

## Response and output controls

| Field | Meaning |
| --- | --- |
| `request_id`, `task`, `duration_ms` | Job identity, selected task, and reported duration. Retain transport IDs and hashes as well. |
| `model`, `runtime` | Checkpoints, backend, dtype, active oracles, software versions, hardware, and deployment limits. |
| `schedule` | Reported integration steps, recycle bands, trunk passes, and partial start times. |
| `search` | Search work report, or null for plain sampling. |
| `designs[]` | Results with opaque IDs, replicate indices, origins, seeds, structures, sequences, metrics, and diagnostics. |
| `designs[].structure` | `{format: "mmcif", filename, content}`, or null when suppressed. |
| `designs[].diagnostics.all_coordinates_finite` | A false value marks an unusable design. The helper checks the flag; independent parsing is separate. |
| `designs[].chain_of_interest` | Context for chain-specific metrics, not an automatic binder-chain assignment. |
| `aggregated_metrics[]` | Reduced metric values and their contributing count `n`. |
| `warnings[]` | Runtime notices and unavailable or adjusted behavior. |

`output.structure_format` accepts `mmcif` or `none`;
`output.include_sequence` defaults to true. Full sequence strings may include
non-protein `X` tokens. The helper assigns safe local filenames rather than
trusting server-supplied paths. It retains the original IDs and suggested names
in `summary.json`.

The published schema describes partial schedules after fraction scaling. Compare
request and response values and report any discrepancy; do not infer executed
work or cost from inconsistent metadata. Search rewards and geometry metrics do
not establish refolding confidence. `model.oracle` lists active predictors;
there is no built-in refolding oracle in the documented API.

## Errors and pending requests

NIM errors may carry `error.code`, `error.message`, and `error.details`; gateways
may return another JSON shape or text/HTML. The helper saves the raw body before
parsing it. An HTTP 200 with no designs or failed validation is not success.

HTTP 202 is pending. The helper retains `NVCF-REQID` and `X-Request-ID` when
present and exits 3. The product's synchronous example does not specify a result
retrieval route; use a documented completion mechanism for the actual deployment
before polling. Do not invent a URL or re-POST to try to obtain a response.

For 401/403, check key validity and product access. For 413, inspect deployment
limits. For 422, check attachments, tasks, schedules, partial/search settings,
and oracle availability. Preserve `Retry-After` on 429/503 and establish service
state before another submission; changing the endpoint is not error recovery.
