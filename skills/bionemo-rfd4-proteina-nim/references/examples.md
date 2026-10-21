# Public request example

The bundled [PD-1 binder request](../assets/requests/16_ppi_binder.json) is the
complete JSON example from the public
[Deploy tab](https://build.nvidia.com/ipd/rfd4-proteina/deploy). It uses
`generation`, annotated mmCIF with a fixed PD-1 target and designable binder,
`thinking_medium`, one replicate, and `metrics.tags=["design"]`.
The [manifest](../assets/source-manifest.json) records the public source and hash.

This example can be prepared without a GPU or credentials. Hosted execution
requires `NVIDIA_API_KEY`; local execution requires a running NIM. It is a
published example, not evidence that this skill revision has run production
inference. Its target and annotations must not replace the user's own inputs.

## Prepare or submit

Run from the skill directory and use a new output directory for each command:

```bash
python scripts/design.py --mode hosted --dry-run \
  --request assets/requests/16_ppi_binder.json \
  --output-dir /tmp/rfd4-binder-prepared
```

When execution is requested, remove `--dry-run` and choose another output
directory. For an existing self-hosted service use `--mode local --base-url
http://localhost:8000`. The hosted API includes `/v1` in its path; local inference
does not. Credentials stay in environment variables, not JSON or CLI arguments.

## Change count and seed

Preserve the complete structure while changing only the requested controls:

```bash
python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("assets/requests/16_ppi_binder.json").read_text())
payload["sampling"].update(num_replicates=3, batch_size=1, seed=42)
Path("ppi-request.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
python scripts/design.py --dry-run --request ppi-request.json \
  --output-dir /tmp/rfd4-ppi-prepared
```

`batch_size=1` allows ordinary sampling replicates to use the documented
incrementing seeds. Changing batching changes replay behavior.

## Prepare search

To prepare a search using the same public structure, add a `search` object
as described in [parameters](parameters.md), using metric keys supported by
that deployment. For the published interchain hydrogen-bond reward example:

```python
payload["sampling"].update(num_replicates=3, batch_size=1, seed=42)
payload["search"] = {
    "step_checkpoints": [0, 30, 60],
    "n_branch": 2,
    "beam_width": 2,
    "reward_metrics_tags": ["interchain_interactions"],
    "weights": {"interchain_interactions.hbond_atom_pair_count_coi": 1.0},
    "clip_values": {"interchain_interactions.hbond_atom_pair_count_coi": {"max": 6.0}},
}
Path("search-request.json").write_text(json.dumps(payload, indent=2) + "\n")
```

This is a skill-authored adaptation of the public input and schema, not a
captured successful search. Check reward availability and whether its chain of
interest matches the intended binder. Use `--dry-run` when only preparation was
requested. Final survivors and retained intermediates can outnumber replicates;
replay the entire search to reproduce a result.

## Other design problems

The [public model card](https://build.nvidia.com/ipd/rfd4-proteina/modelcard)
and [Deploy tab](https://build.nvidia.com/ipd/rfd4-proteina/deploy) describe
motif, enzyme, small-molecule, antibody, nanobody, partial-diffusion, and
inverse-folding designs. Supply a complete input expressing those conditions;
changing the PD-1 example's task name does not create a valid input for another
problem. For a user-provided condition spec, follow [the API reference](request-api.md).

## Extract a saved response

```bash
python scripts/design.py \
  --request /path/to/original-request.json \
  --response-json /path/to/response.json \
  --output-dir /tmp/rfd4-extracted
```

This performs no network request and records `source=saved_response`. Supply
the actual request that produced the response; the helper checks basic
consistency but cannot independently authenticate that relationship.
