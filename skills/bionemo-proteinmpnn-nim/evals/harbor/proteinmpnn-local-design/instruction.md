Use the local ProteinMPNN NIM that is already running in this task container to
design protein sequences for `/workspace/input/1R42.pdb`. Do not call a hosted
API.

Run a real inference request with all of these settings:

- exactly 3 designed sequences
- random seed `1`
- sampling temperature `[0.1]`
- standard model (`use_soluble_model=false`)
- full-backbone mode (`ca_only=false`)

Read the PDB contents into the request; do not send the file path as the input.
You may inspect the local service's health or OpenAPI endpoints if needed.

Create `/workspace/output` and save:

- `request.json`: the exact JSON request body
- `response.json`: the complete JSON response from the NIM
- `designed_sequences.fa`: the response Multi-FASTA
- `summary.json`: the endpoint used, designed-sequence count, and each designed
  sequence with its corresponding score

Execute the request and report a concise summary of the actual response. A
script that is written but not run does not satisfy the task.
