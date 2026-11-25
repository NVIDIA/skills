# Bounded Two-Candidate Example

When the user asks for a bounded number of approaches, for example "try two approaches", initialize with
`--max-candidates 2`; the baseline run does not count toward the cap:

```bash
python "$RUNNER" initialize ./job.py --metric accuracy --env sim --max-candidates 2
python "$RUNNER" prepare ./job.py --name <candidate-1> --hypothesis "<expected improvement>"
python "$RUNNER" evaluate ./job.py --manifest <candidate_manifest.json>
python "$RUNNER" prepare ./job.py --name <candidate-2> --hypothesis "<expected improvement>"
python "$RUNNER" evaluate ./job.py --manifest <candidate_manifest.json>
```

After the second evaluated candidate, campaign state reports `candidate_cap_exhausted` with
`final_response_allowed=true`; write the final summary from `autofl_report.md`.

Abandoned drafts (`status=abandoned`) never consume the cap; only evaluated candidates
(`keep`, `discard`, or `crash` ledger rows) count against `--max-candidates`.
