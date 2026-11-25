# Fixture Source Notes

All fixtures in this directory are synthetic. They were authored for the
literature exploration-batch and candidate-diversification process evals and
carry no data from real campaigns.

The `*_state.json` fixtures follow the `nvflare.autofl.campaign_state.v1`
schema written by `skills/nvflare-autofl/scripts/campaign_guard.py`
(`guard_state_for_rows`). The `*_results.tsv` fixtures follow the ledger
columns in `skills/nvflare-autofl/scripts/run_job_campaign.py`
(`RESULT_FIELDS`). Each state fixture was verified to be byte-for-byte
reproducible (modulo `updated_at`) by replaying its paired ledger through
`guard_state_for_rows` with default thresholds, so the fixtures cannot drift
into states the product code would never emit.

- `develop_literature_batch_state.json` / `develop_literature_batch_results.tsv`:
  an uncapped simulation campaign with a baseline, four scored argument-only
  FedAvg attempts, a recorded literature event `lit-0001` (FedProx,
  arXiv:1812.06127), and one scored source-backed candidate linked to it. The
  batch is incomplete (1 of 3), so the guard derives
  `next_action=develop_literature_batch` with
  `required_exploration=source_backed_exploration` and an unreset plateau
  clock (`last_batch_completion_index=-1`).
- `diversify_candidates_state.json` / `diversify_candidates_results.tsv`:
  the same campaign shape after a completed `lit-0001` exploration batch
  (faithful implementation, tuned variant, ablation; adaptive server
  optimizer, arXiv:2003.00295) followed by six consecutive scored
  argument-only FedAvg attempts. The guard derives
  `next_action=diversify_candidates` from the default
  `family_repeat_limit=6`.

Scores, runtimes, and patch hashes are invented; hashes are fixed hex strings
so evals stay deterministic.

The `agent_instruction` strings are copied verbatim from the product
(`campaign_guard.py`), including the literature-batch instruction whose
trailing list ("a faithful implementation, a tuned variant, and an ablation")
enumerates the required batch *composition*, not the remaining work — the
remaining count is the leading "needs N more". Do not reword the fixtures to
read more naturally; they must stay byte-identical to guard output.
