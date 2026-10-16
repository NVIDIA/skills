# NN-score completion contract

AnomalyGenNext validation writes row-oriented `valid_kpi.csv` files. Read the
`Average` column from the `nn_score` row; higher is better. Do not substitute a
single type column, training loss, or `latest_checkpoint.txt`.

Completion requires all of the following:

- iteration `0` has a finite `Average.nn_score` baseline;
- at least one later score has a matching `iter_NNNNNNNNN.pt` model file;
- `best_checkpoint.txt` selects the maximum eligible NN score;
- that score is strictly greater than baseline plus the configured minimum;
- the published checkpoint and canonical recipe are hash-bound in
  `training_handoff.json` with the recipe's exact anomaly-type order.

A 1000-step run with validation/save at 1000 is the smallest quality smoke.
Shorter wiring tests cannot demonstrate NN improvement.
