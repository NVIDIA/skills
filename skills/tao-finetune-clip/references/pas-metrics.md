# CLIP training and PAS evaluation metrics

With `dataset.val.metadata_match_eval: false`, train logs `val/t2i_mAP`
(the packaged AutoML selection objective), and ordinary standalone evaluate
logs `test/t2i_mAP` for paired-caption retrieval. With
`dataset.val.metadata_match_eval: true`, PAS training instead logs
`val/pas/{easy,medium,hard,overall}_{mAP,rank1,rank5}`. It does not log
`val/t2i_mAP`; configure a PAS-specific selection metric before using AutoML
for this mode.

PAS standalone evaluate writes CSVs and returns before the ordinary test
loop, so it does not emit `test/t2i_mAP` or `test/pas/overall_mAP` scalars.
To compare against `val/pas/*`, evaluate the same checkpoint on the same
validation subset with `evaluate.pas_ground_truth_mode: scalar_attributes`.
Read `nvidia_pas_metadata_metrics_weighted_aggregate.csv` in the evaluation
results. `nvidia_pas_metrics_weighted_aggregate.csv` always contains the
paired-caption result, even when scalar-attribute mode is selected; comparing
it with `val/pas/*` mixes ground-truth modes. If a KPI requires
`test/pas/overall_mAP`, compute the query-count-weighted mean of the `mAP`
column across the `easy`, `medium`, and `hard` rows of the metadata weighted
aggregate CSV, using `num_queries` as weights. Evaluate does not log that
scalar directly.

