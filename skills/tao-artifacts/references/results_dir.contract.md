# results_dir layout contract

Storage-tier-agnostic layout for every job's outputs:

```
<root>/<job_id>/<output_key>/...
```

- `<root>` = `TAO_RESULTS_ROOT` (a persistent mount — tier A/B) **or**
  `s3://$S3_BUCKET_NAME/results` (tier C upload target).
- `<job_id>` = the id minted by `tao_job_record.py open`.
- `<output_key>` = each logical `declared_outputs[].spec_key` from the
  spec-bundle. When that entry declares `relative_path`, its concrete artifact
  is `<job-root>/<relative_path>`; otherwise the legacy concrete path is
  `<job-root>/<output_key>`. `results_dir` itself maps to the job root.

Rules:

1. **Resolved at submit, recorded once, never re-derived.** The concrete
   `results_dir` is written into the job-record by `open` — before launch.
   K8s `ttlSecondsAfterFinished` and `docker --rm` erase the backend object,
   so the record is the only durable pointer to the outputs.
2. The value authored into the spec's `results_dir` field is the
   **compute-frame** path (the mount path inside the container, or the local
   path that tier C uploads from).
3. Tier C: the upload (`aws s3 sync --exclude '.tao/*' <upload_excludes...>`)
   runs **before** the backend object is torn down; on SLURM any upload runs
   on the **login node**, never inside the GPU allocation.
4. `.tao/` (job records) lives **outside** every results tree and is excluded
   from every upload.
5. `relative_path` is a normalized non-absolute path without `..`. Producers
   use it when a stable logical name differs from the executable's real output
   filename; consumers resolve the declaration rather than hardcoding either.
