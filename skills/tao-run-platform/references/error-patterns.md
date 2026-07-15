# SDK error patterns

Read this file when the agent hits an SDK error — the entries map exception
text or status conditions to root cause and fix.

**`CredentialError: Missing BREV_API_TOKEN`**: env var not loaded. Re-export it in your shell (the session inherits your launching shell's environment).

**`CredentialError: S3_BUCKET_NAME env var required`**: any `inputs` or `outputs` argument needs S3 credentials. Set `S3_BUCKET_NAME`, `ACCESS_KEY`, `SECRET_KEY` (and `S3_ENDPOINT_URL` for non-AWS).

**TAO crash: `You need to set ... results_dir`** (or any spec key declared in `skill_info.actions.<action>.outputs`): `build_entrypoint` was called without `outputs=action_cfg["outputs"]`. The script_runner only auto-fills output spec keys it was told about; missing `outputs=` leaves `results_dir: ''` and the TAO entrypoint aborts. Same root cause if S3 input URIs aren't downloaded — `inputs=action_cfg["inputs"]` was also omitted. Mirror both from `skill_info.yaml` exactly.

**`Image pull failed`**: `NGC_KEY` is invalid or expired. Refresh the key and resubmit.

**Double slash in S3 URI**: `dataset_uri.rstrip("/")` before concatenating, or use `os.path.join` (note: not `posixpath.join` — that doesn't strip).

**Brev instance won't start**: GPU type unavailable in the user's region. Try a different `gpu_type` or wait.
