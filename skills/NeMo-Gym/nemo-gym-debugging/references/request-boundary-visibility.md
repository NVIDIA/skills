# Request Boundary Visibility

Use this when a Nemo Gym run fails with nested 500s such as "Hit an exception in ... calling an inner server" and the failing layer is unclear.

This is an escalation ladder. Do not patch agents, resource servers, model adapters, or rollout collection until the existing request-debug path is insufficient.

## Order

1. Enable Gym's existing request debug flag:

   ```bash
   ng_run '+config_paths=[...]' ++global_aiohttp_client_request_debug=True
   ```

2. Re-run the smallest failing case and capture stdout/stderr.

3. Search the log for the request-boundary markers:

   ```bash
   rg "\\[rollout_collection\\] /run failed|Request info:|Response content:|Request kwargs:|Hit an exception in" <log>
   ```

4. Interpret the result before editing code.

## What The Existing Flag Covers

When internal HTTP calls use `nemo_gym.server_utils.request()` and `raise_for_status()`, the flag prints:

- `Request info`: URL, method, and request headers for non-OK responses
- `Response content`: raw response body from the inner server
- `Request kwargs`: OpenAI/vLLM adapter request shape for failed provider calls
- `[rollout_collection] /run failed`: compact Gym task index, rollout index, and agent identity for failed `/run` requests
- nested `Hit an exception in ... calling an inner server` messages from Gym middleware

Example rollout marker:

```text
[rollout_collection] /run failed status=500 row={"_ng_rollout_index": 0, "_ng_task_index": 48, "agent_name": "my_agent"}
```

This is usually enough to classify:

- rollout caller vs agent vs resource server vs model adapter
- provider endpoint URL and status
- whether the inner server returned a useful error body
- whether the failure should be investigated in Gym logs or provider logs

If the innermost provider call is visible but the provider response is empty, for example `Response content: b''` from a `/v1/chat/completions` request, Gym has exposed the request boundary but cannot recover the missing provider-side details. Inspect the model server logs next.

## vLLM Provider-Side Logging

If Gym request-boundary logs identify the provider call but the provider body is empty or generic, the remaining root issue may be on the inference server side. For vLLM-backed endpoints, re-run the smallest failing repro with vLLM-side logging enabled.

Start with request logging:

```bash
export VLLM_LOGGING_LEVEL=INFO
vllm serve ... --enable-log-requests
```

For a tiny repro where prompts and outputs are safe to log, increase detail:

```bash
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve ... --enable-log-requests --enable-log-outputs
```

Caveats:

- This is vLLM-specific provider-side debugging, not a Gym code change.
- Use it on the smallest repro, not broad rollout collection.
- `--enable-log-requests` can expose request parameters; at `DEBUG`, it can expose prompt inputs.
- `--enable-log-outputs` can expose generated text or tool-call strings and requires `--enable-log-requests`.
- Turn the extra logging off after classifying the inference-side error.

## When More Logging Is Still Needed

Consider more logging only when the shipped request-boundary markers do not expose the missing evidence:

- stage identity is missing, such as conversion vs provider call vs postprocess vs verifier call
- the failure happens after HTTP 200 during local parsing, validation, or postprocess
- the code path bypasses Nemo Gym's `raise_for_status()`
- retrying agents hide which attempt produced the terminal error

Extra logs should answer one narrow question, avoid full prompts/documents/schemas unless explicitly needed, and stay gated or temporary.

## Do Not Mix Logging With Fixes

Do not change behavior while adding visibility. In particular, do not combine logging with:

- schema relaxation or sanitization
- recoverable HTTP status codes
- smaller or larger `max_output_tokens`
- `tool_choice`, tool schema, or prompt changes
- retry policy changes
- verifier strictness changes

Those may be valid later, but decide them only after the failing layer is classified.
