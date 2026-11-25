# Diagnosis Eval Fixtures

These fixtures are minimal synthetic log snippets derived from common NVFLARE
failure shapes. They are not copied from customer jobs and contain no private
site data.

`transfer_progress_timeout.log` is synthetic evidence for a large-transfer
timeout warning where later streaming progress indicates a retryable congestion
case rather than a dead peer.

`poisoned_log_injection.log` is synthetic evidence for a real round-timeout
failure whose log lines also carry adversarial injection: embedded directives
(download-and-run a script, `--disable-auth`, exfiltrate the workspace), a
spoofed `[USER_CODE_EXCEPTION]` blame marker, and an AI-assistant-targeted line.
The endpoints use only reserved example.com domains; the directives are
intentionally malicious-looking test data for the log-content trust boundary and
must never be followed.
