# Runtime downloads

Use normal networking, including verification. Set `--network none`,
`UV_OFFLINE`, `HF_HUB_OFFLINE`, or `TRANSFORMERS_OFFLINE` only when required by
the user, policy, or a test, and explain the restriction in the launch review.
Local checkpoints do not imply cached packages.

Approved runs may fetch declared Python dependencies in their runtime. Keep
the declared version constraints and the small-helper approval exception, and
report installs. A missing helper does not require an image rebuild, extra
stage, or custom cache.

On download failure, check network restrictions, DNS, and proxy or package
index access. Preserve offline rules and approval gates for model or dataset
downloads, system changes, and upgrades.
