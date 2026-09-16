# Secrets Handling Across Sibling Skills

Every sibling skill ships a `Verifying secrets safely` block in its
Prerequisites section. Always verify prerequisites by running
`scripts/validate_setup.py` (where it exists) or, for skills without
one (`ncore`, `physical-ai-datasets`, the router), use
`hf auth whoami` or a length-only shell check. Never write ad-hoc
bash that interpolates secret values.

In particular, do not use the bash anti-pattern:

```bash
echo "HF_TOKEN: ${HF_TOKEN:+yes}${HF_TOKEN:-no}"
```

This prints `yes<token-value>` because `${VAR:-no}` only falls back
to `no` when the variable is empty. If you suspect a token was
echoed, rotate it (`huggingface.co/settings/tokens`,
`org.ngc.nvidia.com/setup/api-key`) before continuing.

## Safe verification patterns

```bash
# Hugging Face token
hf auth whoami

# Length-only check (does not echo the value)
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN length=${#HF_TOKEN}" || echo "HF_TOKEN unset"
[ -n "${NGC_CLI_API_KEY:-}" ] && echo "NGC_CLI_API_KEY length=${#NGC_CLI_API_KEY}" || echo "NGC_CLI_API_KEY unset"
[ -n "${NGC_API_KEY:-}" ] && echo "NGC_API_KEY length=${#NGC_API_KEY}" || echo "NGC_API_KEY unset"
```

## NGC key resolution

NRE and the fixer resolve the NGC key in this order — do not prompt
the user until both are exhausted:

1. `$NGC_CLI_API_KEY` (primary; many CI runners and cloud images
   export it automatically)
2. `$NGC_API_KEY` (fallback)
3. Ask the user, pointing them at
   <https://org.ngc.nvidia.com/setup/api-key>

```bash
NGC_KEY="${NGC_CLI_API_KEY:-${NGC_API_KEY:-}}"
printf '%s' "$NGC_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

Always use `--password-stdin` so the key never lands in process
tables or shell history.
