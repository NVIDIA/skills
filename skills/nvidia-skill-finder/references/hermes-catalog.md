# Hermes Agent: Catalog Lookup and Install

Follow this instead of the `npx` commands in SKILL.md when you are running in
Hermes Agent. Hermes has a built-in catalog that is already security scanned and
profile-aware, so `npx` is both unnecessary and far more expensive in context.

`--source nvidia` requires Hermes 0.22 or newer; older versions reject the
flag. If it is rejected, drop `--source nvidia` and filter the results for
`NVIDIA/skills` identifiers yourself.

## Catalog lookup

```bash
hermes skills search <keyword> --source nvidia --json --limit 100
```

Always pass `--limit 100`. The default is 25, which silently truncates the
broad keywords this guidance tells you to use — `jetson` alone matches 39
skills, so the default cuts off real results such as `jetson-quick-start`.

Search one broad keyword at a time — `jetson`, `usd`, `dynamo`, `pandas`,
`cuda`. `--json` returns each candidate's full description; read the
descriptions and choose the match yourself. The search ranks lexically, so it
narrows the catalog but does not do the matching for you. Do not assume the
first row is the answer.

## Why single keywords, not phrases

Hermes matches a query as a plain case-insensitive substring against each
entry's name, description, tags, identifier, and provider joined together. It
does not tokenize, so a multi-word query only matches when those exact words
appear consecutively in that order.

Descriptive phrases therefore collapse to zero results while the obvious
keyword works:

| Query | Result |
| --- | --- |
| `jetson pinmux` | no rows, even though `jetson-customize-pinmux` exists |
| `jetson` | returns `jetson-customize-pinmux` and the rest of the family |
| `OpenUSD` | returns only this finder |
| `usd` | returns the Omniverse and USD skills |

Run two or three separate single-keyword searches, widening by one step each
time:

1. The product name as users say it — `jetson`, `dynamo`, `cuopt`.
2. The product family — `omniverse`, `rapids`, `nemo`, `holoscan`.
3. The taxonomy lane — `usd`, `routing`, `inference`, `genomics`.

## Full-catalog fallback

If several single-keyword searches return nothing, list the whole catalog and
choose from it before giving up:

```bash
curl -sL https://raw.githubusercontent.com/NVIDIA/skills/main/.github/scripts/marketplace/metadata.json \
  | jq -r '.skills[] | "\(.name): \(.description)"'
```

Do not use `hermes skills browse` for this. It costs more context than the
listing above and truncates both names and descriptions, so it is the one
option that is worse on both accuracy and cost.

## Install

```bash
hermes skills install NVIDIA/skills/<skill-name>
```

Casing is what determines trust, not the number of path segments. `NVIDIA/...`
resolves to the trusted first-party source; `nvidia/...` resolves to a
community-trust mirror with a placeholder description. Always use `NVIDIA`.

Prefer the `Identifier` from the search result, copied verbatim, over
assembling one by hand. It may be a longer path such as
`NVIDIA/skills/skills/<skill-name>`; that form and the short
`NVIDIA/skills/<skill-name>` both resolve to the same trusted skill with the
same scan hash.

Confirm an identifier you are unsure about before installing it:

```bash
hermes skills inspect <identifier>
```

Hermes prompts for confirmation itself, so do not pass `--yes` on the user's
behalf.
