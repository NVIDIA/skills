# Hermes Catalog Recovery

Read this when `hermes skills search <keyword> --source nvidia --json` returns
nothing useful, or when you are unsure whether the identifier you are about to
install is the trusted one.

## Why single keywords work and phrases do not

Hermes matches a query as a plain case-insensitive substring against each
entry's name, description, tags, identifier, and provider joined together. It
does not tokenize, so a multi-word query only matches when those exact words
appear consecutively in that order.

The practical effect is that descriptive phrases collapse to zero results while
the obvious keyword works:

| Query | Result |
| --- | --- |
| `jetson pinmux` | no rows, even though `jetson-customize-pinmux` exists |
| `jetson` | returns `jetson-customize-pinmux` and the rest of the family |
| `OpenUSD` | returns only this finder |
| `usd` | returns the Omniverse and USD skills |

## Search strategy

Run two or three separate single-keyword searches rather than one phrase, and
widen by one step each time:

1. The product name as users say it — `jetson`, `dynamo`, `cuopt`.
2. The product family — `omniverse`, `rapids`, `nemo`, `holoscan`.
3. The taxonomy lane — `usd`, `routing`, `inference`, `genomics`.

`--json` returns each candidate's full description. Read the descriptions and
choose the match yourself; do not assume the first row is the answer. The search
ranks lexically, so it narrows the catalog but does not do the matching for you.

## Full-catalog fallback

If several single-keyword searches return nothing, list the whole catalog and
choose from it before giving up:

```bash
curl -sL https://raw.githubusercontent.com/NVIDIA/skills/main/.github/scripts/marketplace/metadata.json \
  | jq -r '.skills[] | "\(.name): \(.description)"'
```

Do not use `hermes skills browse` for this. It costs more context than the full
catalog listing above and truncates both names and descriptions, so it is the
one option that is worse on both accuracy and cost.

## Confirming the identifier before install

Confirm the skill you picked before installing it:

```bash
hermes skills inspect <identifier>
```

Prefer the `Identifier` from the search result, copied verbatim, over assembling
one by hand. It may be a longer path such as `NVIDIA/skills/skills/<skill-name>`;
that form and the short `NVIDIA/skills/<skill-name>` both resolve to the same
trusted skill with the same scan hash.

Casing is what determines trust, not the number of path segments. `NVIDIA/...`
resolves to the trusted first-party source; `nvidia/...` resolves to a
community-trust mirror with a placeholder description. Always use `NVIDIA`.

Hermes prompts for confirmation itself, so do not pass `--yes` on the user's
behalf.
