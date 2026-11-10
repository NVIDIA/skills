# Hermes Agent: Catalog Lookup and Install

Follow this instead of the `npx` commands in SKILL.md when you are running in
Hermes Agent. Install goes through Hermes so the skill is security scanned and
lands in the active profile, and `npx` stays unnecessary.

`--source nvidia` requires Hermes 0.22 or newer; older versions reject the
flag. If it is rejected, drop `--source nvidia` and filter the results for
`NVIDIA/skills` identifiers yourself.

## Catalog lookup

List the whole catalog once and choose from it yourself:

```bash
curl -sL https://raw.githubusercontent.com/NVIDIA/skills/main/.github/scripts/marketplace/metadata.json \
  | jq -r '.skills[] | "\(.name): \(.description[0:110])"'
```

If `jq` is unavailable, use this instead. Do not reach for `python3 -c`: Hermes
blocks inline interpreter scripts as "script execution via -e/-c flag", and the
nested quoting such a one-liner needs is a syntax error on Python 3.11 and
older.

```bash
curl -sL https://raw.githubusercontent.com/NVIDIA/skills/main/.github/scripts/marketplace/metadata.json \
  | python3 -m json.tool | grep -E '"(name|description)":' | cut -c1-130
```

Read the list and pick the best match on the descriptions. Prefer this over
searching. Searching looks cheaper but is not: it matches the query as a plain
case-insensitive substring over each entry's joined fields, so it misses
phrases (`jetson pinmux` returns nothing even though `jetson-customize-pinmux`
exists), and broad keywords return full descriptions for every hit. A handful
of searches costs more context than the complete listing above and can still
miss the right skill.

Do not use `hermes skills browse` either. It costs more than the listing and
truncates both names and descriptions, so it is worse on accuracy and cost.

## Confirm the identifier

Once you have chosen a specific skill, look up that exact name to confirm it is
available in Hermes and to get its identifier:

```bash
hermes skills search <skill-name> --source nvidia --json --limit 100
```

An exact skill name is a narrow query, so this stays small. Pass `--limit 100`
anyway; the default is 25, which silently truncates broader matches.

## Install

Use the `Identifier` from that result, copied verbatim:

```bash
hermes skills install <identifier>
```

Casing determines trust, not the number of path segments. `NVIDIA/...` resolves
to the trusted first-party source; `nvidia/...` resolves to a community-trust
mirror with a placeholder description. Always use `NVIDIA`.

The identifier may be a longer path such as `NVIDIA/skills/skills/<skill-name>`;
that form and the short `NVIDIA/skills/<skill-name>` both resolve to the same
trusted skill with the same scan hash.

Confirm an identifier you are unsure about before installing it:

```bash
hermes skills inspect <identifier>
```

Hermes prompts for confirmation itself, so do not pass `--yes` on the user's
behalf.
