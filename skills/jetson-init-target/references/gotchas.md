# jetson-init-target — Gotchas

Detailed pitfalls callers must respect when authoring a Jetson /
IGX target-platform profile YAML. Linked from `SKILL.md`.

- **Two sources of truth, two domains.**
  `bsp-platforms-catalogue.md` owns product / chip / SKU / flash-conf
  facts (consumed in the "Pick the reference product" and "Pick the
  flash config" steps). `platform_template.yaml` owns the field schema
  and required-vs-optional classification (consumed in the "Optional:
  add custom carrier details" step). Neither is interchangeable; never
  duplicate either into skill prose.
- **Adding or removing a prompted field is a template-only change.**
  The skill iterates the template at runtime for its owned blocks. Do
  not edit skill prose to add per-field prompts unless a field needs
  special validation or a suggested default.
- **Quote marker strings in the template.** A bare
  `path: REQUIRED ...` is ambiguous YAML because of the embedded
  `: ` — always wrap in double quotes.
- **The catalogue is the source of truth for product facts.** Do not
  parse conf filenames or scan the BSP — read
  `bsp-platforms-catalogue.md`.
- **`custom_carrier.module` is NOT prompted for.** The same physical
  Jetson module plugs into both the baseline carrier and the custom
  carrier; the YAML records the module under `reference_devkit.module`
  only.
- **Product-to-conf is many-to-one.** AGX Orin 64 GB and 32 GB share
  `jetson-agx-orin-devkit.conf`; the four Orin NX/Nano rows all share
  `jetson-orin-nano-devkit.conf`. The CVM-SKU on the chosen row
  identifies the actual module recorded in the YAML.
- **Filename collisions for shared confs.** If two reference-only
  products share a default conf, suggest a product-specific filename
  in the "Confirm" step to avoid overwriting an earlier profile.
- **AGX Thor T5000 and IGX Thor T5000 have identical SKUs and conf.**
  They differ only in marketing label — pick the right `name` for the
  YAML based on the chosen product.
- **Raw-conf variants** (catalogue marks `(raw)`). Use the raw
  filename — don't try to append the suffix to the devkit symlink name.
- **Quote `sku`** in YAML (`"0008"`, not `0008`) so numeric-looking
  values and leading zeros survive parsing.
- **`NA` is accepted for any required field** but renders the
  profile partially unusable for that field. Always warn that
  downstream skills may refuse if a missing field is required for
  their edit.
- **Don't invent values to avoid `NA`.** If the user doesn't know a
  SKU or other concrete value, ask whether to record `NA` or cancel —
  never guess.
- **Omit empty optional blocks.** If the "Optional: add custom carrier
  details" step was skipped, drop the `custom_carrier:` key entirely
  — do not write `custom_carrier: {}`, a block of `NA` values, or any
  field the user did not provide.
- **Preserve `active_target.yml`'s comment block** when editing.
- **Don't conflate init with switch.** If the user just wants to make
  an *existing* profile active, stop and route them to
  `jetson-set-target`. Re-authoring an existing profile from scratch
  is wasted prompts.
- **Don't author `bsp_image:`, `source:`, or `documents:` here.** Those
  blocks belong to sibling skills (`jetson-init-image`,
  `jetson-init-source`, `jetson-link-docs`). If the user asks to
  extract the BSP, add documents, or override the source path during
  this skill, point them at the sibling and finish authoring only the
  devkit / carrier fields here.
