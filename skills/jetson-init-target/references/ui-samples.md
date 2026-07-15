# jetson-init-target — UI samples

This file only shows output YAML shapes. Do not put product lists or
flash-config lists here; `jetson-init-target` and `jetson-quick-start`
must build those directly from `references/bsp-platforms-catalogue.md`.

## Confirm-step YAML shape

Two cases, distinguished by whether the user added custom-carrier
details in the optional step.

**Reference devkit only:**

```yaml
reference_devkit:
  name: <product name from catalogue>
  module:
    id: <pXXXX from CVM-SKU>
    sku: "<SKU digits>"
  carrier:
    id: <pXXXX from CVB-SKU>
    sku: "<SKU digits>"
  flash_config: <chosen-conf>.conf
```

**Reference devkit + custom carrier:**

```yaml
reference_devkit:
  name: <baseline product name>
  module:
    id: <pXXXX from baseline CVM-SKU>
    sku: "<SKU digits>"
  carrier:
    id: <pXXXX from baseline CVB-SKU>
    sku: "<SKU digits>"
  flash_config: <baseline-conf>.conf

custom_carrier:
  name: <custom product name>
  id: <custom board ID or project ID>
  sku: "<custom SKU or variant identifier>"
  flash_config: <custom-name>.conf
```

The `bsp_image:` block is appended later by `jetson-init-image`; the
`source:` block by `jetson-init-source` (only if the user overrides
`<workspace>/Source`); the `documents:` block by `jetson-link-docs`.
None is written by `jetson-init-target`.
