# /jetson-modify-usb references

| File | Purpose |
|---|---|
| `usb-architecture.md` | Tegra USB block model: USB2 OTG ports, USB3 SS instances, UPHY lane pool, padctl assignments. |
| `usb-dt-bindings.md` | `tegra_xusb` / `xudc` / `xusb_padctl` DT bindings the agent emits in the overlay. |

The skill is agentic — `AskUserQuestion` payloads are built on the fly
from the per-port diff in step 2 of `SKILL.md`, not from a static
`questions.json`. The Adaptation Guide §"Port the Universal Serial
Bus" (URL pinned to `session.adaptation_guide_version`) is the
authoritative source for every option, mode constraint, and per-port
capability the agent surfaces.
