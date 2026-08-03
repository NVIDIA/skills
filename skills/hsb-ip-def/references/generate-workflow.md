# Generate Workflow

Use this reference when the user asks to create, scaffold, draft, design, or produce a `HOLOLINK_def.svh` for an HSB board.

Operational disclosure: this workflow runs local Python scripts (`scripts/generate_def.py`, which invokes `scripts/validate_def.py`) and writes a `HOLOLINK_def.svh` output file. Before running commands or writing output, state the command and output path, confirm the path with the user, and ask before overwriting any existing file.

Keep `SKILL.md` as the router. This file is the canonical home for the detailed Generate workflow, requirement-discovery order, question style, and per-topic prompt guidance.

## Detailed Steps

**Trigger.** User asks to create, scaffold, draft, design, or produce a `HOLOLINK_def.svh` for their board.

**Approach: requirement-discovery first.** Probe the user about their actual design needs — sensor count and direction, host interfaces, clock frequencies, datapath widths, peripheral counts, etc. — and assemble a profile from their answers. The 6 example configurations in `references/archetypes.md` are *examples* of how real designs combine macro values, not templates the user must pick from. Power users can supply a YAML profile directly to skip the chat.

Steps:

1. **Run preflight once per session.** Follow `references/script-usage.md` to verify bundled scripts, Python version, and PyYAML import before design questioning. During preflight, tell the user that generation will run local Python scripts and write an output `.svh`; confirm the eventual command and output path before running the generator.

2. **Classify what the user already volunteered.** Parse their request and extract any specifics they gave (sensor count, widths, clock frequency, deployment target, etc.).

3. **Probe for the requirements gap, one question per turn.** Ask the user a **single question per message** — never bundle multiple questions or use a structured-question tool that requires the user to answer everything before getting a response. The user must always be able to ask "what do you mean by X?" or revisit an earlier answer at any point without losing context.

   **Never silently default any macro.** Every macro that ends up in the generated file must be the result of an explicit user choice. When you propose a value, **show your suggestion and ask the user to confirm or override** — don't fill it in unilaterally. "I'll just use the default" is the failure mode this rule prevents.

   **Use sensor-agnostic language.** The HSB sensor interface is generic — it can carry image, RF, ADC, JESD-aggregated, audio, or any other byte-aligned sensor data. When explaining sensor-side macros (widths, clocks, packetizer parameters, buffer sizes), do not use camera-specific terms ("pixel clock," "bits-per-pixel," "frames per second," "BPP") unless the user has explicitly told you their sensor is a camera. Frame the math in terms of `bandwidth = width × clock_rate` — the same equation applies to any sensor. The one exception is `DATAUSER_WIDTH`: its per-bit semantics are MIPI CSI-2-specific per the IP docs (see `references/macro-reference.md`), so camera framing is correct there.

   **Don't fabricate use-case rationales.** When a macro has multiple legal values, state what the macro *does* (verified from the IP source or `references/macro-reference.md`) and ask the user what their design needs. **Do not invent "typical setups," "common pairings," or "use cases for value X vs. value Y"** unless they're either documented in the references or directly inferable from the IP behavior. If you don't know what real designs use the feature for, don't make it up — just state the macro's behavior and ask. A short factual question beats a paragraph of fabricated scenarios. Treat "I don't know what's typical here, what does your design need?" as a fully acceptable thing to say.

   **Banned phrases.** Do not write or paraphrase any of these in user-facing chat: *"most designs use X"*, *"typically X"*, *"usually X"*, *"in general, X"*, *"the standard choice is X"*, *"a common choice is X"*, *"most boards use X"*, *"common board-supplied values are X"*, *"boards commonly supply X"*, *"common values are X"*, *"X is a common board frequency"*. They sound authoritative but encode fabricated norms — and the variants that single out one specific number ("common values are exactly X") are the same pattern, just dressed up as fact. Replace every one with a documented anchor: **"the HSB docs list example values X and Y"**, **"the IP legal range is X..Y"**, or **"the doc doesn't state a typical — what does your design need?"**. Do not raise concern about a legal user-provided value solely because it differs from examples bundled with the skill.

   **No meta-commentary about the question or your own answer.** Do not preface answers with phrases like *"Good question"*, *"Great point"*, *"That's a sharp catch"*, or any other compliment to the user's question. Do not narrate your own factuality with phrases like *"there's a clean factual answer here, not a fabricated typical"*, *"to be precise this time"*, *"here's the honest answer"*, or any other self-referential framing about whether you're being accurate or avoiding fabrication. Just state the fact. The user can tell whether the answer is grounded by reading it; meta-commentary that announces it is grounded is noise.

   **No quantitative corpus-frequency claims used as authority.** Do not write phrases like *"6 of 14 designs use X"*, *"most of the corpus uses Y"*, *"X is the dominant value across the captured configurations"*, *"4 out of 6 archetypes pick Z"*. Citing frequencies in user-facing chat reproduces the same fabricated-norm pattern this section bans — it implies "you should pick X because everyone else does." When the user asks what's typical, redirect to behavior: *"the IP supports {set of legal values}; the doc doesn't state a typical — what does your design need?"*

   **No softened-frequency or speculative-usage claims either.** The same ban covers fuzzy variants: *"many designs never use it"*, *"some projects don't use X"*, *"this is rarely used"*, *"only specialized boards need Y"*. These have no factual anchor and are usually wrong. When a feature has a specific, documentable use case, name the use case (e.g., "CoE — IEEE 1722 — is used on AGX Thor for hardware accelerated networking"); when you don't know who uses a feature and why, just describe what it does and stop there.

   **Visually call out the current topic at the start of each question** with a markdown header. Use `### \`MACRO_NAME\` — short description` (or a similar prominent format). Do not bury the topic in prose. For example:

   ```
   ### `ENUM_EEPROM` — board enumeration

   The HSB IP needs to know your board's MAC address and serial number…

   How does your board supply MAC/SN — EEPROM at I²C bus 0, or via input ports?
   ```

   This makes it easy for the user to scan a long chat and see exactly which decision is in front of them.

   Walk through the design in roughly this order, skipping anything already settled:

   sensor RX (count → per-port widths → `SIF_RX_DATA_GEN` toggle if RX is defined → packetizer enable per port — see below) → sensor TX (count → per-port widths → per-port FIFO depths) → `DATAPATH_WIDTH` (system-wide sensor-side maximum; must be ≥ `max(SIF_RX_WIDTH, SIF_TX_WIDTH)`) → `DATAUSER_WIDTH` (sensor `tuser` width, `1` or `2` — see `macro-reference.md`) → host (count → one shared `HOST_WIDTH` for all host interfaces → MTU) → clocks (HIF, then APB, then PTP — see below for context) → peripherals (SPI, then I²C, then UART [max 1], then GPIO bits, then `GPIO_RESET_VALUE`) → enumeration (`ENUM_EEPROM` — see below) → user APB register blocks (`REG_INST` — see below) → `init_reg[]` (empty list disables system init entirely — see below) → UUID (provide one or auto-generate) → optional toggles (`EXT_PTP`, `SYNC_CLK_HIF_APB`/`PTP`, `DISABLE_COE`).

   Leave `PERI_RAM_DEPTH` out of the chat-driven question flow. If a power user explicitly requests it, load `references/advanced-macros.md` and let them provide `peri_ram_depth:` in a YAML profile.

   **Every item in this list must be confirmed by the user** through either a direct macro choice or a stated design requirement that determines the macro. Even `DATAUSER_WIDTH` gets asked. The doc only defines `tuser` semantics for MIPI CSI-2: `tuser[0]` = embedded-data marker, `tuser[1]` = Line End. Ask the user whether their design needs both MIPI markers (`= 2`) or only one (`= 1`); don't infer a "typical" — the doc doesn't state one. For packetizer fields, user ownership comes from the user's data-manipulation description; do not force confirmation of each derived packetizer parameter. The point is user ownership of every macro that lands in the generated file.

   **For these questions, give context BEFORE asking** — they have non-obvious behavior the user needs to know:

   - **`SIF_RX_DATA_GEN` (per-port test-pattern injection capability).** Only ask when `SENSOR_RX_IF_INST` is defined. Explain first: "When this macro is defined at compile time, the IP instantiates a `data_gen` module per Sensor RX interface, each with its own APB register interface. The data_gen's `data_gen_ena` register defaults to 0 at reset — so external sensor data flows through normally. The host software can write to the data_gen's APB registers **at runtime** to enable test-pattern injection per port; when enabled on a port, that port's mux selects the data_gen's output instead of the external `i_sif_axis_*` input, and back-pressure is asserted to the external sensor. When `data_gen_ena = 0` (the reset state), external sensor data flows. When the macro is **undefined** at compile time, the data_gen modules are not instantiated (saves logic and embedded RAM), the mux select is tied to 0, and there is no runtime injection capability. So defining it is a compile-time decision to **include the test-injection capability**, not a decision to bypass external data. Some teams keep it on in production for in-system diagnostics; others remove it to save resources. Want it on (capability available, runtime-controlled by host) or off (capability not present)?"

   - **Packetizer (`SIF_RX_PACKETIZER_EN[]` and the four "do not change" arrays).** Only ask when `SENSOR_RX_IF_INST` is defined. Explain first: "The packetizer is used when **sensor data needs to be manipulated before sending to the host** — rearranged, swizzled, split into different streams, replicated, etc. If your sensor data passes through the IP unchanged from sensor to host, you don't need the packetizer. **Default `SIF_RX_PACKETIZER_EN` to `0` (off) per port.** The four packetizer parameter arrays (`SIF_RX_VP_COUNT`, `SIF_RX_SORT_RESOLUTION`, `SIF_RX_VP_SIZE`, `SIF_RX_NUM_CYCLES`) are then don't-cares — the generator emits placeholder values that the IP ignores when EN=0. If the user needs the packetizer enabled, invoke **`hsb-ip-packetizer`** with the known RX count, RX widths, and the user's data-manipulation description. Let the packetizer skill derive and print the complete `packetizer_profile_overlay` before explaining what the selected fields allow for the described data manipulation. Consume only that skill's `packetizer_profile_overlay` YAML keys (`sif_rx_packetizer_en`, `sif_rx_vp_count`, `sif_rx_sort_resolution`, `sif_rx_vp_size`, `sif_rx_num_cycles`), merge them into this skill's in-progress profile, then continue normal `HOLOLINK_def.svh` generation and validation here. Confirm: do you need any sensor-data manipulation, or shall we leave the packetizer off?"

   - **Sensor widths (`DATAPATH_WIDTH`, `SIF_RX_WIDTH[]`, `SIF_TX_WIDTH[]`).** **The sensor interface is sensor-agnostic.** It can carry image, RF, ADC, JESD-aggregated samples, audio, or any other byte-aligned sensor data. **Do not use camera-specific terminology** — no "pixel clock," "bits-per-pixel," "frames per second," or similar — unless the user has explicitly identified their sensor as a camera. The right framing for per-port widths is generic: *"Interface width is driven by your sensor's data rate, your sensor-side clock, and the total bandwidth requirement. The relationship is `bandwidth = width × sensor_clock`. Pick the smallest power-of-2 byte-aligned width that carries your peak rate with the margins your design requires. Wider per-port widths let you run a slower sensor-domain clock; narrower widths require a faster clock or accept lower bandwidth."* Then ask the user about their actual data rate, or their chosen width and clock directly.

     **`DATAPATH_WIDTH` selection.** Once `SIF_RX_WIDTH[]` and `SIF_TX_WIDTH[]` are known, compute `required_datapath_width = max(SIF_RX_WIDTH[], SIF_TX_WIDTH[])` across the enabled sensor directions. Present that value as the logical setting and ask the user to confirm it. Do **not** present larger example values as equivalent choices, and do **not** frame the exact value as a "tight fit" compromise. Always include one short sentence explaining the higher-value case after the recommendation: set `DATAPATH_WIDTH` higher only when the design intentionally wants to reserve a wider internal sensor datapath now, for example for a later wider RX/TX port without revisiting this IP configuration; the tradeoff is extra logic/RAM, and narrower existing ports leave unused upper bits. Example:

     ```
     ### `DATAPATH_WIDTH` - sensor-side datapath width

     Your widest Sensor RX/TX interface is 512 bits, so `DATAPATH_WIDTH` should be 512. `DATAKEEP_WIDTH` will be `DATAPATH_WIDTH/8`.

     Confirm `DATAPATH_WIDTH = 512`?
     ```

   - **`SIF_TX_BUF_SIZE[]` (per-TX-interface FIFO depth).** Only ask when `SENSOR_TX_IF_INST` is defined. **Do not skip this question when the design has one or more Sensor TX interfaces.** Explain first: "`SIF_TX_BUF_SIZE[i]` sets the per-interface Sensor TX FIFO depth as a count of `SIF_TX_WIDTH[i]`-wide elements, not bytes. The buffer primarily lets the IP absorb host-to-sensor data and apply backpressure toward the host when the sensor side cannot accept data immediately. Larger depths allow more TX-width elements to be stored but consume more FPGA embedded RAM." Ask for one FIFO depth per TX interface, or ask whether one stated depth should apply to every TX interface. Record `sif_tx_buf_size` as a list whose length exactly matches `SENSOR_TX_IF_INST`. If the user asks for a default or says "you choose," state the proposed value in TX-width elements and ask for confirmation before recording it; do not silently use `2048`.

   - **`HOST_WIDTH` (host AXI-Stream tdata width) and `HIF_CLK_FREQ` together.** These two are tightly coupled — the host width is dictated by the host-side Ethernet datapath, and the HIF clock is the clock actually supplied to the HSB host interface after the board/FPGA clocking chain. The Ethernet MAC IP may drive that clock directly, or a board/FPGA PLL or clock divider may derive it from another MAC/transceiver clock. Explain first: "Ethernet line-rate / `HOST_WIDTH` / `HIF_CLK_FREQ` pairings documented in the HSB materials and the in-skill archetypes:
     - **8 bits @ 125 MHz** = **1 GbE** (`gigabit-baseline`)
     - **64 bits @ 156.25 MHz** = **10 GbE** (`mid-bandwidth-baseline`)
     - **128 / 256 bits** at intermediate clocks = **25 GbE / 40 GbE**
     - **512 bits @ ~201 MHz** = **100 GbE** (`high-bandwidth-single-sensor`)
     - **512 bits @ ~322 MHz** = **very-high-speed 100 GbE+** (`very-high-speed`)

     There is exactly one `HOST_WIDTH` macro. It applies to every host interface selected by `HOST_IF_INST`; there is no per-host `HOST_WIDTH[]` array. Do not ask whether host interfaces use the same or different widths. If the user describes multiple host interfaces with different datapath widths, ask them to choose the single shared HSB host-stream width for this IP configuration before continuing.

     Set `HOST_WIDTH` to the shared AXI-Stream width connected to every HSB host interface, and set `HIF_CLK_FREQ` to the frequency of the clock driving that HSB host interface. The source may be the MAC, a transceiver clock path, or a PLL/divider that derives a suitable HIF clock. If you do not know the clocking path, check the MAC/transceiver integration and FPGA clocking plan." Then ask `HOST_WIDTH` and the three clocks (`HIF_CLK_FREQ`, `APB_CLK_FREQ`, `PTP_CLK_FREQ`) one at a time. For each, anchor any suggestions to documented examples or hard legal constraints while allowing other board-supplied values that match the design — **never invent a "typical" or "most designs" claim**.

     **Clock-source relationships (APB and PTP).** Ask whether APB and PTP are independent clocks or deliberately co-sourced with HIF. If the user wants synchronous-clock CDC tightening, load `references/advanced-macros.md` for `SYNC_CLK_HIF_APB` and `SYNC_CLK_HIF_PTP` before discussing those toggles. Do not recommend them without explicit board-level confirmation.

     **APB** can be slower than HIF; its rate sets the I²C divider when `ENUM_EEPROM` is defined. The HSB docs list `APB_CLK_FREQ` examples `19_531_250` and `100_000_000` — present those as examples and ask the user to state what their design provides.

     **PTP** must be in the 95–105 MHz band per the HSB doc. **For `PTP_CLK_FREQ`, use the band as the legal constraint.** The actual frequency is whatever PTP-grade clock source the board provides (oscillator, recovered network clock, etc.). **Do not propose a value, do not single out one number as "common" or "typical," and do not anchor on `100_000_000` (or any other single value) just because it's a round number.** Phrasings like *"common board-supplied values are exactly 100_000_000"*, *"most boards supply 100 MHz"*, or *"100 MHz is the typical"* are banned for this macro — there is no documented typical to cite. Ask the user neutrally: *"What `PTP_CLK_FREQ` (in Hz) does your PTP clock source provide? It must fall within the documented 95–105 MHz band."* If the user asks what's typical, redirect to the doc: *"The HSB doc states only the 95–105 MHz band, not a typical — your board's PTP clock source determines the value."*

   - **`HOST_MTU` (Ethernet packet size).** Explain first: "MTU applies in **both directions** — outgoing packets the IP generates AND incoming packets it accepts. The RX-side `pkt_check` drops Ethernet frames larger than `HOST_MTU`, so set this to at least the largest packet either side will send. The HSB docs list `1500` and `4096` as examples. The IP's internal buffer depth scales linearly via `HOST_BUF_DEPTH = HOST_MTU * 2 / (HOST_WIDTH/8)` — higher values cost embedded RAM."

   - **`GPIO_RESET_VALUE` (boot state of GPIO outputs).** **Never default this silently — always ask.** Explain first: "After picking how many GPIO pins your design exposes (`GPIO_INST`), specify the boot state of each output pin — the value driven on power-up before APB software has configured anything. `GPIO_RESET_VALUE` is a `[GPIO_INST-1:0]`-wide bit vector. Bits set high boot high; bits set low boot low. If your board has any active-low resets or enables that must be deasserted at boot, those bits need to be set high. What boot pattern does your design need?"

   - **`ENUM_EEPROM` (board enumeration).** Explain first: "The HSB IP needs to know your board's MAC address and serial number for BOOTP enumeration. Two supported paths: (a) define `ENUM_EEPROM` so the IP reads MAC/SN at boot from an external EEPROM on I²C bus 0 at 7-bit address `0x50`; `EEPROM_REG_ADDR_BITS` selects the EEPROM internal register-address width, 8 or 16 bits. (b) Leave `ENUM_EEPROM` undefined and have the top wrapper drive `i_mac_addr[]`, `i_board_sn`, and `i_enum_vld` directly. Which path does your design use?"

   - **`REG_INST` (user APB register blocks).** Explain first: "The HSB IP exposes its APB control bus to your user logic via `REG_INST` register-block ports. The IP's APB switch decodes the upper nibble of the 32-bit address: `0x0xxxx_xxxx` is reserved for the IP's own internal registers, and `0x1xxxx_xxxx`, `0x2xxxx_xxxx`, … route to user blocks 0, 1, 2, …. **Each user block sees `paddr[27:0]` — a 256 MB address window** that your logic can decode as needed. This lets your user logic share the IP's control plane — the same one HSB uses for its own MAC/PCS/PTP setup — so the host-side **HSB software driver (`hololink` Python library) can reach your registers via the same ECB/APB protocol** without you building a separate control bus. Range is 1..8; even a design with no user logic still needs `REG_INST = 1` (the IP wires up at least one block). How many do you want?"

   - **`init_reg[]` (system-init register-write sequence).** Explain first: "The HSB IP can run a one-time APB write sequence at boot to configure dependent IP — for example the Ethernet MAC and PCS that the user instantiates around the HSB IP. Each entry is `{32'h<addr>, 32'h<data>}`. The `N_INIT_REG` macro counts the entries. **If you don't need a sequence, leave `N_INIT_REG` undefined entirely and don't include the `init_reg[]` array** — the IP's `\`ifdef N_INIT_REG` then skips the `sys_init` module cleanly. Setting `N_INIT_REG = 0` is invalid and causes elaboration failure. See `init-reg-cookbook.md` for address-range conventions and templates derived from corpus configurations. Do you have a sequence, or skip system init for now?" If the user skips system init, record `init_reg: []` explicitly in the YAML profile; do not omit the key, because archetype/sample defaults may include example init writes.

   - **`DISABLE_COE` (CoE dataplane removal).** Load `references/advanced-macros.md` before discussing this advanced toggle. Ask whether the target host pipeline needs CoE available or should remove the CoE dataplane.

   Accept any of these as valid responses to a question:
   - **A concrete value or number** — record it.
   - **"Yes" / "confirmed" / "use what you suggested"** in response to a stated proposed value — record the suggested value as the user's explicit choice. (You must always *show* the value before asking for confirmation; never assume it.)
   - **"What does X mean?" / "Explain Y"** — answer the question (consult `references/macro-reference.md` if needed), then re-ask the original.
   - **"Not relevant" / "we don't use that"** — record as undefined where appropriate (e.g., omit `SPI_INST` to remove the SPI port group). This is the user explicitly choosing "no" — different from defaulting silently.

   **What is NOT a valid path:** silently picking a value the user hasn't seen. If the user says "you decide" or "use defaults," respond by stating the value you would pick and asking for confirmation: *"For `DATAUSER_WIDTH`: the doc only defines `tuser` semantics for MIPI CSI-2 — `tuser[0]` = embedded-data marker (DT 0x12), `tuser[1]` = Line End marker. Use `2` if your design needs both markers; `1` if you only need `tuser[0]` (or if you're not using MIPI semantics). Which fits your design?"* That stays factual and lets the user pick — no fabricated "typical."

   **Speeding up confirmations.** If the user is consistently confirming proposed values for several topics in a row, you may **batch-confirm the remaining proposals**: list each remaining macro with the value you'd suggest, and ask the user to affirm the batch (or call out specific ones to override). The user must still see every value before it lands in the file.

   **When summarizing the running profile** to the user, show "0 interfaces" rather than "undefined" for gating macros that have no instances (e.g., a design with no Sensor TX has `Sensor TX: 0 interfaces`, not `Sensor TX: undefined`). The underlying SVH still leaves the macro undefined to remove the ports — this rule is about the user-facing summary only.

4. **Optionally show 1–3 example configs** from `assets/metadata/corpus.json` that resemble what the user described, framed as: "examples with similar macro structure" — never "you are a copy of X." Skip this step if the user's design is far from any example.

5. **Build a YAML profile** from the user's answers. Flat keys (e.g., `hif_clk_freq`, `datapath_width`, `host_width`); see `scripts/generate_def.py --help` for the full schema. Do not preselect an archetype unless the user explicitly asked for one.

6. **Run the generator** with the preflight-selected Python interpreter: `<PY> scripts/generate_def.py --profile <yaml-path> -o <output>`. The generator validates internally before writing — if validation fails, it refuses to produce output. The UUID is auto-generated when not in the profile; the generator prints the chosen value to stderr.

7. **Show the result and call out per-field provenance**: which fields came from the user's stated requirements, which from generic defaults, and which were auto-generated (notably the UUID when not supplied).

8. **Offer the top-level handoff.** Ask once: "Do you want to create a matching top-level FPGA scaffold with `hsb-ip-create-top`?" If yes, invoke `hsb-ip-create-top` with the generated and validated `HOLOLINK_def.svh` path or content, the final profile values if available, the known HSB IP source root if available, and the requested output path if the user supplied one. This skill does not generate `FPGA_top.sv`; it hands off to `hsb-ip-create-top`, then that skill owns the wrapper scaffold.

If the user supplies a YAML profile directly (as an attached file or pasted YAML), run preflight, then skip steps 2–5 and go straight to step 6, then continue through steps 7–8.
