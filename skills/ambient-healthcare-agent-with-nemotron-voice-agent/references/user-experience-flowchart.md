# User Experience Flowchart

This is a high-level view of the gated workflow defined in `SKILL.md`. The
skill remains the source of truth for exact prompts, authorization boundaries,
validation rules, and handoff details.

```mermaid
flowchart TB
    A([Start]) --> W[Deliver the required welcome<br/>verbatim, then stop]
    W --> P[/User provides an existing NVA path<br/>or an exact fresh-clone destination/]
    P --> R[Clone if requested and<br/>validate the checkout markers]
    R --> E[Create the environment configuration from its example if absent;<br/>preserve an existing configuration]
    E --> I[Inspect syntax-aware compatibility,<br/>service defaults, catalogs, and deployment skills]
    I --> M[/User reviews the public-endpoint default<br/>and receives the configuration path for NVIDIA_API_KEY;<br/>may select local, existing, or mixed services/]
    M --> D{Docker Compose access passes?}
    D -- No --> Z[Report the exact failed gate<br/>and stop]
    D -- Yes --> S[Disclose the resolved live-validation destination<br/>and bundled fictional fixture]
    S --> Q[/User selects appointment making,<br/>patient intake, or a custom workflow/]
    Q -- Preset --> P1[Load the selected bundled overlay<br/>and its expected histories]
    Q -- Custom workflow --> C1[Define the goal; required, optional, and sensitive fields;<br/>tool preconditions, inputs, reads, writes, and results;<br/>and expected histories]
    C1 --> C2{User approves the exact histories<br/>and validation destination?}
    C2 -- Revise --> C1
    P1 --> O
    C2 -- Approve --> O[Run scenario-aware compatibility preflight<br/>and apply the idempotent overlay]
    O --> T[Run static and scenario tests]
    T --> K{Selected runtime credentials<br/>and endpoints configured?}
    K -- No --> Z
    K -- Yes --> B[Build and start or restart the customized Generic recipe<br/>with one documented profile]
    B --> H2{App and all selected<br/>services healthy?}
    H2 -- No --> Z
    H2 -- Yes --> L[Run only the authorized live expected histories]
    L --> G{Deterministic and semantic<br/>checks pass?}
    G -- No --> X[Correct the prompt, tool, handler,<br/>or expected artifact]
    X --> T
    G -- Yes --> V{Real microphone-to-ASR-to-LLM/tool-to-TTS<br/>round trip passes?}
    V -- No --> Z
    V -- Yes --> J([Handoff the running UI<br/>with exact validation evidence])
```

The main user-controlled moments are the repository location, runtime service
mode, credential setup, scenario selection, and—for custom workflows—the exact
conversation histories and live-validation destination. If the user does not
request a runtime override after disclosure, the skill uses public NVIDIA AI
Endpoints for the LLM, ASR, and TTS. Selecting a bundled preset authorizes only
the disclosed fictional fixture and resolved destination. A custom workflow
requires explicit approval before its histories are transmitted. Every gate
must pass before the workflow moves to the next step; a failed gate is reported
as blocked or pending rather than treated as success.

The appointment-making overlay enforces this patient-facing flow:

```mermaid
flowchart LR
    A[One-time welcome] --> B[/Ask appointment type/]
    B --> C[/Ask date or date range<br/>and morning or afternoon/]
    C --> D[Call find_available_appointments<br/>exactly once]
    D --> E[/Offer at most three slots<br/>and ask user to choose/]
    E --> S[/User selects an offered slot/]
    S --> F[/Ask patient name/]
    F --> G[/Ask date of birth/]
    G --> H[/Ask visit reason/]
    H --> I[Call book_appointment]
    I --> J{Result status is booked?}
    J -- Yes --> K[Report the demo booking<br/>and booking ID]
    J -- No --> L[Do not claim that<br/>an appointment was booked]
```

Knowing the appointment type alone never advances directly to lookup. The
lookup tool also enforces the required morning-or-afternoon preference, so a
prompt error cannot silently broaden the search. Each post-selection booking
field is requested in its own turn, and greetings do not reset or replay the
current slot-selection state.

The patient-intake overlay enforces this patient-facing flow:

```mermaid
flowchart LR
    A[Exact one-time deterministic welcome] --> B[Update shared conversation state]
    B --> C{All five intake<br/>fields known?}
    C -- No --> D[/Ask only the earliest missing field/]
    D --> E[/User answers/]
    E --> B
    C -- Yes --> F[Silently call review_patient_intake<br/>exactly once]
    F --> G[Speak the tool-authored natural review]
    G --> H{User response}
    H -- Correction --> I[Update the corrected field]
    I --> B
    H -- Explicit confirmation --> J[Silently call record_patient_intake<br/>exactly once]
    J --> K[Give the short saved confirmation<br/>and confirmation ID]
```

The five fields are full name, date of birth, current symptoms, current
medications, and preferred pharmacy. The review and save tools author the
spoken text directly. Corrections trigger a complete re-review before another
confirmation can save the record. The runtime speech guard prevents markdown
or field-dump formatting, removes commas between a spoken ordinal day and year,
and discards model preambles before tool results.

## How the Package Is Organized

- `SKILL.md` owns the shared gates, authorization boundaries, and workflow order.
- `references/deployment-modes.md` owns the public, managed-local, existing, and mixed runtime branches.
- The scenario directories own their prompts, tool contracts, expected histories, and scenario tests.
- `scripts/` provides deterministic inspection, overlay application, and live-history validation.
- `tests/` and `evals/` validate the package and publishing contracts; generated reports summarize evidence but do not redefine runtime behavior.
