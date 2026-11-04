# Troubleshoot Spark

Diagnose the smallest useful scope first. Do not treat remembered fixes or a forum post as current product guidance.

## Refresh documentation first

Before asking for more context or diagnostic consent, read
[source-policy.md](references/source-policy.md) and retrieve the current NVIDIA
Known Issues, Release Notes, and documentation for the affected subsystem. For
an update failure, include the OS and Component Update Guide. Read the applicable
workflow references before selecting sources.

Missing model, version, or error details limit applicability and diagnosis; they
do not block retrieving documentation. Read the sources now, state their model
and version scope, then ask for the missing details. Use model-independent guidance
immediately. Label Founders Edition-specific guidance and confirm model
applicability before recommending firmware, recovery, or other model-dependent
procedures. If retrieval fails or is unavailable, report the attempted source
and limitation; do not claim it was verified.

## Establish context

1. Capture the symptom, exact error, impact, onset, and the last relevant change. Ask whether the device still boots and whether the user has a current backup.
2. Establish the model before applying version or recovery guidance. With permission, use the collection's `identify-spark-system` workflow. Otherwise ask the user for the model; do not infer it from a hostname or the word "Spark."
3. Record whether the system is local or headless, whether a display is attached, and whether one or multiple devices are involved. For a cluster, collect the model and software state of every node.
4. Use the refreshed sources to scope the questions and attach links to material claims.

## Consent boundary

Before running commands against the target device or inspecting its state, ask
for explicit permission for a named, minimal diagnostic step and explain what it
reads. Prefer information already provided by the user.

Reading this skill's instruction files and retrieving public documentation do
not require diagnostic consent. Do not defer them until consent to inspect the
device. Respect any explicit user restriction on file or network access.

Ask again before broadening diagnostics, collecting logs or crash dumps, collecting conversation context, installing anything, changing configuration, updating, rebooting, power-cycling, or starting recovery. Authorization to diagnose is not authorization to collect or change. Redact credentials, tokens, keys, usernames, hostnames, addresses, serial numbers, and user data; have the user review any support bundle before it leaves the device.

## Diagnose

1. Check the current [DGX Spark Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html), [Release Notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html), and the NVIDIA page for the affected subsystem. Use the `known-spark-issues` workflow for structured matching.
2. Test one hypothesis at a time. Start with non-mutating observations after consent. Do not prescribe speculative package removal, driver pinning, firmware changes, kernel changes, cache flushing, cable changes, reboots, or power cycles.
3. For update-related symptoms, follow [update-and-recovery.md](references/update-and-recovery.md). Determine whether the user followed the primary DGX Dashboard path or the complete NVIDIA-documented manual fallback, and identify incomplete, mixed, or off-recipe state without inventing a repair.
4. Apply generally valid DGX Spark guidance to a recognized OEM GB10 system only when the NVIDIA source includes that system class or the claim is model-independent. OTA packages are available to all devices when NVIDIA publishes an update; partner OEMs publish firmware independently. Verify firmware versions and firmware or recovery procedures for the specific model.
5. If NVIDIA documentation does not support a conclusion, say so. Use [get-spark-support](../get-spark-support/WORKFLOW.md) for forum searches and author attribution, and keep forum leads subordinate to current documentation.

## Respond

Give the user:

- the observed facts and the leading source-backed explanation;
- model, software-version, operating-mode, and multi-node applicability;
- links beside the claims they support;
- one minimal next step, its risk, and the permission needed before it is performed;
- any uncertainty or source conflict.

For unresolved hardware-health questions, route to `run-spark-diagnostics`. For support escalation, route to `get-spark-support`. Do not run diagnostics or prepare support material until the user consents.
