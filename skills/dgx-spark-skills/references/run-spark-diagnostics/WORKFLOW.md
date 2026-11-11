# Run Spark Diagnostics

Use NVIDIA Field Diagnostics only for DGX Spark hardware-health validation or an
RMA pre-check. It stresses major hardware components, changes the boot/security
state, stops the graphical session, and produces logs that may contain system
information.

## Establish authority and applicability

Before the first substantive answer in a diagnostic case, including an
advice-only prerequisite or remedy question, attempt to retrieve both the current
support page and the Field Diagnostics PDF below. Read the relevant procedure;
retrieval alone does not establish that its contents were reviewed. Do this before
asking for diagnostic consent: documentation retrieval is not device inspection,
installation, or test execution. A user-provided simulated preflight is evidence
for that scenario, not permission to inspect the host.

Track each source separately in the answer as `Reviewed` (with retrieval date),
`Attempted — unavailable/unreadable`, or `Not checked`. Never list an unattempted
PDF as a consulted source. If the PDF cannot be read, accessible official HTML may
support the remedy; state the PDF limitation. Reuse the procedure already reviewed
in the same case for follow-up prerequisite questions unless relevant context or
source currency changes.

Official sources:

- [DGX Spark Field Diagnostic Software](https://docs.nvidia.com/dgx/dgx-spark/support.html#field-diagnostic-software)
- [Field Diagnostics User Guide (PDF)](https://docs.nvidia.com/pdf/userguide-dgx-spark-fieldiag.pdf)
- [NVIDIA DGX Spark Field Diagnostics support article](https://nvidia.custhelp.com/app/answers/detail/a_id/5767)

Treat Customer Help article 5767 as a user-browser source. Automated agent or
script requests may return HTTP 403; this is expected and does not establish
that the link is broken or the service is unavailable. Ask the user to open it
in their browser and share relevant instructions if needed. Do not repeatedly
retry or bypass the restriction. Consult the accessible support page and PDF;
do not claim to have verified the article’s current content without seeing it.

Then read [field-diagnostics.md](references/field-diagnostics.md). It
summarizes the workflow and the details that must be refreshed.

Related OEM GB10 systems will most likely share much of the same software
packaging and filesystem layout. NVIDIA's public Field Diagnostics documentation
does not explicitly exclude these systems. However, package prerequisites, UEFI
behavior, recovery procedures, and RMA routing may differ by OEM and should be
verified. Do not provide undocumented OEM-specific diagnostic or recovery
instructions.

## Preserve explicit authorization

Planning and explaining commands is allowed without execution. Before acting,
obtain explicit user permission for each applicable action class:

1. prerequisite inspection;
2. repository or package installation, upgrade, or removal;
3. rebooting, changing Secure Boot, stopping GDM, or switching runlevel;
4. executing the stress test;
5. discovering, reading, copying, or packaging diagnostic logs.

An approval for one class does not authorize another, and prior troubleshooting
consent does not carry into this workflow. State the exact commands, expected
side effects, approximate duration, and recovery plan before requesting
permission. Do not begin a run unless the user understands and authorizes the
post-run restoration plan.

## Run the workflow

1. **Refresh the procedure.** Reconcile the live NVIDIA sources with the bundled
   reference. Prefer the newer NVIDIA instruction when they differ. Stop and
   report unresolved command, package, or applicability conflicts rather than
   guessing.
2. **Prepare the user.** Ask them to save work, stop important workloads, use
   stable power, and choose a reliable local TTY or SSH path. Explain that the
   GUI will stop, the default run usually takes about 30 minutes but can take
   longer, and the run stresses CPU, GPU, memory, storage, power, thermals, C2C,
   and ConnectX-7.
3. **Inspect prerequisites after permission.** Confirm package state, required
   tools, access, and Secure Boot state. Record whether Secure Boot and GDM were
   enabled so restoration returns only settings changed for this run.
4. **Install only after separate permission.** Follow the current official
   package and CX7 prerequisite sequence. Explain every repository addition and
   package change. Do not overwrite an existing keyring or source definition,
   delete an old install, or remove packages without specific approval.
5. **Guide user to disable Secure Boot if enabled.** Obtain permission and confirm
   the user can interact with UEFI setup before rebooting. Guide them through the
   documented firmware menu, saving changes, and rebooting. Normal OS/SSH access
   cannot operate this menu; resume after reconnection and verify Secure Boot is
   disabled. Do not invent firmware automation or EFI-variable writes.
6. **Execute only after permission.** Use the unmodified full default command
   `sudo ./partnerdiag --field`. Do not edit the Spec JSON or select a partial
   test set unless NVIDIA Support explicitly directs it.
7. **Handle interruption safely.** Never retry automatically. Explain that
   another attempt requires a power cycle and obtain permission. If the user
   declines, stop testing and report any settings still changed from their
   recorded state. Offer separately authorized restoration of GDM and a
   user-guided Secure Boot restoration. If restoration is declined or cannot be
   completed safely, leave the remaining changes explicit in the handoff; do
   not claim the system is restored.
8. **Report results without overclaiming.** Preserve the displayed `PASS`,
   `FAIL`, or `RETEST` banner verbatim. For approved log inspection, identify the
   matching timestamped run directory and summarize `summary.json`, including
   test, virtual ID, notes, and exact error code. A 12-digit suffix other than
   `000000000000` indicates an error. A failure is evidence for NVIDIA review,
   not an automatic RMA authorization or a license to repair hardware.
9. **Restore the system.** Restore both GDM's active and enabled states if this
   workflow changed them. If Secure Boot was enabled before the run, guide the
   user to re-enable it in UEFI, save, and reboot after permission. Resume after
   reconnection and verify both Secure Boot and the
   graphical/remote access path. Do not remove packages that predated the
   workflow. Offer separately authorized cleanup only for components installed
   solely for this run.
10. **Prepare escalation, do not submit it.** With separate permission, hand the
    relevant diagnostic logs to the `get-spark-support` workflow for redaction, review, and
    channel selection. Never upload logs or open an RMA/support case as part of
    this workflow.

## Stop conditions

Stop before mutation or execution when the system is not positively identified,
the current official sources disagree materially, a required tool is missing,
Secure Boot cannot be restored, reliable access after `init 3` is uncertain, or
the user declines any required action. Preserve the current state and provide a
concise next-step plan.
