# Installing or updating NVIDIA Broadcast

Installing software is the most privileged thing this skill can do. It happens **only** after the
user says yes to a direct question, only from NVIDIA's own update service, and only after the
downloaded file proves it is a genuine NVIDIA build.

## When this applies

Exactly two situations, both reached from the version check in `connection.md`:

- `NVIDIA Broadcast.exe` is **not** at the standard path — the app is not installed.
- Its `ProductVersion` is **2.2.x or older** — the build predates the MCP gateway.

Nothing else. A reachable-but-failing gateway, a refused port on a 2.3+ build, NAT-mode WSL, a
rate-limited call, or an effect that would not apply are **not** install problems. Never offer an
install to work around them, and never install as a side effect of an unrelated request — "blur my
background" is not consent to install a large application on someone's machine.

## Ask first — always

Ask in the agent conversation and wait for an explicit answer. Do not pre-download "so it's ready",
do not treat silence, "sure, whatever" or the original task as consent, and do not ask again after a
**decline** — a decline being the user saying no, not an attempt that went wrong. Tell the user what will happen before they answer:

> NVIDIA Broadcast isn't installed on this machine, so I can't drive it over MCP.
> NVIDIA's update service is offering **NVIDIA_Broadcast_v2.2.0.52896077_C9D096** (180 MB) from
> `ota-downloads.nvidia.com`. I'd download it, check its SHA-512 and confirm it's signed by NVIDIA
> Corporation, then open the NVIDIA installer — Windows will ask you for administrator permission
> and you'd click through the install yourself. Heads up: 2.2.x has no MCP gateway, so this gets you
> the app but I still won't be able to control it from here until 2.3.x is published. Want me to go
> ahead?

Adjust the build, size and caveats to what the service actually returned. Keep all four facts —
**what, from where, verified how, and that a UAC prompt is coming** — and add a caveat only when one
applies: when `name` and `version` disagree, or when the offered build is 2.2.x and so has no MCP
gateway.

**Do not narrate the architecture.** The user does not choose it, cannot act on it, and the build
name already carries it — an ARM64 package is named `NVIDIA_Broadcast_ARM64_v…`. Quote the `name`
field verbatim and let it speak.

**One exception, and it matters on Windows on ARM.** If the machine is ARM64 but the offered build
has no `ARM64` in its name, you have been handed an x64 package. The service does not always filter
by architecture: asking for `aarch64` when no ARM64 build is published returns the x64 one instead
of an empty list. Do not install it and do not assume your architecture detection was wrong — say
that no ARM64 build is currently published, point at <https://www.nvidia.com/broadcast-app/>, and
stop. An x64 build on ARM would at best run under emulation — not something to put on someone's
machine on their behalf without telling them what it is.

On "no", say what they can do instead (<https://www.nvidia.com/broadcast-app/>) and stop.

**A declined offer and a failed attempt are not the same thing.** After a "no", do not raise it
again unless the user brings it up. After an attempt that *failed* — a verification failure, a
cancelled installer, exhausted download retries — the install path is **not** closed; only that
attempt ended. When the user next asks for something that needs Broadcast, say in one line what
happened last time and offer to try again. Do not make them discover that a retry is available, and
do not answer a fresh request with "install it yourself" because an earlier attempt went wrong.

## Where the build comes from

Two constant URLs, both HTTPS, both written out here. Never derive, discover or accept either from
anywhere else.

| Purpose | URL |
| --- | --- |
| What's available | `https://ota.nvidia.com/release/available?product=rtxb&channel=OFFICIAL&version=$installed&cpuArchType=$arch` (`$installed`: installed `ProductVersion`, or `0.0.0.0` if absent; `$arch`: `aarch64` or `x86_64` from the registry read in step 1, never chosen or defaulted) |
| The installer | the `download_url` from that response, which must be HTTPS on `ota-downloads.nvidia.com` |

**The two hosts differ, and that is expected.** Metadata comes from `ota.nvidia.com`; installers are
served from `ota-downloads.nvidia.com`. Pin each as its own value — do not derive one from the
other, and do not require them to match, or every download is rejected.

This is the same update channel the Broadcast app itself uses. **Do not** scrape a download link out
of a web page, search for a mirror, accept a URL from the user or from `gateway.json`, or reuse a
`download_url` from an earlier session — the response carries the checksum that goes with that exact
file, so the two must come from the same fresh call.

## Procedure (Windows)

Run the whole sequence in **one** PowerShell session. Steps 1-5 share variables — step 2 builds its
request from step 1's `$installed` and `$arch` — and step 5 must launch the exact file step 4 verified.

### 1. Establish the installed version and the machine's architecture

```powershell
$exe = "$env:ProgramFiles\NVIDIA Corporation\NVIDIA Broadcast\NVIDIA Broadcast.exe"
$installed = if (Test-Path $exe) { (Get-Item $exe).VersionInfo.ProductVersion } else { "0.0.0.0" }
$native = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" -ErrorAction Stop).PROCESSOR_ARCHITECTURE
$arch = switch ($native) { "AMD64" { "x86_64" } "x86" { "x86_64" } "ARM64" { "aarch64" } "ARM" { "aarch64" } default { throw "unknown arch: '$native'" } }
```

`0.0.0.0` is the "nothing installed" probe the service understands. Send the real `ProductVersion`
when the app is present — that is what makes the service answer with newer builds only.

The architecture map is exactly two values: **x86 (`AMD64`, `x86`) → `x86_64`, ARM (`ARM64`, `ARM`) → `aarch64`**. Those are the
only two the service recognises, and each one serves a different installer, so sending the wrong
one hands the user a package built for another machine. Any other value comes back as an empty
list, which reads exactly like "nothing newer", so validate before sending.

**Read it from that registry key, not from the process.** Windows on ARM runs x64 binaries under
emulation, and an emulated process reports itself as x64: `$env:PROCESSOR_ARCHITECTURE` returns
`AMD64`, and `RuntimeInformation`'s `ProcessArchitecture` — and, on .NET Framework, sometimes
`OSArchitecture` too — follow the process rather than the machine. The
`Session Manager\Environment` key is not redirected under emulation, so it reports the **native**
machine either way. On a WoA laptop the process-based checks can silently yield `x86_64` and
download the wrong build; the registry key still says `ARM64`.

From WSL this must likewise be the **Windows** architecture, which is why it is read through
PowerShell and never from `uname` inside the distro.

### 2. Ask the service what it offers

```powershell
try {
  $all = Invoke-RestMethod -ErrorAction Stop -MaximumRedirection 0 -Uri (
    "https://ota.nvidia.com/release/available?product=rtxb&channel=OFFICIAL" +
    "&version=$installed&cpuArchType=$arch")
} catch {
  "update service unreachable: " + $_.Exception.Message; return
}
if (@($all).Count -eq 0) { "nothing newer published"; return }
$rel = @($all)[0]   # the service returns newest-first
```

**"The service said nothing" and "the service never answered" are different, and confusing them is
the worst failure this flow has.** `Invoke-RestMethod` errors are non-terminating by default, so
without `-ErrorAction Stop` a DNS failure, a dropped VPN or a timeout leaves `$all` unset — and a
bare `if (-not $all)` then reports **"nothing newer published"**. The user is told they are already
up to date when nothing was ever reached. Catch the failure explicitly, report it as a network
problem, and stop; that one *is* worth retrying later, unlike a genuine empty list.

An **empty list means there is nothing newer than what is installed** — not a failure to retry.
Tell the user their build is the newest published one, and if it is 2.2.x add that the MCP gateway
first appears in 2.3.x, so there is nothing to install yet. Stop there.

**Count it with `@($all).Count -eq 0`, never `$null -eq $all`.** The service answers `[]`, which
comes back as an empty array: `$null -eq $all` is `$false` for it, so that test falls through to the
install path with no release to install. `@(...)` also normalises the single-record case, which
PowerShell would otherwise hand you as a bare object rather than a list.

**The response is a list, ordered newest-first, and already filtered.** Its length is not something
to reason about: the service returns however many applicable builds it currently publishes for that
architecture, and that changes over time. One record, several, or none are all normal, and no count
is a contract. What does hold is the ordering, and that anything at or below the `version` you sent
has already been excluded. So **take `[0]` and never assume a count.** When more than one record
comes back, say which one you chose.

**Do not re-sort the list on `version` locally.** That field is not always right — see the note
below — and sorting on it would rank a mislabelled build beneath genuinely older releases. The
service's own ordering tracks release recency and is the more reliable signal.

Otherwise read `version`, `size`, `checksum_sha512` and `download_url` from `$rel`, and check three
things before going further:

- `$rel.version` is strictly newer than `$installed`. Never install a build that is the same or
  older, whatever the service says.
- `([uri]$rel.download_url).Scheme` is `https` and `.Host` is exactly `ota-downloads.nvidia.com`.
  Compare against that pinned installer host as its own constant, not against the metadata host.
- `$rel.checksum_sha512` is present and non-empty. No checksum, no install.

If any check fails, stop and report it. Do not "try the URL anyway".

**If `name` and `version` disagree, do not pick one yourself.** The build name embeds a version of
its own (`NVIDIA_Broadcast_v2.2.0.52896077_C9D096`), and the two have been observed to disagree. A
mislabelled `version` cuts two ways: it can trip the downgrade guard above on a build that is really
newer, and it can fire — or wrongly suppress — the 2.2.x-has-no-gateway caveat. It is also why the
selection above uses the service's ordering rather than re-sorting on `version`. Neither guess is
yours to make: report both fields to the user, say which way the discrepancy cuts, and let them
decide. Never silently prefer `name` over `version` to get past a guard.

Now ask the user (above). Everything past this point runs only after a yes.

### 3. Download into a fresh, private directory

```powershell
$dir = Join-Path $env:LOCALAPPDATA ("Temp\nvidia-broadcast-install-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $dir | Out-Null
$pkg = Join-Path $dir "nvidia-broadcast-setup.exe"
try {
  Start-BitsTransfer -Source $rel.download_url -Destination $pkg -ErrorAction Stop
} catch {
  $ProgressPreference = "SilentlyContinue"   # else Windows PowerShell spends longer drawing the bar than downloading
  Invoke-WebRequest -ErrorAction Stop -MaximumRedirection 0 -UseBasicParsing -Uri $rel.download_url -OutFile $pkg
}
```

**Prefer BITS for this transfer.** It is a large file — hundreds of megabytes, and the exact figure
changes with every build — and `Invoke-WebRequest -OutFile` in Windows
PowerShell buffers without resume: one dropped connection leaves a short file on disk and reports
nothing wrong. `Start-BitsTransfer` is the Windows service built for large transfers — it resumes
after an interruption and retries by itself. Fall back to `Invoke-WebRequest` only when BITS is
unavailable or disabled by policy, and expect truncation to be likelier when you do.

`-ErrorAction Stop` on both, for the same reason as step 2: without it a failed transfer is
non-terminating, and the next step would verify a file that is missing or half-written instead of
reporting that the download failed. Note that a transfer can also fail *silently* — ending early
with no error at all — which is why step 4 checks the length before anything else.

This is a large download — the response's `size` field is the exact byte count, so quote that
rather than guessing. Tell the user it is running rather than going quiet on them, and give it
several minutes on a slow link before treating it as stuck.

A brand-new GUID directory under the user's own `%LOCALAPPDATA%\Temp` means nothing can have been
planted at that path in advance and no other user can swap the file between verification and launch.
Do not download to a shared or predictable location, and do not reuse a directory from a previous
attempt.

**The local filename is ours, not theirs.** Write to `nvidia-broadcast-setup.exe`; never build the
path out of the `name` field in the response.

`-MaximumRedirection 0` fails loudly if the download is redirected off the pinned host. If that
happens, the distribution channel has changed — stop and tell the user. Do not follow the redirect.
One retry is reasonable if the transfer itself failed (reset connection, timeout); anything else is
not retryable.

### 4. Verify before running it

All checks must pass before the installer runs — but **check the length first and branch on it.**
The three checks are not equivalent, and collapsing them into one "verification failed" is how a
slow network turns into a false security alarm.

```powershell
$actual = (Get-Item $pkg).Length
if ($rel.size -and $actual -ne $rel.size) {
  "incomplete download: $actual of $($rel.size) bytes"   # transport problem — see below
} else {
  $hashOk = (Get-FileHash -Algorithm SHA512 -Path $pkg).Hash.ToLower() -eq $rel.checksum_sha512.ToLower()
  $sig    = Get-AuthenticodeSignature -FilePath $pkg
  $sigOk  = $sig.Status -eq "Valid" -and $sig.SignerCertificate.Subject -like "CN=NVIDIA Corporation,*"
  "hash=$hashOk signature=$sigOk subject=$($sig.SignerCertificate.Subject)"
}
```

**A short file is an incomplete transfer, not a compromised one.** This is the single most likely
failure in the whole flow — a large file over a real-world link — and it is routine, not suspicious.
Say plainly how far it got, quoting both numbers from `$actual` and `$rel.size`, then **retry the
download**, up to two further attempts, each into a fresh directory from step 3. Only after those
are exhausted do you give up, and even then report it as a network failure, not a security one.

**The only correct comparand is `$rel.size` from this response.** Installer sizes change with every
build, so never sanity-check against a figure from this document, from an earlier release, or from
an earlier session — hard-coding any number here silently rejects the next build. The response that
carried the checksum carries the byte count that goes with it; use that pair and nothing else.

**Do not compute the hash or signature on a short file.** Both fail as a matter of course: the
Authenticode certificate table lives at the end of the image, so a truncated installer always
reports `NotSigned`. Announcing that "size, checksum and signature all failed" makes a dropped
connection look like an attack, and tells the user to distrust a file that is merely unfinished.

**If the response carried no usable `size`,** skip the length branch — that is what the `$rel.size`
guard above does — and let SHA-512 carry both jobs. You then cannot tell a truncated file from a
tampered one, so allow the download **one** retry on a hash mismatch before treating it as final,
and say in your report that the size was unavailable so the distinction could not be made.

**Only a full-size file that fails its hash or signature is a security event.** Correct length with
the wrong content is the tampering signal, and that one is final:

- **Size and SHA-512** confirm you got the file the service described, whole and unmodified.
- **Authenticode** confirms NVIDIA actually signed it, chaining to the machine's own trust store.
  It is the check that still holds if the metadata itself were tampered with, which is why the
  signer subject is matched and not just the status. A genuine build reports
  `CN=NVIDIA Corporation, OU=..., O=NVIDIA Corporation, L=Santa Clara, S=California, C=US`.

In that case: **do not launch the file.** Report which check failed and the full path, tell the user
not to run it, and stop. Do not re-download **this build now** — a full-length file with the wrong
hash will not fix itself, and fetching it again will not change that. Leave it where it is, exactly
as with any other artifact this skill creates.

That ends the current attempt, not the install path. If the user later asks for something needing
Broadcast, offer again as described under *A declined offer and a failed attempt are not the same
thing*.

### 5. Let the user install it

```powershell
Start-Process -FilePath $pkg -Wait
```

No arguments. The NVIDIA installer shows its own UI, Windows raises the UAC prompt, and the user
clicks through — that prompt is the point, and it is the last human checkpoint before a privileged
change. **Never** pass silent or unattended flags (`-s`, `/S`, `/quiet`, `/norestart` and the like),
never add `-Verb RunAs` to elevate on the user's behalf, and never drive the installer UI for them.
If they cancel it, that is a decision — report it and stop. That closes this attempt only; offer
again if they later ask for something that needs Broadcast.

### 6. Confirm, then carry on

Re-read `ProductVersion` from the standard path (step 1). Report the version that is actually
installed now — not the one you expected.

`Start-Process -Wait` returns when the process you launched exits, which for a self-extracting
installer can be before the install itself has finished. If the version has not changed yet, re-read
it for up to 60 seconds before concluding anything. Waiting is fine; launching the installer a second
time is not.

- **2.3.0 or newer** — go back to the normal flow in `connection.md`: launch with
  `scripts/launch_broadcast.ps1`, reread `gateway.json`, probe through your MCP client.
- **Still 2.2.x** — the app is installed and usable on its own, but there is no MCP gateway on that
  build and this skill cannot drive it. Say so plainly, point at
  <https://www.nvidia.com/broadcast-app/>, and stop. Do not launch it, probe for a port, or retry.
- **Unchanged or unreadable** — the install did not complete. Report that; do not reinstall.

## From WSL

Everything above runs on the Windows side through interop, unchanged. Two differences:

- **One invocation.** Each `powershell.exe -NoProfile -Command '...'` is a new process, so
  `$installed`, `$arch`, `$rel`, `$pkg` and `$dir` do not survive between calls. Send steps 1-5 as a
  single command and have it print the offered version, the three verification results and the
  installer path, so the transcript shows what was checked.
- **Interop must exist.** If `powershell.exe` is not on `PATH`, you cannot install anything from
  here. Say so and ask the user to install Broadcast on Windows; do not look for a Linux-side
  installer, package or substitute.

Installing from NAT-mode WSL is allowed and does fix "not installed" — but it does **not** make the
gateway reachable. If `wslinfo --networking-mode` says `nat`, say that too, so the user is not
surprised when a successful install still leaves the gateway unreachable.

## Hard limits

- **One consent, before anything is fetched.** No download, no execution, no system change without an
  explicit yes to the question above.
- **Two URLs, both constant, both HTTPS.** The metadata endpoint on `ota.nvidia.com` and the `download_url`
  on `ota-downloads.nvidia.com`, each pinned as its own value. This is the only
  network access this skill performs from a shell; the rule against reaching the *gateway* from a
  shell is unchanged, and nothing else here licenses fetching, scanning or probing anything.
- **Send the machine's real architecture.** Never hard-code `cpuArchType`, and never fall back to
  the other value when a query comes back empty — an empty result means no build for that machine,
  not "try the other architecture".
- **Never weaken transport security.** No `-SkipCertificateCheck`, no certificate-validation
  callbacks, no falling back to plain HTTP, no proxy the user did not already configure.
- **Never pipe a download into an interpreter.** No `Invoke-Expression`, no `iex`, no executing a
  script fetched over the network. The only downloaded file that ever runs is the verified installer.
- **Never install silently or self-elevate.** Interactive installer, UAC visible, user clicks.
- **Never install anything else.** Not a driver, not a runtime, not a package manager, not the
  NVIDIA App — only the build this endpoint offered.
- **Never uninstall, repair, downgrade or remove an existing install**, and do not delete the
  downloaded installer afterwards. Report where it is and leave it.
- **Retry an incomplete download; never retry a failed hash or signature.** A short file is a
  transport problem and gets up to two more attempts. A full-length file whose hash or signature
  does not match ends the attempt — stop there, but do not treat the install path as closed for the
  session; offer again when the user next asks for something that needs Broadcast. Never install a build the service did not offer for this
  machine's current version.

## Failure modes

| Symptom | Cause | What to do |
| --- | --- | --- |
| Response is `[]` | Nothing newer than the installed build is published | Report it and stop; if installed is 2.2.x, add that MCP arrives in 2.3.x |
| `download_url` host is not `ota-downloads.nvidia.com` | The distribution channel changed, or the response is not trustworthy | Stop and report; never fetch it |
| Response is `[]` on a machine you know has builds | `cpuArchType` was wrong or unrecognised — only `aarch64` and `x86_64` are accepted | Re-read the architecture in step 1; never retry with the other architecture to force a hit |
| `name` says one version, `version` says another | The metadata fields disagree | Report both to the user and let them decide; never override the downgrade guard yourself |
| `Invoke-WebRequest` fails on a redirect | The download was redirected off the pinned host | Stop and report; do not follow it |
| Transfer fails or times out | Transient network problem | At most one retry from step 3, with a fresh directory |
| File is shorter than `size` | Incomplete transfer — the most common failure here | Not a security event. Report how far it got and retry the download, up to two more attempts, each into a fresh directory |
| Full-length file, SHA-512 mismatch | Substituted or corrupted content | Do not run it; report the path and stop. Do not re-download it now — but offer a fresh attempt if the user later asks for something needing Broadcast |
| Signature `NotSigned` on a short file | Expected — the certificate table sits at the end of the image | Ignore it and treat this as an incomplete download; do not report it as a signature problem |
| Full-length file, signature not `Valid` or signer is not NVIDIA | The file is not a genuine, intact NVIDIA build | Do not run it; report the path and stop |
| Signer subject is not `CN=NVIDIA Corporation,...` | Signed, but not by NVIDIA | Do not run it; report the path and stop |
| Installer exits without installing | The user cancelled, or it failed | Report it; do not relaunch or reinstall |
| `ProductVersion` is still 2.2.x afterwards | The installed build predates the gateway | Say MCP needs 2.3.x, link the download page, stop |
| `powershell.exe` missing in WSL | Windows interop is disabled | Ask the user to install on Windows; no Linux-side substitute |
