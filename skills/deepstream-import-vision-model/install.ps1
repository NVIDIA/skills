# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# deepstream-import-vision-model - Windows (PowerShell) installer.
# The native-Windows twin of install.sh, with the IDENTICAL sequence of operations:
#   1. Copy the whole self-contained skill into <Target>\.claude\skills\deepstream-import-vision-model\
#      (and .cursor\skills\ unless -NoCursor), minus machine-local build artifacts.
#   2. Print the next steps (bootstrap IN the container; nothing installs on the host).
# This skill has no plugin dependency, so there's no plugin/scope step (unlike the eval skill).
# The copy is the one host-side action that can't run in a container; everything ELSE runs through
# Docker (see references/windows.md). PowerShell 5.1+ compatible.
#
# Usage (mirror of: bash install.sh --target <p> [--no-cursor] [--dry-run]):
#   .\install.ps1 -Target C:\path\to\project [-NoCursor] [-DryRun]
#   # default -Target = current directory.
param(
    [string]$Target = (Get-Location).Path,
    [switch]$NoCursor,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'

$SkillDir  = $PSScriptRoot
$SkillName = 'deepstream-import-vision-model'

if (-not (Test-Path -LiteralPath $Target -PathType Container)) { throw "Target directory not found: $Target" }
$Target = (Resolve-Path -LiteralPath $Target).Path

# Copy the whole self-contained skill into a skills dir, minus machine-local build artifacts
# (the venv + compiled parser .so are rebuilt in-container by setup.sh).
function Install-SkillTo([string]$SkillsDir) {
    $dest = Join-Path $SkillsDir $SkillName
    if ($DryRun) { Write-Host "  [dry-run] copy $SkillDir -> $dest  (whole skill; minus __pycache__/*.pyc/*.so/*.o)"; return }
    if (Test-Path -LiteralPath $dest) {
        $sourceResolved = (Resolve-Path -LiteralPath $SkillDir).Path
        $destResolved = (Resolve-Path -LiteralPath $dest).Path
        if ([StringComparer]::OrdinalIgnoreCase.Equals($sourceResolved, $destResolved)) {
            Write-Host "  Already installed at $dest; source and destination are identical"
            return
        }
        Remove-Item -LiteralPath $dest -Recurse -Force
    }
    # Create the destination, then copy the source's CHILDREN into it one by one.
    # `Copy-Item -LiteralPath <dir> -Destination <dir> -Recurse` is not portable: whether it copies
    # the folder's contents or the folder itself depends on whether the destination already exists,
    # and on Windows PowerShell 5.1 it has been observed to create the directory tree without
    # copying any leaf files -- producing an empty skill dir and an unloadable skill. Enumerating
    # children and copying each explicitly is well-defined on every version.
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Get-ChildItem -LiteralPath $SkillDir -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Recurse -Force
    }
    Get-ChildItem -LiteralPath $dest -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    # Filter on the extension explicitly instead of -Include. `-Include` is a path filter whose
    # behaviour when combined with -LiteralPath is version-dependent: where it fails to apply,
    # Get-ChildItem returns EVERY file, and this pipeline would then delete the entire installed
    # skill (SKILL.md included) while leaving the directory tree behind -- silently, because the
    # removal suppresses errors. Where-Object cannot over-match, so the blast radius is fixed.
    $strip = @('.pyc', '.so', '.o')
    Get-ChildItem -LiteralPath $dest -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $strip -contains $_.Extension } |
        Remove-Item -Force -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath (Join-Path $dest 'SKILL.md'))) {
        throw "Install failed: SKILL.md is missing from $dest -- the skill would not load. Please report this with your `$PSVersionTable.PSVersion."
    }
    Write-Host "  Copied self-contained skill -> $dest"
}

Write-Host "=== deepstream-import-vision-model Install (Windows) ==="
Write-Host "Skill dir: $SkillDir"
Write-Host "Target:    $Target"
Write-Host "Cursor:    $(if ($NoCursor) { 'disabled (-NoCursor)' } else { 'enabled' })"
Write-Host ""

Write-Host "Claude Code skills -> $Target\.claude\skills\"
Install-SkillTo (Join-Path $Target '.claude\skills')

Write-Host ""
Write-Host "Codex skills -> $Target\.codex\skills\"
Install-SkillTo (Join-Path $Target '.codex\skills')

if (-not $NoCursor) {
    Write-Host ""
    Write-Host "Cursor skills -> $Target\.cursor\skills\"
    Install-SkillTo (Join-Path $Target '.cursor\skills')
}

Write-Host ""
Write-Host "=== Done ==="
Write-Host @"

Next - bootstrap the environment IN the container (nothing installs on the host), from the project root:
  docker run --rm -it --gpus all --shm-size=16g -v "`${PWD}:/work" -w /work ``
    --entrypoint bash nvcr.io/nvidia/deepstream:9.1-triton-multiarch ``
    .claude/skills/$SkillName/setup.sh
  (see .claude/skills/$SkillName/references/windows.md for the full Windows runbook)

Claude Code - invoke the skill:
  Use deepstream-import-vision-model to run this model: https://huggingface.co/onnx-community/yolov8n
"@
if (-not $NoCursor) {
    Write-Host ""
    Write-Host "Cursor - invoke the skill:"
    Write-Host "  @deepstream-import-vision-model run this model: https://huggingface.co/onnx-community/yolov8n"
}
