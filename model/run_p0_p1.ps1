# run_p0_p1.ps1 -- P0 + P1 pipeline (backup runner)
#
# NOTE: run_p0_p1.py is the recommended runner. Use this only if you prefer PowerShell.
# This file MUST stay saved as "UTF-8 with BOM". Windows PowerShell 5.1 decodes a
# BOM-less .ps1 with the system ANSI code page (GBK on Chinese Windows), which
# corrupts non-ASCII bytes and swallows line breaks -> "unexpected token }".
# All comments here are ASCII on purpose, so the file survives a re-save either way.
#
# Usage (PyCharm Terminal, working dir = Model):
#   .\run_p0_p1.ps1
#   .\run_p0_p1.ps1 -Types slope,braking
#   .\run_p0_p1.ps1 -CellVolumes cellV.npy
#   .\run_p0_p1.ps1 -Stride 2
#   .\run_p0_p1.ps1 -SkipRebuild
#
# If PowerShell blocks scripts, run once:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

param(
    [string[]] $Types = @('slope','braking','turning','overtaking'),
    [string]   $CellVolumes = '',
    [int]      $Stride = 1,
    [switch]   $SkipRebuild,
    [switch]   $Fast
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

# It simply delegates to run_p0_p1.py so there is only one place to maintain.
$argsList = @('--types') + $Types
if ($CellVolumes) { $argsList += @('--cell-volumes', $CellVolumes) }
if ($Stride -gt 1) { $argsList += @('--stride', "$Stride") }
if ($SkipRebuild)  { $argsList += '--skip-rebuild' }
if ($Fast)         { $argsList += '--fast' }

Write-Host "python run_p0_p1.py $($argsList -join ' ')" -ForegroundColor Cyan
python run_p0_p1.py @argsList
