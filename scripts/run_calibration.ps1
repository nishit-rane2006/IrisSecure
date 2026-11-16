param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = if (Test-Path ".venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$Arguments = @((Join-Path $PSScriptRoot "calibrate_once.py"))
if ($Force) { $Arguments += "--force" }
& $Python @Arguments
