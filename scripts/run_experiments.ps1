param(
    [int]$Seed = 42,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = if (Test-Path ".venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$Arguments = @((Join-Path $PSScriptRoot "run_experiments.py"), "--seed", $Seed)
if ($Force) { $Arguments += "--force" }
& $Python @Arguments
