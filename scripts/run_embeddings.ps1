param(
    [int]$BatchSize,
    [int]$NumWorkers,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = if (Test-Path ".venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$Arguments = @((Join-Path $PSScriptRoot "extract_embeddings.py"))
if ($PSBoundParameters.ContainsKey("BatchSize")) { $Arguments += @("--batch-size", $BatchSize) }
if ($PSBoundParameters.ContainsKey("NumWorkers")) { $Arguments += @("--num-workers", $NumWorkers) }
if ($Force) { $Arguments += "--force" }
& $Python @Arguments
