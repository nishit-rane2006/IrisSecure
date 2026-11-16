param(
    [int]$Epochs = 8,
    [int]$MaxClasses = 200,
    [int]$BatchSize,
    [int]$GradAccum,
    [int]$NumWorkers,
    [switch]$ApproveTraining,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = if (Test-Path ".venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$Arguments = @((Join-Path $PSScriptRoot "train_once.py"), "--epochs", $Epochs, "--max-classes", $MaxClasses)
if ($PSBoundParameters.ContainsKey("BatchSize")) { $Arguments += @("--batch-size", $BatchSize) }
if ($PSBoundParameters.ContainsKey("GradAccum")) { $Arguments += @("--grad-accum", $GradAccum) }
if ($PSBoundParameters.ContainsKey("NumWorkers")) { $Arguments += @("--num-workers", $NumWorkers) }
if ($ApproveTraining) { $Arguments += "--approve-training" }
if ($Force) { $Arguments += "--force" }
& $Python @Arguments
