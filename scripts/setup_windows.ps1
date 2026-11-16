param(
    [string]$PythonCommand = "python",
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $PythonCommand -m venv .venv
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($CpuOnly) {
    & $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    & $Python -m pip install torch torchvision
}
& $Python -m pip install -r requirements.txt
Write-Host "Environment ready. PyTorch backend selection will follow detected runtime support."
