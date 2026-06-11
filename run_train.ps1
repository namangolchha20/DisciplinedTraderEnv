# Train the GRPO agent with the project venv.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:HF_HOME = Join-Path $PSScriptRoot "hf-cache"
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot "pip-cache"
$env:TMP = Join-Path $PSScriptRoot ".tmp"
$env:TEMP = $env:TMP
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$python = Join-Path $PSScriptRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Missing $python — run: python -m venv .venv312 && .\.venv312\Scripts\pip install -r requirements.txt"
}

& $python inference.py
