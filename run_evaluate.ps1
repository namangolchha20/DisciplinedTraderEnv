# Evaluate trained LLM vs baselines.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:HF_HOME = Join-Path $PSScriptRoot "hf-cache"
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot "pip-cache"

$python = Join-Path $PSScriptRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Missing $python — run: python -m venv .venv312 && .\.venv312\Scripts\pip install -r requirements.txt"
}

& $python evaluate.py
