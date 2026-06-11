# Start the trading terminal with the project venv (torch + unsloth live here).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:HF_HOME = Join-Path $PSScriptRoot "hf-cache"
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot "pip-cache"
$env:TMP = Join-Path $PSScriptRoot ".tmp"
$env:TEMP = $env:TMP

$python = Join-Path $PSScriptRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Missing $python — run: python -m venv .venv312 && .\.venv312\Scripts\pip install -r requirements.txt"
}

& $python -m uvicorn server.app:app --host 127.0.0.1 --port 7860
