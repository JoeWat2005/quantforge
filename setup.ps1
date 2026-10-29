$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Check-Result { if ($LASTEXITCODE -ne 0) { throw 'Setup step failed. See output above.' } }
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    Check-Result
}
& .venv/Scripts/python.exe -m pip install -r requirements.txt
Check-Result
& .venv/Scripts/python.exe scripts/build.py
Check-Result
Push-Location -LiteralPath frontend
try {
    npm.cmd ci
    Check-Result
    npm.cmd run build
    Check-Result
} finally { Pop-Location }
Write-Host 'QUANTFORGE is ready. Run ./start.ps1, then open http://127.0.0.1:8000'
