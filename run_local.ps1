$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $AppDir

Write-Host "==> Running full local app from $AppDir"

if (-not (Test-Path ".\venv")) {
    Write-Host "==> Creating virtualenv"
    py -3 -m venv venv
}

$Python = Join-Path $AppDir "venv\Scripts\python.exe"

Write-Host "==> Installing/updating requirements"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt

$EnvPath = Join-Path $AppDir ".env"
if (Test-Path $EnvPath) {
    Write-Host "==> Loading .env"
    Get-Content $EnvPath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $parts = $line.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($name) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

Write-Host "==> Starting server.py + warm.py through app.py"
& $Python app.py
