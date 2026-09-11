$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$python = 'D:\Anaconda\python.exe'

foreach ($seed in 42, 123, 2026) {
    Write-Host "[$(Get-Date -Format s)] starting Brusselator seed $seed"
    & $python -u (Join-Path $root 'scripts\run_solver_v5_brusselator.py') --stage all --seed $seed
    if ($LASTEXITCODE -ne 0) {
        throw "Brusselator seed $seed failed with exit code $LASTEXITCODE."
    }
    Write-Host "[$(Get-Date -Format s)] completed Brusselator seed $seed"
}
