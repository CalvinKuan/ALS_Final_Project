param(
    [int]$Jobs = 0,
    [int]$WorkersPerCase = 0,
    [ValidateSet("quick", "medium", "high")]
    [string]$Effort = "medium",
    [string]$Python = "python",
    [switch]$CecFinal,
    [string]$ParetoDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$optimizer = Join-Path $root "student\optimizer.py"
$evaluator = Join-Path $root "evaluate.py"

$optimizerArgs = @($optimizer, "--jobs", $Jobs, "--effort", $Effort)
if ($WorkersPerCase -gt 0) {
    $optimizerArgs += @("--max-workers", $WorkersPerCase)
}
if ($CecFinal) {
    $optimizerArgs += "--cec-final"
}
if ($ParetoDir -ne "") {
    $optimizerArgs += @("--pareto-dir", $ParetoDir)
}

Write-Host "Running optimizer: jobs=$Jobs workers_per_case=$WorkersPerCase effort=$Effort"
& $Python @optimizerArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=== All cases done. Running evaluate.py ==="
& $Python $evaluator
exit $LASTEXITCODE
