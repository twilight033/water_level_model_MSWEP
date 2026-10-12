param(
    [string]$MswepCsv = "D:\code\extract_MSWEP_rainfall\results\mswep_14basins_mean_3hourly_1980_2024.csv",
    [string]$Device = "cuda:0",
    [switch]$PreflightOnly
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$entry = Join-Path $PSScriptRoot "src\evaluation\train_nonrating14_external.py"
if (!(Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (!(Test-Path -LiteralPath $MswepCsv)) { throw "MSWEP file not found: $MswepCsv" }
# 正式训练前检查环境、曲线优化器和数据；不悄悄降级到CPU。
& $python -X utf8 -c "import torch,scipy; assert '$Device' == 'cpu' or torch.cuda.is_available(), 'CUDA unavailable; run on the training computer or explicitly select CPU'"
if ($LASTEXITCODE -ne 0) { throw "Training environment check failed" }
foreach ($scale in @("physical", "observed")) {
    & $python -u -X utf8 $entry --mswep-csv $MswepCsv --target-scaling $scale --device $Device --model-seeds 1 2 3 --mask-seeds 42 123 456 --preflight
    if ($LASTEXITCODE -ne 0) { throw "$scale preflight failed; training has not started" }
}
if ($PreflightOnly) { Write-Host "Preflight completed; training has not started"; exit 0 }
foreach ($scale in @("physical", "observed")) {
    $output = Join-Path $PSScriptRoot "results\nonrating99_consistent_v3_${scale}"
    & $python -u -X utf8 $entry --mswep-csv $MswepCsv --out-dir $output --target-scaling $scale --device $Device --model-seeds 1 2 3 --mask-seeds 42 123 456 --rating-baselines
    if ($LASTEXITCODE -ne 0) { throw "$scale run failed; previous results are preserved; fix and resume" }
}
Write-Host "Completed: 150 neural-network runs across two scaling modes, plus piecewise and power-law rating baselines."
