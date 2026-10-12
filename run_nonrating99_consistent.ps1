param(
    [string]$MswepCsv = "D:\code\extract_MSWEP_rainfall\results\mswep_14basins_mean_3hourly_1980_2024.csv",
    [string]$Device = "cuda:0",
    [switch]$PreflightOnly
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$entry = Join-Path $PSScriptRoot "src\evaluation\train_nonrating14_external.py"
if (!(Test-Path -LiteralPath $python)) { throw "缺少虚拟环境：$python" }
if (!(Test-Path -LiteralPath $MswepCsv)) { throw "MSWEP文件不存在：$MswepCsv" }
# 正式训练前检查环境、曲线优化器和数据；不悄悄降级到CPU。
& $python -X utf8 -c "import torch,scipy; assert '$Device' == 'cpu' or torch.cuda.is_available(), 'CUDA不可用，请在训练电脑执行或明确指定CPU'"
if ($LASTEXITCODE -ne 0) { throw "训练环境检查失败" }
foreach ($scale in @("physical", "observed")) {
    & $python -u -X utf8 $entry --mswep-csv $MswepCsv --target-scaling $scale --device $Device --model-seeds 1 2 3 --mask-seeds 42 123 456 --preflight
    if ($LASTEXITCODE -ne 0) { throw "$scale 预检失败，尚未开始训练" }
}
if ($PreflightOnly) { Write-Host "仅预检完成，未启动训练"; exit 0 }
foreach ($scale in @("physical", "observed")) {
    $output = Join-Path $PSScriptRoot "results\nonrating99_consistent_v3_${scale}"
    & $python -u -X utf8 $entry --mswep-csv $MswepCsv --out-dir $output --target-scaling $scale --device $Device --model-seeds 1 2 3 --mask-seeds 42 123 456 --rating-baselines
    if ($LASTEXITCODE -ne 0) { throw "$scale 运行失败；旧结果未覆盖，可以修复后续跑" }
}
Write-Host "完整矩阵完成：两种尺度共150次神经网络训练，另含分段和幂律独立率定对照。"
