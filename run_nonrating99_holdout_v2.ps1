param(
    [string]$MswepCsv = "D:\code\extract_MSWEP_rainfall\results\mswep_14basins_mean_3hourly_1980_2024.csv",
    [string]$Device = "cuda:0"
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$entry = Join-Path $PSScriptRoot "src\evaluation\train_nonrating14_external.py"
if (!(Test-Path -LiteralPath $python)) { throw "缺少虚拟环境：$python" }
if (!(Test-Path -LiteralPath $MswepCsv)) { throw "MSWEP文件不存在：$MswepCsv" }
# 先验证两种尺度；所有预检通过后才启动正式训练。
foreach ($scale in @("physical", "observed")) {
    & $python -u -X utf8 $entry --mswep-csv $MswepCsv --target-scaling $scale --device $Device --model-seeds 1 2 3 --mask-seeds 42 123 456 --random-only --preflight
    if ($LASTEXITCODE -ne 0) { throw "$scale 预检失败，尚未启动正式训练。" }
}
# 新目录与旧协议隔离；中断后再次执行会跳过已完成的新协议运行。
foreach ($scale in @("physical", "observed")) {
    $output = Join-Path $PSScriptRoot "results\nonrating99_random_${scale}_v2"
    & $python -u -X utf8 $entry --mswep-csv $MswepCsv --out-dir $output --target-scaling $scale --device $Device --model-seeds 1 2 3 --mask-seeds 42 123 456 --random-only
    if ($LASTEXITCODE -ne 0) { throw "$scale 训练失败；修复后可重复执行本命令续跑。" }
}
Write-Host "两种尺度随机留出重跑完成，共108次训练。"
