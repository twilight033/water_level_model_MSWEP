param(
    [string]$PythonExe = "",
    [string]$Device = "auto",
    [string]$GeometryMode = "snapshot",
    [string]$Stage = "both",
    [string]$OutDir = "",
    [switch]$PreflightOnly,
    [switch]$Smoke,
    [switch]$Setup
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# 只探测现有解释器；不修改已安装的全局或Conda环境。
$candidates = @()
if ($PythonExe) { $candidates += $PythonExe }
elseif ($env:HYDRO_PYTHON) { $candidates += $env:HYDRO_PYTHON }
else {
    $candidates += (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")
    if ($env:CONDA_PREFIX) { $candidates += (Join-Path $env:CONDA_PREFIX "python.exe") }
    $pythonCommands = @(Get-Command python.exe -All -ErrorAction SilentlyContinue)
    foreach ($command in $pythonCommands) { $candidates += $command.Source }
    $condaCommands = @(Get-Command conda.exe,conda.bat -All -ErrorAction SilentlyContinue)
    foreach ($command in $condaCommands) {
        if ($command.Source) {
            $condaRoot = Split-Path -Parent (Split-Path -Parent $command.Source)
            $candidates += (Join-Path $condaRoot "python.exe")
        }
    }
}
$selectedPython = ""
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (!(Test-Path -LiteralPath $candidate)) { continue }
    if ($candidate -like "*WindowsApps*") { continue }
    $probePreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $candidate -c "import sys; assert (3,10)<=sys.version_info[:2]<(3,13); import torch,numpy,pandas,pyarrow,scipy" 2>$null
    } finally { $ErrorActionPreference = $probePreference }
    if ($LASTEXITCODE -eq 0) { $selectedPython = $candidate; break }
}
if (!$selectedPython -and ($PythonExe -or $env:HYDRO_PYTHON)) {
    throw "指定的Python缺少依赖或版本不在3.10—3.12，请检查PythonExe/HYDRO_PYTHON。"
}
if (!$selectedPython) {
    $uv = Get-Command uv.exe -ErrorAction SilentlyContinue
    if (!$uv) { throw "未找到可用环境或uv。安装uv，或通过-PythonExe指定带PyTorch的Python 3.10—3.12。" }
    Write-Host "建立项目内.venv并安装依赖，不更改现有环境。首次安装需要网络。"
    $selectedPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (!(Test-Path -LiteralPath $selectedPython)) {
        & $uv.Source venv --python 3.12 (Join-Path $PSScriptRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "创建环境失败" }
    }
    # CUDA轮子仅装入新建的项目环境；CPU模式使用PyPI版本。
    if ($Device -ne "cpu") {
        & $uv.Source pip install --python $selectedPython torch --index-url https://download.pytorch.org/whl/cu126
        if ($LASTEXITCODE -ne 0) { throw "安装PyTorch失败，请检查网络或选择现有训练环境" }
    }
    & $uv.Source pip install --python $selectedPython -r (Join-Path $PSScriptRoot "requirements_cross_section86.txt")
    if ($LASTEXITCODE -ne 0) { throw "安装依赖失败" }
}
Write-Host "Python: $selectedPython"
if ($Setup) { Write-Host "环境检查完成，尚未训练。"; exit 0 }
$entry = Join-Path $PSScriptRoot "src\cross_section_experiment\runner.py"
$arguments = @("-u", "-X", "utf8", $entry, "--device", $Device, "--geometry-mode", $GeometryMode, "--stage", $Stage)
if ($OutDir) { $arguments += @("--out-dir", $OutDir) }
if ($Smoke) { $arguments += "--smoke" }
& $selectedPython @arguments --preflight
if ($LASTEXITCODE -ne 0) { throw "预检失败，未启动正式训练。" }
if ($PreflightOnly) { exit 0 }
& $selectedPython @arguments
if ($LASTEXITCODE -ne 0) { throw "运行失败，已完成结果保留；修复后用同一配置续跑。" }
Write-Host "所选两个阶段已完成。"
