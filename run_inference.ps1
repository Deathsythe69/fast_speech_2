param(
    [Parameter(Position=0)]
    [string]$Lang = "english",
    [Parameter(Position=1)]
    [string]$Checkpoint = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$pythonExec = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $pythonExec = ".venv\Scripts\python.exe"
}

if ($Checkpoint -ne "") {
    & $pythonExec tools/run_pipeline.py --lang $Lang --checkpoint $Checkpoint
} else {
    & $pythonExec tools/run_pipeline.py --lang $Lang
}

