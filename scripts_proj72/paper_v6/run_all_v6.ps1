$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$bashScript = "$root/scripts_proj72/paper_v6/run_all_v6.sh"

if (-not (Test-Path $bashScript)) {
    throw "Missing script: $bashScript"
}

$cmd = "cd '$root' && bash '$bashScript'"
Write-Host "[Run] wsl bash -lc `"$cmd`""
wsl bash -lc $cmd
