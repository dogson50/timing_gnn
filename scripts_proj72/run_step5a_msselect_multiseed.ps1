param(
	[string]$Python = "python",
	[string]$DataPath = "output",
	[string]$DatasetPkl = "dataset.pkl",
	[int]$Gpu = 0,
	[double]$Lr = 1e-3,
	[int]$NumEpoch = 500,
	[string]$RunTag = "step5a_msselect",
	[string[]]$Seeds = @("9294", "2025", "6666"),
	[switch]$FreezeHgat,
	[switch]$UseWsl,
	[string]$WslDistro = "Ubuntu-20.04",
	[string]$WslPython = ""
)

$ErrorActionPreference = "Stop"

function Resolve-PythonCommand {
	param([string]$Preferred)

	$venvPy = Join-Path (Resolve-Path ".").Path ".venv\Scripts\python.exe"
	if (Test-Path $venvPy) {
		return $venvPy
	}

	if ($Preferred -and $Preferred.Trim().Length -gt 0) {
		$cmd = Get-Command $Preferred -ErrorAction SilentlyContinue
		if ($cmd) {
			return $Preferred
		}
	}

	$cmdPy = Get-Command python -ErrorAction SilentlyContinue
	if ($cmdPy) {
		return "python"
	}

	throw "Python executable not found. Set -Python to a valid interpreter path, or create .venv in workspace root."
}

function Convert-WindowsPathToWsl {
	param([string]$Path)
	$full = [System.IO.Path]::GetFullPath($Path)
	if ($full -match "^([A-Za-z]):\\(.*)$") {
		$drv = $Matches[1].ToLower()
		$rest = $Matches[2] -replace "\\", "/"
		return "/mnt/$drv/$rest"
	}
	return ($full -replace "\\", "/")
}

function Invoke-Train {
	param(
		[string[]]$TrainArgs,
		[switch]$RunInWsl,
		[string]$PythonCmd,
		[string]$Distro,
		[string]$WslPy
	)

	if (-not $RunInWsl) {
		& $PythonCmd @TrainArgs
		return $LASTEXITCODE
	}

	$pwdWsl = Convert-WindowsPathToWsl -Path (Resolve-Path ".").Path
	$py = $WslPy
	if (-not $py) {
		$py = "python"
	}
	Write-Host "[Info] WSL exec: $py $($TrainArgs -join ' ')"
	& wsl.exe -d $Distro --cd $pwdWsl -- $py @TrainArgs
	return $LASTEXITCODE
}

if ($UseWsl) {
	Write-Host "[Info] Running in WSL distro: $WslDistro"
	if ($WslPython -and $WslPython.Trim().Length -gt 0) {
		Write-Host "[Info] WSL Python: $WslPython"
	}
	else {
		Write-Host "[Info] WSL Python: python (PATH lookup in WSL)"
	}
}
else {
	$Python = Resolve-PythonCommand -Preferred $Python
	Write-Host "[Info] Python: $Python"
}

$SeedInts = @()
foreach ($s in $Seeds) {
	if ($null -eq $s) {
		continue
	}
	$parts = "$s" -split "[,;\s]+" | Where-Object { $_ -and $_.Trim().Length -gt 0 }
	foreach ($p in $parts) {
		$SeedInts += [int]$p
	}
}

if ($SeedInts.Count -eq 0) {
	throw "No valid seeds provided. Example: -Seeds 9294,2025,6666 or -Seeds 9294 2025 6666"
}

function Parse-BestLine {
	param([string]$Path)

	if (-not (Test-Path $Path)) {
		return $null
	}

	$line = Select-String -Path $Path -Pattern "\[Best\] epoch:(\d+), val_r2:([\-0-9\.]+), val_loss:([\-0-9\.eE]+)" | Select-Object -Last 1
	if ($null -eq $line) {
		return $null
	}

	$m = [regex]::Match($line.Line, "\[Best\] epoch:(\d+), val_r2:([\-0-9\.]+), val_loss:([\-0-9\.eE]+)")
	if (-not $m.Success) {
		return $null
	}

	return [pscustomobject]@{
		BestEpoch = [int]$m.Groups[1].Value
		BestR2 = [double]$m.Groups[2].Value
		BestValLoss = [double]$m.Groups[3].Value
	}
}

$results = @()
foreach ($seed in $SeedInts) {
	$modelDir = "model_${RunTag}_seed${seed}"
	Write-Host "[Run] seed=$seed -> $modelDir"
	$extraArgs = @()
	if ($FreezeHgat) {
		$extraArgs += "--freeze_hgat"
	}

	$trainArgs = @(
		"-u",
		"src/train_balanced_sampling_sep_mlp_shared_calib_step5a_msselect_opt.py",
		"--model_saving_dir", $modelDir,
		"--data_save_path", $DataPath,
		"--dataset_pkl_name", $DatasetPkl,
		"--task", "reg",
		"--gpu", "$Gpu",
		"--seed", "$seed",
		"--learning_rate", "$Lr",
		"--num_epoch", "$NumEpoch"
	)
	$trainArgs += $extraArgs

	$exitCode = Invoke-Train -TrainArgs $trainArgs -RunInWsl:$UseWsl -PythonCmd $Python -Distro $WslDistro -WslPy $WslPython
	if ($exitCode -ne 0) {
		Write-Warning "Training command exited with code $exitCode for seed=$seed"
	}

	$stdoutPath = Join-Path $modelDir "stdout.log"
	$best = Parse-BestLine -Path $stdoutPath
	if ($null -eq $best) {
		Write-Warning "Cannot parse [Best] from $stdoutPath"
		$results += [pscustomobject]@{
			Seed = $seed
			BestEpoch = $null
			BestR2 = $null
			BestValLoss = $null
			ModelDir = $modelDir
		}
	}
	else {
		$results += [pscustomobject]@{
			Seed = $seed
			BestEpoch = $best.BestEpoch
			BestR2 = $best.BestR2
			BestValLoss = $best.BestValLoss
			ModelDir = $modelDir
		}
	}
}

$results | Format-Table -AutoSize

$valid = $results | Where-Object { $_.BestR2 -ne $null }
if ($valid.Count -gt 0) {
	$r2 = $valid | Select-Object -ExpandProperty BestR2
	$loss = $valid | Select-Object -ExpandProperty BestValLoss
	$meanR2 = ($r2 | Measure-Object -Average).Average
	$meanLoss = ($loss | Measure-Object -Average).Average

	$stdR2 = 0.0
	if ($r2.Count -gt 1) {
		$sumSq = 0.0
		foreach ($x in $r2) {
			$sumSq += [math]::Pow(($x - $meanR2), 2)
		}
		$stdR2 = [math]::Sqrt($sumSq / ($r2.Count - 1))
	}

	Write-Host "[Summary] runs=$($valid.Count), mean_r2=$([math]::Round($meanR2,6)), std_r2=$([math]::Round($stdR2,6)), mean_val_loss=$([math]::Round($meanLoss,6))"
}

$csvPath = "output/${RunTag}_multiseed_results.csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
Write-Host "[Saved] $csvPath"
