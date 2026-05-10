param(
    [string[]]$Presets = @("A", "B", "C"),
    [string[]]$Seeds = @("9294", "2025", "6666"),

    [string]$ModelPrefix = "model_sep_hgat_ctrl",
    [string]$DataPath = "output",
    [string]$DatasetPkl = "dataset.pkl",
    [int]$Gpu = 0,

    [int]$NumEpoch = 260,
    [int]$BatchSize = 3072,
    [double]$LearningRate = 3e-3,
    [double]$EncLrScale = 0.05,
    [double]$WeightDecay = 2e-5,
    [double]$MlpDropout = 0.15,
    [double]$ZNoiseStd = 0.01,
    [double]$HgatParCapWeight = 0.6,

    [string]$LrScheduler = "plateau",
    [double]$PlateauFactor = 0.5,
    [int]$PlateauPatience = 3,
    [double]$PlateauThreshold = 3e-4,
    [double]$PlateauMinLr = 1e-6,
    [int]$EarlyStopPatience = 10,
    [double]$EarlyStopMinDelta = 8e-4,

    [int]$NumWorkers = 0,
    [int]$ValEvalInterval = 5,
    [int]$TestEvalInterval = 100,
    [bool]$SkipTestEval = $true,

    [switch]$UseWslConda,
    [string]$WslDistro = "Ubuntu-20.04",
    [string]$WslCondaBin = "/home/xiaojun/anaconda3/bin/conda",
    [string]$WslCondaEnvPrefix = "/home/xiaojun/anaconda3/envs/gnn_timing1",

    [string]$Python = "python",
    [string]$CsvOut = "",
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

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

function ConvertTo-SeedList {
    param([string[]]$RawSeeds)
    $out = @()
    foreach ($s in $RawSeeds) {
        if ($null -eq $s) {
            continue
        }
        $parts = "$s" -split "[,;\s]+" | Where-Object { $_ -and $_.Trim().Length -gt 0 }
        foreach ($p in $parts) {
            $out += [int]$p
        }
    }
    return $out
}

function Get-ExpandedPresetList {
    param([string[]]$RawPresets)

    $groupMap = @{
        "A" = @("A0", "A1", "A2")
        "B" = @("B0", "B1", "B2", "B3")
        "C" = @("C1", "C2", "C3")
    }

    $allPresets = @("A0", "A1", "A2", "B0", "B1", "B2", "B3", "C1", "C2", "C3")
    $result = New-Object System.Collections.Generic.List[string]

    foreach ($raw in $RawPresets) {
        if ($null -eq $raw) {
            continue
        }
        $parts = "$raw" -split "[,;\s]+" | Where-Object { $_ -and $_.Trim().Length -gt 0 }
        foreach ($p in $parts) {
            $key = $p.Trim().ToUpper()
            if ($key -eq "ALL" -or $key -eq "ABC") {
                foreach ($x in $allPresets) {
                    if (-not $result.Contains($x)) {
                        $result.Add($x)
                    }
                }
                continue
            }
            if ($groupMap.ContainsKey($key)) {
                foreach ($x in $groupMap[$key]) {
                    if (-not $result.Contains($x)) {
                        $result.Add($x)
                    }
                }
                continue
            }
            if ($allPresets -contains $key) {
                if (-not $result.Contains($key)) {
                    $result.Add($key)
                }
                continue
            }
            throw "Unsupported preset/group token: '$p'. Use A/B/C/ALL/ABC or explicit A0..C3."
        }
    }

    return @($result)
}

function Get-BalancedTransferArgs {
    return @(
        "--loss_weight_45", "1.0",
        "--src_loss_anneal_start", "80",
        "--src_loss_anneal_end", "280",
        "--src_loss_final_scale", "0.8"
    )
}

function Get-PresetConfig {
    param([string]$PresetName)

    $cfg = [ordered]@{
        Preset = $PresetName
        FreezeHgat = $true
        DedupZ = $true
        NumEpoch = $NumEpoch
        BatchSize = $BatchSize
        LearningRate = $LearningRate
        EncLrScale = $EncLrScale
        WeightDecay = $WeightDecay
        MlpDropout = $MlpDropout
        ZNoiseStd = $ZNoiseStd
        EnrichParasitic = $true
        HgatNetFeatMode = "parasitic_append"
        HgatParCapWeight = $HgatParCapWeight
        LrScheduler = $LrScheduler
        PlateauFactor = $PlateauFactor
        PlateauPatience = $PlateauPatience
        PlateauThreshold = $PlateauThreshold
        PlateauMinLr = $PlateauMinLr
        EarlyStopPatience = $EarlyStopPatience
        EarlyStopMinDelta = $EarlyStopMinDelta
        NumWorkers = $NumWorkers
        ValEvalInterval = $ValEvalInterval
        TestEvalInterval = $TestEvalInterval
        SkipTestEval = $SkipTestEval
        ExtraArgs = @()
    }

    switch ($PresetName) {
        "A0" {
            $cfg.ExtraArgs = Get-BalancedTransferArgs
        }
        "A1" {
            $cfg.ExtraArgs = @("--loss_weight_45", "0.0")
        }
        "A2" {
            $cfg.ExtraArgs = @(
                "--loss_weight_45", "1.0",
                "--src_loss_anneal_start", "1",
                "--src_loss_anneal_end", "60",
                "--src_loss_final_scale", "0.0"
            )
        }
        "B0" {
            $cfg.EnrichParasitic = $false
            $cfg.HgatNetFeatMode = "base"
            $cfg.ExtraArgs = Get-BalancedTransferArgs
        }
        "B1" {
            $cfg.HgatNetFeatMode = "parasitic_replace"
            $cfg.ExtraArgs = Get-BalancedTransferArgs
        }
        "B2" {
            $cfg.HgatNetFeatMode = "parasitic_append"
            $cfg.ExtraArgs = Get-BalancedTransferArgs
        }
        "B3" {
            $cfg.HgatNetFeatMode = "parasitic_append_split"
            $cfg.ExtraArgs = Get-BalancedTransferArgs
        }
        "C1" {
            $cfg.FreezeHgat = $false
            $cfg.DedupZ = $false
            $cfg.NumEpoch = 160
            $cfg.BatchSize = 1350
            $cfg.LearningRate = 8e-4
            $cfg.EncLrScale = 0.15
            $cfg.WeightDecay = 1e-5
            $cfg.MlpDropout = 0.1
            $cfg.LrScheduler = "reduce_on_plateau"
            $cfg.PlateauFactor = 0.6
            $cfg.PlateauPatience = 4
            $cfg.PlateauThreshold = 3e-4
            $cfg.PlateauMinLr = 1e-6
            $cfg.EarlyStopPatience = 18
            $cfg.EarlyStopMinDelta = 3e-4
            $cfg.NumWorkers = 4
            $cfg.ValEvalInterval = 1
            $cfg.TestEvalInterval = 50
            $cfg.ExtraArgs = @(
                "--loss_weight_45", "1.0",
                "--src_loss_anneal_start", "80",
                "--src_loss_anneal_end", "220",
                "--src_loss_final_scale", "0.85",
                "--enc_update_interval", "4"
            )
        }
        "C2" {
            $cfg.ExtraArgs = @(
                "--loss_weight_45", "1.0",
                "--src_loss_anneal_start", "80",
                "--src_loss_anneal_end", "280",
                "--src_loss_final_scale", "0.8",
                "--hgat_dual_readout",
                "--hgat_dual_merge", "concat"
            )
        }
        "C3" {
            $cfg.ExtraArgs = @(
                "--loss_weight_45", "1.0",
                "--src_loss_anneal_start", "80",
                "--src_loss_anneal_end", "280",
                "--src_loss_final_scale", "0.8",
                "--hgat_dual_readout",
                "--hgat_dual_merge", "mean"
            )
        }
        default {
            throw "Unsupported preset: $PresetName"
        }
    }

    return $cfg
}

function Get-BestFromLog {
    param([string]$LogPath)

    if (-not (Test-Path $LogPath)) {
        return $null
    }

    $bestLineObj = Select-String -Path $LogPath -Pattern "\[Best\]\s*epoch:(\d+),\s*val_r2:([\-0-9\.]+),\s*val_loss:([\-0-9\.eE]+)" | Select-Object -Last 1
    if ($null -eq $bestLineObj) {
        return $null
    }

    $line = $bestLineObj.Line
    $m = [regex]::Match($line, "\[Best\]\s*epoch:(\d+),\s*val_r2:([\-0-9\.]+),\s*val_loss:([\-0-9\.eE]+)")
    if (-not $m.Success) {
        return $null
    }

    $finalR2 = $null
    $finalLoss = $null
    $mf = [regex]::Match($line, "final_test_r2:([\-0-9\.]+),\s*final_test_loss:([\-0-9\.eE]+)")
    if ($mf.Success) {
        $finalR2 = [double]$mf.Groups[1].Value
        $finalLoss = [double]$mf.Groups[2].Value
    }

    return [pscustomobject]@{
        BestEpoch = [int]$m.Groups[1].Value
        BestValR2 = [double]$m.Groups[2].Value
        BestValLoss = [double]$m.Groups[3].Value
        FinalTestR2 = $finalR2
        FinalTestLoss = $finalLoss
        BestLine = $line
    }
}

function Get-MeanStd {
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -eq 0) {
        return [pscustomobject]@{ Mean = $null; Std = $null }
    }
    $mean = ($Values | Measure-Object -Average).Average
    if ($Values.Count -le 1) {
        return [pscustomobject]@{ Mean = $mean; Std = 0.0 }
    }
    $sumSq = 0.0
    foreach ($v in $Values) {
        $sumSq += [math]::Pow(($v - $mean), 2)
    }
    $std = [math]::Sqrt($sumSq / ($Values.Count - 1))
    return [pscustomobject]@{ Mean = $mean; Std = $std }
}

if (-not (Test-Path $DataPath)) {
    throw "Data path not found: $DataPath"
}

$SeedInts = ConvertTo-SeedList -RawSeeds $Seeds
if ($SeedInts.Count -eq 0) {
    throw "No valid seeds provided. Example: -Seeds 9294,2025,6666"
}

$ExpandedPresets = Get-ExpandedPresetList -RawPresets $Presets
if ($ExpandedPresets.Count -eq 0) {
    throw "No valid presets resolved. Use A/B/C/ALL or explicit A0..C3."
}

if (-not $UseWslConda) {
    $Python = Resolve-PythonCommand -Preferred $Python
    Write-Host "[Info] Local Python: $Python"
}
else {
    Write-Host "[Info] WSL distro: $WslDistro"
    Write-Host "[Info] WSL conda env prefix: $WslCondaEnvPrefix"
}

if (-not $CsvOut -or $CsvOut.Trim().Length -eq 0) {
    $CsvOut = "output/step5a_feat_multiseed_results.csv"
}

$logsDir = Join-Path "copilot_train_logs" "multiseed_step5a_feat"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$csvDir = Split-Path -Parent $CsvOut
if ($csvDir -and -not (Test-Path $csvDir)) {
    New-Item -ItemType Directory -Force -Path $csvDir | Out-Null
}

Write-Host "[Info] Resolved presets: $($ExpandedPresets -join ', ')"
Write-Host "[Info] Seeds: $($SeedInts -join ', ')"

$results = @()

foreach ($preset in $ExpandedPresets) {
    $cfg = Get-PresetConfig -PresetName $preset
    foreach ($seed in $SeedInts) {
        $modelDir = "${ModelPrefix}_${preset}_seed${seed}"
        $runLog = Join-Path $logsDir "${preset}_seed${seed}.log"

        $trainArgs = @(
            "-u",
            "src/train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_opt.py",
            "--model_saving_dir", $modelDir,
            "--data_save_path", $DataPath,
            "--dataset_pkl_name", $DatasetPkl,
            "--gpu", "$Gpu",
            "--seed", "$seed",
            "--num_epoch", "$($cfg.NumEpoch)",
            "--batch_size", "$($cfg.BatchSize)",
            "--learning_rate", "$($cfg.LearningRate)",
            "--enc_lr_scale", "$($cfg.EncLrScale)",
            "--weight_decay", "$($cfg.WeightDecay)",
            "--mlp_dropout", "$($cfg.MlpDropout)",
            "--hgat_l2_norm",
            "--z_noise_std", "$($cfg.ZNoiseStd)",
            "--hgat_net_feat_mode", "$($cfg.HgatNetFeatMode)",
            "--hgat_par_cap_weight", "$($cfg.HgatParCapWeight)",
            "--lr_scheduler", "$($cfg.LrScheduler)",
            "--plateau_factor", "$($cfg.PlateauFactor)",
            "--plateau_patience", "$($cfg.PlateauPatience)",
            "--plateau_threshold", "$($cfg.PlateauThreshold)",
            "--plateau_min_lr", "$($cfg.PlateauMinLr)",
            "--early_stop_patience", "$($cfg.EarlyStopPatience)",
            "--early_stop_min_delta", "$($cfg.EarlyStopMinDelta)",
            "--num_workers", "$($cfg.NumWorkers)",
            "--val_eval_interval", "$($cfg.ValEvalInterval)",
            "--test_eval_interval", "$($cfg.TestEvalInterval)"
        )

        if ($cfg.FreezeHgat) {
            $trainArgs += "--freeze_hgat"
        }
        if ($cfg.DedupZ) {
            $trainArgs += "--dedup_z"
        }
        if ($cfg.EnrichParasitic) {
            $trainArgs += "--enrich_parasitic_net_feat"
        }
        if ($cfg.SkipTestEval) {
            $trainArgs += "--skip_test_eval"
        }

        if ($cfg.ExtraArgs -and $cfg.ExtraArgs.Count -gt 0) {
            $trainArgs += $cfg.ExtraArgs
        }
        if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
            $trainArgs += $ExtraArgs
        }

        Write-Host "[Run] preset=$preset seed=$seed model=$modelDir"
        Write-Host "[Run] log => $runLog"

        # Native commands may write warnings to stderr; don't terminate on those lines.
        $oldErrPref = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            if ($UseWslConda) {
                $wslPwd = Convert-WindowsPathToWsl -Path (Resolve-Path ".").Path
                & wsl.exe -d $WslDistro --cd $wslPwd -- $WslCondaBin run -p $WslCondaEnvPrefix --no-capture-output python @trainArgs 2>&1 | Tee-Object -FilePath $runLog
                $exitCode = $LASTEXITCODE
            }
            else {
                & $Python @trainArgs 2>&1 | Tee-Object -FilePath $runLog
                $exitCode = $LASTEXITCODE
            }
        }
        finally {
            $ErrorActionPreference = $oldErrPref
        }

        $parsed = Get-BestFromLog -LogPath $runLog
        if ($null -eq $parsed) {
            Write-Warning "Cannot parse [Best] from $runLog"
            $results += [pscustomobject]@{
                Preset = $preset
                Seed = $seed
                ExitCode = $exitCode
                BestEpoch = $null
                BestValR2 = $null
                BestValLoss = $null
                FinalTestR2 = $null
                FinalTestLoss = $null
                ModelDir = $modelDir
                LogPath = $runLog
                BestLine = $null
            }
        }
        else {
            $results += [pscustomobject]@{
                Preset = $preset
                Seed = $seed
                ExitCode = $exitCode
                BestEpoch = $parsed.BestEpoch
                BestValR2 = $parsed.BestValR2
                BestValLoss = $parsed.BestValLoss
                FinalTestR2 = $parsed.FinalTestR2
                FinalTestLoss = $parsed.FinalTestLoss
                ModelDir = $modelDir
                LogPath = $runLog
                BestLine = $parsed.BestLine
            }
        }
    }
}

$results | Sort-Object Preset, Seed | Format-Table -AutoSize

$groups = $results | Group-Object Preset
foreach ($g in $groups) {
    $valid = @($g.Group | Where-Object { $_.ExitCode -eq 0 -and $_.BestValR2 -ne $null })
    if ($valid.Count -eq 0) {
        Write-Warning "[Summary][$($g.Name)] no valid run with parsable [Best]."
        continue
    }

    $valR2Stats = Get-MeanStd -Values ([double[]]($valid | Select-Object -ExpandProperty BestValR2))
    $valLossStats = Get-MeanStd -Values ([double[]]($valid | Select-Object -ExpandProperty BestValLoss))
    Write-Host "[Summary][$($g.Name)] runs=$($valid.Count), val_r2_mean=$([math]::Round($valR2Stats.Mean,6)), val_r2_std=$([math]::Round($valR2Stats.Std,6)), val_loss_mean=$([math]::Round($valLossStats.Mean,6))"

    $validWithFinal = @($valid | Where-Object { $_.FinalTestR2 -ne $null })
    if ($validWithFinal.Count -gt 0) {
        $finalR2Stats = Get-MeanStd -Values ([double[]]($validWithFinal | Select-Object -ExpandProperty FinalTestR2))
        $finalLossStats = Get-MeanStd -Values ([double[]]($validWithFinal | Select-Object -ExpandProperty FinalTestLoss))
        Write-Host "[Summary-FinalTest][$($g.Name)] runs=$($validWithFinal.Count), final_test_r2_mean=$([math]::Round($finalR2Stats.Mean,6)), final_test_r2_std=$([math]::Round($finalR2Stats.Std,6)), final_test_loss_mean=$([math]::Round($finalLossStats.Mean,6))"
    }
}

$results | Export-Csv -Path $CsvOut -NoTypeInformation -Encoding UTF8
Write-Host "[Saved] $CsvOut"
