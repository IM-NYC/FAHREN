#Requires -Version 5.1
<#
.SYNOPSIS
    FAHREN interactive CLI for training and evaluating MNIST models on Windows.
#>
param(
    [string] $RepoRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$ConfigDir  = Join-Path $env:USERPROFILE ".fahren"
$ConfigFile = Join-Path $ConfigDir "config.ini"
$CliExe     = Join-Path $RepoRoot "build\tools\Release\fahren_cli.exe"
if (-not (Test-Path $CliExe)) {
    $CliExe = Join-Path $RepoRoot "build\tools\fahren_cli.exe"
}

function Ensure-Config {
    if (-not (Test-Path $ConfigDir)) {
        New-Item -ItemType Directory -Path $ConfigDir | Out-Null
    }
    if (-not (Test-Path $ConfigFile)) {
        @"
# FAHREN configuration
device=cpu
epochs=5
batch_size=64
learning_rate=0.01
train_samples=10000
min_accuracy=0.85
mnist_dir=$RepoRoot\mnist
weights_path=$ConfigDir\mnist_model.bin
build_dir=$RepoRoot\build
"@ | Set-Content -Path $ConfigFile -Encoding UTF8
    }
}

function Read-Config {
    $cfg = @{}
    Get-Content $ConfigFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or $line.StartsWith(";")) { return }
        $p = $line.IndexOf("=")
        if ($p -gt 0) {
            $cfg[$line.Substring(0, $p).Trim()] = $line.Substring($p + 1).Trim()
        }
    }
    return $cfg
}

function Write-ConfigValue($Key, $Value) {
    $lines = Get-Content $ConfigFile
    $found = $false
    $out = foreach ($line in $lines) {
        if ($line -match "^\s*$Key\s*=") {
            $found = $true
            "$Key=$Value"
        } else { $line }
    }
    if (-not $found) { $out += "$Key=$Value" }
    $out | Set-Content $ConfigFile -Encoding UTF8
}

function Show-Header {
    $cfg = Read-Config
    $cuda = "no"
    if (Test-Path $CliExe) {
        $st = & $CliExe status $ConfigFile 2>&1 | Out-String
        if ($st -match "cuda=yes") { $cuda = "yes" }
    }
    $dev = $cfg.device
    if ($dev -eq "cuda" -and $cuda -eq "no") { $dev = "cpu (cuda unavailable)" }

    Write-Host ""
    Write-Host "  FAHREN" -ForegroundColor Cyan -NoNewline
    Write-Host "  v1.0  " -NoNewline
    Write-Host "|  device: " -NoNewline -ForegroundColor DarkGray
    Write-Host $dev -ForegroundColor Green
    Write-Host "  ─────────────────────────────────────" -ForegroundColor DarkGray
}

function Show-StatusBar {
    $cfg = Read-Config
    $mnistOk = Test-Path (Join-Path $cfg.mnist_dir "train-images.idx3-ubyte")
    $modelOk = Test-Path $cfg.weights_path
    $cliOk   = Test-Path $CliExe

    $parts = @()
    $parts += if ($mnistOk) { "MNIST:ok" } else { "MNIST:missing" }
    $parts += if ($modelOk) { "model:ready" } else { "model:none" }
    $parts += if ($cliOk) { "cli:ok" } else { "cli:build first" }
    Write-Host ""
    Write-Host "  $($parts -join '  |  ')" -ForegroundColor DarkGray
    Write-Host ""
}

function Invoke-Cli($Command) {
    if (-not (Test-Path $CliExe)) {
        Write-Host "  Building fahren_cli..." -ForegroundColor Yellow
        $cmake = Get-Command cmake -ErrorAction SilentlyContinue
        if (-not $cmake) { throw "cmake not found" }
        & cmake --build (Join-Path $RepoRoot "build") --config Release --target fahren_cli
    }
    & $CliExe $Command $ConfigFile
    return $LASTEXITCODE
}

function Show-MainMenu {
    while ($true) {
        Show-Header
        Write-Host "  [1] Train MNIST model"
        Write-Host "  [2] Evaluate model"
        Write-Host "  [3] Settings"
        Write-Host "  [4] Build library"
        Write-Host "  [0] Exit"
        Show-StatusBar
        $choice = Read-Host "  Select"
        switch ($choice) {
            "1" {
                $cfg = Read-Config
                Write-Host ""
                Write-Host "  Architecture: 784 -> 128 -> 64 -> 10" -ForegroundColor Yellow
                Write-Host "  Device: $($cfg.device)  Epochs: $($cfg.epochs)  Batch: $($cfg.batch_size)"
                $confirm = Read-Host "  Proceed? (Y/n)"
                if ($confirm -eq "" -or $confirm -eq "Y" -or $confirm -eq "y") {
                    $code = Invoke-Cli "train"
                    if ($code -eq 0) { Write-Host "  Done." -ForegroundColor Green }
                    else { Write-Host "  Failed (exit $code)" -ForegroundColor Red }
                }
                Read-Host "  Press Enter"
            }
            "2" {
                Invoke-Cli "eval" | Out-Null
                Read-Host "  Press Enter"
            }
            "3" { Show-SettingsMenu }
            "4" {
                & cmake --build (Join-Path $RepoRoot "build") --config Release
                Read-Host "  Press Enter"
            }
            "0" { return }
        }
    }
}

function Show-SettingsMenu {
    while ($true) {
        Show-Header
        Write-Host "  Settings" -ForegroundColor Cyan
        $cfg = Read-Config
        Write-Host "  [1] device         = $($cfg.device)"
        Write-Host "  [2] epochs         = $($cfg.epochs)"
        Write-Host "  [3] batch_size     = $($cfg.batch_size)"
        Write-Host "  [4] learning_rate  = $($cfg.learning_rate)"
        Write-Host "  [5] train_samples  = $($cfg.train_samples)"
        Write-Host "  [6] mnist_dir      = $($cfg.mnist_dir)"
        Write-Host "  [7] weights_path   = $($cfg.weights_path)"
        Write-Host "  [0] Back"
        Show-StatusBar
        $c = Read-Host "  Select"
        switch ($c) {
            "1" {
                Write-Host "  cpu | cuda"
                $v = Read-Host "  device"
                if ($v -eq "cuda" -or $v -eq "cpu") { Write-ConfigValue "device" $v }
            }
            "2" { Write-ConfigValue "epochs" (Read-Host "  epochs") }
            "3" { Write-ConfigValue "batch_size" (Read-Host "  batch_size") }
            "4" { Write-ConfigValue "learning_rate" (Read-Host "  learning_rate") }
            "5" { Write-ConfigValue "train_samples" (Read-Host "  train_samples") }
            "6" { Write-ConfigValue "mnist_dir" (Read-Host "  mnist_dir") }
            "7" { Write-ConfigValue "weights_path" (Read-Host "  weights_path") }
            "0" { return }
        }
    }
}

Ensure-Config
Show-MainMenu
