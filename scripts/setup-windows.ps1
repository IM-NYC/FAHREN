#Requires -Version 5.1
<#
.SYNOPSIS
    Configure and build Novaflow on Windows with CMake.
#>
[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string] $BuildType = "Release",
    [string] $Generator = "",
    [string] $BuildDir = "build",
    [string] $InstallPrefix = "",
    [switch] $Shared,
    [switch] $Static,
    [switch] $NoTest,
    [switch] $Install,
    [switch] $EnableVerbose,
    [switch] $Reconfigure,
    [switch] $Help
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "OK  $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "WARN $Message" -ForegroundColor Yellow
}

function Write-Err([string]$Message) {
    Write-Host "ERROR $Message" -ForegroundColor Red
}

function Show-Help {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

if ($Help) { Show-Help }

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

Write-Host ""
Write-Host "Novaflow Windows setup" -ForegroundColor Green
Write-Host "Repository: $RepoRoot"
Write-Host ""

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Find-CMakeExe {
    if (Test-Command cmake) { return (Get-Command cmake).Source }
    $candidates = @(
        "${env:ProgramFiles}\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\CMake\bin\cmake.exe",
        "$env:LOCALAPPDATA\Programs\CMake\bin\cmake.exe"
    )
    foreach ($path in $candidates) {
        if ($path -and (Test-Path $path)) { return $path }
    }
    return $null
}

$cmakeExe = Find-CMakeExe
if (-not $cmakeExe) {
    Write-Err "CMake not found. Install from https://cmake.org/download/"
    exit 1
}

$cmakeVersion = (& $cmakeExe --version | Select-Object -First 1)
Write-Ok $cmakeVersion

$compilerHint = ""
if (Test-Command cl) { $compilerHint = "MSVC (cl.exe)" }
elseif (Test-Command gcc) { $compilerHint = "GCC ($((gcc --version | Select-Object -First 1)))" }
elseif (Test-Command clang) { $compilerHint = "Clang ($((clang --version | Select-Object -First 1)))" }
else { Write-Warn "No C compiler detected. Use Developer PowerShell for VS or install MinGW-w64." }

if ($compilerHint) { Write-Ok "Compiler: $compilerHint" }

function Find-VsWhereExe {
    if (Test-Command vswhere) { return (Get-Command vswhere).Source }
    $path = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $path) { return $path }
    return $null
}

function Get-CMakeVSGenerators([string]$Cmake) {
    $help = & $Cmake --help 2>&1 | Out-String
    $names = [regex]::Matches($help, 'Visual Studio \d+ \d{4}') |
        ForEach-Object { $_.Value } | Select-Object -Unique
    return @($names | ForEach-Object {
        if ($_ -match 'Visual Studio (\d+) (\d{4})') {
            [PSCustomObject]@{ Name = $_; Major = [int]$Matches[1]; Year = [int]$Matches[2] }
        }
    } | Sort-Object Major, Year)
}

function Get-LatestVSInstall([string]$VsWhere) {
    $json = & $VsWhere -latest -sort `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -format json 2>$null
    if (-not $json) { $json = & $VsWhere -latest -sort -format json 2>$null }
    if (-not $json) { return $null }
    $info = $json | ConvertFrom-Json
    if ($info -is [System.Array]) { if ($info.Count -eq 0) { return $null }; $info = $info[0] }
    $major = 0
    if ($info.installationVersion -match '^(\d+)\.') { $major = [int]$Matches[1] }
    return [PSCustomObject]@{ Path = [string]$info.installationPath; Major = $major }
}

function Resolve-VSGenerator {
    param([string]$Cmake, [string]$RequestedGenerator, [object]$VsInstall)
    $available = Get-CMakeVSGenerators -Cmake $Cmake
    if ($available.Count -eq 0) { throw "No VS generators found" }
    if ($RequestedGenerator) {
        $match = $available | Where-Object { $_.Name -eq $RequestedGenerator }
        if (-not $match) { throw "Generator '$RequestedGenerator' not found" }
        return [PSCustomObject]@{ Generator = $RequestedGenerator; Arch = "x64" }
    }
    if (-not $VsInstall) {
        $pick = $available[-1]
        return [PSCustomObject]@{ Generator = $pick.Name; Arch = "x64" }
    }
    $match = $available | Where-Object { $_.Major -eq $VsInstall.Major } | Select-Object -Last 1
    if (-not $match) { $match = $available[-1] }
    return [PSCustomObject]@{ Generator = $match.Name; Arch = "x64" }
}

$vsWhereExe = Find-VsWhereExe
$vsInstall = $null
if ($vsWhereExe) { $vsInstall = Get-LatestVSInstall -VsWhere $vsWhereExe }

$wantVS = ($Generator -match '^Visual Studio') -or (-not $Generator -and $null -ne $vsInstall)

if ($Generator -and -not $wantVS) {
    Write-Ok "Using generator: $Generator"
} elseif ($wantVS) {
    $vsGen = Resolve-VSGenerator -Cmake $cmakeExe -RequestedGenerator $Generator -VsInstall $vsInstall
    $Generator = $vsGen.Generator
    Write-Ok "Generator: $Generator"
} elseif (-not $Generator -and (Test-Command ninja)) {
    $Generator = if (Test-Command cl) { "Ninja Multi-Config" } else { "Ninja" }
} else {
    Write-Err "No suitable build environment found."
    exit 1
}

$BuildPath = Join-Path $RepoRoot $BuildDir
if ($Reconfigure -and (Test-Path $BuildPath)) {
    Write-Step "Removing existing build directory"
    Remove-Item -Recurse -Force $BuildPath
}
if (-not (Test-Path $BuildPath)) { New-Item -ItemType Directory -Path $BuildPath | Out-Null }

$sharedLibs = $true
if ($Static) { $sharedLibs = $false }
if ($Shared) { $sharedLibs = $true }

$cmakeArgs = @(
    "-S", $RepoRoot, "-B", $BuildPath, "-G", $Generator,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DNOVA_BUILD_TESTS=ON",
    "-DNOVA_INSTALL=ON"
)

if ($sharedLibs) { $cmakeArgs += "-DNOVA_BUILD_SHARED_LIBS=ON" }
else { $cmakeArgs += "-DNOVA_BUILD_SHARED_LIBS=OFF" }

if ($EnableVerbose) { $cmakeArgs += "-DNOVA_ENABLE_VERBOSE=ON" }

if ($InstallPrefix) { $cmakeArgs += "-DCMAKE_INSTALL_PREFIX=$InstallPrefix" }

$isMultiConfig = $Generator -match "Visual Studio"
if ($isMultiConfig) { $cmakeArgs = $cmakeArgs | Where-Object { $_ -ne "-DCMAKE_BUILD_TYPE=$BuildType" } }

Write-Step "Configuring CMake"
& $cmakeExe @cmakeArgs
if ($LASTEXITCODE -ne 0) { Write-Err "Configuration failed"; exit $LASTEXITCODE }

Write-Step "Building"
if ($isMultiConfig) { & $cmakeExe --build $BuildPath --config $BuildType }
else { & $cmakeExe --build $BuildPath }
if ($LASTEXITCODE -ne 0) { Write-Err "Build failed"; exit $LASTEXITCODE }
Write-Ok "Build finished"

if (-not $NoTest) {
    Write-Step "Running tests"
    if ($isMultiConfig) { & ctest --test-dir $BuildPath -C $BuildType --output-on-failure }
    else { & ctest --test-dir $BuildPath --output-on-failure }
    if ($LASTEXITCODE -ne 0) { Write-Err "Tests failed"; exit $LASTEXITCODE }
    Write-Ok "All tests passed"
}

if ($Install) {
    Write-Step "Installing"
    if ($isMultiConfig) { & $cmakeExe --install $BuildPath --config $BuildType }
    else { & $cmakeExe --install $BuildPath }
    Write-Ok "Install complete"
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ""
