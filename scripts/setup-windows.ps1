#Requires -Version 5.1
<#
.SYNOPSIS
    Configure and build FAHREN on Windows with CMake.

.DESCRIPTION
    Detects Visual Studio or Ninja+MinGW, configures an out-of-tree build,
    compiles the library and smoke tests, and optionally runs ctest.

.PARAMETER BuildType
    CMake build type: Debug, Release, RelWithDebInfo, or MinSizeRel.

.PARAMETER Generator
    CMake generator (e.g. "Visual Studio 18 2026", "Ninja"). Auto-detected from
    installed Visual Studio via vswhere when omitted.

.PARAMETER BuildDir
    Directory for build artifacts (default: build).

.PARAMETER InstallPrefix
    CMAKE_INSTALL_PREFIX for `cmake --install` when -Install is used.

.PARAMETER Shared
    Build shared library (DLL). Default: on.

.PARAMETER Static
    Build static library (.lib) instead of shared.

.PARAMETER NoTest
    Skip running tests after a successful build.

.PARAMETER Install
    Run `cmake --install` after build.

.PARAMETER EnableVerbose
    Enable FAHREN_VERBOSE in the library.

.PARAMETER Addons
    Enable FAHREN_ENABLE_ADDONS (placeholder hook).

.PARAMETER Reconfigure
    Delete the build directory before configuring.

.PARAMETER Help
    Show usage.

.EXAMPLE
    .\scripts\setup-windows.ps1

.EXAMPLE
    .\scripts\setup-windows.ps1 -BuildType Debug -Static -NoTest
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
    [switch] $Addons,
    [switch] $Cuda,
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
Write-Host "FAHREN Windows setup" -ForegroundColor Green
Write-Host "Repository: $RepoRoot"
Write-Host ""

# --- Tool checks ---
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
    Write-Err "CMake not found on PATH or in standard install locations."
    Write-Host "Install options:"
    Write-Host "  winget install Kitware.CMake"
    Write-Host "  https://cmake.org/download/"
    exit 1
}

if ($cmakeExe -ne (Get-Command cmake -ErrorAction SilentlyContinue).Source) {
    $env:PATH = "$(Split-Path $cmakeExe -Parent);$env:PATH"
}

$cmakeVersion = (& $cmakeExe --version | Select-Object -First 1)
Write-Ok $cmakeVersion

$compilerHint = ""
if (Test-Command cl) {
    $compilerHint = "MSVC (cl.exe)"
} elseif (Test-Command gcc) {
    $compilerHint = "GCC ($((gcc --version | Select-Object -First 1)))"
} elseif (Test-Command clang) {
    $compilerHint = "Clang ($((clang --version | Select-Object -First 1)))"
} else {
    Write-Warn "No C compiler detected on PATH. Use a 'Developer PowerShell for VS' or install MinGW-w64."
}

if ($compilerHint) { Write-Ok "Compiler: $compilerHint" }

# --- Visual Studio / generator auto-detection ---
function Find-VsWhereExe {
    if (Test-Command vswhere) { return (Get-Command vswhere).Source }
    $path = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $path) { return $path }
    return $null
}

function Get-CMakeVisualStudioGenerators([string]$Cmake) {
    $help = & $Cmake --help 2>&1 | Out-String
    $names = [regex]::Matches($help, 'Visual Studio \d+ \d{4}') |
        ForEach-Object { $_.Value } |
        Select-Object -Unique
    return @($names | ForEach-Object {
        if ($_ -match 'Visual Studio (\d+) (\d{4})') {
            [PSCustomObject]@{
                Name  = $_
                Major = [int]$Matches[1]
                Year  = [int]$Matches[2]
            }
        }
    } | Sort-Object Major, Year)
}

function Get-LatestVisualStudioInstall([string]$VsWhere) {
    $json = & $VsWhere -latest -sort `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -format json 2>$null
    if (-not $json) {
        $json = & $VsWhere -latest -sort -format json 2>$null
    }
    if (-not $json) { return $null }

    $info = $json | ConvertFrom-Json
    if ($info -is [System.Array]) {
        if ($info.Count -eq 0) { return $null }
        $info = $info[0]
    }

    $major = 0
    if ($info.installationVersion -match '^(\d+)\.') {
        $major = [int]$Matches[1]
    }

    return [PSCustomObject]@{
        Path        = [string]$info.installationPath
        Version     = [string]$info.installationVersion
        Major       = $major
        DisplayName = [string]$info.displayName
    }
}

function Resolve-VisualStudioGenerator {
    param(
        [string]$Cmake,
        [string]$RequestedGenerator,
        [object]$VsInstall
    )

    $available = Get-CMakeVisualStudioGenerators -Cmake $Cmake
    if ($available.Count -eq 0) {
        throw "CMake reports no Visual Studio generators. Upgrade CMake or pass -Generator explicitly."
    }

    if ($RequestedGenerator) {
        $match = $available | Where-Object { $_.Name -eq $RequestedGenerator }
        if (-not $match) {
            $list = ($available.Name -join ', ')
            throw "Generator '$RequestedGenerator' is not supported by this CMake. Available: $list"
        }
        return [PSCustomObject]@{
            Generator = $RequestedGenerator
            Instance  = $VsInstall.Path
            Arch      = "x64"
        }
    }

    if (-not $VsInstall) {
        $pick = $available[-1]
        Write-Warn "Visual Studio not found via vswhere; using newest CMake VS generator: $($pick.Name)"
        return [PSCustomObject]@{
            Generator = $pick.Name
            Instance  = $null
            Arch      = "x64"
        }
    }

    $match = $available | Where-Object { $_.Major -eq $VsInstall.Major } | Select-Object -Last 1
    if (-not $match) {
        $match = $available | Where-Object { $_.Major -le $VsInstall.Major } | Select-Object -Last 1
    }
    if (-not $match) {
        $match = $available[-1]
        Write-Warn "No exact CMake generator for VS $($VsInstall.Major); using $($match.Name)"
    }

    return [PSCustomObject]@{
        Generator = $match.Name
        Instance  = $VsInstall.Path
        Arch      = "x64"
    }
}

$vsWhereExe = Find-VsWhereExe
$vsInstall = $null
if ($vsWhereExe) {
    $vsInstall = Get-LatestVisualStudioInstall -VsWhere $vsWhereExe
    if ($vsInstall) {
        Write-Ok "Visual Studio: $($vsInstall.DisplayName) ($($vsInstall.Version))"
    }
}

$cmakeArch = $null
$cmakeGeneratorInstance = $null

$wantVisualStudio = ($Generator -match '^Visual Studio') -or (
    -not $Generator -and $null -ne $vsInstall
)

if ($Generator -and -not $wantVisualStudio) {
    Write-Ok "Using explicit generator: $Generator"
} elseif ($wantVisualStudio) {
    try {
        $vsGen = Resolve-VisualStudioGenerator -Cmake $cmakeExe `
            -RequestedGenerator $(if ($Generator) { $Generator } else { "" }) `
            -VsInstall $vsInstall
        $Generator = $vsGen.Generator
        $cmakeArch = $vsGen.Arch
        $cmakeGeneratorInstance = $vsGen.Instance
    } catch {
        Write-Err $_.Exception.Message
        exit 1
    }
} elseif (-not $Generator -and (Test-Command ninja)) {
    $Generator = if (Test-Command cl) { "Ninja Multi-Config" } else { "Ninja" }
} else {
    Write-Err "No suitable build environment found."
    Write-Host "Install Visual Studio with 'Desktop development with C++', or install Ninja + a C compiler."
    Write-Host "You can also pass -Generator explicitly (e.g. -Generator 'Visual Studio 18 2026')."
    exit 1
}

Write-Ok "Generator: $Generator"
if ($cmakeGeneratorInstance) {
    Write-Ok "VS instance: $cmakeGeneratorInstance"
}

# --- Build directory ---
$BuildPath = Join-Path $RepoRoot $BuildDir
if ($Reconfigure -and (Test-Path $BuildPath)) {
    Write-Step "Removing existing build directory: $BuildPath"
    Remove-Item -Recurse -Force $BuildPath
}

if (-not (Test-Path $BuildPath)) {
    New-Item -ItemType Directory -Path $BuildPath | Out-Null
}

# --- CMake options ---
$sharedLibs = $true
if ($Static) { $sharedLibs = $false }
if ($Shared) { $sharedLibs = $true }

$cmakeArgs = @(
    "-S", $RepoRoot,
    "-B", $BuildPath,
    "-G", $Generator,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DFAHREN_BUILD_TESTS=ON",
    "-DFAHREN_INSTALL=ON"
)

if ($cmakeArch) {
    $cmakeArgs += "-A", $cmakeArch
}

if ($cmakeGeneratorInstance) {
    $cmakeArgs += "-DCMAKE_GENERATOR_INSTANCE=$cmakeGeneratorInstance"
}

if ($sharedLibs) {
    $cmakeArgs += "-DFAHREN_BUILD_SHARED_LIBS=ON"
} else {
    $cmakeArgs += "-DFAHREN_BUILD_SHARED_LIBS=OFF"
}

if ($EnableVerbose) {
    $cmakeArgs += "-DFAHREN_ENABLE_VERBOSE=ON"
}

if ($Addons -or $Cuda) {
    $cmakeArgs += "-DFAHREN_ENABLE_ADDONS=ON"
}

if ($Cuda) {
    $cmakeArgs += "-DFAHREN_ENABLE_CUDA=ON"
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if (-not $nvcc -and $env:CUDA_PATH) {
        $nvccPath = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
        if (Test-Path $nvccPath) {
            $env:PATH = "$(Split-Path $nvccPath -Parent);$env:PATH"
            Write-Ok "CUDA toolkit: $env:CUDA_PATH"
        }
    } elseif ($nvcc) {
        Write-Ok "CUDA compiler: $($nvcc.Source)"
    } else {
        Write-Warn "CUDA requested but nvcc not found. Install CUDA Toolkit 12.8+ and set CUDA_PATH."
    }
}

if ($InstallPrefix) {
    $cmakeArgs += "-DCMAKE_INSTALL_PREFIX=$InstallPrefix"
}

# Multi-config generators (Visual Studio) ignore CMAKE_BUILD_TYPE at configure time
$isMultiConfig = $Generator -match "Visual Studio"

Write-Step "Configuring CMake"
Write-Host "cmake $($cmakeArgs -join ' ')"
& $cmakeExe @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Err "CMake configuration failed."
    exit $LASTEXITCODE
}

Write-Step "Building"
if ($isMultiConfig) {
    & $cmakeExe --build $BuildPath --config $BuildType
} else {
    & $cmakeExe --build $BuildPath
}
if ($LASTEXITCODE -ne 0) {
    Write-Err "Build failed."
    exit $LASTEXITCODE
}
Write-Ok "Build finished"

if (-not $NoTest) {
    Write-Step "Running tests (ctest)"
    if ($isMultiConfig) {
        & $cmakeExe --build $BuildPath --config $BuildType --target fahren_test 2>$null | Out-Null
        & ctest --test-dir $BuildPath -C $BuildType --output-on-failure
    } else {
        & ctest --test-dir $BuildPath --output-on-failure
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Tests failed."
        exit $LASTEXITCODE
    }
    Write-Ok "All tests passed"
}

if ($Install) {
    Write-Step "Installing"
    if ($isMultiConfig) {
        & $cmakeExe --install $BuildPath --config $BuildType
    } else {
        & $cmakeExe --install $BuildPath
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Install failed."
        exit $LASTEXITCODE
    }
    Write-Ok "Install complete"
}

# Optional: fahren CLI shim on PATH
$FahrenBin = Join-Path $env:USERPROFILE ".local\bin"
if (-not (Test-Path $FahrenBin)) {
    New-Item -ItemType Directory -Path $FahrenBin -Force | Out-Null
}
$FahrenCmd = Join-Path $FahrenBin "fahren.cmd"
$FahrenPs1 = Join-Path $RepoRoot "scripts\fahren.ps1"
if (Test-Path $FahrenPs1) {
    @"
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "$FahrenPs1" %*
"@ | Set-Content -Path $FahrenCmd -Encoding ASCII
    Write-Ok "CLI shim: $FahrenCmd (add $FahrenBin to PATH if needed)"
}

$FahrenConfigDir = Join-Path $env:USERPROFILE ".fahren"
if (-not (Test-Path $FahrenConfigDir)) {
    New-Item -ItemType Directory -Path $FahrenConfigDir | Out-Null
}
$FahrenIni = Join-Path $FahrenConfigDir "config.ini"
if (-not (Test-Path $FahrenIni)) {
    @"
device=cpu
epochs=5
batch_size=64
learning_rate=0.01
train_samples=10000
min_accuracy=0.85
mnist_dir=$RepoRoot\mnist
weights_path=$FahrenConfigDir\mnist_model.bin
build_dir=$BuildPath
"@ | Set-Content -Path $FahrenIni -Encoding UTF8
    Write-Ok "Default config: $FahrenIni"
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "  Library:  $BuildPath"
Write-Host "  Link:     target_link_libraries(your_app PRIVATE Fahren::fahren)"
Write-Host "  Include:  #include <fahren/fahren.h>  or  <fahren/fahren_easy.h>"
Write-Host ""
Write-Host "Next steps:"
if ($isMultiConfig) {
    Write-Host "  cmake --build $BuildDir --config $BuildType --target run_test"
    Write-Host "  .\$BuildDir\test\$BuildType\fahren_test.exe"
    Write-Host "  .\scripts\fahren.ps1"
    Write-Host "  fahren   (if $FahrenBin is on PATH)"
} else {
    Write-Host "  cmake --build $BuildDir --target run_test"
    Write-Host "  .\$BuildDir\test\fahren_test.exe"
    Write-Host "  .\scripts\fahren.ps1"
}
Write-Host ""
