# Run with powershell.exe or pwsh.exe. The launcher fixture produces local smoke logs.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$temp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path.TrimEnd([IO.Path]::DirectorySeparatorChar)
$fixture = Join-Path $temp ('xcl-smoke-artifacts-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture

# Keep the no-report GPU fallback independent of which machine runs the test.
function Get-CimInstance {
    [pscustomobject]@{ Name = 'Fixture fallback GPU' }
}

$envNames = @('XCL_PRESS_AT', 'XCL_DATA_ROOT', 'XCL_DXIL_CACHE', 'XCL_NETWORK')
$originalEnv = @{}
foreach ($name in $envNames) { $originalEnv[$name] = [Environment]::GetEnvironmentVariable($name) }

try {
    $xcl = Join-Path $fixture 'xcl.cmd'
    $log = Join-Path $fixture 'xcl.log'
    $report = Join-Path $fixture 'report.json'
    $artifacts = Join-Path $fixture 'artifacts'
    [IO.File]::WriteAllText($xcl, "@echo off`r`necho Log: %~dp0xcl.log`r`n")
    $firstLog = "(7 frames shown, 7 PresentX calls)`n"
    $firstReport = '{"d3d12":{"adapter":"Fixture GPU (1 GB VRAM)"}}'
    [IO.File]::WriteAllText($log, $firstLog)
    [IO.File]::WriteAllText($report, $firstReport)
    $env:XCL_PRESS_AT = '0.25:A,0.75:MENU'
    $env:XCL_DATA_ROOT = Join-Path $fixture 'data'
    $env:XCL_DXIL_CACHE = Join-Path $fixture 'dxil'
    $common = @{ Game = 'D:\Games\Fixture'; Seconds = 1; Xcl = $xcl; Title = 'smoke fixture' }

    $plain = & $smoke @common
    $first = & $smoke @common -ArtifactsDir $artifacts -CacheState fresh
    if ($first -cne $plain) { throw 'Saving artifacts changed the smoke block' }
    $firstDir = Get-ChildItem -LiteralPath $artifacts -Directory
    $firstManifest = Get-Content -LiteralPath (Join-Path $firstDir.FullName 'manifest.json') -Raw | ConvertFrom-Json
    if ($firstManifest.gpu -ne 'Fixture GPU' -or $firstManifest.seconds -ne 1 -or
        $firstManifest.title -ne 'smoke fixture' -or $firstManifest.commit -notmatch '^[0-9a-f]{40}$' -or
        $firstManifest.input.press_at -ne $env:XCL_PRESS_AT -or $firstManifest.cache.state -ne 'fresh' -or
        $firstManifest.cache.dxil_root -ne $env:XCL_DXIL_CACHE -or $firstManifest.data_root -ne $env:XCL_DATA_ROOT) {
        throw 'The first manifest does not describe the run'
    }

    $secondLog = "(3 frames shown, 3 PresentX calls)`n"
    [IO.File]::WriteAllText($log, $secondLog)
    Remove-Item -LiteralPath $report
    $second = & $smoke @common -ArtifactsDir $artifacts -CacheState seeded
    if ($second -notmatch '(?m)^gpu: Fixture fallback GPU\r?$') { throw 'Missing-report fallback failed' }
    $runs = @(Get-ChildItem -LiteralPath $artifacts -Directory)
    if ($runs.Count -ne 2) { throw "Expected two artifact directories, found $($runs.Count)" }
    $secondDir = $runs | Where-Object FullName -ne $firstDir.FullName
    $secondManifest = Get-Content -LiteralPath (Join-Path $secondDir.FullName 'manifest.json') -Raw | ConvertFrom-Json
    if ($secondManifest.cache.state -ne 'seeded' -or $secondManifest.report -ne $null -or
        (Test-Path -LiteralPath (Join-Path $secondDir.FullName 'report.json'))) { throw 'The second run reused the earlier report' }
    if ($firstManifest.report -ne 'report.json' -or
        [IO.File]::ReadAllText((Join-Path $firstDir.FullName 'report.json')) -cne $firstReport -or
        [IO.File]::ReadAllText((Join-Path $firstDir.FullName 'xcl.log')) -cne $firstLog -or
        [IO.File]::ReadAllText((Join-Path $secondDir.FullName 'xcl.log')) -cne $secondLog) {
        throw 'The preserved logs or report do not match their invocations'
    }
    $rejected = $false
    try {
        & $smoke @common -ArtifactsDir (Join-Path $PSScriptRoot '.') | Out-Null
    } catch {
        if ($_.Exception.Message -notmatch 'outside the XCL checkout') { throw }
        $rejected = $true
    }
    if (-not $rejected) { throw 'An artifacts directory inside the checkout was accepted' }
    'smoke-artifacts: ok'
} finally {
    foreach ($name in $envNames) { [Environment]::SetEnvironmentVariable($name, $originalEnv[$name]) }
    $resolvedFixture = (Resolve-Path -LiteralPath $fixture).Path
    if ((Split-Path -Parent $resolvedFixture) -ne $temp) { throw 'Unexpected fixture location' }
    Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
}
