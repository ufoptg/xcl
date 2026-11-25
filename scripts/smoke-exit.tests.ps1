# Run with powershell.exe or pwsh.exe. Fake the child process to exercise both wait results.
param([string]$Smoke)
$ErrorActionPreference = 'Stop'
if (!$Smoke) { $Smoke = Join-Path $PSScriptRoot 'smoke.ps1' }
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-exit-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$log = Join-Path $fixture 'xcl.log'
$report = Join-Path $fixture 'report.json'
$oldNetwork = $env:XCL_NETWORK
$oldSeconds = $env:XCL_RUN_SECONDS
$oldActivity = $env:XCL_ACTIVITY

function git {
    if ($args[2] -eq 'rev-parse') { '1111111222222233333334444444555555566666' }
}

function New-Object([string]$TypeName) {
    if ($TypeName -eq 'System.Diagnostics.Process') { return $process }
    Microsoft.PowerShell.Utility\New-Object -TypeName $TypeName @args
}

function Assert-Exit([bool]$ExitedInTime, [string]$Expected) {
    $process = [pscustomobject]@{
        StartInfo = $null
        StandardOutput = [IO.StringReader]::new("Log: $log")
        StandardError = [IO.StringReader]::new('')
        ExitedInTime = $ExitedInTime
        HasExited = $ExitedInTime
        ExitCode = 0
    }
    $process | Add-Member ScriptMethod Start { $true }
    $process | Add-Member ScriptMethod WaitForExit { if ($args.Count) { $this.ExitedInTime } }
    $process | Add-Member ScriptMethod Kill { $this.HasExited = $true }

    $output = & $Smoke -Game $fixture -Xcl $PSCommandPath
    if (($output -split "\r?\n") -notcontains "exit: $Expected") {
        throw "Expected exit: $Expected; smoke output: $output"
    }
    if (!$process.HasExited) { throw 'The timed-out child was not killed.' }
}

try {
    Set-Content -LiteralPath $log -Value '(7 frames shown, 7 PresentX calls)' -Encoding UTF8
    Set-Content -LiteralPath $report -Value '{"d3d12":{"adapter":"Test adapter"}}' -Encoding UTF8

    Set-StrictMode -Version Latest
    Assert-Exit $true '0x0'
    $timedOut = $true
    Assert-Exit $true '0x0'
    Assert-Exit $false 'killed (no exit)'
} finally {
    $env:XCL_NETWORK = $oldNetwork
    $env:XCL_RUN_SECONDS = $oldSeconds
    $env:XCL_ACTIVITY = $oldActivity
    foreach ($path in $log, $report, $fixture) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
}
