# Regression run of a title for a pull request: runs it for a fixed time, reads XCL's log and
# prints an `xcl-smoke` block to paste into the PR's "Regression run" section – once built
# from main, once from the PR's head commit (CONTRIBUTING.md, "What a PR needs to be merged").
#
#   .\scripts\smoke.ps1 -Game 'D:\Games\SomeTitle' [-Seconds 60] [-Xcl .\target\release\xcl.exe] [-Title <label>]
#
# The block's `title:` line is the first 8 hex digits of the SHA-256 of the game folder's name
# (trimmed, UTF-8), not the name itself, so PR descriptions do not name anyone's copy of a game (#677);
# -Title prints the given label instead. Use the same one for the main and the PR run.
# -ArtifactsDir preserves each run's log, report and manifest in a new local directory.
#
# Networking is off (XCL_NETWORK=0). Build first: cargo build --release -p xcl
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [int]$Seconds = 60,
    [string]$Xcl = (Join-Path $PSScriptRoot '..\target\release\xcl.exe'),
    [string]$Title,
    [ValidateNotNullOrEmpty()][string]$ArtifactsDir,
    [ValidateSet('fresh', 'seeded', 'unspecified')][string]$CacheState = 'unspecified'
)
$ErrorActionPreference = 'Stop'

if ($PSBoundParameters.ContainsKey('Title')) {
    $Title = $Title.Trim()
    # control, format and line/paragraph separator characters would split the line for pr_check.py
    if (-not $Title -or $Title -match '[\p{Cc}\p{Cf}\p{Zl}\p{Zp}`]') { throw '-Title must be one line of text without backticks' }
} else {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $digest = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes((Split-Path $Game -Leaf).Trim()))
    $sha.Dispose()
    $Title = -join ($digest[0..3] | ForEach-Object { $_.ToString('x2') })
}

if (-not (Test-Path $Xcl)) { throw "no $Xcl - build it first (cargo build --release -p xcl)" }
$root = Split-Path $PSScriptRoot -Parent
$commit = (git -C $root rev-parse HEAD).Trim()
$dirty = [bool](git -C $root status --porcelain --untracked-files=no -- crates Cargo.toml Cargo.lock)
if ($ArtifactsDir) {
    $ArtifactsDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ArtifactsDir)
    $rootPrefix = $root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $destination = $ArtifactsDir.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if ($destination.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw '-ArtifactsDir must be outside the XCL checkout'
    }
}

# Use ProcessStartInfo instead of Start-Process: the latter cannot build its
# environment block on Windows PowerShell 5.1 when PATH is present twice with
# different casing (which is possible on Windows).
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Xcl
$psi.Arguments = 'run "' + $Game + '"'
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
$psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
$env:XCL_NETWORK = '0'
$env:XCL_RUN_SECONDS = "$Seconds"
$env:XCL_ACTIVITY = '1'
$started = [DateTime]::UtcNow
$null = $p.Start()
$stdoutTask = $p.StandardOutput.ReadToEndAsync()
$stderrTask = $p.StandardError.ReadToEndAsync()
if (-not $p.WaitForExit(($Seconds + 120) * 1000)) {
    if (-not $p.HasExited) { $p.Kill() }
    $timedOut = $true
}
$p.WaitForExit()
$stdout = $stdoutTask.Result
$null = $stderrTask.Result
Remove-Item Env:XCL_RUN_SECONDS, Env:XCL_ACTIVITY

$logLine = ($stdout -split "`n" | Where-Object { $_ -match '^Log:\s+(.+)$' } | Select-Object -Last 1)
if (-not $logLine) { throw "XCL printed no log path; its output:`n$stdout" }
$log = ($logLine -replace '^Log:\s+', '').Trim()
$text = Get-Content -LiteralPath $log -Raw -Encoding utf8

$frames = if ($text -match '\((\d+) frames shown, (\d+) PresentX calls\)') { "$($Matches[1]) (PresentX $($Matches[2]))" } else { 'none logged (the run did not reach its end)' }
$presents = [regex]::Matches($text, 'activity \(2 s\): (\d+) presents') | ForEach-Object { [int]$_.Groups[1].Value }
$lastPresents = ($presents | Select-Object -Last 3) -join ' '
$commits = [regex]::Matches($text, 'memory: (\d+) MB committed') | ForEach-Object { [int]$_.Groups[1].Value }
$peak = if ($commits) { ($commits | Measure-Object -Maximum).Maximum } else { 'n/a' }
$removals = ([regex]::Matches($text, 'device removed')).Count
$crashes = ([regex]::Matches($text, 'unhandled exception|process ends with status')).Count
# the adapter XCL ran on (report.json next to the log; XCL removes the last run's at start),
# not the first one Windows lists – on a machine with an iGPU that is often not the one used
$gpu = $null
$report = Join-Path (Split-Path $log -Parent) 'report.json'
$hasReport = Test-Path -LiteralPath $report -PathType Leaf
if ($hasReport) {
    try { $gpu = ((Get-Content -LiteralPath $report -Raw -Encoding utf8 | ConvertFrom-Json).d3d12.adapter -replace '\s*\([^()]*VRAM\)$', '') } catch { }
}
if (-not $gpu) {
    try {
        $gpu = (Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -notmatch 'Basic' } | Select-Object -First 1).Name
    } catch { }
}
$code = if ($timedOut) { 'killed (no exit)' } else { '0x{0:x}' -f $p.ExitCode }

if ($ArtifactsDir) {
    $runName = $started.ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N')
    $run = Join-Path $ArtifactsDir $runName
    $null = [IO.Directory]::CreateDirectory($run)
    Copy-Item -LiteralPath $log -Destination (Join-Path $run 'xcl.log')
    if ($hasReport) { Copy-Item -LiteralPath $report -Destination (Join-Path $run 'report.json') }
    $manifest = [ordered]@{
        version = 1
        started_utc = $started.ToString('o')
        commit = $commit
        dirty = $dirty
        gpu = $gpu
        title = $Title
        seconds = $Seconds
        exit = $code
        log = 'xcl.log'
        report = if ($hasReport) { 'report.json' } else { $null }
        input = @{
            press_at = $env:XCL_PRESS_AT
            press_a_at = $env:XCL_PRESS_A_AT
        }
        data_root = if ($env:XCL_DATA_ROOT) { $env:XCL_DATA_ROOT } else { Join-Path $env:LOCALAPPDATA 'XCL' }
        cache = @{
            state = $CacheState
            dxil_root = if ($env:XCL_DXIL_CACHE) { $env:XCL_DXIL_CACHE } else { Join-Path $env:LOCALAPPDATA 'xcl\dxil' }
            translated = $env:XCL_TRANSLATED_CACHE
            pso = $env:XCL_PSO_DISK_CACHE
        }
    }
    [IO.File]::WriteAllText((Join-Path $run 'manifest.json'), ($manifest | ConvertTo-Json -Depth 4))
    Write-Verbose "Saved smoke artifacts to $run"
}

@"
``````
xcl-smoke v1
commit: $commit$(if ($dirty) { ' (uncommitted changes)' })
gpu: $gpu
title: $Title
seconds: $Seconds
exit: $code
frames shown: $frames
presents per 2 s (last 3): $lastPresents
device removals: $removals
crashes: $crashes
peak commit: $peak MB
``````
"@
