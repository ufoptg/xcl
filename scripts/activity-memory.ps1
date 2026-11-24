# Summarize XCL_ACTIVITY=1 memory samples without printing the source logs or their paths.
# Compare two runs at the same elapsed time:
#   .\scripts\activity-memory.ps1 -Log main.log,pr.log -AtSeconds 60 -Json
param(
    [Parameter(Mandatory = $true, Position = 0)][string[]]$Log,
    [ValidateRange(0, 86400)][double]$AtSeconds,
    [ValidateRange(0, 60)][double]$MaxLagSeconds = 3,
    [switch]$Json
)
$ErrorActionPreference = 'Stop'

$atTime = $PSBoundParameters.ContainsKey('AtSeconds')
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$pattern = [regex]('^\[\s*(?<elapsed>\d+(?:\.\d+)?)\]\[\s*\d+\]\[INFO\s*\]\s+memory: ' +
    '(?<ProcessCommitMiB>\d+) MB committed by the process; video memory ' +
    '(?<LocalMiB>\d+) MB local(?: \(budget (?<LocalBudgetMiB>\d+) MB\))?, (?<NonLocalMiB>\d+) MB non-local; target pages ' +
    '(?<TexturePagesMiB>\d+) MB \(pools (?<TexturePoolsMiB>\d+) MB\); arenas ' +
    '(?<ArenasMiB>\d+) MB in use, (?<ArenasMadeMiB>\d+) MB made; sparse pools ' +
    '(?<SparsePoolsMiB>\d+) MB \((?<SparseMappedMiB>\d+) MB mapped\)' +
    '(?:; DXGI budget (?<LocalBudgetMiB>\d+) MB local, (?<NonLocalBudgetMiB>\d+) MB non-local)?(?:;|$)')
$metrics = 'ProcessCommitMiB', 'LocalMiB', 'NonLocalMiB', 'TexturePagesMiB', 'TexturePoolsMiB',
    'ArenasMiB', 'ArenasMadeMiB', 'SparsePoolsMiB', 'SparseMappedMiB', 'LocalBudgetMiB', 'NonLocalBudgetMiB'

$rows = @(for ($i = 0; $i -lt $Log.Count; $i++) {
    $path = (Resolve-Path -LiteralPath $Log[$i]).ProviderPath
    $best = $null
    $seconds = $null
    # XCL can still have the log open for writing when the user takes a sample.
    $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $stream = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $share)
    $reader = [IO.StreamReader]::new($stream)
    try {
        while ($null -ne ($line = $reader.ReadLine())) {
            $match = $pattern.Match($line)
            if (-not $match.Success) { continue }
            $elapsed = [double]::Parse($match.Groups['elapsed'].Value, $culture)
            if ($atTime -and $elapsed -gt $AtSeconds) { continue }
            if ($null -eq $seconds -or $elapsed -ge $seconds) {
                $best = $match
                $seconds = $elapsed
            }
        }
    } finally {
        $reader.Dispose()
    }
    $lag = if ($atTime -and $best) { [Math]::Round($AtSeconds - $seconds, 3) } else { $null }
    $status = if (-not $best) { 'no_sample' } elseif ($null -ne $lag -and $lag -gt $MaxLagSeconds) { 'stale' } else { 'ok' }
    $row = [ordered]@{
        Input = $i + 1
        RequestedSeconds = if ($atTime) { $AtSeconds } else { $null }
        SampleSeconds = $seconds
        LagSeconds = $lag
        Status = $status
    }
    foreach ($key in $metrics) {
        $row[$key] = if ($status -eq 'ok' -and $best.Groups[$key].Success) { [long]$best.Groups[$key].Value } else { $null }
    }
    [pscustomobject]$row
})

if ($Json) { ConvertTo-Json -InputObject $rows -Compress } else { $rows }
