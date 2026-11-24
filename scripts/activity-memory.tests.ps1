# Windows PowerShell and pwsh, no GPU required.
$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'activity-memory.ps1'
$first = [IO.Path]::GetTempFileName()
$second = [IO.Path]::GetTempFileName()
$writer = $null
try {
    $line = 'memory: 9137 MB committed by the process; video memory 1300 MB local, 950 MB non-local; target pages 116 MB (pools 128 MB); arenas 93 MB in use, 102 MB made; sparse pools 832 MB (806 MB mapped)'
    [IO.File]::WriteAllLines($first, @(
        'G:\private\name\game.exe',
        "[   4.125][  421][INFO ] $line",
        "[   6.000][  421][INFO ] $($line.Replace('1300 MB local', '1570 MB local (budget 3344 MB)'))",
        '[   6.900][  421][INFO ] memory: incomplete sample'
    ))
    [IO.File]::WriteAllText($second, "[   2.002][  421][INFO ] $line; DXGI budget 3344 MB local, 15289 MB non-local")

    # Like an XCL session, allow concurrent readers while holding a write handle.
    $writer = [IO.File]::Open($second, [IO.FileMode]::Open, [IO.FileAccess]::Write, [IO.FileShare]::Read)

    $json = & $script -Log $first,$second -AtSeconds 6 -Json
    if ($json -match 'private|game\.exe|[\\/]') { throw 'A summary contains a source path' }
    $rows = ConvertFrom-Json -InputObject $json
    if ($rows.Count -ne 2 -or $rows[0].Input -ne 1 -or $rows[0].Status -ne 'ok' -or $rows[0].LocalMiB -ne 1570) {
        throw 'The sample at the requested time was not selected'
    }
    if ($rows[0].TexturePoolsMiB -ne 128 -or $rows[0].SparseMappedMiB -ne 806 -or $rows[0].ArenasMiB -ne 93 -or $rows[0].ArenasMadeMiB -ne 102 -or $rows[0].LocalBudgetMiB -ne 3344 -or $null -ne $rows[0].NonLocalBudgetMiB) {
        throw 'Memory categories were not parsed correctly'
    }
    if ($rows[1].Status -ne 'stale' -or $null -ne $rows[1].LocalMiB -or $rows[1].LagSeconds -ne 3.998) {
        throw 'An old sample was reported as comparable'
    }
    $earlier = & $script $first -AtSeconds 5
    if ($earlier.SampleSeconds -ne 4.125 -or $earlier.LocalMiB -ne 1300 -or $null -ne $earlier.LocalBudgetMiB) { throw 'A later sample was used' }
    $missing = & $script $first -AtSeconds 0
    if ($missing.Status -ne 'no_sample' -or $null -ne $missing.SampleSeconds) { throw 'An absent sample was not reported' }
    $last = & $script $second
    if ($last.Status -ne 'ok' -or $last.SampleSeconds -ne 2.002 -or $last.NonLocalMiB -ne 950 -or $last.LocalBudgetMiB -ne 3344 -or $last.NonLocalBudgetMiB -ne 15289) { throw 'The optional budget suffix was not parsed' }
    'activity-memory: ok'
} finally {
    if ($writer) { $writer.Dispose() }
    [IO.File]::Delete($first)
    [IO.File]::Delete($second)
}
