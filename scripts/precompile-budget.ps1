# Stop an offline shader precompile if the child's sampled private commit reaches a budget.
param(
    [Parameter(Mandatory = $true)][string]$ShaderInput,
    [Parameter(Mandatory = $true)][ValidateRange(1, 2147483647)][int]$MaxCommitMiB,
    [string]$Xcl,
    [string[]]$XclArgs = @()
)
$ErrorActionPreference = 'Stop'
if (-not $Xcl) { $Xcl = Join-Path $PSScriptRoot '..\target\release\xcl.exe' }
$inputPath = (Resolve-Path -LiteralPath $ShaderInput).ProviderPath
$limit = [long]$MaxCommitMiB * 1MB

# ProcessStartInfo.Arguments is also available in Windows PowerShell 5.1. For the Windows
# argument parser, double each run of backslashes before a quote or the closing quote.
function Quote-Argument([string]$Value) {
    $escaped = $Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

$start = [System.Diagnostics.ProcessStartInfo]::new()
$start.FileName = (Resolve-Path -LiteralPath $Xcl).ProviderPath
$nativeArgs = @('shaders', $inputPath, '--precompile') + $XclArgs
$start.Arguments = ($nativeArgs | ForEach-Object { Quote-Argument $_ }) -join ' '
$start.UseShellExecute = $false

$child = [System.Diagnostics.Process]::Start($start)
try {
    $exceeded = $false
    while (-not $child.WaitForExit(100)) {
        $child.Refresh()
        try {
            $privateBytes = $child.PrivateMemorySize64
        } catch {
            if ($child.HasExited) { break }
            throw
        }
        if ($privateBytes -ge $limit) {
            try {
                $child.Kill()
                $exceeded = $true
            } catch {
                if (-not $child.HasExited) { throw }
            }
            $child.WaitForExit()
            break
        }
    }
    if ($exceeded) {
        [Console]::Error.WriteLine("XCL precompile stopped at $([Math]::Ceiling($privateBytes / 1MB)) MiB private commit (limit $MaxCommitMiB MiB). Completed cache entries can be reused on the next run.")
        exit 4
    }
    exit $child.ExitCode
} finally {
    try {
        if (-not $child.HasExited) {
            # A monitoring error or cancellation must not leave an unmonitored precompile.
            try { $child.Kill() } catch { if (-not $child.HasExited) { throw } }
            $child.WaitForExit()
        }
    } finally {
        $child.Dispose()
    }
}
