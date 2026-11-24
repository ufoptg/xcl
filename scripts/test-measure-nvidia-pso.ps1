$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'measure-nvidia-pso.ps1')

foreach ($failure in 'none', 'create', 'measure', 'remove') {
    $events = [System.Collections.Generic.List[string]]::new()
    $caught = $null
    try {
        Invoke-ColdPsoScope -Create {
            $events.Add('create')
            if ($failure -eq 'create') { throw 'creation failed' }
        } -Measure {
            $events.Add('measure')
            if ($failure -eq 'measure') { throw 'measurement failed' }
        } -Remove {
            $events.Add('remove')
            if ($failure -eq 'remove') { throw 'cleanup failed' }
        }
    } catch { $caught = $_.Exception.Message }
    $expected = if ($failure -eq 'create') { 'create' } else { 'create,measure,remove' }
    if (($events -join ',') -ne $expected) { throw "Wrong profile lifetime for $failure`: $events" }
    if ($failure -eq 'none' -and $caught) { throw $caught }
    if ($failure -eq 'create' -and $caught -ne 'creation failed') { throw 'Creation failure lost.' }
    if ($failure -eq 'measure' -and $caught -ne 'measurement failed') { throw 'Measurement failure lost.' }
    if ($failure -eq 'remove' -and $caught -ne 'cleanup failed') { throw 'Cleanup failure lost.' }
}
Write-Output '4 profile lifetime checks passed (no driver settings touched).'
