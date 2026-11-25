# Run with Windows PowerShell 5.1 or PowerShell 7. Uses a synthetic child, no XCL build or GPU.
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'precompile-budget.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-budget-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$source = Join-Path $fixture 'fake.cs'
$exe = Join-Path $fixture 'fake xcl.exe'
$input = Join-Path $fixture 'shader pack.rpf'
$argvFile = Join-Path $fixture 'argv.txt'
$cache = Join-Path $fixture 'complete-cache-entry.bin'
$oldMode = $env:XCL_BUDGET_FIXTURE_MODE
$oldDir = $env:XCL_BUDGET_FIXTURE_DIR

try {
    @'
using System;
using System.IO;
using System.Threading;

static class TestChild {
    static int Main(string[] args) {
        var folder = Environment.GetEnvironmentVariable("XCL_BUDGET_FIXTURE_DIR");
        File.WriteAllLines(Path.Combine(folder, "argv.txt"), args);
        if (Environment.GetEnvironmentVariable("XCL_BUDGET_FIXTURE_MODE") == "fast")
            return 23;
        File.WriteAllText(Path.Combine(folder, "complete-cache-entry.bin"), "synthetic cache");
        var memory = new byte[64 * 1024 * 1024];
        for (int i = 0; i < memory.Length; i += 4096)
            memory[i] = 1;
        Thread.Sleep(5000);
        GC.KeepAlive(memory);
        return 0;
    }
}
'@ | Set-Content -LiteralPath $source -Encoding ASCII
    New-Item -ItemType File -Path $input | Out-Null
    $compiler = Join-Path ([Environment]::GetFolderPath('Windows')) 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    & $compiler /nologo "/out:$exe" $source
    if ($LASTEXITCODE -ne 0) { throw 'Could not build the synthetic child.' }

    $env:XCL_BUDGET_FIXTURE_DIR = $fixture
    $env:XCL_BUDGET_FIXTURE_MODE = 'fast'
    $extra = @('--only', 'cs', '--pipelines', 'quoted "value"', 'ends with slash\', '')
    & $runner -ShaderInput $input -Xcl $exe -MaxCommitMiB 256 -XclArgs $extra
    if ($LASTEXITCODE -ne 23) { throw "Expected the child's exit code 23, got $LASTEXITCODE." }
    $expected = @('shaders', $input, '--precompile') + $extra
    $actual = [IO.File]::ReadAllLines($argvFile)
    if ($actual.Count -ne $expected.Count) { throw 'Incorrect native argument count.' }
    for ($i = 0; $i -lt $expected.Count; $i++) {
        if ($actual[$i] -cne $expected[$i]) { throw "Incorrect native argument at index $i." }
    }

    $env:XCL_BUDGET_FIXTURE_MODE = 'heavy'
    & $runner -ShaderInput $input -Xcl $exe -MaxCommitMiB 48
    if ($LASTEXITCODE -ne 4) { throw "Expected the budget exit code 4, got $LASTEXITCODE." }
    if ((Get-Content -LiteralPath $cache -Raw) -ne 'synthetic cache') {
        throw 'The completed fixture cache entry was not preserved.'
    }
    Write-Host 'Both precompile budget checks passed.'
} finally {
    $env:XCL_BUDGET_FIXTURE_MODE = $oldMode
    $env:XCL_BUDGET_FIXTURE_DIR = $oldDir
    foreach ($path in $source, $exe, $input, $argvFile, $cache, $fixture) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
}
