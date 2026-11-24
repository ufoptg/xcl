# NVIDIA cold PSO measurements for #1269. Requires the JSON/backend support in #1802.
param(
    [string]$Pack,
    [string]$Hash,
    [ValidateSet('hlsl', 'direct')][string]$Backend = 'hlsl',
    [ValidateRange(1, 20)][int]$Repeat = 3,
    [string]$Xcl = (Join-Path $PSScriptRoot '..\target\release\xcl.exe')
)
$ErrorActionPreference = 'Stop'

# Keep the profile lifetime testable without touching the driver. A failed creation
# has not saved anything; a failed measurement must still remove the saved profile.
function Invoke-ColdPsoScope {
    param([scriptblock]$Create, [scriptblock]$Measure, [scriptblock]$Remove)
    & $Create
    try { & $Measure } finally { & $Remove }
}

function Initialize-NvidiaPsoProfile {
    if ('XclColdPso.Profile' -as [type]) { return }
    # Independently written ABI bindings; signatures, IDs and layout facts from
    # NVIDIA/nvapi nvapi.h, nvapi_interface.h, NvApiDriverSettings.h (MIT).
    # https://github.com/NVIDIA/nvapi . No SDK files or driver binaries are bundled.
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
namespace XclColdPso {
    public static class Profile {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        static extern IntPtr LoadLibraryExW(string name, IntPtr file, uint flags);
        [DllImport("kernel32.dll", CharSet = CharSet.Ansi, ExactSpelling = true)]
        static extern IntPtr GetProcAddress(IntPtr module, string name);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate IntPtr Query(uint id);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int Init();
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int NewSession(out IntPtr s);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int SessionCall(IntPtr s);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int Find(IntPtr s, IntPtr name, out IntPtr p);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int Pair(IntPtr s, IntPtr p);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int Triple(IntPtr s, IntPtr p, IntPtr data);
        static readonly Query query;
        static Profile() {
            IntPtr dll = LoadLibraryExW("nvapi64.dll", IntPtr.Zero, 0x800); // System32 only
            if (dll == IntPtr.Zero) throw new Exception("nvapi64.dll is unavailable");
            IntPtr entry = GetProcAddress(dll, "nvapi_QueryInterface");
            if (entry == IntPtr.Zero) throw new Exception("nvapi_QueryInterface is unavailable");
            query = (Query)Marshal.GetDelegateForFunctionPointer(entry, typeof(Query));
            Check("initialize", Fn<Init>(0x0150e828)());
        }
        static T Fn<T>(uint id) where T : class {
            IntPtr p = query(id);
            if (p == IntPtr.Zero) throw new Exception("NVAPI entry unavailable: " + id.ToString("x"));
            return (T)(object)Marshal.GetDelegateForFunctionPointer(p, typeof(T));
        }
        static void Check(string operation, int status) {
            if (status != 0) throw new Exception("NVAPI " + operation + ": " + status);
        }
        sealed class Buffer : IDisposable {
            public readonly IntPtr P;
            public Buffer(int bytes, bool versioned) {
                P = Marshal.AllocHGlobal(bytes);
                Marshal.Copy(new byte[bytes], 0, P, bytes);
                if (versioned) Marshal.WriteInt32(P, bytes | (1 << 16));
            }
            public void Text(int offset, string value) {
                if (value.Length >= 2048 || value.IndexOf('\0') >= 0) throw new ArgumentException("NVAPI string too long or contains NUL");
                byte[] bytes = Encoding.Unicode.GetBytes(value + "\0");
                Marshal.Copy(bytes, 0, IntPtr.Add(P, offset), bytes.Length);
            }
            public void Dispose() { Marshal.FreeHGlobal(P); }
        }
        public static void Change(string name, string exe, bool create) {
            if (!System.Text.RegularExpressions.Regex.IsMatch(name, @"\AXCL cold PSO [0-9a-f]{32}\z"))
                throw new ArgumentException("Not a temporary cold PSO profile name");
            IntPtr session;
            Check("create session", Fn<NewSession>(0x0694d52e)(out session));
            try {
                Check("load settings", Fn<SessionCall>(0x375dbd6b)(session));
                using (var text = new Buffer(4096, false)) {
                    text.Text(0, name);
                    IntPtr profile;
                    int found = Fn<Find>(0x7e4a9a0b)(session, text.P, out profile);
                    if (!create) {
                        if (found == -163) return; // profile not found
                        Check("find temporary profile", found);
                        Check("delete temporary profile", Fn<Pair>(0x17093206)(session, profile));
                    } else {
                        if (found != -163) throw new Exception("Refusing to replace an existing profile");
                        using (var info = new Buffer(4116, true)) { // NVDRS_PROFILE_V1
                            info.Text(4, name);
                            Check("create temporary profile", Fn<Find>(0xcc176068)(session, info.P, out profile));
                        }
                        using (var app = new Buffer(12296, true)) { // NVDRS_APPLICATION_V1
                            app.Text(8, exe);
                            Check("register probe exe (existing matches are left unchanged)", Fn<Triple>(0x4347a9de)(session, profile, app.P));
                        }
                        using (var setting = new Buffer(12320, true)) { // NVDRS_SETTING_V1, pack 4
                            Marshal.WriteInt32(setting.P, 4100, 0x00198fff); // PS_SHADERDISKCACHE_ID
                            // DWORD type = 0; current value at 8220 = 0 (OFF).
                            Check("disable probe disk cache", Fn<Triple>(0x577dd202)(session, profile, setting.P));
                        }
                    }
                    Check("save settings", Fn<SessionCall>(0xfcbc7e14)(session));
                }
            } finally { Check("destroy session", Fn<SessionCall>(0xdad9cff8)(session)); }
        }
    }
}
'@
}

function Invoke-NvidiaColdPso {
    if ($Hash -notmatch '\A[0-9a-fA-F]{16}\z') { throw 'Hash must be exactly 16 hexadecimal digits.' }
    $packPath = (Resolve-Path -LiteralPath $Pack).ProviderPath
    $source = (Resolve-Path -LiteralPath $Xcl).ProviderPath
    $id = [guid]::NewGuid().ToString('N')
    $profile = "XCL cold PSO $id"
    # Avoid x*.exe: NVIDIA predefined wildcard profiles can match those names.
    $probe = Join-Path (Split-Path $source -Parent) "coldpso$id.exe"
    Initialize-NvidiaPsoProfile
    Copy-Item -LiteralPath $source -Destination $probe
    try {
        Invoke-ColdPsoScope -Create {
            [XclColdPso.Profile]::Change($profile, $probe, $true)
            Write-Host "Temporary NVIDIA profile: $profile"
        } -Measure {
            $raw = & $probe shaders $packPath --measure-pso $Hash --backend $Backend --windows 4 --split 1 --repeat $Repeat --json
            if ($LASTEXITCODE -ne 0) { throw "xcl measurement failed: $LASTEXITCODE" }
            $result = ($raw -join "`n") | ConvertFrom-Json
            if ($result.backend -ne $Backend -or $result.adapter -notlike '*NVIDIA*' -or $result.driver_seconds.Count -ne $Repeat) {
                throw 'Expected NVIDIA/backend JSON from #1802; this result is not a supported measurement.'
            }
            $result | Add-Member -NotePropertyName cache_control -NotePropertyValue 'temporary_nvidia_profile_off'
            $result | ConvertTo-Json -Depth 8 -Compress
        } -Remove {
            try {
                [XclColdPso.Profile]::Change($profile, $probe, $false)
                Write-Host "Removed temporary NVIDIA profile: $profile"
            }
            catch { throw "Could not remove temporary profile '$profile': $_" }
        }
    } finally {
        # The only file removed is this invocation's new, exact sibling of xcl.exe.
        Remove-Item -LiteralPath $probe
    }
}

if ($MyInvocation.InvocationName -ne '.') { Invoke-NvidiaColdPso }
