//! The host process: started by the launcher with the game's command line. Sets up
//! logging, the VFS, hooks and HLE, maps the title and jumps to its entry point.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::Boot;

pub static HOST_MODE: AtomicBool = AtomicBool::new(false);

pub fn host_main(boot_path: &Path) -> i32 {
    let boot: Boot = match std::fs::read_to_string(boot_path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string())) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("xcl host: cannot read {}: {e}", boot_path.display());
            return 0x7C1_0001;
        }
    };
    crate::log::init(
        Some(&boot.logs_dir().join("xcl.log")),
        crate::log::Level::parse(&boot.config.log_level),
        boot.config.log_to_stderr,
    );
    crate::xlog_info!("XCL {} – host for '{}'", env!("CARGO_PKG_VERSION"), boot.title);
    crate::xlog_info!("game: {} (read-only folder: {})", boot.exe_path.display(), boot.game_root.display());
    crate::xlog_info!("XCL data: {}", boot.data_dir.display());

    // Keep the job object handle alive for the lifetime of the host run.
    let _commit_cap = match crate::config::title_commit_cap_mb_from_env() {
        Ok(Some(cap_mb)) => match apply_commit_cap_mb(cap_mb) {
            Ok(job) => {
                crate::xlog_info!("process commit cap: {cap_mb} MB ({})", crate::config::ENV_TITLE_COMMIT_CAP_MB);
                Some(job)
            }
            Err(e) => {
                crate::xlog_warn!("process commit cap ({cap_mb} MB) not applied: {e}");
                None
            }
        },
        Ok(None) => None,
        Err(e) => {
            crate::xlog_warn!("{e}");
            None
        }
    };

    if std::env::var_os("XCL_UMA_MB").is_none() && boot.config.uma_mb != 0 {
        std::env::set_var("XCL_UMA_MB", boot.config.uma_mb.to_string());
    }
    crate::state::set_boot(boot.clone());
    // `XCL_ACTIVITY=1`: every 2 s what the title did since (a timeline of where it stops)
    if std::env::var_os("XCL_ACTIVITY").is_some() {
        let _ = std::thread::Builder::new().name("xcl-activity".into()).spawn(|| {
            let snap = || {
                let (p, e) = crate::hle::d3d12x::present::activity();
                let (d, x) = crate::hle::d3d12x::bind::activity();
                let f = crate::hle::d3d12x::frame::activity();
                let (r, b, fail) = crate::hle::dstorage::counters();
                [p, e, d, x, f[0], f[1], f[2], f[3], r, b >> 20, fail]
            };
            let mut last = snap();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let now = snap();
                let d: Vec<u64> = now.iter().zip(last).map(|(a, b)| a - b).collect();
                crate::xlog_info!(
                    "activity (2 s): {} presents, {} submits, {} draws, {} dispatches, {} frame-event waits ({} timed out), {} frame statistics ({} not flipped); DirectStorage {} requests, {} MB, {} failed",
                    d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10]
                );
                // where the memory is: the process's commit, video memory, and XCL's parts of it
                let private = unsafe {
                    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
                    let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
                    c.cb = std::mem::size_of_val(&c) as u32;
                    K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb);
                    c.PrivateUsage
                };
                let (local, non_local) = crate::hle::d3d12x::real::video_memory().unwrap_or((0, 0));
                let (arenas, arenas_made) = crate::hle::d3d12x::vram::arena_bytes();
                let (pools, pools_mapped) = crate::hle::d3d12x::sparse::pool_bytes();
                let (pages, page_pools) = crate::hle::d3d12x::vram::page_bytes();
                let mb = |b: u64| b >> 20;
                crate::xlog_info!(
                    "memory: {} MB committed by the process; video memory {} MB local, {} MB non-local; target pages {} MB (pools {} MB); arenas {} MB in use, {} MB made; sparse pools {} MB ({} MB mapped)",
                    mb(private as u64),
                    mb(local),
                    mb(non_local),
                    mb(pages as u64),
                    mb(page_pools as u64),
                    mb(arenas as u64),
                    mb(arenas_made as u64),
                    mb(pools as u64),
                    mb(pools_mapped as u64)
                );
                last = now;
            }
        });
    }
    // `XCL_THREAD_DUMP_AT=s1,s2…`: where every thread is at those seconds (a stalled title)
    if let Ok(v) = std::env::var("XCL_THREAD_DUMP_AT") {
        let at: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let _ = std::thread::Builder::new().name("xcl-thread-dump".into()).spawn(move || {
            let t0 = std::time::Instant::now();
            for s in at {
                let left = s - t0.elapsed().as_secs_f64();
                if left > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(left));
                }
                crate::eh::dump_threads();
            }
        });
    }
    // `XCL_RUN_SECONDS=n`: the title ends itself after n seconds – the GPU drained, the log and
    // report written (a process killed from outside with GPU work in flight can hang in the driver)
    if let Some(secs) = std::env::var("XCL_RUN_SECONDS").ok().and_then(|v| v.parse::<u64>().ok()) {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            let (shown, presented) = crate::hle::d3d12x::present::counters();
            crate::xlog_info!("XCL_RUN_SECONDS: {secs} s – ending the title ({shown} frames shown, {presented} PresentX calls)");
            crate::report::write_final(None);
            crate::log::flush();
            crate::hle::d3d12x::real::end_title(0);
        });
    }
    crate::stubs::init_db(&boot.config.stub_returns);
    crate::hle::init(&boot.config);
    crate::hle::kernelx::fine_timer();
    crate::loader::set_cache_dir(boot.cache_dir());
    let mut dirs = vec![boot.game_root.clone()];
    if let Some(p) = boot.exe_path.parent() {
        if p != boot.game_root {
            dirs.insert(0, p.to_path_buf());
        }
    }
    crate::resolver::set_search_dirs(dirs);

    let mounts: Vec<(String, String)> = boot.config.mounts.iter().map(|(k, v)| (k.clone(), boot.expand(v))).collect();
    // (the shader cache is XCL's too: on a drive a console mount names it must stay where it is)
    crate::vfs::init(crate::vfs::Vfs::new(&boot.game_root, &boot.data_dir, mounts).own_folder(&crate::gpu::recomp::dxc::cache_dir()));
    let _ = std::fs::create_dir_all(boot.data_dir.join("temp"));

    if let Err(e) = crate::vfs_hooks::install() {
        crate::xlog_error!("VFS: not all hooks installed: {e}");
    }
    if let Err(e) = crate::eh::install_pc_hook() {
        crate::xlog_error!("hook RtlPcToFileHeader: {e} – C++ exceptions in the game may not work");
    }
    crate::eh::install_crash_handler();
    crate::eh::start_probes();
    HOST_MODE.store(true, Ordering::Release);

    let main = match crate::loader::load_main(&boot.exe_path) {
        Ok(m) => m,
        Err(e) => {
            crate::xlog_error!("loading the game failed: {e}");
            crate::report::write_final(Some(format!("loader: {e}")));
            return 0x7C1_0002;
        }
    };
    unsafe { crate::hle::kernelx::adopt_title_hooks(&main) };
    crate::report::write_final(None);
    let code = crate::loader::run_main(&main);
    crate::xlog_info!("game entry point returned code {code}");
    crate::report::write_final(None);
    crate::log::flush();
    code as i32
}

/// Live Windows Job Object handle enforcing `XCL_TITLE_COMMIT_CAP_MB` for the host run.
pub struct CommitCapJob(windows_sys::Win32::Foundation::HANDLE);

unsafe impl Send for CommitCapJob {}
unsafe impl Sync for CommitCapJob {}

impl Drop for CommitCapJob {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0) };
        }
    }
}

/// Applies a Windows Job Object commit cap (`JOB_OBJECT_LIMIT_PROCESS_MEMORY`) of `cap_mb` MiB
/// to the current process.
pub fn apply_commit_cap_mb(cap_mb: u64) -> Result<CommitCapJob, String> {
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, IsProcessInJob, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    if cap_mb == 0 {
        return Err("commit cap must be > 0 MiB".into());
    }
    let cap_bytes = cap_mb
        .checked_mul(1024 * 1024)
        .and_then(|b| usize::try_from(b).ok())
        .ok_or_else(|| format!("{}: {} MB overflows process commit limit", crate::config::ENV_TITLE_COMMIT_CAP_MB, cap_mb))?;

    unsafe {
        let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if handle.is_null() {
            let err = GetLastError();
            return Err(format!("CreateJobObjectW failed: error {err} ({err:#x})"));
        }
        let job = CommitCapJob(handle);

        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = cap_bytes;
        if SetInformationJobObject(
            job.0,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            std::mem::size_of_val(&info) as u32,
        ) == 0
        {
            let err = GetLastError();
            return Err(format!(
                "SetInformationJobObject(JOB_OBJECT_LIMIT_PROCESS_MEMORY = {cap_mb} MB) failed: error {err} ({err:#x})"
            ));
        }

        if AssignProcessToJobObject(job.0, GetCurrentProcess()) == 0 {
            let err = GetLastError();
            let mut in_job = 0i32;
            let nested_note = if IsProcessInJob(GetCurrentProcess(), std::ptr::null_mut(), &mut in_job) != 0 && in_job != 0 {
                " (process is already in a job object that forbids nesting or breakaway)"
            } else {
                ""
            };
            return Err(format!(
                "AssignProcessToJobObject failed: error {err} ({err:#x}){nested_note}"
            ));
        }

        Ok(job)
    }
}

/// Reads `XCL_TITLE_COMMIT_CAP_MB` and, when set to a positive MiB value, applies a Windows Job
/// Object commit cap (`JOB_OBJECT_LIMIT_PROCESS_MEMORY`) to the current process.
pub fn apply_commit_cap_from_env() -> Result<Option<CommitCapJob>, String> {
    match crate::config::title_commit_cap_mb_from_env()? {
        Some(cap_mb) => apply_commit_cap_mb(cap_mb).map(Some),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_and_overflow_commit_cap() {
        assert!(apply_commit_cap_mb(0).is_err());
        assert!(apply_commit_cap_mb(u64::MAX).is_err());
    }

    #[test]
    fn title_commit_cap_enforced_only_when_positive() {
        const CHILD_MODE: &str = "XCL_TEST_COMMIT_CAP_MODE";
        if let Ok(mode) = std::env::var(CHILD_MODE) {
            use windows_sys::Win32::Foundation::{GetLastError, ERROR_COMMITMENT_LIMIT, ERROR_NOT_ENOUGH_MEMORY};
            use windows_sys::Win32::System::Memory::{
                VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
            };

            const ALLOC_BYTES: usize = 128 * 1024 * 1024;
            match mode.as_str() {
                "unset" | "zero" => {
                    let job = apply_commit_cap_from_env().expect("disabled cap should not error");
                    assert!(job.is_none(), "expected no job handle when cap is disabled ({mode})");
                    unsafe {
                        let ptr = VirtualAlloc(std::ptr::null(), ALLOC_BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                        assert!(!ptr.is_null(), "uncapped 128 MB VirtualAlloc failed with {}", GetLastError());
                        assert_ne!(VirtualFree(ptr, 0, MEM_RELEASE), 0);
                    }
                }
                "capped" => {
                    let _job = apply_commit_cap_from_env()
                        .expect("64 MB cap should succeed")
                        .expect("expected live job handle for 64 MB cap");
                    unsafe {
                        let ptr = VirtualAlloc(std::ptr::null(), ALLOC_BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                        let err = GetLastError();
                        assert!(ptr.is_null(), "128 MB VirtualAlloc unexpectedly succeeded under 64 MB cap");
                        assert!(
                            err == ERROR_NOT_ENOUGH_MEMORY || err == ERROR_COMMITMENT_LIMIT,
                            "expected ERROR_NOT_ENOUGH_MEMORY ({ERROR_NOT_ENOUGH_MEMORY}) or ERROR_COMMITMENT_LIMIT ({ERROR_COMMITMENT_LIMIT}), got {err}"
                        );
                        if !ptr.is_null() {
                            VirtualFree(ptr, 0, MEM_RELEASE);
                        }
                    }
                    let mut buf = Vec::<u8>::new();
                    assert!(
                        buf.try_reserve(ALLOC_BYTES).is_err(),
                        "128 MB heap reservation unexpectedly succeeded under 64 MB cap"
                    );
                }
                other => panic!("unexpected child mode: {other}"),
            }
            return;
        }

        let exe = std::env::current_exe().unwrap();
        let test_name = "host::tests::title_commit_cap_enforced_only_when_positive";

        for (mode, env_val) in [("unset", None), ("zero", Some("0")), ("capped", Some("64"))] {
            let mut cmd = std::process::Command::new(&exe);
            cmd.args(["--exact", test_name, "--nocapture"]).env(CHILD_MODE, mode);
            match env_val {
                Some(v) => {
                    cmd.env(crate::config::ENV_TITLE_COMMIT_CAP_MB, v);
                }
                None => {
                    cmd.env_remove(crate::config::ENV_TITLE_COMMIT_CAP_MB);
                }
            }
            let out = cmd.output().unwrap();
            assert!(
                out.status.success(),
                "child mode {mode} failed (status {:?}):\nstdout:\n{}\nstderr:\n{}",
                out.status.code(),
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}
