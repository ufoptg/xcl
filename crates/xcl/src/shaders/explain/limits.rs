//! Limits of an `--explain` run: a job object caps the commit of a process (this one, or a
//! `dxc.exe`) so a program that needs more fails instead of growing the page file, and a
//! sampler records the peak commit of this process while a stage runs.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// A job object with a commit limit per process; its processes end when it is closed.
pub(super) struct Job(windows_sys::Win32::Foundation::HANDLE);

impl Job {
    pub(super) fn new(limit: u64) -> Option<Job> {
        use windows_sys::Win32::System::JobObjects::*;
        unsafe {
            let h = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if h.is_null() {
                return None;
            }
            let job = Job(h);
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            info.ProcessMemoryLimit = limit as usize;
            let ok = SetInformationJobObject(h, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32);
            (ok != 0).then_some(job)
        }
    }

    pub(super) fn assign(&self, process: windows_sys::Win32::Foundation::HANDLE) -> bool {
        unsafe { windows_sys::Win32::System::JobObjects::AssignProcessToJobObject(self.0, process) != 0 }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

/// Peak private bytes of the process while a stage runs, above what it had when the stage
/// started (sampled every 5 ms).
pub(super) struct Peak {
    base: u64,
    max: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn private_bytes() -> u64 {
    unsafe {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of_val(&c) as u32;
        K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb);
        // Wine fills PagefileUsage, not PrivateUsage (`kernelx`)
        if c.PrivateUsage != 0 { c.PrivateUsage as u64 } else { c.PagefileUsage as u64 }
    }
}

impl Peak {
    pub(super) fn start() -> Peak {
        let base = private_bytes();
        let (max, stop) = (Arc::new(AtomicU64::new(base)), Arc::new(AtomicBool::new(false)));
        let (m, s) = (max.clone(), stop.clone());
        let thread = std::thread::Builder::new()
            .name("xcl-explain-mem".into())
            .spawn(move || {
                while !s.load(Ordering::Relaxed) {
                    m.fetch_max(private_bytes(), Ordering::Relaxed);
                    std::thread::sleep(Duration::from_millis(5));
                }
            })
            .ok();
        Peak { base, max, stop, thread }
    }

    /// MB above the start
    pub(super) fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.max.fetch_max(private_bytes(), Ordering::Relaxed);
        self.max.load(Ordering::Relaxed).saturating_sub(self.base) >> 20
    }
}
