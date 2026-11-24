//! Global host state (set once at host start; defaults in unit tests).

use std::sync::OnceLock;

use crate::config::{Boot, Config};

static BOOT: OnceLock<Boot> = OnceLock::new();
static DEFAULT_CONFIG: OnceLock<Config> = OnceLock::new();
static DEFAULT_DATA_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

pub fn set_boot(b: Boot) {
    let _ = BOOT.set(b);
}

pub fn boot() -> Option<&'static Boot> {
    BOOT.get()
}

pub fn config() -> &'static Config {
    match BOOT.get() {
        Some(b) => &b.config,
        None => DEFAULT_CONFIG.get_or_init(Config::default),
    }
}

/// Folder for title data (saves, PLS, temp…) – a temp folder when no boot record.
pub fn data_dir() -> std::path::PathBuf {
    match BOOT.get() {
        Some(b) => b.data_dir.clone(),
        None => DEFAULT_DATA_DIR
            .get_or_init(|| {
                let temp = std::env::temp_dir();
                cleanup_stale_default_data_dirs(&temp);
                let nonce = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                default_data_dir(&temp, std::process::id(), nonce)
            })
            .clone(),
    }
}

fn default_data_dir(temp: &std::path::Path, pid: u32, nonce: u128) -> std::path::PathBuf {
    temp.join(format!("xcl_unit_data_{pid}_{nonce}"))
}

fn default_data_dir_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("xcl_unit_data_") else { return false };
    let Some((pid, nonce)) = rest.split_once('_') else { return false };
    pid.parse::<u32>().is_ok() && nonce.parse::<u128>().is_ok()
}

/// Test processes cannot rely on static destructors running at exit. Bound the leftovers
/// instead: the next test process removes only directories with our exact generated name
/// that have not been touched for a day. All failures are best-effort and leave data alone.
fn cleanup_stale_default_data_dirs(temp: &std::path::Path) {
    const MAX_AGE: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);
    let Ok(entries) = std::fs::read_dir(temp) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !default_data_dir_name(name) {
            continue;
        }
        let stale = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age >= MAX_AGE);
        if stale {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_data_folders_are_stable_per_process_and_separate_between_processes() {
        let temp = std::path::Path::new("temp");
        assert_eq!(default_data_dir(temp, 17, 42), temp.join("xcl_unit_data_17_42"));
        assert_eq!(default_data_dir(temp, 17, 42), default_data_dir(temp, 17, 42));
        assert_ne!(default_data_dir(temp, 17, 42), default_data_dir(temp, 18, 42));
        assert_ne!(default_data_dir(temp, 17, 42), default_data_dir(temp, 17, 43));
        assert!(default_data_dir_name("xcl_unit_data_17_42"));
        assert!(!default_data_dir_name("xcl_unit_data_17"));
        assert!(!default_data_dir_name("xcl_unit_data_17_42_extra"));
        assert!(!default_data_dir_name("xcl_unit_data_bad_42"));
    }
}
