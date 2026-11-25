use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn version_flag_prints_version_and_exits_zero() {
    for flag in ["--version", "-V"] {
        let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg(flag)
            .output()
            .unwrap();
        assert!(output.status.success(), "flag {flag}: {output:?}");
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(
            text.starts_with(&format!("xcl {}", env!("CARGO_PKG_VERSION"))),
            "flag {flag}: unexpected output {text}"
        );
    }
}

#[test]
fn help_flag_prints_usage_and_exits_zero() {
    for flag in ["--help", "-h"] {
        let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg(flag)
            .output()
            .unwrap();
        assert!(output.status.success(), "flag {flag}: {output:?}");
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.contains("XCL"), "flag {flag}: {text}");
        assert!(text.contains("Usage:"), "flag {flag}: {text}");
        assert!(text.contains("xcl run"), "flag {flag}: {text}");
        assert!(text.contains("xcl inspect"), "flag {flag}: {text}");
        assert!(text.contains("xcl disasm"), "flag {flag}: {text}");
        assert!(text.contains("xcl shaders"), "flag {flag}: {text}");
        assert!(text.contains("xcl rpf"), "flag {flag}: {text}");
    }
}

#[test]
fn unknown_subcommand_exits_code_two_with_help() {
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("unknown_subcommand")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("Usage:"));
}

#[test]
fn inspect_missing_argument_exits_code_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("inspect")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("Usage:"));
}

#[test]
fn inspect_nonexistent_file_exits_code_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("inspect")
        .arg("nonexistent_binary.exe")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.contains("xcl inspect: cannot read"));
}

#[test]
fn inspect_invalid_pe_exits_code_three() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_file = std::env::temp_dir().join(format!("xcl-bad-pe-{}-{nonce}.bin", std::process::id()));
    std::fs::write(&temp_file, b"NOT_A_PE_BINARY_CONTENT").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("inspect")
        .arg(&temp_file)
        .output();
    let _ = std::fs::remove_file(&temp_file);

    let output = output.unwrap();
    assert_eq!(output.status.code(), Some(3));
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.contains("xcl inspect:"));
    assert!(err.contains("XVC/MSIXVC"));
}

#[test]
fn inspect_valid_pe_text_and_json() {
    let exe = env!("CARGO_BIN_EXE_xcl");

    // Text format
    let text_output = Command::new(exe)
        .arg("inspect")
        .arg(exe)
        .output()
        .unwrap();
    assert!(text_output.status.success());
    let stdout = String::from_utf8(text_output.stdout).unwrap();
    assert!(stdout.contains("File:"));
    assert!(stdout.contains("Type:"));
    assert!(stdout.contains("ImageBase:"));

    // JSON format
    let json_output = Command::new(exe)
        .arg("inspect")
        .arg(exe)
        .arg("--json")
        .output()
        .unwrap();
    assert!(json_output.status.success());
    let json_str = String::from_utf8(json_output.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!(v.get("file").is_some());
    assert!(v.get("machine").is_some());
    assert!(v.get("image_base").is_some());
    assert!(v.get("imports").is_some());
}
