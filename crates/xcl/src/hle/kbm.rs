//! Keyboard and mouse as the gamepad (`keyboard_mouse` in `xcl.toml`, on by default).
//!
//! Console titles read only gamepads. While the XCL window has the focus the keyboard and
//! the mouse play as the one gamepad XCL reports (on top of a real controller):
//!
//! | PC | gamepad |
//! |---|---|
//! | W A S D | left stick |
//! | mouse movement | right stick (the cursor is held in the window; Alt+Tab frees it) |
//! | left / right mouse button | right / left trigger |
//! | wheel up / down | right / left bumper (one press a notch) |
//! | middle mouse button | right stick click |
//! | Left Shift / Left Ctrl | left / right stick click |
//! | E / Q | Y / X |
//! | Space, Enter / Escape, Backspace | A / B |
//! | arrows / Tab / P | D-pad / View / Menu |
//!
//! The mouse is read with Raw Input on the XCL window (movement in mouse counts, not the
//! cursor); its speed becomes a stick deflection – the stick turns the camera at a rate,
//! so a mouse moving at a given speed turns it at a matching one (`mouse_sensitivity`).

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicIsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use super::gameinput::{
    GamepadState, A, B, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT, DPAD_UP, LEFT_SHOULDER, LEFT_STICK, MENU,
    RIGHT_SHOULDER, RIGHT_STICK, VIEW, X, Y,
};

// ---------------------------------------------------------------------------------------
// the mapping (pure, tested)
// ---------------------------------------------------------------------------------------

/// Host inputs use Windows virtual keys. Physical variants preserve the built-in
/// layout's positions on non-QWERTY keyboards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(i32)]
enum InputKey {
    MouseLeft = 0x1,
    MouseRight = 0x2,
    MouseMiddle = 0x4,
    MouseX1 = 0x5,
    MouseX2 = 0x6,
    Backspace = 0x8,
    Tab = 0x9,
    Enter = 0xd,
    Escape = 0x1b,
    Space = 0x20,
    Left = 0x25,
    Up = 0x26,
    Right = 0x27,
    Down = 0x28,
    Insert = 0x2d,
    Delete = 0x2e,
    LeftShift = 0xa0,
    RightShift = 0xa1,
    LeftCtrl = 0xa2,
    RightCtrl = 0xa3,
    LeftAlt = 0xa4,
    RightAlt = 0xa5,
    Comma = 0xbc,
    Period = 0xbe,
    A = 0x41,
    B = 0x42,
    C = 0x43,
    D = 0x44,
    E = 0x45,
    F = 0x46,
    G = 0x47,
    H = 0x48,
    I = 0x49,
    J = 0x4a,
    K = 0x4b,
    L = 0x4c,
    M = 0x4d,
    N = 0x4e,
    O = 0x4f,
    P = 0x50,
    Q = 0x51,
    R = 0x52,
    S = 0x53,
    T = 0x54,
    U = 0x55,
    V = 0x56,
    W = 0x57,
    X = 0x58,
    Y = 0x59,
    Z = 0x5a,
    Digit0 = 0x30,
    Num0 = 0x60,
    Digit1 = 0x31,
    Num1 = 0x61,
    Digit2 = 0x32,
    Num2 = 0x62,
    Digit3 = 0x33,
    Num3 = 0x63,
    Digit4 = 0x34,
    Num4 = 0x64,
    Digit5 = 0x35,
    Num5 = 0x65,
    Digit6 = 0x36,
    Num6 = 0x66,
    Digit7 = 0x37,
    Num7 = 0x67,
    Digit8 = 0x38,
    Num8 = 0x68,
    Digit9 = 0x39,
    Num9 = 0x69,
    F1 = 0x70,
    F2 = 0x71,
    F3 = 0x72,
    F4 = 0x73,
    F5 = 0x74,
    F6 = 0x75,
    F7 = 0x76,
    F8 = 0x77,
    F9 = 0x78,
    F10 = 0x79,
    F11 = 0x7a,
    F12 = 0x7b,
    PhysicalQ = 0x1051,
    PhysicalW = 0x1157,
    PhysicalE = 0x1245,
    PhysicalA = 0x1e41,
    PhysicalS = 0x1f53,
    PhysicalD = 0x2044,
}

impl InputKey {
    fn virtual_key(self, layout: *mut c_void) -> i32 {
        let code = self as i32;
        if code > 0xFF {
            scancode_to_vk((code >> 8) as u32, code & 0xFF, layout)
        } else {
            code
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
enum WheelButton {
    Unbound = 0,
    A = A,
    B = B,
    X = X,
    Y = Y,
    DpadUp = DPAD_UP,
    DpadDown = DPAD_DOWN,
    DpadLeft = DPAD_LEFT,
    DpadRight = DPAD_RIGHT,
    View = VIEW,
    Menu = MENU,
    LeftBumper = LEFT_SHOULDER,
    RightBumper = RIGHT_SHOULDER,
    LeftStick = LEFT_STICK,
    RightStick = RIGHT_STICK,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct InputProfile {
    move_up: Vec<InputKey>,
    move_left: Vec<InputKey>,
    move_down: Vec<InputKey>,
    move_right: Vec<InputKey>,
    a: Vec<InputKey>,
    b: Vec<InputKey>,
    x: Vec<InputKey>,
    y: Vec<InputKey>,
    dpad_up: Vec<InputKey>,
    dpad_down: Vec<InputKey>,
    dpad_left: Vec<InputKey>,
    dpad_right: Vec<InputKey>,
    view: Vec<InputKey>,
    menu: Vec<InputKey>,
    left_bumper: Vec<InputKey>,
    right_bumper: Vec<InputKey>,
    left_stick: Vec<InputKey>,
    right_stick: Vec<InputKey>,
    left_trigger: Vec<InputKey>,
    right_trigger: Vec<InputKey>,
    wheel_up: WheelButton,
    wheel_down: WheelButton,
}

impl Default for InputProfile {
    fn default() -> Self {
        use InputKey::*;
        Self {
            move_up: vec![PhysicalW],
            move_left: vec![PhysicalA],
            move_down: vec![PhysicalS],
            move_right: vec![PhysicalD],
            a: vec![Space, Enter],
            b: vec![Escape, Backspace],
            x: vec![PhysicalQ],
            y: vec![PhysicalE],
            dpad_up: vec![Up],
            dpad_down: vec![Down],
            dpad_left: vec![Left],
            dpad_right: vec![Right],
            view: vec![Tab],
            menu: vec![P],
            left_bumper: vec![],
            right_bumper: vec![],
            left_stick: vec![LeftShift],
            right_stick: vec![LeftCtrl, MouseMiddle],
            left_trigger: vec![MouseRight],
            right_trigger: vec![MouseLeft],
            wheel_up: WheelButton::RightBumper,
            wheel_down: WheelButton::LeftBumper,
        }
    }
}

impl InputProfile {
    pub(crate) fn is_default(&self) -> bool {
        self == &Self::default()
    }

    fn navigation_buttons(&self, held: impl Fn(InputKey) -> bool) -> u32 {
        Self::buttons([
            (&self.a, A),
            (&self.b, B),
            (&self.dpad_up, DPAD_UP),
            (&self.dpad_down, DPAD_DOWN),
            (&self.dpad_left, DPAD_LEFT),
            (&self.dpad_right, DPAD_RIGHT),
            (&self.view, VIEW),
            (&self.menu, MENU),
        ], held)
    }

    fn buttons<const N: usize>(bindings: [(&[InputKey], u32); N], held: impl Fn(InputKey) -> bool) -> u32 {
        bindings.into_iter().filter(|(keys, _)| keys.iter().copied().any(&held)).fold(0, |mask, (_, bit)| mask | bit)
    }

    fn apply_keys(&self, s: &mut GamepadState, held: impl Fn(InputKey) -> bool) {
        let down = |keys: &[InputKey]| keys.iter().copied().any(&held);
        let (lx, ly) = stick_from_keys(down(&self.move_up), down(&self.move_left), down(&self.move_down), down(&self.move_right));
        if (lx != 0.0 || ly != 0.0) && lx * lx + ly * ly >= s.left_x * s.left_x + s.left_y * s.left_y {
            s.left_x = lx;
            s.left_y = ly;
        }
        s.buttons |= Self::buttons([
            (&self.x, X),
            (&self.y, Y),
            (&self.left_bumper, LEFT_SHOULDER),
            (&self.right_bumper, RIGHT_SHOULDER),
            (&self.left_stick, LEFT_STICK),
            (&self.right_stick, RIGHT_STICK),
        ], &held);
        if down(&self.left_trigger) {
            s.left_trigger = 1.0;
        }
        if down(&self.right_trigger) {
            s.right_trigger = 1.0;
        }
    }
}

/// The left stick from the movement keys: full deflection, diagonals on the unit circle.
pub fn stick_from_keys(up: bool, left: bool, down: bool, right: bool) -> (f32, f32) {
    let x = (right as i32 - left as i32) as f32;
    let y = (up as i32 - down as i32) as f32;
    let len = (x * x + y * y).sqrt();
    if len == 0.0 {
        (0.0, 0.0)
    } else {
        (x / len, y / len)
    }
}

/// mouse counts a second that give full deflection at sensitivity 1
const FULL_SPEED: f32 = 4000.0;
/// slower than this (counts a second) the stick rests (sensor noise, a hand on the mouse)
const REST_SPEED: f32 = 20.0;
/// deflection the slowest movement starts at: past the inner dead zone games give sticks
const START: f32 = 0.15;
/// time constant of the speed's smoothing (s): the mouse reports in bursts, a stick is read
/// once a frame
const SMOOTHING: f32 = 0.03;

/// The right stick for a mouse velocity (counts a second, screen axes: y grows downwards).
pub fn stick_from_mouse(vx: f32, vy: f32, sensitivity: f32) -> (f32, f32) {
    let speed = (vx * vx + vy * vy).sqrt();
    if speed < REST_SPEED || sensitivity <= 0.0 {
        return (0.0, 0.0);
    }
    let m = (speed * sensitivity / FULL_SPEED).min(1.0);
    let out = START + (1.0 - START) * m;
    (vx / speed * out, -vy / speed * out)
}

/// The mouse's velocity, smoothed: counts since the last update over `dt` seconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Velocity {
    pub x: f32,
    pub y: f32,
}

impl Velocity {
    pub fn update(&mut self, dx: i32, dy: i32, dt: f32) {
        let dt = dt.clamp(0.001, 0.1);
        let a = 1.0 - (-dt / SMOOTHING).exp();
        self.x += a * (dx as f32 / dt - self.x);
        self.y += a * (dy as f32 / dt - self.y);
    }
}

/// press time of a wheel notch and the gap after it (µs): long enough for a title that reads
/// the gamepad once a frame to see each press
const NOTCH_HOLD_US: u64 = 50_000;
const NOTCH_GAP_US: u64 = 50_000;

/// Wheel notches as bumper presses, one after the other.
#[derive(Debug, Default)]
pub struct Wheel {
    /// notches still to press (+ up, - down)
    pending: i32,
    /// the press under way: (button, µs since it started)
    current: Option<(u32, u64)>,
}

impl Wheel {
    pub fn notches(&mut self, n: i32) {
        self.pending = (self.pending + n).clamp(-8, 8);
    }

    /// The bumper held now, `dt` seconds after the last call.
    pub fn step(&mut self, dt: f32, up: u32, down: u32) -> u32 {
        if let Some((b, t)) = self.current.as_mut() {
            *t += (dt.max(0.0) * 1e6).round() as u64;
            if *t < NOTCH_HOLD_US {
                return *b;
            }
            if *t < NOTCH_HOLD_US + NOTCH_GAP_US {
                return 0;
            }
            self.current = None;
        }
        if self.pending == 0 {
            return 0;
        }
        let b = if self.pending > 0 { up } else { down };
        self.pending -= self.pending.signum();
        self.current = Some((b, 0));
        b
    }
}

// ---------------------------------------------------------------------------------------
// input from the PC
// ---------------------------------------------------------------------------------------

#[link(name = "user32")]
extern "system" {
    fn GetAsyncKeyState(key: i32) -> i16;
    fn GetForegroundWindow() -> *mut c_void;
    fn GetWindowThreadProcessId(hwnd: *mut c_void, pid: *mut u32) -> u32;
    fn GetKeyboardLayout(id_thread: u32) -> *mut c_void;
    fn MapVirtualKeyExW(code: u32, map_type: u32, dwhkl: *mut c_void) -> u32;
    fn RegisterRawInputDevices(devices: *const RawInputDevice, n: u32, size: u32) -> i32;
    fn GetRawInputData(input: isize, command: u32, data: *mut c_void, size: *mut u32, header: u32) -> u32;
    fn ClipCursor(rect: *const Rect) -> i32;
    fn GetClientRect(hwnd: *mut c_void, rect: *mut Rect) -> i32;
    fn ClientToScreen(hwnd: *mut c_void, point: *mut Point) -> i32;
    fn SetCursor(cursor: *mut c_void) -> *mut c_void;
}

#[repr(C)]
struct RawInputDevice {
    usage_page: u16,
    usage: u16,
    flags: u32,
    target: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

#[repr(C)]
#[derive(Default)]
struct Point {
    x: i32,
    y: i32,
}

/// RAWINPUTHEADER (24 bytes) + RAWMOUSE
#[repr(C)]
#[derive(Default)]
struct RawMouseInput {
    kind: u32,
    size: u32,
    device: usize,
    wparam: usize,
    flags: u16,
    _pad: u16,
    button_flags: u16,
    button_data: u16,
    raw_buttons: u32,
    last_x: i32,
    last_y: i32,
    extra: u32,
}

const WM_INPUT: u32 = 0x00FF;
const WM_ACTIVATE: u32 = 0x0006;
const WM_SETCURSOR: u32 = 0x0020;
const WM_SIZE: u32 = 0x0005;
const WM_MOVE: u32 = 0x0003;
const WM_LBUTTONDOWN: u32 = 0x0201;
const WM_ENTERSIZEMOVE: u32 = 0x0231;
const WM_EXITSIZEMOVE: u32 = 0x0232;
const RID_INPUT: u32 = 0x1000_0003;
const RIM_TYPEMOUSE: u32 = 0;
const MOUSE_MOVE_ABSOLUTE: u16 = 1;
const RI_MOUSE_WHEEL: u16 = 0x0400;
const HTCLIENT: isize = 1;
const MAPVK_VSC_TO_VK_EX: u32 = 3;

/// Physical AT Set 1 keyboard scancodes for the left-hand cluster, paired with their
/// QWERTY virtual-key fallbacks if `MapVirtualKeyExW` returns 0.
#[cfg(test)]
const SC_Q: (u32, i32) = (0x10, 0x51);
#[cfg(test)]
const SC_W: (u32, i32) = (0x11, 0x57);
#[cfg(test)]
const SC_E: (u32, i32) = (0x12, 0x45);
#[cfg(test)]
const SC_A: (u32, i32) = (0x1E, 0x41);
#[cfg(test)]
const SC_S: (u32, i32) = (0x1F, 0x53);
#[cfg(test)]
const SC_D: (u32, i32) = (0x20, 0x44);

/// the XCL window (raw input goes there; the cursor is held in it)
static WINDOW: AtomicIsize = AtomicIsize::new(0);
/// the cursor is held in the window (it has the focus)
static CAPTURED: AtomicBool = AtomicBool::new(false);
/// the user is dragging or resizing the window (WM_ENTERSIZEMOVE … WM_EXITSIZEMOVE):
/// the cursor must stay free, or the 1-pixel clip teleports it back mid-drag
static SIZING: AtomicBool = AtomicBool::new(false);
/// mouse counts not yet taken by `apply`
static DX: AtomicI32 = AtomicI32::new(0);
static DY: AtomicI32 = AtomicI32::new(0);
/// wheel movement not yet taken (WHEEL_DELTA = 120 a notch)
static WHEEL: AtomicI32 = AtomicI32::new(0);

struct Mouse {
    last: Instant,
    velocity: Velocity,
    wheel: Wheel,
    wheel_rest: i32,
}

impl Mouse {
    fn reset(&mut self, now: Instant) {
        self.last = now;
        self.velocity = Velocity::default();
        self.wheel = Wheel::default();
        self.wheel_rest = 0;
    }
}

static MOUSE: Mutex<Option<Mouse>> = Mutex::new(None);

fn reset_mouse_state() {
    DX.store(0, Ordering::Relaxed);
    DY.store(0, Ordering::Relaxed);
    WHEEL.store(0, Ordering::Relaxed);
    let mut g = MOUSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = g.as_mut() {
        m.reset(Instant::now());
    }
}

pub fn enabled() -> bool {
    crate::state::config().keyboard_mouse
}

/// The XCL window is up: the mouse reports to it.
pub fn attach(hwnd: *mut c_void) {
    if !enabled() {
        return;
    }
    WINDOW.store(hwnd as isize, Ordering::Release);
    // generic desktop mouse
    let dev = RawInputDevice { usage_page: 1, usage: 2, flags: 0, target: hwnd };
    let ok = unsafe { RegisterRawInputDevices(&dev, 1, std::mem::size_of::<RawInputDevice>() as u32) } != 0;
    crate::xlog_info!(
        "keyboard and mouse play as the gamepad (WASD left stick, mouse right stick, mouse buttons triggers, wheel bumpers){}",
        if ok { "" } else { " – raw mouse input unavailable, the mouse is not read" }
    );
}

unsafe fn hold_cursor(hwnd: *mut c_void, on: bool) {
    // a drag/resize loop owns the cursor: clipping it to the window's centre would teleport
    // it back on every WM_MOVE of the drag
    if on && SIZING.load(Ordering::Acquire) {
        if CAPTURED.swap(false, Ordering::AcqRel) {
            ClipCursor(std::ptr::null());
        }
        return;
    }
    if !on {
        if CAPTURED.swap(false, Ordering::AcqRel) {
            ClipCursor(std::ptr::null());
        }
        return;
    }
    // a point at the client area's centre: the cursor stays there, hidden
    let mut r = Rect::default();
    GetClientRect(hwnd, &mut r);
    let mut c = Point { x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 };
    ClientToScreen(hwnd, &mut c);
    let at = Rect { left: c.x, top: c.y, right: c.x + 1, bottom: c.y + 1 };
    ClipCursor(&at);
    CAPTURED.store(true, Ordering::Release);
}

/// The process is ending: the cursor is let go (a held cursor outlives its window).
pub fn release() {
    unsafe { hold_cursor(WINDOW.load(Ordering::Acquire) as *mut c_void, false) };
}

/// The XCL window's messages: true when handled (the window procedure returns 0).
pub unsafe fn on_message(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> bool {
    if WINDOW.load(Ordering::Acquire) != hwnd as isize {
        return false;
    }
    match msg {
        WM_INPUT => {
            let mut raw = RawMouseInput::default();
            let mut size = std::mem::size_of::<RawMouseInput>() as u32;
            let n = GetRawInputData(lp, RID_INPUT, &mut raw as *mut _ as *mut c_void, &mut size, 24);
            if n != u32::MAX && n as usize >= std::mem::size_of::<RawMouseInput>() && raw.kind == RIM_TYPEMOUSE && CAPTURED.load(Ordering::Acquire) {
                if raw.flags & MOUSE_MOVE_ABSOLUTE == 0 {
                    DX.fetch_add(raw.last_x, Ordering::Relaxed);
                    DY.fetch_add(raw.last_y, Ordering::Relaxed);
                }
                if raw.button_flags & RI_MOUSE_WHEEL != 0 {
                    WHEEL.fetch_add(raw.button_data as i16 as i32, Ordering::Relaxed);
                }
            }
            false // DefWindowProc cleans up
        }
        WM_ACTIVATE => {
            let active = wp & 0xFFFF != 0;
            hold_cursor(hwnd, active);
            if !active {
                reset_mouse_state();
            }
            false
        }
        WM_ENTERSIZEMOVE => {
            SIZING.store(true, Ordering::Release);
            hold_cursor(hwnd, false);
            reset_mouse_state();
            false
        }
        WM_EXITSIZEMOVE => {
            SIZING.store(false, Ordering::Release);
            if GetForegroundWindow() == hwnd {
                hold_cursor(hwnd, true);
            }
            false
        }
        WM_LBUTTONDOWN | WM_SIZE | WM_MOVE => {
            if GetForegroundWindow() == hwnd {
                hold_cursor(hwnd, true);
            }
            false
        }
        WM_SETCURSOR if CAPTURED.load(Ordering::Acquire) && (lp & 0xFFFF) == HTCLIENT => {
            SetCursor(std::ptr::null_mut());
            true
        }
        _ => false,
    }
}

fn key(vk: i32) -> bool {
    // SAFETY: `GetAsyncKeyState` accepts any virtual-key code with no memory invariants.
    unsafe { GetAsyncKeyState(vk) as u16 & 0x8000 != 0 }
}

/// Active keyboard layout (`HKL`) of `hwnd`'s thread, falling back to the current thread.
fn keyboard_layout_for_window(hwnd: *mut c_void) -> *mut c_void {
    // SAFETY: `GetWindowThreadProcessId` and `GetKeyboardLayout` accept null/valid handles
    // and thread ID 0 (calling thread) with no pointer aliasing requirements.
    unsafe {
        let tid = if hwnd.is_null() { 0 } else { GetWindowThreadProcessId(hwnd, std::ptr::null_mut()) };
        let hkl = GetKeyboardLayout(tid);
        if hkl.is_null() && tid != 0 {
            GetKeyboardLayout(0)
        } else {
            hkl
        }
    }
}

/// Translate a physical hardware scancode to a virtual-key code for `layout`, falling
/// back to `fallback_vk` (the QWERTY VK) when `layout` is null or unmapped.
pub(crate) fn scancode_to_vk(scancode: u32, fallback_vk: i32, layout: *mut c_void) -> i32 {
    if layout.is_null() {
        return fallback_vk;
    }
    // SAFETY: `MapVirtualKeyExW` is a pure Win32 lookup on `scancode` and `layout`, returning 0
    // when no translation exists.
    let vk = unsafe { MapVirtualKeyExW(scancode, MAPVK_VSC_TO_VK_EX, layout) };
    if vk != 0 {
        vk as i32
    } else {
        fallback_vk
    }
}

/// Buttons traditionally available before the XCL rendering window is attached.
pub(crate) fn navigation_buttons(hwnd: *mut c_void) -> u32 {
    let layout = keyboard_layout_for_window(hwnd);
    crate::state::config().input_profile.navigation_buttons(|input| key(input.virtual_key(layout)))
}

/// The keyboard and the mouse on top of `s` (the real controller), while the XCL window
/// has the focus.
pub fn apply(s: &mut GamepadState) {
    if !enabled() {
        return;
    }
    let hwnd = WINDOW.load(Ordering::Acquire);
    let focused = hwnd != 0 && unsafe { GetForegroundWindow() } as isize == hwnd;
    if !focused {
        // Alt+Tab, another window: the cursor is free, nothing is pressed
        // SAFETY: `hold_cursor(..., false)` only releases the cursor clip when captured.
        unsafe { hold_cursor(hwnd as *mut c_void, false) };
        reset_mouse_state();
        return;
    }
    let layout = keyboard_layout_for_window(hwnd as *mut c_void);
    let config = crate::state::config();
    let profile = &config.input_profile;
    profile.apply_keys(s, |input| key(input.virtual_key(layout)));
    let mut g = MOUSE.lock().unwrap_or_else(|e| e.into_inner());
    let m = g.get_or_insert_with(|| Mouse { last: Instant::now(), velocity: Velocity::default(), wheel: Wheel::default(), wheel_rest: 0 });
    let now = Instant::now();
    let dt = now.duration_since(m.last).as_secs_f32();
    m.last = now;
    m.velocity.update(DX.swap(0, Ordering::Relaxed), DY.swap(0, Ordering::Relaxed), dt);
    let (rx, ry) = stick_from_mouse(m.velocity.x, m.velocity.y, config.mouse_sensitivity);
    if (rx != 0.0 || ry != 0.0) && rx * rx + ry * ry >= s.right_x * s.right_x + s.right_y * s.right_y {
        s.right_x = rx;
        s.right_y = ry;
    }
    m.wheel_rest += WHEEL.swap(0, Ordering::Relaxed);
    let notches = m.wheel_rest / 120;
    m.wheel_rest -= notches * 120;
    m.wheel.notches(notches);
    s.buttons |= m.wheel.step(dt.min(0.1), profile.wheel_up as u32, profile.wheel_down as u32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_preserves_keyboard_and_mouse_bindings() {
        use InputKey::{Backspace, Down, Enter, Escape, Left, LeftCtrl, LeftShift, MouseLeft, MouseMiddle,
            MouseRight, PhysicalA, PhysicalD, PhysicalE, PhysicalQ, PhysicalS, PhysicalW, Right, Space, Tab, Up};
        let profile = InputProfile::default();
        for (key, expected) in [
            (Space, A), (Enter, A), (Escape, B), (Backspace, B), (Up, DPAD_UP),
            (Down, DPAD_DOWN), (Left, DPAD_LEFT), (Right, DPAD_RIGHT), (Tab, VIEW), (InputKey::P, MENU),
        ] {
            assert_eq!(profile.navigation_buttons(|input| input == key), expected);
        }
        for (key, expected) in [(PhysicalQ, X), (PhysicalE, Y), (LeftShift, LEFT_STICK),
            (LeftCtrl, RIGHT_STICK), (MouseMiddle, RIGHT_STICK)] {
            let mut state = GamepadState::default();
            profile.apply_keys(&mut state, |input| input == key);
            assert_eq!(state.buttons, expected);
        }
        for (key, expected) in [(PhysicalW, (0.0, 1.0)), (PhysicalA, (-1.0, 0.0)),
            (PhysicalS, (0.0, -1.0)), (PhysicalD, (1.0, 0.0))] {
            let mut state = GamepadState::default();
            profile.apply_keys(&mut state, |input| input == key);
            assert_eq!((state.left_x, state.left_y), expected);
        }
        let mut state = GamepadState::default();
        profile.apply_keys(&mut state, |input| input == MouseLeft);
        assert_eq!((state.left_trigger, state.right_trigger), (0.0, 1.0));
        profile.apply_keys(&mut state, |input| input == MouseRight);
        assert_eq!((state.left_trigger, state.right_trigger), (1.0, 1.0));
        assert_eq!(profile.wheel_up as u32, RIGHT_SHOULDER);
        assert_eq!(profile.wheel_down as u32, LEFT_SHOULDER);
    }

    #[test]
    fn profile_inherits_defaults_and_can_replace_or_clear_bindings() {
        let legacy: crate::config::Config = toml::from_str("mouse_sensitivity = 2.0").unwrap();
        assert!(legacy.input_profile.is_default());
        let config: crate::config::Config = toml::from_str(r#"
            [input_profile]
            move_up = ["up"]
            dpad_up = []
            a = ["num0"]
            right_trigger = ["mouse_right"]
            left_trigger = ["mouse_left"]
            wheel_up = "y"
            wheel_down = "unbound"
        "#).unwrap();
        let p = &config.input_profile;
        let mut state = GamepadState { buttons: MENU, left_trigger: 0.25, ..Default::default() };
        p.apply_keys(&mut state, |key| matches!(key, InputKey::Up | InputKey::MouseRight));
        assert_eq!(state, GamepadState {
            buttons: MENU, left_y: 1.0, left_trigger: 0.25, right_trigger: 1.0, ..Default::default()
        });
        assert_eq!(p.navigation_buttons(|key| matches!(key, InputKey::Up | InputKey::Space)), 0);
        assert_eq!(p.navigation_buttons(|key| key == InputKey::Num0), A);
        assert_eq!(p.navigation_buttons(|key| key == InputKey::Escape), B);
        let mut wheel = Wheel::default();
        wheel.notches(1);
        assert_eq!(wheel.step(0.01, p.wheel_up as u32, p.wheel_down as u32), Y);
        let mut wheel = Wheel::default();
        wheel.notches(-1);
        assert_eq!(wheel.step(0.01, p.wheel_up as u32, p.wheel_down as u32), 0);

        let boot_json = serde_json::to_string(&config).unwrap();
        let host_config: crate::config::Config = serde_json::from_str(&boot_json).unwrap();
        assert_eq!(host_config.input_profile, *p);
        assert!(toml::from_str::<crate::config::Config>("[input_profile]\na = [\"typo\"]").is_err());
        assert!(toml::from_str::<crate::config::Config>("[input_profile]\nunknown = []").is_err());
    }

    #[test]
    fn movement_keys_make_a_unit_stick() {
        assert_eq!(stick_from_keys(false, false, false, false), (0.0, 0.0));
        assert_eq!(stick_from_keys(true, false, false, false), (0.0, 1.0));
        assert_eq!(stick_from_keys(false, true, false, false), (-1.0, 0.0));
        let (x, y) = stick_from_keys(true, false, false, true);
        assert!((x - y).abs() < 1e-6 && (x * x + y * y - 1.0).abs() < 1e-6);
        // opposite keys cancel
        assert_eq!(stick_from_keys(true, true, true, true), (0.0, 0.0));
    }

    #[test]
    fn mouse_speed_becomes_a_deflection() {
        // resting
        assert_eq!(stick_from_mouse(5.0, 0.0, 1.0), (0.0, 0.0));
        // slow movement starts past the dead zone, to the right
        let (x, y) = stick_from_mouse(100.0, 0.0, 1.0);
        assert!(x > START && x < 0.2 && y == 0.0);
        // fast movement: full deflection; the mouse moving down tilts the stick down
        let (x, y) = stick_from_mouse(0.0, 10_000.0, 1.0);
        assert!(x == 0.0 && (y + 1.0).abs() < 1e-6);
        // sensitivity scales the speed
        let (a, _) = stick_from_mouse(1000.0, 0.0, 1.0);
        let (b, _) = stick_from_mouse(1000.0, 0.0, 2.0);
        assert!(b > a);
    }

    #[test]
    fn velocity_follows_the_counts_and_settles() {
        let mut v = Velocity::default();
        for _ in 0..50 {
            v.update(8, -4, 0.004); // 2000 counts/s right, 1000 up
        }
        assert!((v.x - 2000.0).abs() < 20.0 && (v.y + 1000.0).abs() < 10.0);
        for _ in 0..50 {
            v.update(0, 0, 0.004);
        }
        assert!(v.x.abs() < REST_SPEED && v.y.abs() < REST_SPEED);
    }

    #[test]
    fn wheel_notches_press_bumpers_one_by_one() {
        let mut w = Wheel::default();
        w.notches(2);
        w.notches(-1); // net one notch up
        let mut presses = Vec::new();
        let mut last = 0;
        for _ in 0..100 {
            let b = w.step(0.01, RIGHT_SHOULDER, LEFT_SHOULDER);
            if b != 0 && b != last {
                presses.push(b);
            }
            last = b;
        }
        assert_eq!(presses, vec![RIGHT_SHOULDER]);
        w.notches(-2);
        let held: Vec<u32> = (0..30).map(|_| w.step(0.01, RIGHT_SHOULDER, LEFT_SHOULDER)).collect();
        // held for NOTCH_HOLD, released for NOTCH_GAP, then the second press
        assert_eq!(held.iter().filter(|&&b| b == LEFT_SHOULDER).count(), 10);
        assert!(held[..5].iter().all(|&b| b == LEFT_SHOULDER) && held[5..10].iter().all(|&b| b == 0));
    }

    #[test]
    fn scancodes_map_to_distinct_vks_and_fall_back_safely() {
        const MAPVK_VK_TO_VSC: u32 = 0;
        const MAPVK_VSC_TO_VK: u32 = 1;
        let layout = keyboard_layout_for_window(std::ptr::null_mut());
        assert!(!layout.is_null(), "current thread should have an active Windows keyboard layout");

        let cluster = [SC_Q, SC_W, SC_E, SC_A, SC_S, SC_D];
        let mut seen_vks = Vec::new();
        for &(sc, fb) in &cluster {
            let vk = scancode_to_vk(sc, fb, layout);
            // SAFETY: pure Win32 lookup on the current thread's HKL.
            let vk_basic = unsafe { MapVirtualKeyExW(sc, MAPVK_VSC_TO_VK, layout) } as i32;
            assert_eq!(vk, vk_basic, "MAPVK_VSC_TO_VK and MAPVK_VSC_TO_VK_EX should agree for scancode {sc:#x}");
            assert_ne!(vk, 0);
            assert!(!seen_vks.contains(&vk), "scancode {sc:#x} collided with another key's VK {vk:#x}");
            seen_vks.push(vk);

            // Round-trip VK -> scancode verifies we resolved the physical key position.
            // SAFETY: pure Win32 lookup on the current thread's HKL.
            let roundtrip = unsafe { MapVirtualKeyExW(vk as u32, MAPVK_VK_TO_VSC, layout) };
            assert_eq!(roundtrip, sc, "VK {vk:#x} should round-trip back to physical scancode {sc:#x}");
        }

        // Verify MAPVK_VSC_TO_VK_EX distinguishes Left Shift (0x2A -> VK_LSHIFT 0xA0) whereas
        // MAPVK_VSC_TO_VK collapses it to generic VK_SHIFT (0x10).
        // SAFETY: pure Win32 lookup on the current thread's HKL.
        let lshift_basic = unsafe { MapVirtualKeyExW(0x2A, MAPVK_VSC_TO_VK, layout) };
        let lshift_ex = scancode_to_vk(0x2A, 0xA0, layout) as u32;
        assert_eq!(lshift_basic, 0x10);
        assert_eq!(lshift_ex, 0xA0);

        // Fallback when layout is null or scancode has no mapping.
        assert_eq!(scancode_to_vk(SC_W.0, SC_W.1, std::ptr::null_mut()), SC_W.1);
        assert_eq!(scancode_to_vk(0xFFFF, SC_Q.1, layout), SC_Q.1);
    }

    #[test]
    fn focus_loss_resets_mouse_velocity_and_wheel_state() {
        // Seed raw deltas and active MOUSE state (velocity, in-flight & pending wheel notches, remainder).
        DX.store(250, Ordering::Relaxed);
        DY.store(-180, Ordering::Relaxed);
        WHEEL.store(240, Ordering::Relaxed);
        {
            let mut g = MOUSE.lock().unwrap_or_else(|e| e.into_inner());
            let mut v = Velocity::default();
            for _ in 0..20 {
                v.update(40, -20, 0.004);
            }
            let mut w = Wheel::default();
            w.notches(3);
            assert_eq!(w.step(0.01, RIGHT_SHOULDER, LEFT_SHOULDER), RIGHT_SHOULDER);
            *g = Some(Mouse {
                last: Instant::now(),
                velocity: v,
                wheel: w,
                wheel_rest: 90,
            });
            let m = g.as_ref().unwrap();
            assert!(m.velocity.x.abs() > REST_SPEED && m.velocity.y.abs() > REST_SPEED);
        }

        // With no focused XCL window (`WINDOW == 0`), `apply` treats focus as lost and resets everything.
        let mut s = GamepadState::default();
        apply(&mut s);
        assert_eq!(s, GamepadState::default());
        assert_eq!(DX.load(Ordering::Relaxed), 0);
        assert_eq!(DY.load(Ordering::Relaxed), 0);
        assert_eq!(WHEEL.load(Ordering::Relaxed), 0);

        let mut g = MOUSE.lock().unwrap_or_else(|e| e.into_inner());
        let m = g.as_mut().expect("mouse state should remain allocated");
        assert_eq!(m.velocity, Velocity::default());
        assert_eq!(m.wheel_rest, 0);
        for _ in 0..20 {
            assert_eq!(m.wheel.step(0.01, RIGHT_SHOULDER, LEFT_SHOULDER), 0, "stale wheel notches must not fire after refocusing");
        }
    }
}
