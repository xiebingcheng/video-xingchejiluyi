//! GPU 调速器：防止 ffmpeg 批处理时 GPU 长时间持续满载。
//!
//! 原理（满载工时累计 + 强制休息）：
//!   1. 每个 ffmpeg 子进程启动时把进程句柄登记进注册表，结束时注销；
//!   2. 调速线程每秒用 nvidia-smi 采样 GPU 占用率，占用 ≥90% 时累计
//!      "满载工时"（GPU 本来就不忙时不干预，零速度损失）；
//!   3. 满载工时累计满 BREATH_AFTER_BUSY_MS 就把所有 ffmpeg 进程挂起
//!      SUSPEND_MS 强制休息，然后清零重新累计。
//!
//! 为什么不能用更细的占空比精确钉在 95%：解码/降噪/编码管线里缓冲着
//! 1 秒以上的帧，短挂起会被 GPU 缓冲完全吸收（实测恢复运行后 <0.5 秒
//! 就重新打满）。所以休息时长按"能真实压出 GPU 空闲"设计，效果是
//! 平均占用 ~85%、持续满载最长十几秒必被打断——满足护卡目的。
//!
//! 挂起用 ntdll 的 NtSuspendProcess / NtResumeProcess（动态加载，仅影响
//! 本程序自己启动的 ffmpeg 子进程）；单次 1.5 秒远小于看门狗的 120 秒
//! 无输出阈值，不会误杀任务。

use crate::detect::Caps;
use crate::queue::{Ev, Notifier};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// 视为"满载"的占用率阈值（%）。
const BUSY_UTIL_PCT: f32 = 90.0;
/// 满载工时累计到此值就强制休息。
const BREATH_AFTER_BUSY_MS: u64 = 10_000;
/// 单次强制休息时长：实测解码/降噪/编码管线的帧缓冲约 0.5~1 秒消化完，
/// 挂起 1.5 秒能真实产生 ~1 秒 GPU 空闲。
const SUSPEND_MS: u64 = 1500;

type NtSuspendFn = unsafe extern "system" fn(process: isize) -> i32;

/// 活动中的 ffmpeg 进程句柄（isize 便于跨线程复制）。
static REGISTRY: OnceLock<Mutex<HashSet<isize>>> = OnceLock::new();
/// 从 ntdll 动态解析出的挂起/恢复函数，解析失败则整体禁用调速。
static NTDLL: OnceLock<Option<(NtSuspendFn, NtSuspendFn)>> = OnceLock::new();

fn registry() -> &'static Mutex<HashSet<isize>> {
    REGISTRY.get_or_init(|| Mutex::new(HashSet::new()))
}

/// 忽略毒化（poisoned）的锁：调速器绝不能因 panic 而把进程留在挂起状态。
fn lock() -> std::sync::MutexGuard<'static, HashSet<isize>> {
    registry().lock().unwrap_or_else(|p| p.into_inner())
}

fn ntdll() -> Option<&'static (NtSuspendFn, NtSuspendFn)> {
    NTDLL
        .get_or_init(|| {
            use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
            unsafe {
                let name: Vec<u16> = "ntdll.dll".encode_utf16().chain(Some(0)).collect();
                let module = GetModuleHandleW(name.as_ptr());
                if module.is_null() {
                    return None;
                }
                let suspend = b"NtSuspendProcess\0".as_ptr();
                let resume = b"NtResumeProcess\0".as_ptr();
                Some((
                    std::mem::transmute(GetProcAddress(module, suspend)?),
                    std::mem::transmute(GetProcAddress(module, resume)?),
                ))
            }
        })
        .as_ref()
}

/// 把所有已登记 ffmpeg 挂起；返回是否有进程被实际挂起。
fn suspend_all() -> bool {
    let Some((suspend, _)) = ntdll() else { return false };
    let handles: Vec<isize> = lock().iter().copied().collect();
    let mut any = false;
    for h in handles {
        // 进程可能刚好退出：调用失败无妨，继续处理其余进程
        if unsafe { suspend(h) } == 0 {
            any = true;
        }
    }
    any
}

/// 恢复所有已登记 ffmpeg（不注销，进程仍在运行）。
fn resume_all_running() {
    let Some((_, resume)) = ntdll() else { return };
    let handles: Vec<isize> = lock().iter().copied().collect();
    for h in handles {
        unsafe { resume(h) };
    }
}

/// 兜底恢复：注销并恢复登记表里所有进程（批次结束/异常退出时调用）。
pub fn resume_all() {
    let Some((_, resume)) = ntdll() else { return };
    let handles: Vec<isize> = lock().drain().collect();
    for h in handles {
        unsafe { resume(h) };
    }
}

/// ffmpeg 进程在调速器注册表中的守卫：Drop 时自动注销并确保恢复运行。
pub struct Guard(isize);

impl Guard {
    /// 登记一个 ffmpeg 子进程（进程句柄由 Child 持有，这里只记数值）。
    pub fn register(child: &std::process::Child) -> Self {
        use std::os::windows::io::AsRawHandle;
        let h = child.as_raw_handle() as isize;
        lock().insert(h);
        Guard(h)
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if lock().remove(&self.0) {
            // 已注销才恢复：避免对已被 resume_all 处理过的句柄重复操作
            if let Some((_, resume)) = ntdll() {
                unsafe { resume(self.0) };
            }
        }
    }
}

/// 启动 GPU 调速线程：批处理期间防止 GPU 持续满载。
/// 没有 ffmpeg 在跑时低频空转；nvidia-smi 不可用时不启动。
pub fn spawn_governor(
    caps: std::sync::Arc<Caps>,
    stop: std::sync::Arc<AtomicBool>,
    tx: Notifier,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        // 调试：设置环境变量后把调速决策写到 %TEMP%\vf-governor.log
        let debug = std::env::var("VIDEO_FIXER_GPU_DEBUG").is_ok();
        let log_dbg = |msg: String| {
            if debug {
                let p = std::env::temp_dir().join("vf-governor.log");
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
                    use std::io::Write;
                    let _ = writeln!(f, "{msg}");
                }
            }
        };
        let t0 = Instant::now();
        let mut busy_ms: f32 = 0.0; // 自上次强制休息以来累计的满载工时
        let mut logged = false;
        let mut last = Instant::now();
        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            if lock().is_empty() {
                // 没有活动 ffmpeg（扫描/间歇期）不需要调速
                busy_ms = 0.0;
                last = Instant::now();
                std::thread::sleep(Duration::from_millis(1000));
                continue;
            }
            let Some(util) = caps.gpu_graphics_util() else {
                // 采样失败：短暂退避，持续失败时本轮批次等于不限载
                std::thread::sleep(Duration::from_millis(2000));
                continue;
            };
            let now = Instant::now();
            let dt = now.duration_since(last).as_millis() as f32;
            last = now;
            if util >= BUSY_UTIL_PCT {
                busy_ms += dt;
            }
            log_dbg(format!(
                "t={:.1}s util={util:.0} busy_ms={busy_ms:.0}",
                t0.elapsed().as_secs_f32()
            ));
            if busy_ms >= BREATH_AFTER_BUSY_MS as f32 {
                if !logged {
                    logged = true;
                    let _ = tx.send(Ev::Log(
                        "GPU 已持续高负载，自动降载保护显卡（满载约 10 秒休息 1.5 秒）".into(),
                    ));
                }
                if suspend_all() {
                    log_dbg(format!("  -> 强制休息 {SUSPEND_MS}ms"));
                    std::thread::sleep(Duration::from_millis(SUSPEND_MS));
                    resume_all_running();
                }
                busy_ms = 0.0;
                last = Instant::now();
            }
            std::thread::sleep(Duration::from_millis(1000));
        }
        // 退出兜底：绝不把 ffmpeg 留在挂起状态
        resume_all();
    })
}
