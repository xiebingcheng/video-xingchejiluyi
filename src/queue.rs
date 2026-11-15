//! 目录扫描、批量处理队列、缩略图后台生成。

use crate::config::thumbs_dir;
use crate::detect::{quiet_low_priority, Caps};
use crate::job::{self, JobEvent};
use crate::model::{FileEntry, FileStatus, VIDEO_EXTS};
use eframe::egui;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Instant;

/// 后台事件（线程 → UI）。
pub enum Ev {
    Scanned(FileEntry),
    ScanDone,
    Thumb { path: PathBuf, jpeg: Vec<u8> },
    Progress { path: PathBuf, frac: f32, speed: f32 },
    Status { path: PathBuf, status: FileStatus },
    Log(String),
    BatchDone { done: usize, failed: usize, skipped: usize, cancelled: bool },
}

/// 事件发送器：发出事件的同时唤醒 egui 界面。
/// 关键点：eframe 在无系统事件时完全休眠，后台线程必须主动 request_repaint，
/// 否则进度事件堆积在通道里、界面永远停留在旧画面。
#[derive(Clone)]
pub struct Notifier {
    tx: Sender<Ev>,
    ctx: Option<egui::Context>,
}

impl Notifier {
    pub fn new(tx: Sender<Ev>, ctx: Option<egui::Context>) -> Self {
        Self { tx, ctx }
    }

    /// 发送事件并唤醒界面；返回 false 表示接收端已关闭。
    pub fn send(&self, ev: Ev) -> bool {
        let ok = self.tx.send(ev).is_ok();
        if let Some(c) = &self.ctx {
            c.request_repaint();
        }
        ok
    }
}

/// 扫描输入目录（可选递归），跳过输出目录自身内容与系统目录。
pub fn spawn_scan(
    input: &Path,
    output_root: &Path,
    include_subdirs: bool,
    tx: Notifier,
) -> std::thread::JoinHandle<()> {
    let root = input.to_path_buf();
    let out_root = output_root.to_path_buf();
    std::thread::spawn(move || {
        let min_depth = 1;
        let max_depth = if include_subdirs { usize::MAX } else { 1 };
        let mut walker = walkdir::WalkDir::new(&root)
            .min_depth(min_depth)
            .max_depth(max_depth)
            .follow_links(false)
            .into_iter();
        let mut found: Vec<FileEntry> = Vec::new();
        while let Some(entry) = walker.next() {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if entry.file_type().is_dir() {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if name.starts_with('.')
                    || name == "$recycle.bin"
                    || name == "system volume information"
                {
                    walker.skip_current_dir();
                }
                continue;
            }
            let path = entry.path().to_path_buf();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase())
                .unwrap_or_default();
            if !VIDEO_EXTS.contains(&ext.as_str()) {
                continue;
            }
            // 输出目录在输入目录内时，避免把输出文件也扫进来
            if path.starts_with(&out_root) {
                continue;
            }
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let rel = path.strip_prefix(&root).unwrap_or(&path).to_path_buf();
            found.push(FileEntry::new(path, rel, size));
        }
        found.sort_by(|a, b| a.rel.cmp(&b.rel));
        for e in found {
            if !tx.send(Ev::Scanned(e)) {
                break;
            }
        }
        let _ = tx.send(Ev::ScanDone);
    })
}

/// 待处理任务。
pub struct BatchItem {
    pub input: PathBuf,
    pub rel: PathBuf,
}

pub struct BatchSpec {
    pub items: Vec<BatchItem>,
    pub output_root: PathBuf,
    pub flip: crate::model::FlipMode,
    pub denoise: crate::model::DenoiseLevel,
    pub skip_existing: bool,
    pub max_threads: u32,
    /// 同时处理的文件数（1~4）
    pub jobs: usize,
    /// 自动模式：按 GPU 实时占用动态增减并发
    pub auto_jobs: bool,
    /// 输出体积百分比（100 = 与原片相当）
    pub size_pct: u32,
    /// 用 HEVC 编码（体积更小）
    pub prefer_hevc: bool,
}

pub fn spawn_batch(
    caps: Arc<Caps>,
    spec: BatchSpec,
    tx: Notifier,
    cancel: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let items = Arc::new(spec.items);
        let total = items.len();
        let next_idx = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));
        let skipped = Arc::new(AtomicUsize::new(0));
        // 期望并发数；自动模式由采样线程按 GPU 占用动态更新
        let desired = Arc::new(AtomicUsize::new(spec.jobs.max(1)));
        let stop_sampler = Arc::new(AtomicBool::new(false));

        // GPU 调速线程：把 GPU 平均占用限制在上限（默认 95%）以内，
        // 避免长时间 100% 满载。与并发数无关，始终生效。
        let mut governor = None;
        if caps.nvidia_smi {
            let caps2 = caps.clone();
            let stop2 = stop_sampler.clone();
            let tx2 = tx.clone();
            governor = Some(crate::throttle::spawn_governor(caps2, stop2, tx2));
        }

        // GPU 占用采样线程（自动模式）：每 5 秒采样一次，动态增减并发。
        // 用户在游戏等高负载时图形占用高 → 降并发让资源给前台；GPU 空闲 → 提高并发。
        let mut sampler = None;
        if spec.auto_jobs && caps.nvidia_smi {
            let caps2 = caps.clone();
            let desired2 = desired.clone();
            let stop2 = stop_sampler.clone();
            let cancel2 = cancel.clone();
            let tx2 = tx.clone();
            sampler = Some(std::thread::spawn(move || {
                let mut last = 255u32;
                loop {
                    if cancel2.load(Ordering::Relaxed) || stop2.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Some(util) = caps2.gpu_graphics_util() {
                        let want: usize = if util < 25.0 {
                            3
                        } else if util < 60.0 {
                            2
                        } else {
                            1
                        };
                        let rounded = (util as u32).min(100);
                        if rounded != last {
                            last = rounded;
                            let _ = tx2.send(Ev::Log(format!(
                                "GPU 占用 {}% → 并发 {} 路",
                                rounded, want
                            )));
                        }
                        desired2.store(want, Ordering::Relaxed);
                    }
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
            }));
        }

        // 单个 worker：按共享索引领取任务；编号达到期望并发数时主动退出
        let run_worker = |id: usize| {
            let items = items.clone();
            let next_idx = next_idx.clone();
            let done = done.clone();
            let failed = failed.clone();
            let skipped = skipped.clone();
            let desired = desired.clone();
            let tx = tx.clone();
            let cancel = cancel.clone();
            let caps = caps.clone();
            let out_root = spec.output_root.clone();
            let flip = spec.flip;
            let denoise = spec.denoise;
            let skip_existing = spec.skip_existing;
            let size_pct = spec.size_pct;
            let prefer_hevc = spec.prefer_hevc;
            // 总线程预算按并发数分摊，避免多路并发吃满 CPU
            let max_threads = (((spec.max_threads as f32) / (spec.jobs.max(1) as f32)).ceil() as u32).max(1);
            std::thread::spawn(move || 'worker: loop {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                if id >= desired.load(Ordering::Relaxed) {
                    break; // 并发收缩，多余 worker 退出
                }
                let idx = next_idx.fetch_add(1, Ordering::SeqCst);
                if idx >= items.len() {
                    break;
                }
                let item = &items[idx];
                let out_ext = out_ext_for(&item.rel);
                let final_path = with_ext(&out_root.join(&item.rel), &out_ext);
                let part_path = with_ext(&final_path, "part");

                // 断点复用：输出已存在则跳过
                if skip_existing && final_path.is_file() {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    let _ = tx.send(Ev::Status {
                        path: item.input.clone(),
                        status: FileStatus::Skipped,
                    });
                    continue;
                }
                if let Some(parent) = final_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let t0 = Instant::now();
                let _ = tx.send(Ev::Status {
                    path: item.input.clone(),
                    status: FileStatus::Processing,
                });
                let _ = tx.send(Ev::Log(format!("▶ 处理 {} ...", item.rel.display())));

                let (ptx, prx) = std::sync::mpsc::channel::<JobEvent>();
                let input = item.input.clone();
                let part = part_path.clone();
                let caps2 = caps.clone();
                let cancel2 = cancel.clone();
                let jh = std::thread::spawn(move || {
                    job::run(
                        &caps2,
                        flip,
                        denoise,
                        max_threads,
                        &input,
                        &part,
                        &ptx,
                        &cancel2,
                        size_pct,
                        prefer_hevc,
                    )
                });
                // 汇聚进度事件
                loop {
                    match prx.recv_timeout(std::time::Duration::from_millis(200)) {
                        Ok(JobEvent::Progress(p)) => {
                            let _ = tx.send(Ev::Progress {
                                path: item.input.clone(),
                                frac: p.frac,
                                speed: p.speed,
                            });
                        }
                        Ok(JobEvent::Heartbeat(msg)) => {
                            let _ = tx.send(Ev::Log(msg));
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                        Err(_) => break,
                    }
                    if jh.is_finished() {
                        break;
                    }
                }
                let result = jh.join().unwrap_or_else(|_| Err(anyhow::anyhow!("工作线程崩溃")));

                match result {
                    Ok(()) => {
                        // .part → 最终文件
                        if let Err(e) = std::fs::rename(&part_path, &final_path) {
                            failed.fetch_add(1, Ordering::Relaxed);
                            let _ = tx.send(Ev::Status {
                                path: item.input.clone(),
                                status: FileStatus::Failed(format!("改名失败: {e}")),
                            });
                        } else {
                            done.fetch_add(1, Ordering::Relaxed);
                            let secs = t0.elapsed().as_secs_f32();
                            let _ = tx.send(Ev::Status {
                                path: item.input.clone(),
                                status: FileStatus::Done,
                            });
                            let _ = tx.send(Ev::Progress {
                                path: item.input.clone(),
                                frac: 1.0,
                                speed: 0.0,
                            });
                            let _ = tx.send(Ev::Log(format!(
                                "✓ 完成 {}（耗时 {:.0}s）",
                                item.rel.display(),
                                secs
                            )));
                        }
                    }
                    Err(e) => {
                        let _ = std::fs::remove_file(&part_path);
                        let msg = format!("{e:#}");
                        let was_cancelled = msg.contains("已取消");
                        failed.fetch_add(1, Ordering::Relaxed);
                        let _ = tx.send(Ev::Status {
                            path: item.input.clone(),
                            status: if was_cancelled {
                                FileStatus::Pending
                            } else {
                                FileStatus::Failed(msg)
                            },
                        });
                        if was_cancelled {
                            break 'worker; // 用户取消，此 worker 停止领新任务
                        }
                    }
                }
            })
        };

        // 协调线程：按期望并发启动/回收 worker，直到任务取尽且全部结束
        let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::new();
        let mut next_id = 0usize;
        loop {
            handles.retain(|h| !h.is_finished());
            let active = handles.len();
            let cancelled = cancel.load(Ordering::Relaxed);
            let want = if cancelled {
                0
            } else {
                desired.load(Ordering::Relaxed).clamp(1, 4)
            };
            if !cancelled && active < want && next_idx.load(Ordering::SeqCst) < total {
                handles.push(run_worker(next_id));
                next_id += 1;
            }
            let drained = next_idx.load(Ordering::SeqCst) >= total;
            if handles.is_empty() && (drained || cancelled) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(400));
        }
        stop_sampler.store(true, Ordering::Relaxed);
        crate::throttle::resume_all(); // 兜底：确保没有 ffmpeg 被留在挂起状态
        if let Some(s) = sampler {
            let _ = s.join();
        }
        if let Some(g) = governor {
            let _ = g.join();
        }
        let _ = tx.send(Ev::BatchDone {
            done: done.load(Ordering::Relaxed),
            failed: failed.load(Ordering::Relaxed),
            skipped: skipped.load(Ordering::Relaxed),
            cancelled: cancel.load(Ordering::Relaxed),
        });
    })
}

fn out_ext_for(rel: &Path) -> String {
    match rel
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("mkv") => "mkv".into(),
        Some("mov") => "mov".into(),
        Some("webm") => "webm".into(),
        _ => "mp4".into(),
    }
}

fn with_ext(path: &Path, ext: &str) -> PathBuf {
    path.with_extension(ext)
}

/// 缩略图后台线程：单任务串行、低优先级，磁盘缓存。
pub struct ThumbRequest {
    pub path: PathBuf,
    pub size: u64,
    pub mtime: u64,
    pub flip: crate::model::FlipMode,
}

pub fn spawn_thumb_worker(
    caps: Arc<Caps>,
    rx: std::sync::mpsc::Receiver<ThumbRequest>,
    tx: Notifier,
    cancel: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    let _ = std::fs::create_dir_all(thumbs_dir());
    std::thread::spawn(move || {
        while let Ok(req) = rx.recv() {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let cache = cache_path(&req);
            if let Some(bytes) = std::fs::read(&cache).ok().filter(|b| !b.is_empty()) {
                if !tx.send(Ev::Thumb { path: req.path, jpeg: bytes }) {
                    break;
                }
                continue;
            }
            match generate(&caps, &req) {
                Ok(bytes) => {
                    let _ = std::fs::write(&cache, &bytes);
                    if !tx.send(Ev::Thumb { path: req.path, jpeg: bytes }) {
                        break;
                    }
                }
                Err(_) => {
                    // 缩略图失败不影响主流程，发空表示无图
                    let _ = tx.send(Ev::Thumb { path: req.path, jpeg: Vec::new() });
                }
            }
        }
    })
}

fn cache_path(req: &ThumbRequest) -> PathBuf {
    // 用路径+大小+修改时间+翻转模式的组合哈希做缓存键
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |b: &[u8]| {
        for &byte in b {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    };
    feed(req.path.to_string_lossy().as_bytes());
    feed(&req.size.to_le_bytes());
    feed(&req.mtime.to_le_bytes());
    feed(format!("{:?}", req.flip).as_bytes());
    thumbs_dir().join(format!("{h:016x}.jpg"))
}

fn generate(caps: &Caps, req: &ThumbRequest) -> anyhow::Result<Vec<u8>> {
    let tmp = cache_path(req).with_extension("tmp.jpg");
    // 与主流程相同的转向滤镜，预览即最终效果；缩略图单帧走 CPU，开销可忽略
    let gen_once = |ss: bool| -> anyhow::Result<()> {
        let mut cmd = Command::new(&caps.ffmpeg);
        quiet_low_priority(&mut cmd);
        cmd.args(["-hide_banner", "-loglevel", "error", "-y"]);
        if ss {
            cmd.args(["-ss", "1"]);
        }
        cmd.arg("-i")
            .arg(&req.path)
            .args(["-frames:v", "1", "-vf"])
            .arg(format!("{},scale=240:-2", flip_vf(&req.flip)))
            .args(["-q:v", "5"])
            .stdin(std::process::Stdio::null())
            .arg(&tmp);
        let out = cmd.output()?;
        if !out.status.success() || !tmp.is_file() {
            return Err(anyhow::anyhow!("缩略图生成失败"));
        }
        Ok(())
    };
    gen_once(true).or_else(|_| gen_once(false))?;
    let bytes = std::fs::read(&tmp)?;
    let _ = std::fs::remove_file(&tmp);
    Ok(bytes)
}

fn flip_vf(flip: &crate::model::FlipMode) -> &'static str {
    flip.cpu_filter()
}

