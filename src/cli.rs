//! 命令行批量模式：无界面直接转码（也便于自动化/验证）。
//!
//! 用法：
//!   video-fixer-cli <输入目录> [输出目录] [--angle 180|90cw|90ccw|hflip|vflip]
//!                   [--denoise off|light|medium] [--threads N] [--overwrite] [--flat]

use crate::config::Config;
use crate::detect::Caps;
use crate::model::{DenoiseLevel, FileStatus, FlipMode};
use crate::queue::{self, BatchItem, BatchSpec, Ev};
use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

pub fn run(args: &[String]) -> Result<()> {
    let mut input = String::new();
    let mut output: Option<String> = None;
    let mut angle = "180".to_string();
    let mut denoise = "light".to_string();
    let mut threads = 4u32;
    let mut jobs = 1usize;
    let mut size_pct = 100u32;
    let mut hevc = false;
    let mut overwrite = false;
    let mut flat = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--angle" => angle = it.next().cloned().unwrap_or_default(),
            "--denoise" => denoise = it.next().cloned().unwrap_or_default(),
            "--threads" => {
                threads = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| anyhow!("--threads 需要数字参数"))?
            }
            "--jobs" => {
                jobs = it
                    .next()
                    .and_then(|s| s.parse::<usize>().ok())
                    .ok_or_else(|| anyhow!("--jobs 需要数字参数"))?
                    .clamp(1, 4)
            }
            "--size" => {
                size_pct = it
                    .next()
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| anyhow!("--size 需要数字参数（10~100）"))?
                    .clamp(10, 100)
            }
            "--hevc" => hevc = true,
            "--overwrite" => overwrite = true,
            "--flat" => flat = true,
            other => {
                if input.is_empty() {
                    input = other.to_string();
                } else if output.is_none() {
                    output = Some(other.to_string());
                }
            }
        }
    }
    if input.is_empty() {
        return Err(anyhow!(
            "用法: video-fixer-cli <输入目录> [输出目录] [--angle 180|90cw|90ccw|hflip|vflip] [--denoise off|light|medium] [--threads N] [--overwrite] [--flat]"
        ));
    }
    let flip = match angle.as_str() {
        "180" => FlipMode::Rotate180,
        "90cw" => FlipMode::Rotate90cw,
        "90ccw" => FlipMode::Rotate90ccw,
        "hflip" => FlipMode::MirrorH,
        "vflip" => FlipMode::MirrorV,
        other => return Err(anyhow!("未知角度: {other}")),
    };
    let dn = match denoise.as_str() {
        "off" => DenoiseLevel::Off,
        "light" => DenoiseLevel::Light,
        "medium" => DenoiseLevel::Medium,
        other => return Err(anyhow!("未知降噪档位: {other}")),
    };

    let caps = Arc::new(Caps::locate()?);
    println!("硬件加速: {}", caps.hardware_desc());
    println!("ffmpeg: {}", caps.ffmpeg.display());

    let input_path = PathBuf::from(&input);
    let output_path = output.map(PathBuf::from).unwrap_or_else(|| input_path.join("调正输出"));
    println!("输入: {}", input_path.display());
    println!("输出: {}", output_path.display());

    // 同步扫描
    let (tx, rx): (std::sync::mpsc::Sender<queue::Ev>, Receiver<queue::Ev>) =
        std::sync::mpsc::channel();
    let note = queue::Notifier::new(tx.clone(), None);
    queue::spawn_scan(&input_path, &output_path, !flat, note.clone());
    let mut items: Vec<BatchItem> = Vec::new();
    loop {
        match rx.recv() {
            Ok(Ev::Scanned(e)) => items.push(BatchItem { input: e.path, rel: e.rel }),
            Ok(Ev::ScanDone) | Err(_) => break,
            _ => {}
        }
    }
    if items.is_empty() {
        println!("未发现视频文件。");
        return Ok(());
    }
    let total_size: u64 = {
        let mut s = 0u64;
        for it in &items {
            s += std::fs::metadata(&it.input).map(|m| m.len()).unwrap_or(0);
        }
        s
    };
    println!("发现 {} 个视频，共 {:.1} GB。开始处理…", items.len(), total_size as f64 / 1e9);

    let mut cfg = Config::default();
    cfg.skip_existing = !overwrite;
    let spec = BatchSpec {
        items,
        output_root: output_path,
        flip,
        denoise: dn,
        skip_existing: cfg.skip_existing,
        max_threads: threads,
        jobs,
        auto_jobs: false,
        size_pct,
        prefer_hevc: hevc,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let jh = queue::spawn_batch(caps, spec, note.clone(), cancel);

    // 控制台进度显示
    let mut current = String::new();
    while let Ok(ev) = rx.recv() {
        match ev {
            Ev::Scanned(_) | Ev::ScanDone => {}
            Ev::Thumb { .. } => {}
            Ev::Progress { path, frac, speed } => {
                if frac >= 0.0 {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    current = name.clone();
                    print!("\r  {} {:.0}%{}", name, frac * 100.0, " ".repeat(12));
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
                if speed > 0.0 {
                    print!(" {speed:.1}x ");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
            }
            Ev::Status { path, status } => {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                match status {
                    FileStatus::Done => println!("\r  {name} ✓ 完成{}", " ".repeat(20)),
                    FileStatus::Skipped => println!("\r  {name} 已存在，跳过"),
                    FileStatus::Failed(e) => println!("\r  {name} ✗ 失败: {e}"),
                    _ => {}
                }
            }
            Ev::Log(msg) => println!("{msg}"),
            Ev::BatchDone { done, failed, skipped, cancelled } => {
                println!(
                    "\n批次结束：完成 {done}，跳过 {skipped}，失败 {failed}{}",
                    if cancelled { "（已取消）" } else { "" }
                );
                break;
            }
        }
        let _ = &current;
    }
    let _ = jh.join();
    let _ = cfg;
    Ok(())
}
