//! 单个视频的转码任务：探测信息 → 组装 ffmpeg 参数 → 执行并回报进度。

use crate::detect::{quiet_low_priority, Caps};
use crate::model::{DenoiseLevel, FlipMode};
use anyhow::{anyhow, Context, Result};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ProbeInfo {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub duration_s: f64,
    /// 视频流码率（bps）。可能来自格式总码率兜底。
    pub video_bitrate: u64,
}

fn parse_csv_field(s: &str) -> String {
    s.trim().trim_matches('"').to_string()
}

/// 用 ffprobe 获取视频流信息（csv 输出，避免引入 JSON 解析依赖）。
pub fn probe(caps: &Caps, input: &Path) -> Result<ProbeInfo> {
    let mut cmd = Command::new(&caps.ffprobe);
    let out = quiet_low_priority(&mut cmd)
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,bit_rate",
            "-of", "csv=p=0",
        ])
        .arg(input)
        .stdin(Stdio::null())
        .output()
        .context("运行 ffprobe 失败")?;
    let line = String::from_utf8_lossy(&out.stdout);
    let mut it = line.lines().filter(|l| !l.trim().is_empty());
    let first = it.next().unwrap_or_default();
    let f: Vec<&str> = first.split(',').collect();
    if f.is_empty() || f[0].trim().is_empty() {
        return Err(anyhow!("无法读取视频流信息（可能不是视频文件）: {}", input.display()));
    }
    let codec = parse_csv_field(f[0]);
    let width: u32 = f.get(1).map(|s| parse_csv_field(s).parse().unwrap_or(0)).unwrap_or(0);
    let height: u32 = f.get(2).map(|s| parse_csv_field(s).parse().unwrap_or(0)).unwrap_or(0);
    let mut bitrate: u64 = f.get(3).and_then(|s| parse_csv_field(s).parse().ok()).unwrap_or(0);

    let mut cmd = Command::new(&caps.ffprobe);
    let out = quiet_low_priority(&mut cmd)
        .args([
            "-v", "error",
            "-show_entries", "format=duration,size",
            "-of", "csv=p=0",
        ])
        .arg(input)
        .stdin(Stdio::null())
        .output()
        .context("运行 ffprobe 失败")?;
    let line = String::from_utf8_lossy(&out.stdout);
    let g: Vec<&str> = line.lines().next().unwrap_or_default().split(',').collect();
    let duration_s: f64 = g
        .first()
        .and_then(|s| parse_csv_field(s).parse().ok())
        .unwrap_or(0.0);
    let size: u64 = g
        .get(1)
        .and_then(|s| parse_csv_field(s).parse().ok())
        .unwrap_or(0);

    if bitrate == 0 && duration_s > 0.5 && size > 0 {
        // 兜底：总码率（含音频，偏差可忽略）
        bitrate = ((size as f64 * 8.0) / duration_s) as u64;
    }
    if duration_s <= 0.0 {
        return Err(anyhow!("无法读取视频时长: {}", input.display()));
    }
    Ok(ProbeInfo { codec, width, height, duration_s, video_bitrate: bitrate })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Encoder {
    NvencH264,
    NvencHevc,
    Libx264,
}

impl Encoder {
    fn name(&self) -> &'static str {
        match self {
            Encoder::NvencH264 => "h264_nvenc",
            Encoder::NvencHevc => "hevc_nvenc",
            Encoder::Libx264 => "libx264",
        }
    }
}

fn pick_encoder(caps: &Caps, codec: &str, prefer_hevc: bool) -> Encoder {
    // 用户选择 HEVC 时优先 hevc_nvenc（同画质体积更小；需要 GPU 编码器支持）
    if prefer_hevc && caps.nvenc_hevc {
        return Encoder::NvencHevc;
    }
    match codec {
        "h264" if caps.nvenc_h264 => Encoder::NvencH264,
        "hevc" | "h265" if caps.nvenc_hevc => Encoder::NvencHevc,
        _ => Encoder::Libx264,
    }
}

/// 目标封装格式：ts/avi 等老封装统一输出 mp4，其余保持原样。
fn out_format(rel: &Path) -> &'static str {
    match rel
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("mkv") => "mkv",
        Some("mov") => "mov",
        Some("webm") => "webm",
        _ => "mp4",
    }
}

/// 生成 ffmpeg 参数表。
/// 管线设计（实测验证过）：
///   NVDEC 硬解 → (若支持)GPU 翻转 → 数据回内存 → CPU 轻量翻转(兜底) → 轻度降噪 → NVENC 硬编
/// 解码+编码这两块重负载始终在 GPU 上；CPU 只做像素搬运与轻量降噪。
pub fn build_args(
    caps: &Caps,
    flip: FlipMode,
    denoise: DenoiseLevel,
    max_threads: u32,
    input: &Path,
    output: &Path,
    probe: &ProbeInfo,
    audio_to_aac: bool,
    size_pct: u32,
    prefer_hevc: bool,
) -> Vec<String> {
    let mut args: Vec<String> = [
        "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1", "-y",
    ].iter().map(|s| s.to_string()).collect();

    args.push("-threads".to_string());
    args.push(max_threads.to_string());

    let encoder = pick_encoder(caps, &probe.codec, prefer_hevc);
    let hw_decode = matches!(probe.codec.as_str(), "h264" | "hevc")
        && !matches!(encoder, Encoder::Libx264);
    // Vulkan 全 GPU 管线：解码/降噪/翻转全在显卡上（按编码器实测可用性门控）
    let use_vulkan = hw_decode
        && match probe.codec.as_str() {
            "h264" => caps.vulkan_h264,
            "hevc" | "h265" => caps.vulkan_hevc,
            _ => false,
        };

    // 输入选项必须位于 -i 之前
    if use_vulkan {
        args.push("-init_hw_device".to_string());
        args.push("vulkan=vk".to_string());
        args.push("-hwaccel".to_string());
        args.push("vulkan".to_string());
        args.push("-hwaccel_device".to_string());
        args.push("vk".to_string());
        args.push("-hwaccel_output_format".to_string());
        args.push("vulkan".to_string());
    } else if hw_decode {
        args.push("-hwaccel".to_string());
        args.push("cuda".to_string());
        args.push("-hwaccel_output_format".to_string());
        args.push("cuda".to_string());
    }
    args.push("-i".to_string());
    args.push(input.to_string_lossy().into_owned());

    let mut vf: Vec<String> = Vec::new();

    if use_vulkan {
        // Vulkan 降噪 + 翻转（帧始终在 GPU）
        if let Some(g) = denoise.vulkan_filter() {
            vf.push(g.to_string());
        }
        if let Some(f) = flip.vulkan_filter() {
            vf.push(f.to_string());
        } else {
            // 镜像模式：CPU 单轴翻转
            vf.push("hwdownload".into());
            vf.push("format=nv12".into());
            for part in flip.cpu_filter().split(',') {
                vf.push(part.to_string());
            }
        }
        // 数据回内存交给 NVENC（驱动 DMA 上传，CPU 开销极小）
        if !vf.iter().any(|s| s == "hwdownload") {
            vf.push("hwdownload".into());
            vf.push("format=nv12".into());
        }
    } else if hw_decode {
        // CUDA 路径
        if caps.cuda_flip {
            if let Some(f) = flip.cuda_filter() {
                vf.push(f.to_string());
            }
        }
        if vf.iter().any(|s| s.contains("transpose_cuda")) {
            if denoise.filter().is_some() {
                vf.push("hwdownload".into());
                vf.push("format=nv12".into());
            }
        } else {
            vf.push("hwdownload".into());
            vf.push("format=nv12".into());
            for part in flip.cpu_filter().split(',') {
                vf.push(part.to_string());
            }
        }
    } else {
        for part in flip.cpu_filter().split(',') {
            vf.push(part.to_string());
        }
    }
    // CPU 降噪兜底（Vulkan 降噪已在上面单独处理）
    if !use_vulkan {
        if let Some(d) = denoise.filter() {
            vf.push(d.to_string());
        }
    }
    if !vf.is_empty() {
        args.push("-vf".to_string());
        args.push(vf.join(","));
    }

    // 码率 = 源片码率 × 压缩比例，钳制到合理区间；降噪后的画面同码率下观感更好
    let mut pct = size_pct.clamp(10, 100) as f64 / 100.0;
    // HEVC 压缩效率更高：自动再降 30% 码率，同画质体积更小
    if prefer_hevc {
        pct *= 0.7;
    }
    let bv = ((probe.video_bitrate as f64 * pct) as u64).clamp(500_000, 50_000_000);
    let compressed = size_pct < 100;
    args.push("-c:v".to_string());
    args.push(encoder.name().to_string());
    match encoder {
        Encoder::Libx264 => {
            args.push("-preset".into());
            args.push("veryfast".into());
        }
        _ => {
            // 压缩模式下用更高效的 p5 预设 + 自适应量化（编码器有富余算力）
            args.push("-preset".into());
            args.push(if compressed { "p5" } else { "p4" }.into());
            args.push("-rc".into());
            args.push("vbr".into());
            if compressed {
                args.push("-spatial-aq".into());
                args.push("1".into());
                args.push("-temporal-aq".into());
                args.push("1".into());
            }
        }
    }
    args.push("-b:v".into());
    args.push(bv.to_string());
    args.push("-maxrate".into());
    args.push((bv * 2).to_string());
    args.push("-bufsize".into());
    args.push((bv * 2).to_string());

    // 只保留视频与音频流；音频默认直拷，封装不兼容时转 AAC
    args.push("-map".into());
    args.push("0:v:0".into());
    args.push("-map".into());
    args.push("0:a?".into());
    // HEVC 在 mp4 里使用 hvc1 标签，保证苹果系播放器兼容
    if matches!(encoder, Encoder::NvencHevc) && out_format(output) == "mp4" {
        args.push("-tag:v".into());
        args.push("hvc1".into());
    }
    if audio_to_aac {
        args.push("-c:a".into());
        args.push("aac".into());
        args.push("-b:a".into());
        args.push("160k".into());
    } else {
        args.push("-c:a".into());
        args.push("copy".into());
    }

    args.push("-f".into());
    args.push(out_format(output).into());
    args.push(output.to_string_lossy().into_owned());
    args
}

pub struct JobProgress {
    pub frac: f32,
    pub speed: f32,
}

pub enum JobEvent {
    Progress(JobProgress),
    /// 转码中周期输出（供 UI 刷新日志）
    Heartbeat(String),
}

/// 执行一个文件：返回 Ok(()) 表示成功产出输出文件。
pub fn run(
    caps: &Caps,
    flip: FlipMode,
    denoise: DenoiseLevel,
    max_threads: u32,
    input: &Path,
    output_part: &Path,
    tx: &Sender<JobEvent>,
    cancel: &Arc<std::sync::atomic::AtomicBool>,
    size_pct: u32,
    prefer_hevc: bool,
) -> Result<()> {
    let info = probe(caps, input)?;
    let run_once = |audio_to_aac: bool| -> Result<(i32, String)> {
        let args = build_args(
            caps, flip, denoise, max_threads, input, output_part, &info, audio_to_aac,
            size_pct, prefer_hevc,
        );
        let mut cmd = Command::new(&caps.ffmpeg);
        cmd.args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null());
        quiet_low_priority(&mut cmd);

        let mut child = cmd.spawn().context("启动 ffmpeg 失败")?;
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        // GPU 调速器登记：批处理时限制 GPU 占用上限（Drop 时注销并确保恢复）
        let mut child_opt: Option<Child> = Some(child);
        let _gpu_guard = crate::throttle::Guard::register(child_opt.as_ref().unwrap());

        let stderr_tail: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        {
            let tail = stderr_tail.clone();
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    let mut t = tail.lock().unwrap();
                    if t.len() > 4000 {
                        t.clear();
                    }
                    t.push_str(&line);
                    t.push('\n');
                }
            });
        }

        // 进度读取（stdout 的 -progress 机器可读输出）
        let mut last_event = Instant::now() - Duration::from_secs(1);
        let mut last_output;
        let mut status_code: Option<i32> = None;

        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        loop {
            // 检查取消：非阻塞轮询子进程退出
            line.clear();
            let n = reader.read_line(&mut line).unwrap_or(0);
            if n == 0 {
                // EOF：进程应已退出
                if let Some(c) = child_opt.as_mut() {
                    status_code = c.wait().ok().and_then(|s| s.code());
                }
                break;
            }
            last_output = Instant::now();

            if let Some(kv) = line.trim().strip_prefix("out_time_us=") {
                if let Ok(us) = kv.trim().parse::<f64>() {
                    let frac = (us / 1e6 / info.duration_s).clamp(0.0, 1.0) as f32;
                    let now = Instant::now();
                    if now.duration_since(last_event) >= Duration::from_millis(400)
                        && tx.send(JobEvent::Progress(JobProgress { frac, speed: 0.0 })).is_err()
                    {
                        // 接收端已关闭
                    }
                    last_event = now;
                }
            }
            if let Some(kv) = line.trim().strip_prefix("speed=") {
                let sp = kv.trim().trim_end_matches('x').trim();
                if let Ok(v) = sp.parse::<f32>() {
                    // 记录在最近的 Progress 后补充；直接发送
                    let _ = tx.send(JobEvent::Progress(JobProgress { frac: -1.0, speed: v }));
                }
            }

            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(c) = child_opt.as_mut() {
                    let _ = c.kill();
                    let _ = c.wait();
                }
                return Err(anyhow!("已取消"));
            }
            // 看门狗：2 分钟无任何输出视为卡死
            if last_output.elapsed() > Duration::from_secs(120) {
                if let Some(c) = child_opt.as_mut() {
                    let _ = c.kill();
                    let _ = c.wait();
                }
                return Err(anyhow!("ffmpeg 长时间无输出，已终止（文件可能损坏）"));
            }
        }

        let code = status_code.unwrap_or(-1);
        let tail = stderr_tail.lock().unwrap().clone();
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(anyhow!("已取消"));
        }
        if code != 0 {
            return Err(anyhow!("ffmpeg 退出码 {}：{}", code, tail_tail(&tail)));
        }
        Ok((code, tail))
    };

    match run_once(false) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            // 仅当音频封装不兼容（如 mp4 不支持 PCM 等）时改用 AAC 重试一次
            let audio_issue = msg.contains("Could not find tag for codec")
                || msg.contains("codec not currently supported in container");
            if audio_issue {
                tx.send(JobEvent::Heartbeat("音频直拷失败，改用 AAC 重试...".into())).ok();
                run_once(true)?;
            } else {
                return Err(e);
            }
        }
    }

    // 成功：确认输出文件存在
    if !output_part.is_file() {
        return Err(anyhow!("转码结束但未找到输出文件"));
    }
    Ok(())
}

fn tail_tail(s: &str) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(3);
    lines[start..].join(" | ")
}
