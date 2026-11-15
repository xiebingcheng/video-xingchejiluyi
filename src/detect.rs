//! 定位 ffmpeg / ffprobe 并探测硬件加速能力。

use anyhow::{anyhow, Result};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const LOW_PRIORITY_FLAGS: u32 = BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW;

/// 给子进程设置：低优先级（让出 CPU 给前台软件）+ 不弹出控制台窗口。
pub fn quiet_low_priority(c: &mut Command) -> &mut Command {
    c.creation_flags(LOW_PRIORITY_FLAGS)
}

#[derive(Debug, Clone)]
pub struct Caps {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    /// ffmpeg 构建是否包含 transpose_cuda（GPU 翻转）
    pub cuda_flip: bool,
    /// Vulkan 全 GPU 管线实测可用（解码 + GPU 降噪 + GPU 翻转），按编码器分
    pub vulkan_h264: bool,
    pub vulkan_hevc: bool,
    pub nvenc_h264: bool,
    pub nvenc_hevc: bool,
    /// nvidia-smi 可用（用于动态并发采样 GPU 占用）
    pub nvidia_smi: bool,
}

impl Caps {
    /// 探测顺序：exe 同目录 bin\ → exe 同目录 → 工作目录 tools\（开发布局）→ PATH。
    /// 发布时把 ffmpeg.exe 放在软件旁即可复用。
    pub fn locate() -> Result<Self> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(d) = exe.parent() {
                candidates.push(d.join("bin").join("ffmpeg.exe"));
                candidates.push(d.join("ffmpeg.exe"));
                candidates.push(d.join("tools").join("ffmpeg.exe"));
                // 开发布局：target/release → 项目根 tools/
                if let Some(root) = d.parent() {
                    candidates.push(root.join("tools").join("ffmpeg.exe"));
                    if let Some(root2) = root.parent() {
                        candidates.push(root2.join("tools").join("ffmpeg.exe"));
                    }
                }
            }
        }
        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join("tools").join("ffmpeg.exe"));
        }
        if let Ok(path) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path) {
                candidates.push(dir.join("ffmpeg.exe"));
            }
        }

        for cand in candidates {
            if !cand.is_file() {
                continue;
            }
            let ffprobe = cand.with_file_name("ffprobe.exe");
            if !ffprobe.is_file() {
                continue;
            }
            let ok = Command::new(&cand)
                .arg("-version")
                .creation_flags(CREATE_NO_WINDOW)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if !ok {
                continue;
            }
            let nvenc_h264 = has_encoder(&cand, "h264_nvenc");
            let nvenc_hevc = has_encoder(&cand, "hevc_nvenc");
            // Vulkan 全 GPU 管线可达性：滤镜存在 + 用合成视频实测解码/降噪/翻转链路
            let has_vk_filters =
                has_filter(&cand, "nlmeans_vulkan") && has_filter(&cand, "transpose_vulkan");
            let (vulkan_h264, vulkan_hevc) = if has_vk_filters {
                (
                    probe_vulkan_pipeline(&cand, false),
                    probe_vulkan_pipeline(&cand, true),
                )
            } else {
                (false, false)
            };
            return Ok(Self {
                cuda_flip: has_filter(&cand, "transpose_cuda"),
                vulkan_h264,
                vulkan_hevc,
                nvenc_h264,
                nvenc_hevc,
                nvidia_smi: nvidia_smi_available(),
                ffmpeg: cand,
                ffprobe,
            });
        }
        Err(anyhow!(
            "未找到 ffmpeg.exe。请将 ffmpeg.exe 与 ffprobe.exe 放到本程序同目录的 bin 文件夹（或同目录），并确保构建支持 NVENC。"
        ))
    }

    pub fn hardware_desc(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.nvenc_h264 || self.nvenc_hevc {
            parts.push("NVENC 硬件编码".into());
        } else {
            parts.push("软件编码（未检测到 NVENC）".into());
        }
        if self.vulkan_h264 || self.vulkan_hevc {
            parts.push("Vulkan 解码+降噪+翻转全 GPU（CPU 近零占用）".into());
        } else if self.cuda_flip {
            parts.push("GPU 翻转 + CPU 降噪".into());
        } else {
            parts.push("CPU 轻量翻转/降噪".into());
        }
        if self.nvidia_smi {
            parts.push("GPU 自动降载（防持续满载）".into());
        }
        parts.join(" + ")
    }
}

fn run_capture(exe: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new(exe)
        .args(args)
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn has_filter(ffmpeg: &Path, name: &str) -> bool {
    run_capture(ffmpeg, &["-hide_banner", "-filters"]).map_or(false, |s| {
        s.lines().any(|l| l.split_whitespace().any(|w| w == name))
    })
}

fn has_encoder(ffmpeg: &Path, name: &str) -> bool {
    run_capture(ffmpeg, &["-hide_banner", "-encoders"]).map_or(false, |s| {
        s.lines().any(|l| l.split_whitespace().any(|w| w == name))
    })
}

fn nvidia_smi_available() -> bool {
    Command::new("nvidia-smi")
        .arg("-L")
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// 实测 Vulkan 全 GPU 链路："Vulkan 解码 → nlmeans_vulkan 降噪 → transpose_vulkan 翻转 → 回内存"。
/// 避免按版本号猜：不同构建/驱动对 Vulkan Video 解码的支持差异很大。
fn probe_vulkan_pipeline(ffmpeg: &Path, hevc: bool) -> bool {
    // 1) 生成一个小测试视频（320x180，0.2 秒）
    let tmp = std::env::temp_dir().join(if hevc {
        "vf_probe_hevc.mp4"
    } else {
        "vf_probe_h264.mp4"
    });
    let encoder = if hevc { "libx265" } else { "libx264" };
    let gen_out = Command::new(ffmpeg)
        .args([
            "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=320x180:d=0.2",
            "-c:v", encoder, "-qp", "45",
        ])
        .arg(&tmp)
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .output();
    if gen_out.map(|o| !o.status.success()).unwrap_or(true) {
        return false;
    }
    // 2) Vulkan 解码 + GPU 降噪 + GPU 翻转 + 回内存
    let out = Command::new(ffmpeg)
        .args([
            "-v", "error", "-y",
            "-init_hw_device", "vulkan=vk",
            "-hwaccel", "vulkan",
            "-hwaccel_device", "vk",
            "-hwaccel_output_format", "vulkan",
            "-i",
        ])
        .arg(&tmp)
        .args([
            "-vf",
            "nlmeans_vulkan=s=1.5:p=3:r=5,transpose_vulkan=dir=1,transpose_vulkan=dir=1,hwdownload,format=nv12",
            "-frames:v", "3",
            "-f", "null", "-",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let _ = std::fs::remove_file(&tmp);
    out
}

impl Caps {
    /// 查询 NVIDIA GPU 的图形占用率（%）。不可用时返回 None。
    /// 注意：NVENC 编码占用与 3D 图形占用是独立引擎；用户在游戏等高负载时
    /// 图形占用高，此时应降低并发把资源让给前台。
    pub fn gpu_graphics_util(&self) -> Option<f32> {
        if !self.nvidia_smi {
            return None;
        }
        let out = Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(std::process::Stdio::null())
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        s.lines()
            .next()
            .and_then(|l| l.trim().parse::<f32>().ok())
    }
}
