//! 共享数据模型：转向模式、降噪档位、文件条目。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 画面转向模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FlipMode {
    /// 180° 上下颠倒（默认：镜头翻转拍摄）
    #[default]
    Rotate180,
    /// 顺时针 90°
    Rotate90cw,
    /// 逆时针 90°
    Rotate90ccw,
    /// 水平镜像
    MirrorH,
    /// 垂直镜像
    MirrorV,
}

impl FlipMode {
    pub fn label(&self) -> &'static str {
        match self {
            FlipMode::Rotate180 => "180° 上下颠倒",
            FlipMode::Rotate90cw => "90° 顺时针",
            FlipMode::Rotate90ccw => "90° 逆时针",
            FlipMode::MirrorH => "水平镜像",
            FlipMode::MirrorV => "垂直镜像",
        }
    }

    /// CPU 滤镜链（对系统内存帧操作）。
    pub fn cpu_filter(&self) -> &'static str {
        match self {
            FlipMode::Rotate180 => "transpose=1,transpose=1",
            FlipMode::Rotate90cw => "transpose=1",
            FlipMode::Rotate90ccw => "transpose=2",
            FlipMode::MirrorH => "hflip",
            FlipMode::MirrorV => "vflip",
        }
    }

    /// CUDA 滤镜链（对 GPU 帧操作）。镜像模式无 CUDA 滤镜，回退 CPU。
    pub fn cuda_filter(&self) -> Option<&'static str> {
        match self {
            FlipMode::Rotate180 => Some("transpose_cuda=dir=1,transpose_cuda=dir=1"),
            FlipMode::Rotate90cw => Some("transpose_cuda=dir=1"),
            FlipMode::Rotate90ccw => Some("transpose_cuda=dir=2"),
            FlipMode::MirrorH | FlipMode::MirrorV => None,
        }
    }

    /// Vulkan 滤镜链（对 GPU 帧操作）。镜像模式回退 CPU（flip_vulkan 无法单轴翻转）。
    pub fn vulkan_filter(&self) -> Option<&'static str> {
        match self {
            FlipMode::Rotate180 => Some("transpose_vulkan=dir=1,transpose_vulkan=dir=1"),
            FlipMode::Rotate90cw => Some("transpose_vulkan=dir=1"),
            FlipMode::Rotate90ccw => Some("transpose_vulkan=dir=2"),
            FlipMode::MirrorH | FlipMode::MirrorV => None,
        }
    }
}

/// 降噪档位（hqdn3d 参数：亮度空域:色度空域:亮度时域:色度时域）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DenoiseLevel {
    Off,
    #[default]
    Light,
    Medium,
}

impl DenoiseLevel {
    pub fn label(&self) -> &'static str {
        match self {
            DenoiseLevel::Off => "关闭",
            DenoiseLevel::Light => "轻度（保细节）",
            DenoiseLevel::Medium => "中度",
        }
    }

    pub fn filter(&self) -> Option<&'static str> {
        match self {
            // 轻度：只压噪点，不动纹理与边缘（实测夜间路牌/仪表清晰可辨）
            DenoiseLevel::Light => Some("hqdn3d=1.5:1.5:3:3"),
            // 中度：hqdn3d 默认强度
            DenoiseLevel::Medium => Some("hqdn3d=4:3:6:4.5"),
            DenoiseLevel::Off => None,
        }
    }

    /// GPU 降噪滤镜参数（Vulkan 非局部均值，边缘保持不糊车牌）。
    /// 轻参数 p=3 r=7 实测 2K 下约 1x 实时、画质与 hqdn3d 轻度相当。
    pub fn vulkan_filter(&self) -> Option<&'static str> {
        match self {
            DenoiseLevel::Light => Some("nlmeans_vulkan=s=1.5:p=3:r=5"),
            DenoiseLevel::Medium => Some("nlmeans_vulkan=s=4:p=5:r=7"),
            DenoiseLevel::Off => None,
        }
    }
}

/// 单个文件的处理状态。
#[derive(Debug, Clone, PartialEq)]
pub enum FileStatus {
    Pending,
    Processing,
    Done,
    Failed(String),
    Skipped,
}

impl FileStatus {
    pub fn label(&self) -> String {
        match self {
            FileStatus::Pending => "待处理".into(),
            FileStatus::Processing => "处理中".into(),
            FileStatus::Done => "✓ 完成".into(),
            FileStatus::Failed(e) => format!("✗ 失败: {e}"),
            FileStatus::Skipped => "已存在，跳过".into(),
        }
    }
}

/// 扫描到的视频文件。
#[derive(Debug, Clone)]
pub struct FileEntry {
    pub path: PathBuf,
    /// 相对输入根目录的路径，输出时保持同样结构。
    pub rel: PathBuf,
    pub size: u64,
    pub selected: bool,
    pub status: FileStatus,
    /// 0.0~1.0 处理进度。
    pub progress: f32,
    pub speed: f32,
}

impl FileEntry {
    pub fn new(path: PathBuf, rel: PathBuf, size: u64) -> Self {
        Self {
            path,
            rel,
            size,
            selected: true,
            status: FileStatus::Pending,
            progress: 0.0,
            speed: 0.0,
        }
    }
}

pub const VIDEO_EXTS: &[&str] = &[
    "mp4", "mov", "mkv", "avi", "ts", "m2ts", "webm", "wmv", "mpg", "mpeg", "flv",
];
