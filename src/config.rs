//! 配置持久化（%APPDATA%\video-fixer\config.toml）与缓存目录。

use crate::model::{DenoiseLevel, FlipMode};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub input_dir: String,
    pub output_dir: String,
    pub flip: FlipMode,
    pub denoise: DenoiseLevel,
    pub include_subdirs: bool,
    pub skip_existing: bool,
    /// 限制 ffmpeg 使用的 CPU 线程数（越小越省 CPU，但转码越慢）。
    pub max_threads: u32,
    /// 同时处理文件数：0 = 自动（按 GPU 占用动态 1~3 路），1~4 = 固定并发。
    pub jobs: u32,
    /// 输出体积百分比：100 = 与原片相当，70/50/30 = 压缩到原片码率的比例。
    pub compress_pct: u32,
    /// 用 H.265/HEVC 编码（同画质体积再省 30% 左右，兼容性略窄）。
    pub hevc: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_dir: String::new(),
            output_dir: String::new(),
            flip: FlipMode::Rotate180,
            denoise: DenoiseLevel::Light,
            include_subdirs: true,
            skip_existing: true,
            max_threads: 4,
            jobs: 0,
            compress_pct: 100,
            hevc: false,
        }
    }
}

fn base_dir() -> PathBuf {
    let root = std::env::var("APPDATA").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into());
    PathBuf::from(root).join("video-fixer")
}

pub fn thumbs_dir() -> PathBuf {
    let root = std::env::var("LOCALAPPDATA")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into());
    PathBuf::from(root).join("video-fixer").join("thumbs")
}

impl Config {
    fn path() -> PathBuf {
        base_dir().join("config.toml")
    }

    pub fn load() -> Self {
        match std::fs::read_to_string(Self::path()) {
            Ok(s) => toml::from_str(&s).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> Result<()> {
        let dir = base_dir();
        std::fs::create_dir_all(&dir).context("创建配置目录失败")?;
        let s = toml::to_string_pretty(self).context("序列化配置失败")?;
        std::fs::write(Self::path(), s).context("写入配置失败")?;
        Ok(())
    }
}

/// 输出盘剩余可用空间（字节）。路径不存在时返回其最近的现存祖先所在盘。
pub fn free_space(path: &std::path::Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    let mut probe = path.to_path_buf();
    loop {
        if probe.as_os_str().is_empty() {
            return None;
        }
        if probe.exists() {
            break;
        }
        probe = probe.parent()?.to_path_buf();
    }
    let wide: Vec<u16> = probe
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    unsafe {
        let mut free_to_caller: u64 = 0;
        let ok = windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
            wide.as_ptr(),
            null_mut_u64(),
            null_mut_u64(),
            &mut free_to_caller,
        );
        if ok != 0 { Some(free_to_caller) } else { None }
    }
}

fn null_mut_u64() -> *mut u64 {
    std::ptr::null_mut()
}
