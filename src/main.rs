//! GUI 入口（Windows 下不弹控制台窗口）。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use video_fixer::config::Config;
use video_fixer::detect::Caps;
use video_fixer::ui::App;

fn main() -> anyhow::Result<()> {
    let caps = Caps::locate();
    let mut cfg = Config::load();

    // 首次运行：预填默认输出目录
    if !cfg.input_dir.is_empty() && cfg.output_dir.is_empty() {
        cfg.output_dir = std::path::Path::new(&cfg.input_dir)
            .join("调正输出")
            .to_string_lossy()
            .into_owned();
    }

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("行车记录仪视频调正工具")
            .with_inner_size([1100.0, 720.0])
            .with_min_inner_size([480.0, 380.0]),
        ..Default::default()
    };

    match eframe::run_native(
        "video-fixer",
        native_options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, caps)))),
    ) {
        Ok(()) => Ok(()),
        Err(e) => Err(anyhow::anyhow!("GUI 启动失败: {e}")),
    }
}
