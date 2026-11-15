//! egui 图形界面：中文、低刷新、缩略图预览、批量进度。

use eframe::egui;

use crate::config::{self, Config};
use crate::detect::Caps;
use crate::model::{DenoiseLevel, FileEntry, FileStatus, FlipMode};
use crate::queue::{self, BatchItem, BatchSpec, Ev, ThumbRequest};
use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

struct Row {
    entry: FileEntry,
    thumb: Option<egui::TextureHandle>,
    thumb_done: bool,
}

pub struct App {
    /// egui 上下文：供后台线程唤醒界面刷新
    ctx: egui::Context,
    caps: Option<Caps>,
    caps_err: Option<String>,
    cfg: Config,
    rows: Vec<Row>,
    tx: Sender<Ev>,
    rx: Receiver<Ev>,
    thumb_tx: Option<Sender<ThumbRequest>>,
    thumb_outstanding: usize,
    scanning: bool,
    processing: bool,
    cancel: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
    batch_total: usize,
    batch_done: usize,
    batch_failed: usize,
    batch_skipped: usize,
    /// 当前正在处理文件的进度（0~1），用于总进度显示
    cancelled_flag: bool,
    logs: Vec<String>,
    dirty_cfg: bool,
    space_warn: Option<String>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, caps: Result<Caps, anyhow::Error>) -> Self {
        install_cjk_font(&cc.egui_ctx);
        let cfg = Config::load();
        let (caps_ok, caps_err) = match caps {
            Ok(c) => (Some(c), None),
            Err(e) => (None, Some(e.to_string())),
        };
        let (tx, rx) = std::sync::mpsc::channel::<Ev>();
        let mut app = Self {
            ctx: cc.egui_ctx.clone(),
            caps: caps_ok,
            caps_err,
            cfg,
            rows: Vec::new(),
            tx,
            rx,
            thumb_tx: None,
            thumb_outstanding: 0,
            scanning: false,
            processing: false,
            cancel: Arc::new(AtomicBool::new(false)),
            worker: None,
            batch_total: 0,
            batch_done: 0,
            batch_failed: 0,
            batch_skipped: 0,
            cancelled_flag: false,
            logs: Vec::new(),
            dirty_cfg: false,
            space_warn: None,
        };
        app.spawn_thumb_worker();
        // 上次目录自动扫描
        if !app.cfg.input_dir.is_empty() {
            app.rescan();
        }
        app
    }

    fn spawn_thumb_worker(&mut self) {
        if let Some(caps) = self.caps.clone() {
            let (ttx, trx) = std::sync::mpsc::channel::<ThumbRequest>();
            self.thumb_tx = Some(ttx);
            let caps = Arc::new(caps);
            let tx = queue::Notifier::new(self.tx.clone(), Some(self.ctx.clone()));
            let cancel = self.cancel.clone();
            queue::spawn_thumb_worker(caps, trx, tx, cancel);
        }
    }

    fn rescan(&mut self) {
        if self.scanning || self.processing {
            return;
        }
        self.rows.clear();
        self.thumb_outstanding = 0;
        let input = PathBuf::from(&self.cfg.input_dir);
        if !input.is_dir() {
            self.logs.push(format!("输入目录不存在: {}", self.cfg.input_dir));
            return;
        }
        self.scanning = true;
        self.batch_done = 0;
        self.batch_failed = 0;
        self.batch_skipped = 0;
        let out_root = PathBuf::from(&self.cfg.output_dir);
        let tx = queue::Notifier::new(self.tx.clone(), Some(self.ctx.clone()));
        queue::spawn_scan(&input, &out_root, self.cfg.include_subdirs, tx);
    }

    fn request_thumb(&mut self, entry: &FileEntry) {
        if self.thumb_tx.is_none() {
            return;
        }
        let mtime = std::fs::metadata(&entry.path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let req = ThumbRequest {
            path: entry.path.clone(),
            size: entry.size,
            mtime,
            flip: self.cfg.flip,
        };
        self.thumb_outstanding += 1;
        let _ = self.thumb_tx.as_ref().unwrap().send(req);
    }

    fn start_batch(&mut self) {
        if self.processing || self.caps.is_none() {
            return;
        }
        let _ = self.cfg.save();
        let items: Vec<BatchItem> = self
            .rows
            .iter()
            .filter(|r| r.entry.selected)
            .map(|r| BatchItem {
                input: r.entry.path.clone(),
                rel: r.entry.rel.clone(),
            })
            .collect();
        if items.is_empty() {
            return;
        }
        // 重置选中项状态
        for row in &mut self.rows {
            if row.entry.selected {
                row.entry.status = FileStatus::Pending;
                row.entry.progress = 0.0;
            }
        }
        // 磁盘空间预警
        let need: u64 = self
            .rows
            .iter()
            .filter(|r| r.entry.selected)
            .map(|r| r.entry.size)
            .sum();
        self.space_warn = None;
        if let Some(free) = config::free_space(&PathBuf::from(&self.cfg.output_dir)) {
            if free < need {
                self.space_warn = Some(format!(
                    "警告：输出盘剩余空间 {:.1} GB，小于预计所需 {:.1} GB！",
                    free as f64 / 1e9,
                    need as f64 / 1e9
                ));
            }
        }
        self.batch_total = items.len();
        self.batch_done = 0;
        self.batch_failed = 0;
        self.batch_skipped = 0;
        self.cancelled_flag = false;
        self.cancel.store(false, Ordering::Relaxed);
        self.processing = true;
        let caps = Arc::new(self.caps.clone().unwrap());
        let spec = BatchSpec {
            items,
            output_root: PathBuf::from(&self.cfg.output_dir),
            flip: self.cfg.flip,
            denoise: self.cfg.denoise,
            skip_existing: self.cfg.skip_existing,
            max_threads: self.cfg.max_threads,
            jobs: self.cfg.jobs.max(1) as usize,
            auto_jobs: self.cfg.jobs == 0,
            size_pct: self.cfg.compress_pct,
            prefer_hevc: self.cfg.hevc,
        };
        self.worker = Some(queue::spawn_batch(
            caps,
            spec,
            queue::Notifier::new(self.tx.clone(), Some(self.ctx.clone())),
            self.cancel.clone(),
        ));
    }

    fn row_by_path_mut(&mut self, path: &std::path::Path) -> Option<&mut Row> {
        self.rows.iter_mut().find(|r| r.entry.path == path)
    }

    fn handle_event(&mut self, ctx: &egui::Context, ev: Ev) {
        match ev {
            Ev::Scanned(e) => {
                self.request_thumb(&e);
                self.rows.push(Row { entry: e, thumb: None, thumb_done: false });
            }
            Ev::ScanDone => {
                self.scanning = false;
                self.rows.sort_by(|a, b| a.entry.rel.cmp(&b.entry.rel));
            }
            Ev::Thumb { path, jpeg } => {
                self.thumb_outstanding = self.thumb_outstanding.saturating_sub(1);
                if let Some(row) = self.row_by_path_mut(&path) {
                    row.thumb_done = true;
                    if !jpeg.is_empty() {
                        if let Ok(img) = image::load_from_memory(&jpeg) {
                            let rgb = img.to_rgb8();
                            let size = [rgb.width() as usize, rgb.height() as usize];
                            let pixels: Vec<egui::Color32> = rgb
                                .pixels()
                                .map(|p| egui::Color32::from_rgb(p[0], p[1], p[2]))
                                .collect();
                            let tex = ctx.load_texture(
                                format!("thumb-{}", row.entry.rel.display()),
                                egui::ColorImage::new(size, pixels),
                                egui::TextureOptions::LINEAR,
                            );
                            row.thumb = Some(tex);
                        }
                    }
                }
            }
            Ev::Progress { path, frac, speed } => {
                if let Some(row) = self.row_by_path_mut(&path) {
                    if frac >= 0.0 {
                        row.entry.progress = frac;
                    }
                    if speed > 0.0 {
                        row.entry.speed = speed;
                    }
                }
                if frac >= 0.0 {
                }
            }
            Ev::Status { path, status } => {
                let is_done = matches!(status, FileStatus::Done);
                let is_skipped = matches!(status, FileStatus::Skipped);
                let is_failed = matches!(status, FileStatus::Failed(_));
                if let Some(row) = self.row_by_path_mut(&path) {
                    row.entry.status = status;
                }
                if is_done {
                    self.batch_done += 1;
                }
                if is_skipped {
                    self.batch_skipped += 1;
                }
                if is_failed {
                    self.batch_failed += 1;
                }
            }
            Ev::Log(msg) => {
                self.logs.push(msg);
                if self.logs.len() > 300 {
                    self.logs.drain(..100);
                }
            }
            Ev::BatchDone { done, failed, skipped, cancelled } => {
                self.processing = false;
                self.cancelled_flag = cancelled;
                self.batch_done = done;
                self.batch_failed = failed;
                self.batch_skipped = skipped;
                self.logs.push(format!(
                    "批次结束：完成 {done}，跳过 {skipped}，失败 {failed}{}",
                    if cancelled { "（已取消）" } else { "" }
                ));
                if let Some(h) = self.worker.take() {
                    let _ = h.join();
                }
            }
        }
    }

}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 处理后台事件（此阶段不允许绘制 UI）
        let mut got = false;
        while let Ok(ev) = self.rx.try_recv() {
            got = true;
            self.handle_event(ctx, ev);
        }
        // 需要后续刷新的情形
        if self.processing || self.scanning || self.thumb_outstanding > 0 || got {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }

        if self.dirty_cfg {
            self.dirty_cfg = false;
            let _ = self.cfg.save();
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        // 顺序很重要：顶部/底部面板先于中央面板创建，中央面板填充剩余空间
        self.draw_top(ui, frame);
        self.draw_bottom(ui);
        self.draw_center(ui);
    }
}

// ---------- 绘制 ----------

impl App {
    fn draw_top(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(6.0);
            // 窗口变窄时允许换行，避免内容被裁掉
            ui.horizontal_wrapped(|ui| {
                ui.heading("行车记录仪视频调正工具");
                ui.separator();
                match &self.caps {
                    Some(caps) => {
                        ui.label(
                            egui::RichText::new(format!("硬件加速：{}", caps.hardware_desc()))
                                .weak(),
                        );
                    }
                    None => {
                        ui.label(egui::RichText::new("⚠ 未找到 ffmpeg")
                            .color(egui::Color32::from_rgb(230, 90, 90)));
                    }
                }
            });
            if let Some(err) = &self.caps_err {
                ui.colored_label(egui::Color32::from_rgb(230, 90, 90), err);
            }
            if let Some(warn) = &self.space_warn {
                ui.colored_label(egui::Color32::from_rgb(240, 180, 60), warn);
            }
            ui.add_space(4.0);

            // 输入目录
            ui.horizontal_wrapped(|ui| {
                ui.label("输入目录：");
                // 预留标签+三个按钮的宽度，保证按钮与本行文本框同行；
                // 窗口不够宽时文本框折到下一行（horizontal_wrapped 兜底）
                let w = (ui.available_width() - 330.0).max(140.0);
                let resp = ui.add_sized(
                    [w, 20.0],
                    egui::TextEdit::singleline(&mut self.cfg.input_dir)
                        .hint_text("选择或粘贴文件夹路径"),
                );
                if resp.changed() {
                    self.dirty_cfg = true;
                }
                if ui
                    .button("浏览…")
                    .on_hover_text("在资源管理器中打开输入目录，可直接查看、整理其中的视频")
                    .clicked()
                {
                    let dir = self.cfg.input_dir.clone();
                    self.open_in_explorer(&dir, "输入目录");
                }
                if ui
                    .button("选择文件夹…")
                    .on_hover_text("弹出对话框选择输入目录")
                    .clicked()
                {
                    self.pick_input_dir(frame);
                }
                let scan_enabled = !self.scanning && !self.processing;
                if ui.add_enabled(scan_enabled, egui::Button::new("重新扫描")).clicked() {
                    self.rescan();
                }
            });
            // 输出目录
            ui.horizontal_wrapped(|ui| {
                ui.label("输出目录：");
                // 预留标签+浏览+选择按钮的宽度，保证按钮与本行文本框同行
                let w = (ui.available_width() - 240.0).max(140.0);
                let resp = ui.add_sized(
                    [w, 20.0],
                    egui::TextEdit::singleline(&mut self.cfg.output_dir)
                        .hint_text("调正后的视频保存到此处（原文件保留）"),
                );
                if resp.changed() {
                    self.dirty_cfg = true;
                }
                if ui
                    .button("浏览…")
                    .on_hover_text("在资源管理器中打开输出目录，可直接查看、整理生成的视频")
                    .clicked()
                {
                    let dir = self.cfg.output_dir.clone();
                    self.open_in_explorer(&dir, "输出目录");
                }
                if ui
                    .button("选择文件夹…")
                    .on_hover_text("弹出对话框选择输出目录")
                    .clicked()
                {
                    self.pick_output_dir(frame);
                }
            });
            ui.add_space(2.0);
            // 参数：窗口不够宽时自动折行
            ui.horizontal_wrapped(|ui| {
                egui::ComboBox::from_label("转向")
                    .selected_text(self.cfg.flip.label())
                    .show_ui(ui, |ui| {
                        for f in [
                            FlipMode::Rotate180,
                            FlipMode::Rotate90cw,
                            FlipMode::Rotate90ccw,
                            FlipMode::MirrorH,
                            FlipMode::MirrorV,
                        ] {
                            if ui
                                .selectable_value(&mut self.cfg.flip, f, f.label())
                                .changed()
                            {
                                self.on_flip_changed();
                            }
                        }
                    });
                egui::ComboBox::from_label("降噪")
                    .selected_text(self.cfg.denoise.label())
                    .show_ui(ui, |ui| {
                        for d in [DenoiseLevel::Off, DenoiseLevel::Light, DenoiseLevel::Medium] {
                            ui.selectable_value(&mut self.cfg.denoise, d, d.label());
                        }
                    });
                egui::ComboBox::from_label("输出体积")
                    .selected_text(size_label(self.cfg.compress_pct))
                    .show_ui(ui, |ui| {
                        for p in [100u32, 70, 50, 30] {
                            ui.selectable_value(&mut self.cfg.compress_pct, p, size_label(p));
                        }
                    })
                    .response
                    .on_hover_text("压缩比例（相对原片码率）。降噪后的画面压缩空间更大；白天高动态画面建议不低于 50%");
                egui::ComboBox::from_label("编码格式")
                    .selected_text(if self.cfg.hevc { "H.265/HEVC（更小）" } else { "H.264（兼容最好）" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.cfg.hevc, false, "H.264（兼容最好）");
                        ui.selectable_value(&mut self.cfg.hevc, true, "H.265/HEVC（更小）");
                    })
                    .response
                    .on_hover_text("HEVC 同画质体积再省 30% 左右；现代播放器/手机/电视基本都支持，老设备可能不认");
                egui::ComboBox::from_label("同时处理")
                    .selected_text(jobs_label(self.cfg.jobs))
                    .show_ui(ui, |ui| {
                        for j in [0u32, 1, 2, 3, 4] {
                            ui.selectable_value(&mut self.cfg.jobs, j, jobs_label(j));
                        }
                    })
                    .response
                    .on_hover_text("自动：GPU 空闲时开多路并行吃满显卡，GPU 忙时自动降回 1 路");
                ui.checkbox(&mut self.cfg.skip_existing, "跳过已完成");
                ui.checkbox(&mut self.cfg.include_subdirs, "含子目录");
                ui.separator();
                ui.label("CPU 总线程预算：").on_hover_text("所有并行任务共享此预算（自动按并发数分摊）。全 GPU 管线下 CPU 本就接近空闲。");
                ui.add(egui::Slider::new(&mut self.cfg.max_threads, 1..=8).text("个"));
            });
            ui.add_space(4.0);
        });
    }

    /// 在资源管理器中打开文件所在文件夹并选中（高亮）该文件。
    fn reveal_in_explorer(&mut self, path: &std::path::Path) {
        if !path.is_file() {
            self.logs.push(format!("文件不存在，无法定位: {}", path.display()));
            return;
        }
        // explorer 的 /select 参数不接受正斜杠，统一转成反斜杠
        let win = path.to_string_lossy().replace('/', "\\");
        // raw_arg 保证 `/select,"路径"` 原样进入命令行：常规 arg 在路径含空格时会把
        // 整串再包一层引号，explorer 自己的参数解析处理不了；这里的引号护住路径本身
        use std::os::windows::process::CommandExt;
        if let Err(e) = std::process::Command::new("explorer")
            .raw_arg(format!("/select,\"{win}\""))
            .spawn()
        {
            self.logs.push(format!("定位文件失败: {e}"));
        }
    }

    /// 在资源管理器中打开文件夹，供用户直接查看、整理其中的文件。
    /// 目录不存在时不自动创建，只记一条日志（打开一个错误的空窗口更容易让人困惑）。
    fn open_in_explorer(&mut self, dir: &str, what: &str) {
        if dir.trim().is_empty() {
            self.logs.push(format!("尚未设置{what}，无法打开"));
            return;
        }
        if !PathBuf::from(dir).is_dir() {
            self.logs.push(format!("{what}不存在，无法打开: {dir}"));
            return;
        }
        // explorer 不接受正斜杠路径，统一转成反斜杠
        let win = dir.replace('/', "\\");
        if let Err(e) = std::process::Command::new("explorer").arg(&win).spawn() {
            self.logs.push(format!("打开{what}失败: {e}"));
        }
    }

    /// 输入目录选择（"选择文件夹…"按钮）：弹出"选择视频文件"对话框（挂在主窗口上，模态）。
    /// 不用文件夹选择框——它只显示子文件夹、隐藏文件，在存满视频的目录里
    /// 会显示成一片空白（"没有与搜索条件匹配的项"），让人以为出了问题；
    /// 改为直接列出视频，选中任意一个视频文件，其所在文件夹即作为输入目录。
    fn pick_input_dir(&mut self, frame: &eframe::Frame) {
        let mut start = PathBuf::from(&self.cfg.input_dir);
        if !start.is_dir() {
            start = std::env::var_os("USERPROFILE")
                .map(PathBuf::from)
                .unwrap_or_default();
        }
        let mut dialog = rfd::FileDialog::new()
            .set_parent(frame)
            .set_title("选择任意一个视频文件（将扫描其所在文件夹）");
        // Shell API（SHCreateItemFromParsingName）不接受正斜杠路径，统一转成反斜杠
        let start_str = start.to_string_lossy().replace('/', "\\");
        let start = PathBuf::from(&start_str);
        if start.is_dir() {
            dialog = dialog.set_directory(&start);
        }
        dialog = dialog.add_filter("视频文件", crate::model::VIDEO_EXTS);
        dialog = dialog.add_filter("所有文件", &["*"]);
        // 模态对话框，阻塞在本帧内是标准行为；父窗口此时由系统禁用输入
        if let Some(file) = dialog.pick_file() {
            let Some(dir) = file.parent() else { return };
            self.cfg.input_dir = dir.to_string_lossy().into_owned();
            if self.cfg.output_dir.is_empty() {
                self.cfg.output_dir = std::path::Path::new(&self.cfg.input_dir)
                    .join("调正输出")
                    .to_string_lossy()
                    .into_owned();
            }
            self.dirty_cfg = true;
            self.rescan();
        }
    }

    /// 输出目录选择（"选择文件夹…"按钮）：弹出与输入目录同风格的资源管理器
    /// 对话框（文件夹模式），打开时自动定位到当前输出目录。
    /// 用文件夹模式而非文件模式，是因为输出目录常常是空的（首次使用/新建），
    /// 文件模式里没有文件可选、确认不了；文件夹模式里"选择文件夹"按钮直接
    /// 确认当前打开的目录，"新建文件夹"按钮也在工具栏上。
    fn pick_output_dir(&mut self, frame: &eframe::Frame) {
        let mut start = PathBuf::from(&self.cfg.output_dir);
        if !start.is_dir() {
            // 输出目录尚不存在：定位到其所在目录
            start = std::path::Path::new(&self.cfg.output_dir)
                .parent()
                .map(PathBuf::from)
                .unwrap_or_default();
        }
        if !start.is_dir() {
            start = std::env::var_os("USERPROFILE")
                .map(PathBuf::from)
                .unwrap_or_default();
        }
        let mut dialog = rfd::FileDialog::new()
            .set_parent(frame)
            .set_title("选择输出文件夹（调正后的视频保存到此处）");
        // Shell API 不接受正斜杠路径，统一转成反斜杠
        let start_str = start.to_string_lossy().replace('/', "\\");
        let start = PathBuf::from(&start_str);
        if start.is_dir() {
            dialog = dialog.set_directory(&start);
        }
        // 模态对话框，阻塞在本帧内是标准行为；父窗口此时由系统禁用输入
        if let Some(dir) = dialog.pick_folder() {
            self.cfg.output_dir = dir.to_string_lossy().into_owned();
            self.dirty_cfg = true;
        }
    }

    fn on_flip_changed(&mut self) {
        self.dirty_cfg = true;
        // 转向变了，缩略图缓存键随之变化：全部重取
        let entries: Vec<FileEntry> = self.rows.iter().map(|r| r.entry.clone()).collect();
        for r in &mut self.rows {
            r.thumb = None;
            r.thumb_done = false;
        }
        for e in entries {
            self.request_thumb(&e);
        }
    }

    fn draw_center(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show(ui, |ui| {
            // 汇总与操作
            let mut selected_count = 0usize;
            let mut need = 0u64;
            let mut all_sel = !self.rows.is_empty();
            let mut none_sel = true;
            for r in &self.rows {
                if r.entry.selected {
                    selected_count += 1;
                    need += r.entry.size;
                    none_sel = false;
                } else {
                    all_sel = false;
                }
            }
            ui.horizontal(|ui| {
                let tri = if all_sel {
                    Some(true)
                } else if none_sel {
                    Some(false)
                } else {
                    None
                };
                let est = need as f64
                    * (self.cfg.compress_pct as f64 / 100.0)
                    * if self.cfg.hevc { 0.7 } else { 1.0 };
                let size_info = if est + 0.01 < need as f64 / 1e9 {
                    format!(
                        "{:.1} GB → 预计 {:.1} GB",
                        need as f64 / 1e9,
                        est / 1e9
                    )
                } else {
                    format!("{:.1} GB", need as f64 / 1e9)
                };
                ui.label(format!(
                    "共 {} 个视频，已选 {} 个（{}）",
                    self.rows.len(),
                    selected_count,
                    size_info
                ));
                if ui.button("全选").clicked() {
                    for r in &mut self.rows {
                        r.entry.selected = true;
                    }
                }
                if ui.button("全不选").clicked() {
                    for r in &mut self.rows {
                        r.entry.selected = false;
                    }
                }
                let _ = tri;
            });
            ui.add_space(4.0);

            if self.rows.is_empty() {
                if self.scanning {
                    ui.centered_and_justified(|ui| ui.label("正在扫描目录…"));
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.weak("选择输入目录扫描视频文件；缩略图中可直接看出哪些画面是颠倒的。")
                    });
                }
                return;
            }

            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| {
                    // 本帧内点击的文件：滚动区绘制完再统一在资源管理器中定位
                    let mut reveal: Vec<PathBuf> = Vec::new();
                    for row in &mut self.rows {
                        let path = row.entry.path.clone();
                        let mut locate = false;
                        let mut row_hovered = false;
                        ui.horizontal(|ui| {
                            // 预占一个绘制位：悬停高亮垫在整行内容之下，行末再回填
                            let hl = ui.painter().add(egui::Shape::Noop);
                            ui.checkbox(&mut row.entry.selected, "");
                            let tip = format!(
                                "{}\n点击在资源管理器中定位此文件",
                                path.display()
                            );
                            // 缩略图（点击定位文件所在文件夹）
                            let thumb_resp = if let Some(tex) = &row.thumb {
                                ui.add(
                                    egui::Image::new((tex.id(), egui::vec2(96.0, 54.0)))
                                        .sense(egui::Sense::click()),
                                )
                            } else {
                                let (rect, resp) = ui.allocate_exact_size(
                                    egui::vec2(96.0, 54.0),
                                    egui::Sense::click(),
                                );
                                if !row.thumb_done {
                                    ui.painter().rect_filled(
                                        rect,
                                        2.0,
                                        egui::Color32::from_gray(40),
                                    );
                                }
                                resp
                            };
                            if thumb_resp.clicked() {
                                locate = true;
                            }
                            row_hovered |= thumb_resp.hovered();
                            thumb_resp
                                .on_hover_text(tip.clone())
                                .on_hover_cursor(egui::CursorIcon::PointingHand);
                            ui.vertical(|ui| {
                                // 给右侧状态列预留固定宽度，避免文件名挤占导致裁切；
                                // 窗口很窄时允许文件名列收缩到最小宽度（自动截断）
                                let name_w =
                                    (ui.available_width() - 230.0).max(60.0);
                                let name_resp = ui.add_sized(
                                    [name_w, 14.0],
                                    egui::Label::new(
                                        egui::RichText::new(
                                            row.entry.rel.to_string_lossy(),
                                        )
                                        .strong(),
                                    )
                                    .truncate()
                                    .sense(egui::Sense::click()),
                                );
                                let size_resp = ui.add(
                                    egui::Label::new(egui::RichText::new(format!(
                                        "{:.0} MB",
                                        row.entry.size as f64 / 1e6
                                    )).weak())
                                    .sense(egui::Sense::click()),
                                );
                                for resp in [name_resp, size_resp] {
                                    if resp.clicked() {
                                        locate = true;
                                    }
                                    row_hovered |= resp.hovered();
                                    resp.on_hover_text(tip.clone())
                                        .on_hover_cursor(egui::CursorIcon::PointingHand);
                                }
                            });
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    match &row.entry.status {
                                        FileStatus::Processing => {
                                            let bar = egui::ProgressBar::new(
                                                row.entry.progress.clamp(0.0, 1.0),
                                            )
                                            .show_percentage()
                                            .desired_width(160.0);
                                            ui.add(bar);
                                            if row.entry.speed > 0.0 {
                                                ui.label(egui::RichText::new(format!(
                                                    "{:.1}x",
                                                    row.entry.speed
                                                )).weak());
                                            }
                                        }
                                        st => {
                                            let txt = st.label();
                                            let color = match st {
                                                FileStatus::Done => {
                                                    egui::Color32::from_rgb(120, 220, 130)
                                                }
                                                FileStatus::Failed(_) => {
                                                    egui::Color32::from_rgb(235, 100, 100)
                                                }
                                                FileStatus::Skipped => {
                                                    egui::Color32::from_rgb(160, 190, 235)
                                                }
                                                _ => egui::Color32::from_gray(160),
                                            };
                                            let short = if matches!(st, FileStatus::Failed(_)) {
                                                "✗ 失败（悬停查看）"
                                            } else {
                                                txt.as_str()
                                            };
                                            ui.add_sized(
                                                [140.0, 16.0],
                                                egui::Label::new(
                                                    egui::RichText::new(short).color(color),
                                                ),
                                            )
                                            .on_hover_text(&txt);
                                        }
                                    }
                                },
                            );
                            // 行末回填悬停高亮：垫在整行内容之下，指示整行可点击定位
                            if row_hovered {
                                ui.painter().set(
                                    hl,
                                    egui::Shape::rect_filled(
                                        ui.min_rect().expand2(egui::vec2(2.0, 4.0)),
                                        3.0,
                                        ui.visuals().widgets.hovered.bg_fill,
                                    ),
                                );
                            }
                        });
                        if locate {
                            reveal.push(path);
                        }
                        ui.separator();
                    }
                    for p in &reveal {
                        self.reveal_in_explorer(p);
                    }
                });
        });
    }

    fn draw_bottom(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("bottom").show(ui, |ui| {
            ui.add_space(6.0);
            // 窗口变窄时按钮/进度/状态允许换行
            ui.horizontal_wrapped(|ui| {
                let can_start =
                    !self.processing && !self.rows.is_empty() && self.caps.is_some();
                if ui.add_enabled(can_start, egui::Button::new(egui::RichText::new("▶ 开始处理").strong()))
                    .clicked()
                {
                    self.start_batch();
                }
                if ui
                    .add_enabled(self.processing, egui::Button::new("■ 取消"))
                    .clicked()
                {
                    self.cancel.store(true, Ordering::Relaxed);
                }
                ui.separator();
                if self.processing {
                    let finished = self.batch_done + self.batch_failed + self.batch_skipped;
                    // 多文件并行时，总进度 = 已完成数 + 各处理中文件的进度之和
                    let mut cur = 0f32;
                    let mut processing_n = 0usize;
                    for r in &self.rows {
                        if let FileStatus::Processing = r.entry.status {
                            processing_n += 1;
                            cur += r.entry.progress.clamp(0.0, 1.0);
                        }
                    }
                    let frac = if self.batch_total > 0 {
                        ((finished as f32 + cur) / self.batch_total as f32).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .show_percentage()
                            .desired_width(220.0),
                    );
                    ui.label(format!(
                        "总进度 {}/{}（完成 {}，跳过 {}，失败 {}）—— {} 路并行中",
                        finished,
                        self.batch_total,
                        self.batch_done,
                        self.batch_skipped,
                        self.batch_failed,
                        processing_n
                    ));
                } else if self.scanning {
                    ui.label("扫描中…");
                } else {
                    let sel: usize = self.rows.iter().filter(|r| r.entry.selected).count();
                    ui.label(format!("空闲 —— 已选 {sel} 个待处理"));
                }
            });
            if let Some(last) = self.logs.last() {
                ui.label(egui::RichText::new(last).weak().small());
            }
            ui.add_space(2.0);
            egui::CollapsingHeader::new("详细日志")
                .default_open(false)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                        for l in &self.logs {
                            ui.label(egui::RichText::new(l).small().weak());
                        }
                    });
                });
            ui.add_space(4.0);
        });
    }
}

fn install_cjk_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    for path in [
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\msyh.ttf",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\simsun.ttc",
    ] {
        if let Ok(data) = std::fs::read(path) {
            fonts
                .font_data
                .insert("cjk".to_owned(), egui::FontData::from_owned(data).into());
            for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts
                    .families
                    .entry(family)
                    .or_default()
                    .push("cjk".to_owned());
            }
            break;
        }
    }
    ctx.set_fonts(fonts);
}

fn jobs_label(jobs: u32) -> &'static str {
    match jobs {
        1 => "1 路（最省）",
        2 => "2 路",
        3 => "3 路",
        4 => "4 路（最快）",
        _ => "自动（按 GPU 占用）",
    }
}

fn size_label(pct: u32) -> &'static str {
    match pct {
        70 => "压缩约 30%",
        50 => "压缩约 50%",
        30 => "压缩约 70%",
        _ => "与原片相当",
    }
}
