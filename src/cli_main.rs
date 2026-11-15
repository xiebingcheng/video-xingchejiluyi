//! CLI 入口（保留控制台输出）。

fn set_console_utf8() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Console::{GetConsoleOutputCP, SetConsoleOutputCP};
        if GetConsoleOutputCP() != 65001 {
            SetConsoleOutputCP(65001);
        }
    }
}

fn main() -> anyhow::Result<()> {
    set_console_utf8();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|s| s.as_str()) == Some("--print-config") {
        println!("{:?}", video_fixer::config::Config::load());
        return Ok(());
    }
    video_fixer::cli::run(&args)
}
