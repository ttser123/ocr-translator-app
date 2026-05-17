pub mod commands;
pub mod ocr;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let ocr_engine = match ocr::OnnxOcrEngine::new("../models/en_PP-OCRv4_rec_infer.onnx") {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("Failed to initialize OCR Engine: {}", e);
            // Ignore error and instantiate with empty session
            ocr::OnnxOcrEngine::new("dummy").unwrap()
        }
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .manage(ocr_engine)
        .invoke_handler(tauri::generate_handler![commands::process_screen_area])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
