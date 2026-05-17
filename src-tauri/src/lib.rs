pub mod commands;
pub mod ocr;
use tauri::{Manager, Emitter};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};
use crate::ocr::{OcrEngine, OnnxOcrDetector, OnnxOcrRecognizer};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize 2-Stage OCR Engine
    let det = OnnxOcrDetector::new("../models/en_PP-OCRv3_det_infer.onnx").expect("Failed to load detector");
    let rec = OnnxOcrRecognizer::new("../models/en_PP-OCRv4_rec_infer.onnx", "../models/en_dict.txt").expect("Failed to load recognizer");
    
    let ocr_engine = OcrEngine {
        detector: Box::new(det),
        recognizer: Box::new(rec),
    };

    let capture_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyS);

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, shortcut, event| {
                    if shortcut == &capture_shortcut && event.state() == ShortcutState::Pressed {
                        let window = app.get_webview_window("main").unwrap();
                        window.set_fullscreen(true).unwrap();
                        window.set_resizable(false).unwrap();
                        window.show().unwrap();
                        window.set_focus().unwrap();
                        window.emit("start-capture", ()).unwrap();
                    }
                })
                .build(),
        )
        .setup(move |app| {
            app.global_shortcut().register(capture_shortcut)?;
            Ok(())
        })
        .manage(ocr_engine)
        .invoke_handler(tauri::generate_handler![
            commands::process_screen_area,
            hide_window
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command]
async fn hide_window(window: tauri::WebviewWindow) {
    window.set_fullscreen(false).unwrap();
    window.hide().unwrap();
}
