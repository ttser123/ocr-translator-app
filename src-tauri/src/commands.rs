use serde::Serialize;
use screenshots::Screen; 
use std::fs;
use std::io::Cursor;
use image::{load_from_memory, ImageFormat, ImageOutputFormat};
use tauri::State;
use crate::ocr::OnnxOcrEngine;

// กำหนด Type ข้อมูลให้ชัดเจน
#[derive(Serialize)]
pub struct TranslationResult {
    pub original_text: String,
    pub translated_text: String,
    pub status: String,
}

// ฟังก์ชันสำหรับทำความสะอาดรูปภาพก่อนส่งให้ OCR
fn preprocess_image_for_ocr(raw_buffer: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let img = load_from_memory(raw_buffer)?;
    let grayscale = img.grayscale();
    
    let mut result_buffer: Vec<u8> = Vec::new();
    grayscale.write_to(&mut Cursor::new(&mut result_buffer), ImageOutputFormat::Png)?;
    
    Ok(result_buffer)
}

// Facade Command รอรับคำสั่งจาก TypeScript
#[tauri::command]
pub async fn process_screen_area(
    x: i32, 
    y: i32, 
    width: u32, 
    height: u32,
    ocr_engine: State<'_, OnnxOcrEngine>
) -> Result<TranslationResult, String> {
    println!("Capturing screen at: x={}, y={}, w={}, h={}", x, y, width, height);
    
    // --- (1) NATIVE SCREEN CAPTURE ---
    let screen = Screen::from_point(x, y).map_err(|e| format!("หาหน้าจอไม่เจอ: {}", e))?;
    let image = screen.capture_area(x, y, width, height).map_err(|e| format!("แคปจอพัง: {}", e))?;
    
    let mut raw_buffer = Cursor::new(Vec::new());
    image.write_to(&mut raw_buffer, ImageFormat::Png)
        .map_err(|e| format!("แปลงรูปพัง: {}", e))?;
    
    let raw_bytes = raw_buffer.into_inner();

    // --- (2) IMAGE PREPROCESSOR ---
    let processed_bytes = preprocess_image_for_ocr(&raw_bytes)
        .map_err(|e| format!("ทำรูปขาวดำพัง: {}", e))?;

    // *DEBUG MODE 2*: เซฟรูปขาวดำเพื่อเปรียบเทียบ
    fs::write("../debug_bw.png", &processed_bytes).map_err(|e| format!("เซฟรูปขาวดำไม่ได้: {}", e))?;
    println!("DEBUG: เซฟรูป debug_bw.png สำเร็จ!");
    // ------------------------------

    // --- (3) OCR ENGINE (ONNX) ---
    let extracted_text = ocr_engine.extract_text(&processed_bytes)
        .map_err(|e| format!("OCR พัง: {}", e))?;
    println!("DEBUG: สกัดข้อความได้ -> {}", extracted_text);
    // ------------------------------
    
    Ok(TranslationResult {
        original_text: extracted_text,
        translated_text: "Warrior".to_string(),
        status: "success".to_string(),
    })
}
