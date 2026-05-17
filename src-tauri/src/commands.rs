use serde::Serialize;
use screenshots::Screen; 
use std::fs;
use std::io::Cursor;
use image::ImageFormat;
use tauri::State;
use crate::ocr::OcrEngine;

// กำหนด Type ข้อมูลที่ส่งกลับไปให้หน้าบ้าน
#[derive(Serialize)]
pub struct TranslationResult {
    pub original_text: String,
    pub translated_text: String,
    pub status: String,
}

// Facade Command รอรับคำสั่งจาก TypeScript
#[tauri::command]
pub async fn process_screen_area(
    x: i32, 
    y: i32, 
    width: u32, 
    height: u32,
    ocr_engine: State<'_, OcrEngine>
) -> Result<TranslationResult, String> {
    println!("Capturing screen at: x={}, y={}, w={}, h={}", x, y, width, height);
    
    // --- (1) NATIVE SCREEN CAPTURE ---
    let screen = Screen::from_point(x, y).map_err(|e| format!("หาหน้าจอไม่เจอ: {}", e))?;
    let image = screen.capture_area(x, y, width, height).map_err(|e| format!("แคปจอพัง: {}", e))?;
    
    let mut raw_buffer = Cursor::new(Vec::new());
    image.write_to(&mut raw_buffer, ImageFormat::Png)
        .map_err(|e| format!("แปลงรูปพัง: {}", e))?;
    
    let raw_bytes = raw_buffer.into_inner();

    // *DEBUG*: เซฟรูปดิบไว้เช็ค
    fs::write("../debug_capture.png", &raw_bytes).map_err(|e| format!("เซฟรูปลงเครื่องไม่ได้: {}", e))?;

    // --- (2) 2-STAGE OCR PIPELINE (Detect + Recognize) ---
    let extracted_text = ocr_engine.process_image(&raw_bytes)
        .map_err(|e| format!("OCR พัง: {}", e))?;
    
    println!("DEBUG: สกัดข้อความได้ทั้งหมด -> \n{}", extracted_text);
    
    Ok(TranslationResult {
        original_text: extracted_text,
        translated_text: "Warrior (2-Stage)".to_string(), // Mock แปลภาษาไว้ก่อน
        status: "success".to_string(),
    })
}
