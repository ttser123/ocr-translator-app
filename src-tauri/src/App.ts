import { invoke } from '@tauri-apps/api/core';

// สร้าง Interface ให้ตรงกับที่ Rust พ่นออกมา (DRY Principle - ห้ามลืมเด็ดขาด)
interface TranslationResult {
    original_text: string;
    translated_text: string;
    status: string;
}

async function handleTranslate() {
    try {
        // สมมติว่ามึงลากเมาส์คลุมจอได้พิกัดมาแล้ว
        const result = await invoke<TranslationResult>('process_screen_area', {
            x: 100, y: 100, width: 300, height: 50
        });

        console.log("ได้คำแปลมาแล้วเว้ย:", result.translated_text);
        // TODO: เอา result.translated_text ไปเรนเดอร์ลง UI ให้ลอยทับเกม

    } catch (error) {
        console.error("ชิบหายแล้ว เออเร่อ:", error);
    }
}