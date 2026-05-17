use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::inputs;
use ort::value::Value;
use image::{load_from_memory, imageops::FilterType, GenericImageView};
use ndarray::Array4;
use std::error::Error;
use std::sync::Mutex;

pub struct OnnxOcrEngine {
    session: Option<Mutex<Session>>,
}

impl OnnxOcrEngine {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        // Init ort environment (needs to be done once globally)
        let _ = ort::init()
            .with_name("ocr_env")
            .commit();

        // Attempt to load the model. If it fails (e.g. file not found), we store None
        let session = match Session::builder() {
            Ok(builder) => {
                match builder.with_optimization_level(GraphOptimizationLevel::Level3) {
                    Ok(mut b) => {
                        match b.commit_from_file(model_path) {
                            Ok(s) => Some(Mutex::new(s)),
                            Err(e) => {
                                println!("Warning: Failed to load ONNX model from {}: {}", model_path, e);
                                None
                            }
                        }
                    },
                    Err(e) => {
                        println!("Warning: Failed to set optimization level: {}", e);
                        None
                    }
                }
            },
            Err(e) => {
                println!("Warning: Failed to create Session builder: {}", e);
                None
            }
        };

        Ok(Self { session })
    }

    pub fn extract_text(&self, processed_image_buffer: &[u8]) -> Result<String, Box<dyn Error>> {
        let session_mutex = self.session.as_ref().ok_or("ONNX model is not loaded. Please check the model path.")?;
        let mut session = session_mutex.lock().map_err(|_| "Failed to lock OCR session")?;

        println!("เริ่มหั่นรูปเป็น Tensor...");

        // 1. โหลดรูปที่ผ่านการทำขาวดำมาจากท่อก่อนหน้า
        let img = load_from_memory(processed_image_buffer)?;

        // 2. บีบรูป/ขยายรูป (Resize) ให้ได้ขนาด 320x48 ตามที่โมเดลต้องการเป๊ะๆ
        let resized = img.resize_exact(320, 48, FilterType::Triangle);

        // 3. สร้าง Array 1 มิติ
        let mut tensor_data = vec![0.0f32; 3 * 48 * 320];

        // 4. ลูปดูดค่าสีทีละพิกเซล
        for (x, y, pixel) in resized.pixels() {
            let r = (pixel[0] as f32 / 255.0 - 0.5) / 0.5;
            let g = (pixel[1] as f32 / 255.0 - 0.5) / 0.5;
            let b = (pixel[2] as f32 / 255.0 - 0.5) / 0.5;

            let c0_idx = (0 * 48 * 320) + (y as usize * 320) + (x as usize);
            let c1_idx = (1 * 48 * 320) + (y as usize * 320) + (x as usize);
            let c2_idx = (2 * 48 * 320) + (y as usize * 320) + (x as usize);

            tensor_data[c0_idx] = r;
            tensor_data[c1_idx] = g;
            tensor_data[c2_idx] = b;
        }

        // 5. ปั้น Vector ให้กลายเป็น Tensor แบบ 4 มิติ
        let tensor = Array4::from_shape_vec((1, 3, 48, 320), tensor_data)?;

        // 6. โยนเข้าปากโมเดล (Hardcode "x" ตามที่ซีเนียร์สั่ง!)
        let input_value = Value::from_array(tensor)?;
        let inputs = inputs!["x" => input_value];
        
        // 7. รัน!
        let _outputs = session.run(inputs)?;

        println!("AI รันเสร็จแล้วเว้ย!");
        
        Ok("รัน Tensor ผ่านโว้ย!".to_string())
    }
}
