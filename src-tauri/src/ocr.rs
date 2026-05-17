use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::inputs;
use ort::value::Value;
use image::{load_from_memory, imageops::{self, FilterType}, GenericImageView, DynamicImage, GrayImage, Luma};
use ndarray::{Array4, Axis, ArrayViewD, ArrayView2};
use std::error::Error;
use std::sync::Mutex;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor};
use imageproc::contours::find_contours;
use geo::{Coord, Polygon, ConvexHull, MinimumRotatedRect};
use clipper2_rust::{inflate_paths_64, make_path64, JoinType, EndType};

// --- (1) DATA STRUCTURES ---
#[derive(Debug, Clone)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

// --- (2) INTERFACES (SOLID) ---
pub trait IOcrDetector {
    fn find_bounding_boxes(&self, image: &DynamicImage) -> Result<Vec<Rect>, Box<dyn Error>>;
}

pub trait IOcrRecognizer {
    fn extract_text(&self, cropped_image: &DynamicImage) -> Result<String, Box<dyn Error>>;
}

// --- (3) DETECTOR IMPLEMENTATION ---
pub struct OnnxOcrDetector {
    session: Mutex<Session>,
}

impl OnnxOcrDetector {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Ok(Self { session: Mutex::new(session) })
    }
}

impl IOcrDetector for OnnxOcrDetector {
    fn find_bounding_boxes(&self, image: &DynamicImage) -> Result<Vec<Rect>, Box<dyn Error>> {
        let (orig_w, orig_h) = image.dimensions();
        
        let det_size = 960;
        let resized = image.resize_exact(det_size, det_size, FilterType::Triangle);
        
        let mut tensor_data = vec![0.0f32; 3 * det_size as usize * det_size as usize];
        for (x, y, pixel) in resized.pixels() {
            tensor_data[(0 * det_size as usize * det_size as usize) + (y as usize * det_size as usize) + (x as usize)] = (pixel[0] as f32 / 255.0 - 0.485) / 0.229;
            tensor_data[(1 * det_size as usize * det_size as usize) + (y as usize * det_size as usize) + (x as usize)] = (pixel[1] as f32 / 255.0 - 0.456) / 0.224;
            tensor_data[(2 * det_size as usize * det_size as usize) + (y as usize * det_size as usize) + (x as usize)] = (pixel[2] as f32 / 255.0 - 0.406) / 0.225;
        }

        let tensor = Array4::from_shape_vec((1, 3, det_size as usize, det_size as usize), tensor_data)?;
        let mut session = self.session.lock().map_err(|_| "Failed to lock det session")?;
        
        let input_value = Value::from_array(tensor)?;
        let inputs = inputs!["x" => input_value];
        let outputs = session.run(inputs)?;
        
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (shape, data) = output_tensor;
        let nd_shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let heatmap = ArrayViewD::from_shape(nd_shape, data)?;
        
        // Fix: Lifetime and dimensionality
        let heatmap_slice = heatmap.index_axis(Axis(0), 0);
        let heatmap_2d_raw = heatmap_slice.index_axis(Axis(0), 0);
        let heatmap_2d = heatmap_2d_raw.into_dimensionality::<ndarray::Ix2>()?;

        let boxes = self.post_process(&heatmap_2d, orig_w, orig_h)?;
        Ok(boxes)
    }
}

impl OnnxOcrDetector {
    fn post_process(&self, heatmap: &ArrayView2<f32>, orig_w: u32, orig_h: u32) -> Result<Vec<Rect>, Box<dyn Error>> {
        let thresh = 0.3;
        let box_thresh = 0.6;
        let unclip_ratio = 1.5;
        
        let (rows, cols) = heatmap.dim();
        let mut mask = GrayImage::new(cols as u32, rows as u32);
        for ((y, x), &val) in heatmap.indexed_iter() {
            if val > thresh {
                mask.put_pixel(x as u32, y as u32, Luma([255]));
            }
        }

        let contours = find_contours::<i32>(&mask);
        let mut rects = Vec::new();

        for contour in contours {
            if contour.points.len() < 3 { continue; }
            
            let points: Vec<Coord<f32>> = contour.points.iter()
                .map(|p| Coord { x: p.x as f32, y: p.y as f32 })
                .collect();
            
            let poly = Polygon::new(geo::LineString::new(points), vec![]);
            let hull = poly.convex_hull();
            
            if let Some(mrr_poly) = hull.minimum_rotated_rect() {
                let box_pts: Vec<[f32; 2]> = mrr_poly.exterior().points()
                    .take(4)
                    .map(|p| [p.x(), p.y()])
                    .collect();

                let score = self.box_score_fast(heatmap, &box_pts);
                if score < box_thresh { continue; }

                let unclipped = self.unclip(&box_pts, unclip_ratio);
                
                let min_x = unclipped.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
                let max_x = unclipped.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
                let min_y = unclipped.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
                let max_y = unclipped.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);

                let rx = orig_w as f32 / cols as f32;
                let ry = orig_h as f32 / rows as f32;

                rects.push(Rect {
                    x: (min_x * rx) as i32,
                    y: (min_y * ry) as i32,
                    width: ((max_x - min_x) * rx) as u32,
                    height: ((max_y - min_y) * ry) as u32,
                });
            }
        }
        
        rects.sort_by_key(|r| r.y);
        Ok(rects)
    }

    fn box_score_fast(&self, heatmap: &ArrayView2<f32>, box_pts: &[[f32; 2]]) -> f32 {
        let (rows, cols) = heatmap.dim();
        let min_x = box_pts.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min).max(0.0) as usize;
        let max_x = box_pts.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max).min((cols - 1) as f32) as usize;
        let min_y = box_pts.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min).max(0.0) as usize;
        let max_y = box_pts.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max).min((rows - 1) as f32) as usize;

        let mut sum = 0.0;
        let mut count = 0;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                sum += heatmap[[y, x]];
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f32 }
    }

    fn unclip(&self, box_pts: &[[f32; 2]], ratio: f32) -> Vec<[f32; 2]> {
        let path = make_path64(&box_pts.iter().flat_map(|p| vec![p[0] as i64, p[1] as i64]).collect::<Vec<_>>());
        let inflated = inflate_paths_64(&vec![path], ratio as f64 * 2.0, JoinType::Round, EndType::Polygon, 2.0, 0.0);
        if inflated.is_empty() || inflated[0].is_empty() {
            return box_pts.to_vec();
        }
        inflated[0].iter().map(|p| [p.x as f32, p.y as f32]).collect()
    }
}

// --- (4) RECOGNIZER IMPLEMENTATION ---
pub struct OnnxOcrRecognizer {
    session: Mutex<Session>,
    dict: Vec<String>,
}

impl OnnxOcrRecognizer {
    pub fn new(model_path: &str, dict_path: &str) -> Result<Self, Box<dyn Error>> {
        let mut dict = Vec::new();
        let file = File::open(dict_path)?;
        for line in BufReader::new(file).lines() {
            dict.push(line?);
        }
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Ok(Self { session: Mutex::new(session), dict })
    }
}

impl IOcrRecognizer for OnnxOcrRecognizer {
    fn extract_text(&self, cropped_image: &DynamicImage) -> Result<String, Box<dyn Error>> {
        let resized = cropped_image.resize_exact(320, 48, FilterType::Triangle);
        let mut tensor_data = vec![0.0f32; 3 * 48 * 320];
        for (x, y, pixel) in resized.pixels() {
            let r = (pixel[0] as f32 / 255.0 - 0.5) / 0.5;
            let g = (pixel[1] as f32 / 255.0 - 0.5) / 0.5;
            let b = (pixel[2] as f32 / 255.0 - 0.5) / 0.5;
            tensor_data[(0 * 48 * 320) + (y as usize * 320) + (x as usize)] = r;
            tensor_data[(1 * 48 * 320) + (y as usize * 320) + (x as usize)] = g;
            tensor_data[(2 * 48 * 320) + (y as usize * 320) + (x as usize)] = b;
        }

        let tensor = Array4::from_shape_vec((1, 3, 48, 320), tensor_data)?;
        let mut session = self.session.lock().map_err(|_| "Failed to lock rec session")?;
        
        let input_value = Value::from_array(tensor)?;
        let inputs = inputs!["x" => input_value];
        let outputs = session.run(inputs)?;
        
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (shape, data) = output_tensor;
        let nd_shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let view = ArrayViewD::from_shape(nd_shape, data)?;
        
        let seq_len = view.shape()[1];
        let num_classes = view.shape()[2];
        let mut prev_idx = -1i32;
        let mut result_text = String::new();
        
        for t in 0..seq_len {
            let row = view.index_axis(Axis(1), t);
            let mut max_val = -f32::INFINITY;
            let mut max_idx = 0usize;
            for i in 0..num_classes {
                let val = row[[0, i]];
                if val > max_val { max_val = val; max_idx = i; }
            }
            if max_idx != 0 && max_idx as i32 != prev_idx {
                if let Some(char) = self.dict.get(max_idx - 1) {
                    result_text.push_str(char);
                }
            }
            prev_idx = max_idx as i32;
        }
        Ok(result_text)
    }
}

// --- (5) ORCHESTRATOR ---
pub struct OcrEngine {
    pub detector: Box<dyn IOcrDetector + Send + Sync>,
    pub recognizer: Box<dyn IOcrRecognizer + Send + Sync>,
}

impl OcrEngine {
    pub fn process_image(&self, raw_buffer: &[u8]) -> Result<String, Box<dyn Error>> {
        let img = load_from_memory(raw_buffer)?;
        
        let boxes = self.detector.find_bounding_boxes(&img)?;
        println!("Detected {} lines.", boxes.len());
        
        let mut results = Vec::new();
        for (i, box_rect) in boxes.iter().enumerate() {
            let cropped = imageops::crop_imm(&img, 
                box_rect.x.max(0) as u32, 
                box_rect.y.max(0) as u32, 
                box_rect.width.min(img.width()), 
                box_rect.height.min(img.height())
            ).to_image();
            
            // Debug save
            let mut buf = Cursor::new(Vec::new());
            DynamicImage::ImageRgba8(cropped.clone()).write_to(&mut buf, image::ImageFormat::Png)?;
            std::fs::write(format!("../debug_line_{}.png", i), buf.into_inner())?;

            let text = self.recognizer.extract_text(&DynamicImage::ImageRgba8(cropped))?;
            if !text.trim().is_empty() {
                results.push(text);
            }
        }
        
        Ok(results.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_pipeline() {
        // ให้มันรันเฉพาะตอนที่มีไฟล์โมเดลครบเท่านั้น
        let det_path = "../models/en_PP-OCRv3_det_infer.onnx";
        let rec_path = "../models/en_PP-OCRv4_rec_infer.onnx";
        let dict_path = "../models/en_dict.txt";
        let img_path = "../debug_capture.png";

        if !std::path::Path::new(det_path).exists() || !std::path::Path::new(rec_path).exists() {
            println!("Skipping test: Models not found.");
            return;
        }

        let det = OnnxOcrDetector::new(det_path).unwrap();
        let rec = OnnxOcrRecognizer::new(rec_path, dict_path).unwrap();
        let engine = OcrEngine {
            detector: Box::new(det),
            recognizer: Box::new(rec),
        };

        let img_bytes = std::fs::read(img_path).expect("Failed to read test image");
        let result = engine.process_image(&img_bytes);

        match result {
            Ok(text) => {
                println!("--- OCR RESULT ---");
                println!("{}", text);
                println!("------------------");
                assert!(!text.is_empty(), "OCR result should not be empty");
            },
            Err(e) => panic!("OCR Pipeline failed: {}", e),
        }
    }
}
