use super::char_set::CharacterSet;
use super::error::ConvertError;
use super::font_loader::{FontLoader, FontSettings};
use crate::image_manip::edge_detect::{EdgeDetect, Sobel};
use crate::image_manip::edge_processor::EdgeDownscaler;
use crate::image_manip::processing::{DoG, MedianBlur, Processor, SharpenGaussian, Threshold};
use crate::image_manip::util::bufr_to_arr;
use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel, Rgb};
use imageproc::drawing::draw_text_mut;
use ndarray::{ArrayView2, Zip};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

pub struct Converter {
    font_settings: FontSettings,
    pixel_mapping: CharacterSet,
    tile_preprocessors: Vec<Box<dyn Processor<u8, u8>>>,
    edge_preprocessors: Vec<Box<dyn Processor<u8, u8>>>,
    edge_detector: Box<dyn EdgeDetect<u8, u8>>,
    bg_color: Rgb<u8>,
    // If use_image_color is true, then when drawing image, the drawer will use the color of the
    // pixel in the original image instead
    use_image_color: bool,
    color: Rgb<u8>,
}

// TODO: Remove color banding

impl Converter {
    pub fn default() -> Self {
        Converter {
            font_settings: FontSettings::default(),
            pixel_mapping: CharacterSet::default(),
            tile_preprocessors: vec![],
            edge_preprocessors: vec![
                Box::new(SharpenGaussian::default()),
                Box::new(DoG::default()),
                // No need for Bilateral as the threshold is doing most of the work
                // Box::new(BilateralFilter::default()),
                Box::new(MedianBlur::default()),
                Box::new(Threshold::default()),
            ],
            edge_detector: Box::new(Sobel::new()),
            bg_color: Rgb([117, 33, 141]),
            use_image_color: true,
            color: Rgb([255, 255, 255]),
        }
    }

    pub fn new(
        font_settings: FontSettings,
        pixel_mapping: CharacterSet,
        tile_preprocessors: Vec<Box<dyn Processor<u8, u8>>>,
        edge_preprocessors: Vec<Box<dyn Processor<u8, u8>>>,
        edge_detector: Box<dyn EdgeDetect<u8, u8>>,
        bg_color: Rgb<u8>,
        use_image_color: bool,
        color: Rgb<u8>,
    ) -> Self {
        Converter {
            font_settings,
            pixel_mapping,
            tile_preprocessors,
            edge_preprocessors,
            edge_detector,
            bg_color,
            use_image_color,
            color,
        }
    }

    fn arr_to_img(
        &self,
        arr: &ArrayView2<char>,
        arr_img: &DynamicImage,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, ConvertError> {
        let (h, w) = (
            arr.shape()[0] as u32 * self.font_settings.font_size,
            arr.shape()[1] as u32 * self.font_settings.font_size,
        );

        let ascii_bufr = Arc::new(Mutex::new(ImageBuffer::<Rgb<u8>, Vec<u8>>::from_pixel(
            w,
            h,
            self.bg_color.clone(),
        )));

        let (font, scale) = FontLoader::load_font_from_settings(&self.font_settings)?;

        let font_size = self.font_settings.font_size;
        let use_image_color = self.use_image_color;
        let bg_color = self.bg_color;
        let color = self.color;

        arr.outer_iter()
            .enumerate()
            .collect::<Vec<_>>() // Collect rows to maintain order since par_iter might not preserve order
            .par_iter() // Process rows in parallel
            .for_each(|(y, row)| {
                let mut local_bufr = ImageBuffer::from_pixel(w, font_size, bg_color.clone());
                for (x, &ch) in row.iter().enumerate() {
                    let x_pos = (x as u32 * font_size) as i32;
                    let y_pos = 0; // local y position in the row buffer

                    let mut local_color = color.clone();
                    if use_image_color {
                        local_color = arr_img.get_pixel(x as u32, *y as u32).to_rgb();
                    }
                    draw_text_mut(
                        &mut local_bufr,
                        local_color,
                        x_pos,
                        y_pos,
                        scale,
                        &font,
                        &ch.to_string(),
                    );
                }

                let mut ascii_bufr_lock = ascii_bufr.lock().unwrap();

                // Calculate the starting Y position for this row in the final image buffer
                let start_y = *y as u32 * font_size;

                // Merge the processed row into the final image buffer at the correct position
                for (x, y, pixel) in local_bufr.enumerate_pixels() {
                    ascii_bufr_lock.put_pixel(x, y + start_y, *pixel);
                }
            });

        let final_bufr = Arc::try_unwrap(ascii_bufr)
            .expect("Arc unwrap failed")
            .into_inner()
            .unwrap();

        Ok(final_bufr)
    }

    pub fn convert_img(
        &self,
        path: &str,
        out: &str,
        sharpen_thres: f32,
    ) -> Result<(), ConvertError> {
        /*
         * Read an image given file path and convert that image into an ascii image / txt file / or
         * print it depending on settings
         */

        // Calculate the new size of the image for downscaling
        let ori_img: DynamicImage = ImageReader::open(path)?
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        let (ori_w, ori_h): (f32, f32) = (ori_img.width() as f32, ori_img.height() as f32);
        let (new_w, new_h): (u32, u32) = (
            (ori_w / self.font_settings.font_size as f32).floor() as u32,
            (ori_h / self.font_settings.font_size as f32).floor() as u32,
        );

        // Downscaling and grayscale the image for preprocessing
        // Maybe let user choose resize algorithm
        let resized_img = ori_img.resize_exact(new_w, new_h, FilterType::Triangle);
        let mut gs_resized_img = resized_img.to_luma8();

        // Apply preprocessors before quantization
        for preproc in self.tile_preprocessors.iter() {
            gs_resized_img = preproc.apply(&gs_resized_img)?;
        }

        // Normalize and quantize the img
        // Convert to array for easy processing
        let qt_tile_arr = (bufr_to_arr(&gs_resized_img)).mapv(|x: u8| -> char {
            let index = ((x as f32 / 255.0)
                * (self.pixel_mapping.get_tile_mapping_size() - 1) as f32)
                .floor() as usize;
            self.pixel_mapping.tile[index].clone()
        });

        // Find edges
        let mut gs_ori_img = ori_img.to_luma8();

        // Apply preprocessors on gs_ori_img
        for preproc in self.edge_preprocessors.iter() {
            gs_ori_img = preproc.apply(&gs_ori_img)?;
        }

        let qt_edge = self.edge_detector.apply(&gs_ori_img, 5)?;
        let qt_edge_arr = bufr_to_arr(&qt_edge);

        // Apply edge sharpening and map to edge char
        let mut ds_edge_arr = EdgeDownscaler::hist_downscale(
            &qt_edge_arr,
            self.font_settings.font_size as usize,
            sharpen_thres,
            (new_h as usize, new_w as usize),
        )
        .mapv(|x: u8| -> char { self.pixel_mapping.edge[x as usize].clone() });

        // Combine tile arr and edge arr
        Zip::from(&qt_tile_arr)
            .and(&mut ds_edge_arr)
            .par_for_each(|&tile_val, edge_val| {
                if *edge_val == ' ' {
                    *edge_val = tile_val;
                }
            });

        let ascii_img = self.arr_to_img(&ds_edge_arr.view(), &resized_img)?;

        // Save image
        ascii_img.save(out)?;

        Ok(())
    }
}
