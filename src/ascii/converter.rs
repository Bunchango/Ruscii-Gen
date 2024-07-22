use super::char_set::CharacterSet;
use super::error::ConvertError;
use super::font_loader::{FontLoader, FontSettings};
use crate::image_manip::edge_detect::{EdgeDetect, Sobel};
use crate::image_manip::edge_processor::EdgeDownscaler;
use crate::image_manip::processing::{
    BilateralFilter, DoG, MedianBlur, Processor, SharpenGaussian, Threshold,
};
use crate::image_manip::util::bufr_to_arr;
use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use image::{DynamicImage, ImageBuffer, Rgba};
use imageproc::drawing::draw_text_mut;
use ndarray::{ArrayView2, Zip};

pub struct Converter {
    font_settings: FontSettings,
    pixel_mapping: CharacterSet,
    tile_preprocessors: Vec<Box<dyn Processor<u8, u8>>>,
    edge_preprocessors: Vec<Box<dyn Processor<u8, u8>>>,
    edge_detector: Box<dyn EdgeDetect<u8, u8>>,
    bg_color: Rgba<u8>,
    color: Rgba<u8>,
}

impl Converter {
    pub fn default() -> Self {
        Converter {
            font_settings: FontSettings::default(),
            pixel_mapping: CharacterSet::default(),
            tile_preprocessors: vec![],
            edge_preprocessors: vec![
                Box::new(SharpenGaussian::default()),
                Box::new(DoG::default()),
                Box::new(BilateralFilter::default()),
                Box::new(MedianBlur::default()),
                Box::new(Threshold::default()),
            ],
            edge_detector: Box::new(Sobel::new()),
            bg_color: Rgba([0, 0, 0, 255]),
            color: Rgba([255, 255, 255, 255]),
        }
    }

    pub fn new(
        font_settings: FontSettings,
        pixel_mapping: CharacterSet,
        tile_preprocessors: Vec<Box<dyn Processor<u8, u8>>>,
        edge_preprocessors: Vec<Box<dyn Processor<u8, u8>>>,
        edge_detector: Box<dyn EdgeDetect<u8, u8>>,
        bg_color: Rgba<u8>,
        color: Rgba<u8>,
    ) -> Self {
        Converter {
            font_settings,
            pixel_mapping,
            tile_preprocessors,
            edge_preprocessors,
            edge_detector,
            bg_color,
            color,
        }
    }

    fn arr_to_img(
        &self,
        arr: &ArrayView2<char>,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, ConvertError> {
        let (h, w): (u32, u32) = (
            arr.shape()[0] as u32 * self.font_settings.font_size,
            arr.shape()[1] as u32 * self.font_settings.font_size,
        );

        let mut ascii_bufr =
            ImageBuffer::<Rgba<u8>, Vec<u8>>::from_fn(w, h, |_, _| self.bg_color.clone());

        // Load font
        let (font, scale) = FontLoader::load_font_from_settings(&self.font_settings)?;

        for (y, row) in arr.outer_iter().enumerate() {
            for (x, &ch) in row.iter().enumerate() {
                draw_text_mut(
                    &mut ascii_bufr,
                    self.color.clone(),
                    (x as u32 * self.font_settings.font_size) as i32,
                    (y as u32 * self.font_settings.font_size) as i32,
                    scale,
                    &font,
                    &ch.to_string(),
                );
            }
        }

        Ok(ascii_bufr)
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
        let resized_img = ori_img.resize_exact(new_w, new_h, FilterType::CatmullRom);
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

        let ascii_img = self.arr_to_img(&ds_edge_arr.view())?;

        // Save image
        ascii_img.save(out)?;

        Ok(())
    }

    pub fn convert_vid(&self, path: &str) {
        /*
         * Read a video given file path and convert each frame of video into ascii image. Then combine
         * every frame to form a complete video
         */
    }
}
