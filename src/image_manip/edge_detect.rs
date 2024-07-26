use crate::ascii::error::ConvertError;
use image::{ImageBuffer, Luma, Primitive};
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use ndarray::{Array2, Zip};
use num_traits::Num;

use std::{f32::consts::PI, usize};

use super::util::{arr_to_bufr, bufr_to_arr};

/*
* Detect edges and quantize the image to a number of allowed values
*/
pub trait EdgeDetect<T: Num + Copy + Primitive, U: Copy + Num + Primitive> {
    fn apply(
        &self,
        bufr: &ImageBuffer<Luma<T>, Vec<T>>,
        val_num: u8,
    ) -> Result<ImageBuffer<Luma<U>, Vec<U>>, ConvertError>;
}

pub struct Sobel {}

impl Sobel {
    pub fn new() -> Self {
        Sobel {}
    }
}

impl EdgeDetect<u8, u8> for Sobel {
    fn apply(
        &self,
        bufr: &ImageBuffer<Luma<u8>, Vec<u8>>,
        // For now we don't use _val_num
        _val_num: u8,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ConvertError> {
        /*
         * Apply the Sobel filter on an image buffer and quantize the result
         */
        let gx = horizontal_sobel(bufr);
        let gy = vertical_sobel(bufr);
        let gx_arr = bufr_to_arr(&gx);
        let gy_arr = bufr_to_arr(&gy);
        let (w, h) = bufr.dimensions();

        let mut theta_arr_norm = Array2::zeros((h as usize, w as usize));
        Zip::from(&gx_arr)
            .and(&gy_arr)
            .and(&mut theta_arr_norm)
            .par_for_each(|&gx, &gy, theta_norm| {
                let gx_val = gx as f32;
                let gy_val = gy as f32;

                let theta = gy_val.atan2(gx_val);
                *theta_norm = (theta / PI) * 0.5 + 0.5;
            });

        let edges = theta_arr_norm.mapv(|x: f32| -> u8 {
            // Default as 0 representing space character
            let mut res = 0;
            if x == 0.5 {
                res = 0;
            } else if x >= 0.95 && x <= 1.0 {
                res = 2;
            } else if ((x >= 0.25) && (x < 0.27)) || ((x >= 0.75) && (x < 0.77)) {
                res = 1;
            } else if ((x >= 0.28) && (x < 0.55)) || ((x >= 0.78) && (x < 1.0)) {
                res = 3;
            } else if ((x >= 0.28) && (x < 0.55)) || ((x >= 0.78) && (x < 1.0)) {
                res = 4;
            }
            res
        });

        Ok(arr_to_bufr(&edges))
    }
}
