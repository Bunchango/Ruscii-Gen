use super::util::{arr_to_bufr, bufr_to_arr};
use crate::ascii::error::ConvertError;
use image::{ImageBuffer, Luma, Primitive};
use imageproc::contrast::{threshold, ThresholdType};
use imageproc::filter::{
    bilateral_filter, gaussian_blur_f32, median_filter, sharpen3x3, sharpen_gaussian,
};
use num_traits::Num;

pub trait Processor<T: Num + Copy + Primitive, U: Copy + Num + Primitive> {
    fn apply(
        &self,
        bufr: &ImageBuffer<Luma<T>, Vec<T>>,
    ) -> Result<ImageBuffer<Luma<U>, Vec<U>>, ConvertError>;
}

#[derive(Clone, Debug)]
pub struct DoG {
    pub sigma_1: f32,
    pub sigma_2: f32,
}

impl DoG {
    pub fn default() -> Self {
        DoG {
            sigma_1: 1.0,
            sigma_2: 3.5,
        }
    }

    pub fn new(sigma_1: f32, sigma_2: f32) -> Self {
        DoG { sigma_1, sigma_2 }
    }
}

impl Processor<u8, u8> for DoG {
    fn apply(
        &self,
        bufr: &ImageBuffer<Luma<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ConvertError> {
        /*
         * Apply difference of gaussian on an image buffer.
         * This function accepts only u8 because in this situation, it needs to be applied on a
         * grayscaled image
         */
        let blur_1 = gaussian_blur_f32(bufr, self.sigma_1);
        let blur_2 = gaussian_blur_f32(bufr, self.sigma_2);

        let blur_1_arr = bufr_to_arr(&blur_1).mapv(|x| x as i32);
        let blur_2_arr = bufr_to_arr(&blur_2).mapv(|x| x as i32);

        let dog_arr = &blur_2_arr - &blur_1_arr;

        // Convert to u8 and handle saturation
        Ok(arr_to_bufr(&dog_arr.mapv(|x| x.max(0).min(255) as u8)))
    }
}

pub struct MedianBlur {
    pub kernel_size: u32,
}

impl MedianBlur {
    pub fn default() -> Self {
        MedianBlur { kernel_size: 2 }
    }

    pub fn new(kernel_size: u32) -> Self {
        MedianBlur { kernel_size }
    }
}

impl Processor<u8, u8> for MedianBlur {
    fn apply(
        &self,
        bufr: &ImageBuffer<Luma<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ConvertError> {
        Ok(median_filter(bufr, self.kernel_size, self.kernel_size))
    }
}

pub struct BilateralFilter {
    pub window_size: u32,
    pub sigma_color: f32,
    pub sigma_spatial: f32,
}

impl BilateralFilter {
    pub fn default() -> Self {
        BilateralFilter {
            window_size: 10,
            sigma_color: 2.0,
            sigma_spatial: 5.0,
        }
    }

    pub fn new(window_size: u32, sigma_color: f32, sigma_spatial: f32) -> Self {
        BilateralFilter {
            window_size,
            sigma_color,
            sigma_spatial,
        }
    }
}

impl Processor<u8, u8> for BilateralFilter {
    fn apply(
        &self,
        bufr: &ImageBuffer<Luma<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ConvertError> {
        Ok(bilateral_filter(
            bufr,
            self.window_size,
            self.sigma_color,
            self.sigma_spatial,
        ))
    }
}

pub struct Threshold {
    pub threshold: u8,
}

impl Threshold {
    pub fn new(threshold: u8) -> Self {
        Threshold { threshold }
    }

    pub fn default() -> Self {
        Threshold { threshold: 10 }
    }
}

impl Processor<u8, u8> for Threshold {
    fn apply(
        &self,
        bufr: &ImageBuffer<Luma<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ConvertError> {
        Ok(threshold(
            bufr,
            self.threshold,
            ThresholdType::ToZeroInverted,
        ))
    }
}

pub struct Sharpen3x3 {}

impl Sharpen3x3 {
    pub fn new() -> Self {
        Sharpen3x3 {}
    }
}

impl Processor<u8, u8> for Sharpen3x3 {
    fn apply(
        &self,
        bufr: &ImageBuffer<Luma<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ConvertError> {
        Ok(sharpen3x3(bufr))
    }
}

pub struct SharpenGaussian {
    pub sigma: f32,
    pub amount: f32,
}

impl SharpenGaussian {
    pub fn default() -> Self {
        SharpenGaussian {
            sigma: 1.0,
            amount: 1.0,
        }
    }

    pub fn new(sigma: f32, amount: f32) -> Self {
        SharpenGaussian { sigma, amount }
    }
}

impl Processor<u8, u8> for SharpenGaussian {
    fn apply(
        &self,
        bufr: &ImageBuffer<Luma<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, ConvertError> {
        Ok(sharpen_gaussian(bufr, self.sigma, self.amount))
    }
}
