use image::{ImageBuffer, Luma, Primitive};
use ndarray::{Array, Array2};
use num_traits::Num;

pub fn bufr_to_arr<T: Num + Copy + 'static + Primitive>(
    bufr: &ImageBuffer<Luma<T>, Vec<T>>,
) -> Array2<T> {
    let (w, h) = bufr.dimensions();
    let raw = bufr.clone().into_raw(); // If performance is paramount, then don't clone
    Array::from_shape_vec((h as usize, w as usize), raw).unwrap()
}

pub fn arr_to_bufr<T: Copy + Num + 'static + Primitive>(
    arr: &Array2<T>,
) -> ImageBuffer<Luma<T>, Vec<T>> {
    let (h, w) = arr.dim();
    let raw: Vec<T> = arr.iter().cloned().collect();
    ImageBuffer::from_raw(w as u32, h as u32, raw).unwrap()
}
