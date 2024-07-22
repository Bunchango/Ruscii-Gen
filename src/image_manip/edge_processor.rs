use ndarray::{s, Array2};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    usize,
};

/*
* Sharpen edge when downscale images
*/
pub struct EdgeDownscaler {}

impl EdgeDownscaler {
    pub fn hist_downscale(
        qt_edge_arr: &Array2<u8>,
        tile_size: usize,
        thres_ratio: f32,
        new_size: (usize, usize), // new_h , new_w
    ) -> Array2<u8> {
        let ds_edge_arr = Arc::new(Mutex::new(Array2::zeros(new_size)));

        (0..new_size.0).into_par_iter().for_each(|i| {
            (0..new_size.1).into_par_iter().for_each(|j| {
                // Create a local histogram of tile
                let mut hist: HashMap<u8, usize> = HashMap::new();
                let mut max_val = 0; // To store the maximum occurring value
                let mut max_count = 0; // To store the count of the maximum occurring value

                // Get a tile
                let tile = qt_edge_arr.slice(s![
                    (i * tile_size) as usize..((i + 1) * tile_size) as usize,
                    (j * tile_size) as usize..((j + 1) * tile_size) as usize
                ]);

                for &value in tile.iter() {
                    *hist.entry(value).or_insert(0) += 1;

                    // Update max_value and max_count if current value has a higher count
                    if hist[&value] > max_count {
                        max_count = hist[&value];
                        max_val = value;
                    }
                }

                // Check if the number of edge values in the histogram pass the threshold
                if hist.values().filter(|&&x| x != 0).count() as f32
                    / (tile_size * tile_size) as f32
                    >= thres_ratio
                {
                    // Set the value in ds_edge_arr to the value that occurs most in the histogram
                    let mut ds_edge_arr = ds_edge_arr.lock().unwrap();
                    ds_edge_arr[(i as usize, j as usize)] = max_val as u8;
                }
            });
        });

        Arc::try_unwrap(ds_edge_arr).unwrap().into_inner().unwrap()
    }
}
