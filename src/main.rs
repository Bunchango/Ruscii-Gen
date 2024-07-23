use ascii::converter::Converter;

mod ascii;
mod image_manip;
use std::time::Instant;

fn main() {
    let start = Instant::now();
    let converter: Converter = ascii::converter::Converter::default();
    let path = "4.png";
    let _ = converter
        .convert_img(
            &format!("test/{}", path),
            &format!("test_out/{}", path),
            0.0,
        )
        .expect("Error");
    let duration = start.elapsed();
    println!("Produced ascii art in {:?}", duration);
}
