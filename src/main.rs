use ascii::converter::Converter;

mod ascii;
mod image_manip;

fn main() {
    let converter: Converter = ascii::converter::Converter::default();
    let path = "3.png";
    let _ = converter
        .convert_img(
            &format!("test/{}", path),
            &format!("test_out/{}", path),
            0.0,
        )
        .expect("Error");
}
