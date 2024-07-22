use super::error::ConvertError;
use ab_glyph::{FontVec, PxScale};
use std::fs;

#[derive(Debug, Clone)]
pub struct FontSettings {
    pub font_size: u32,
    pub font_path: String,
}

impl FontSettings {
    pub fn new(font_size: u32, font_path: &str) -> Self {
        FontSettings {
            font_size,
            font_path: font_path.to_string(),
        }
    }

    pub fn default() -> Self {
        FontSettings {
            // 4 is the optimal and smallest font size for displaying the characters correctly
            // with the default settings
            font_size: 4,
            font_path: "font.ttf".to_string(),
        }
    }
}

pub struct FontLoader {}

impl FontLoader {
    pub fn load_font_from_settings(
        settings: &FontSettings,
    ) -> Result<(FontVec, PxScale), ConvertError> {
        // Load font data
        let font_dat = fs::read(&settings.font_path)?;
        let font = match FontVec::try_from_vec(font_dat) {
            Ok(ft) => ft,
            Err(_) => {
                // Attempt fallback to default font if custom font fails
                let fallback_font_dat = fs::read("font.ttf")?;
                FontVec::try_from_vec(fallback_font_dat)?
            }
        };
        let scale = PxScale::from(settings.font_size as f32);

        Ok((font, scale))
    }
}
