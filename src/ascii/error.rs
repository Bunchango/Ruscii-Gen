use ab_glyph::InvalidFont;
use image::ImageError;
use ndarray::ShapeError;
use std::io::Error;

#[derive(Debug)]
pub enum ConvertError {
    ImageError,
    FileError,
    NdArrayShapeError,
    InvalidFont,
}

impl From<ImageError> for ConvertError {
    fn from(_: ImageError) -> Self {
        ConvertError::ImageError
    }
}

impl From<ShapeError> for ConvertError {
    fn from(_: ShapeError) -> Self {
        ConvertError::NdArrayShapeError
    }
}

impl From<InvalidFont> for ConvertError {
    fn from(_: InvalidFont) -> Self {
        ConvertError::InvalidFont
    }
}

impl From<Error> for ConvertError {
    fn from(_: Error) -> Self {
        ConvertError::FileError
    }
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::ImageError => write!(f, "Failed processing image"),
            ConvertError::FileError => write!(f, "Failed reading file"),
            ConvertError::NdArrayShapeError => write!(
                f,
                "Failed converting image to array with given shape or layout"
            ),
            ConvertError::InvalidFont => write!(f, "Failed reading font data"),
        }
    }
}

impl std::error::Error for ConvertError {}
