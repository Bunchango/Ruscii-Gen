#[derive(Clone, Debug)]
pub struct CharacterSet {
    pub tile: Vec<char>,
    pub edge: Vec<char>,
}

impl CharacterSet {
    pub fn default() -> Self {
        CharacterSet {
            tile: vec![
                ' ', '.', ',', '*', ':', 'c', 'o', 'P', 'O', '?', '%', '&', '@',
            ],
            // For now, the edge selection is fixed until edge detection quantization code depends
            // on allowed edge characters
            edge: vec![' ', '_', '|', '/', '\\'],
        }
    }

    pub fn new(tile: &Vec<char>) -> Self {
        CharacterSet {
            tile: tile.clone(),
            edge: vec![' ', '_', '|', '/', '\\'],
        }
    }

    pub fn get_tile_mapping_size(&self) -> u8 {
        self.tile.len() as u8
    }

    pub fn get_edge_mapping_size(&self) -> u8 {
        self.edge.len() as u8
    }

    pub fn find_edge_char_index(&self, character: &char) -> Option<usize> {
        self.edge.iter().position(|&r| r == *character)
    }

    pub fn find_tile_char_index(&self, character: &char) -> Option<usize> {
        self.tile.iter().position(|&r| r == *character)
    }
}
