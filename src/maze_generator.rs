//! Map generation

use rand::{rngs::StdRng, Rng, SeedableRng};

/// Maze generator for additional maps.
pub struct MazeGenerator {
    hero: char,
    goal: char,
    valid: char,
    wall: char,
    dragon: char,
}

impl Default for MazeGenerator {
    fn default() -> Self {
        Self {
            hero: '🏃',
            goal: '❎',
            valid: '🟩',
            wall: '🟫',
            dragon: '🐉',
        }
    }
}

impl MazeGenerator {
    /// Generate simple imperfet maze (maze with loops)
    ///
    /// Inspired by https://github.com/Yassineelg/maze_runner/
    pub fn generate_maze(&self, height: usize, width: usize, seed: Option<u64>) -> Vec<Vec<char>> {
        let mut grid: Vec<Vec<char>> = (0..height)
            .map(|_| (0..width).map(|_| self.wall).collect())
            .collect();

        let mut random = if let Some(state) = seed {
            StdRng::seed_from_u64(state)
        } else {
            StdRng::from_entropy()
        };

        // Open odd cells
        for i in (0..height).step_by(2) {
            for j in (0..width).step_by(2) {
                grid[i][j] = self.valid;

                // Open neighboring cells horizontally or vertically
                if random.gen_bool(0.5) {
                    if i > 0 {
                        grid[i - 1][j] = self.valid;
                    }
                    if i < height - 1 {
                        grid[i + 1][j] = self.valid;
                    }
                } else {
                    if j > 0 {
                        grid[i][j - 1] = self.valid;
                    }
                    if j < width - 1 {
                        grid[i][j + 1] = self.valid;
                    }
                }
            }
        }

        grid[0][random.gen_range(1..width - 1)] = self.hero;
        grid[height - 1][random.gen_range(1..width - 1)] = self.goal;
        let mut d_pos = random.gen_range(((width * height) / 2)..((width * height) * 3 / 4));
        while grid[d_pos / height][d_pos % width] != self.valid {
            d_pos -= 1;
        }
        grid[d_pos / height][d_pos % width] = self.dragon;

        grid
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::maze_generator::MazeGenerator;

    #[test]
    fn get_some() {
        let gen = MazeGenerator::default();
        let res = gen.generate_maze(16, 16);
        println!("{}", res.iter().map(|row| row.iter().join("")).join("\n"));

        assert!(false);
    }
}
