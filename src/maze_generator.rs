//! Map generation

use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

/// Maze generator for additional maps.
pub struct MazeGenerator {
    random: StdRng,
}

impl MazeGenerator {
    const DIRECTIONS: [(i32, i32); 4] = [(0, 2), (2, 0), (0, -2), (-2, 0)];
    const S_HERO: char = '🏃';
    const S_GOAL: char = '❎';
    const S_VALID: char = '🟩';
    const S_WALL: char = '🟫';
    const S_DRAGON: char = '🐉';

    pub fn new(seed: Option<u64>) -> Self {
        Self {
            random: if let Some(state) = seed {
                StdRng::seed_from_u64(state)
            } else {
                StdRng::from_entropy()
            },
        }
    }

    /// Generate simple imperfet maze (maze with loops)
    ///
    /// Inspired by https://github.com/Yassineelg/maze_runner/
    pub fn generate_maze(&mut self, height: usize, width: usize) -> Vec<Vec<char>> {
        let mut grid: Vec<Vec<char>> = (0..height)
            .map(|_| (0..width).map(|_| Self::S_WALL).collect())
            .collect();

        // Generate random starting position in an odd cell
        let start_x = 1 + self.random.gen_range(1..(width / 2 - 1)) * 2;
        let start_y = 1 + self.random.gen_range(1..(height / 2 - 1)) * 2;
        grid[start_y][start_x] = Self::S_VALID;

        self.build_maze(&mut grid, start_x, start_y, width, height);

        // Add hero, goal and dragon
        if height >= width {
            grid[1][self.random.gen_range(1..width - 1)] = Self::S_HERO;
            grid[height - 2][self.random.gen_range(1..width - 1)] = Self::S_GOAL;
        } else {
            grid[self.random.gen_range(1..height - 1)][1] = Self::S_HERO;
            grid[self.random.gen_range(1..height - 1)][width - 2] = Self::S_GOAL;
        }
        let mut d_pos = self
            .random
            .gen_range(((width * height) / 2)..((width * height) * 3 / 4));
        while grid[d_pos / width][d_pos % width] != Self::S_VALID {
            d_pos -= 1;
        }
        grid[d_pos / width][d_pos % width] = Self::S_DRAGON;

        grid
    }

    /// Build maze recursively
    ///
    /// From current position, go into random directions. Carve out walls
    /// if there is wall begind carved area (or at random, skip this check).
    /// This randomness allows creation of imperfect mazes.
    fn build_maze(
        &mut self,
        grid: &mut Vec<Vec<char>>,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) {
        let mut directions = Self::DIRECTIONS.to_vec();
        directions.shuffle(&mut self.random);

        for (dx, dy) in directions {
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;

            if nx < width
                && ny < height
                && (grid[ny][nx] == Self::S_WALL || self.random.gen_bool(0.05))
            {
                // Remove wall between current cell and neighbor
                grid[(y as i32 + dy / 2) as usize][(x as i32 + dx / 2) as usize] = Self::S_VALID;
                grid[ny][nx] = Self::S_VALID;

                // Recurse to continue generating maze
                self.build_maze(grid, nx, ny, width, height);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::{maze_generator::MazeGenerator, Maze};

    #[test]
    fn generate_parseable_emoji_maze() {
        let mut gen = MazeGenerator::new(Some(0));
        let res = gen.generate_maze(15, 15);

        let emojis = res.iter().map(|row| row.iter().join("")).join("\n");

        let parsed_result = Maze::parse_emojis(&emojis);
        assert!(parsed_result.is_ok());
    }
}
