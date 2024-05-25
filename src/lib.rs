//! Find a way out of the maze inhabited by a dragon
//!
//! # Examples
//! ## Example maze 1 (the shortest path is 16)
//! ```
//! use wundernut_vol13::Maze;
//!
//! let maze_emojis = "
//! 🟫🏃🟫🟫🟫🟫🟫
//! 🟫🟩🟩🟩🟩🟩🟫
//! 🟫🟩🟫🟫🟫🟩🟫
//! 🟫🟩🟩🟩🟫🟩🟫
//! 🟫🐉🟫🟩🟫🟩🟫
//! 🟫🟩🟩🟩🟫🟩🟫
//! 🟫🟩🟫🟫🟫🟩🟫
//! 🟫🟩🟩🟩🟩🟩🟫
//! 🟫❎🟫🟫🟫🟫🟫";
//! let maze = Maze::parse_emojis(maze_emojis.trim()).unwrap();
//! // let solution = maze.solve().unwrap();
//! // solution.print_report();
//! // assert_eq!(solution.shortest_path, 16)
//! ```
//!
//! ## Example maze 2
//! ```
//! use wundernut_vol13::Maze;
//!
//! let maze_emojis = "
//! 🟫🟫🟫🟫🟩🟩🏃🟫🟫🟩🟩🟩
//! 🟩🟩🟩🟫🟩🟫🟫🟫🟫🟩🟫🟩
//! 🟩🟫🟩🟫🟩🟩🟩🟩🟩🟩🟫🟩
//! 🟩🟫🟩🟫🟫🟫🟫🟫🟫🟩🟫🟩
//! 🟩🟫🟩🟩🟩🟩🟩🟩🟫🟩🟫🟩
//! 🟩🟫🟫🟫🟫🟫🟫🟩🟫🟩🟫🟩
//! 🟩🟩🟩🟩🟩🟩🟩🟩🟫🟩🟫🟩
//! 🟩🟫🟩🟫🟫🟫🟫🟫🟫🟩🟫🟩
//! 🟩🟫🟩🐉🟩🟩🟩🟩🟩🟩🟫🟩
//! 🟩🟫🟫🟫🟫🟫🟫🟫🟫🟩🟫🟩
//! 🟩🟩🟫🟫🟫🟩❎🟫🟩🟩🟩🟩
//! 🟫🟩🟩🟩🟩🟩🟫🟩🟩🟫🟫🟩";
//! let maze = Maze::parse_emojis(maze_emojis.trim()).unwrap();
//! // let solution = maze.solve().unwrap();
//! // solution.print_report();
//! ```

use anyhow::anyhow;

/// Location in the maze
#[derive(PartialEq, Debug)]
struct Point {
    y: usize,
    x: usize,
}

/// Representation of the hero-dragon maze
pub struct Maze {
    /// Hero starting position
    hero_start: Point,
    /// Dragon starting position
    dragon_start: Point,
    /// Location of the final target
    goal: Point,
    /// Allowed access squares
    nodes: Vec<Point>,
}

/// Solution to the maze
pub struct MazeSolution {
    /// Shortest path from start to finish
    pub shortest_path: usize,
}

impl Maze {
    const S_HERO: char = '🏃';
    const S_GOAL: char = '❎';
    const S_VALID: char = '🟩';
    const S_WALL: char = '🟫';
    const S_DRAGON: char = '🐉';

    /// Parse maze representation from string
    ///
    ///
    /// - `emojis`: Emoji representation of the maze, as per Wundernut
    ///   instructions.
    ///
    /// Returns error, if maze contains unknown characters.
    ///
    /// # Examples
    /// ```
    /// use wundernut_vol13::Maze;
    /// let maze_emojis = "
    /// 🟫🏃🟫🟫🟫🟫🟫
    /// 🟫🟩🟩🟩🟩🟩🟫
    /// 🟫🟩🟫🟫🟫🟩🟫
    /// 🟫🟩🟩🟩🟫🟩🟫
    /// 🟫🐉🟫🟩🟫🟩🟫
    /// 🟫🟩🟩🟩🟫🟩🟫
    /// 🟫🟩🟫🟫🟫🟩🟫
    /// 🟫🟩🟩🟩🟩🟩🟫
    /// 🟫❎🟫🟫🟫🟫🟫";
    /// let maze = Maze::parse_emojis(maze_emojis.trim());
    /// ```
    pub fn parse_emojis(emojis: &str) -> anyhow::Result<Self> {
        let mut hero_start = None;
        let mut dragon_start = None;
        let mut goal = None;

        let mut nodes = Vec::new();
        for (y, row) in emojis.split('\n').enumerate() {
            let mut row_nodes = Vec::new();
            for (x, c) in row.chars().enumerate() {
                match c {
                    Self::S_VALID => row_nodes.push(Point { y, x }),
                    Self::S_HERO => {
                        hero_start = Some(Point { y, x });
                        row_nodes.push(Point { y, x });
                    }
                    Self::S_DRAGON => {
                        dragon_start = Some(Point { y, x });
                        row_nodes.push(Point { y, x })
                    }
                    Self::S_GOAL => {
                        goal = Some(Point { y, x });
                        row_nodes.push(Point { y, x })
                    }
                    Self::S_WALL => (),
                    val => {
                        return Err(anyhow!(format!(
                            "Unexpected character `{}` at x={}, y={}",
                            val, x, y
                        )))
                    }
                }
            }

            nodes.push(row_nodes);
        }

        Ok(Maze {
            hero_start: hero_start.ok_or_else(|| anyhow!("Hero not found in maze"))?,
            dragon_start: dragon_start.ok_or_else(|| anyhow!("Dragon not found in maze"))?,
            goal: goal.ok_or_else(|| anyhow!("Goal not found in maze"))?,
            nodes: nodes.into_iter().flatten().collect(),
        })
    }

    /// Solve maze
    ///
    /// Find the shortest path that the hero can take to reach the exit,
    /// without being caught by the dragon.
    pub fn solve(&self) -> anyhow::Result<MazeSolution> {
        Err(anyhow!("Not implemented"))
    }
}

impl MazeSolution {
    /// Print report
    pub fn print_report(&self) {
        println!("The shortest path is {} steps.", self.shortest_path)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Maze, Point};

    #[test]
    fn foo() {
        let emojis = "
🟫🏃🟫🟫🟫🟫🟫
🟫🟩🟩🟩🟩🟩🟫
🟫🟩🟫🟫🟫🟩🟫
🟫🟩🟩🟩🟫🟩🟫
🟫🐉🟫🟩🟫🟩🟫
🟫🟩🟩🟩🟫🟩🟫
🟫🟩🟫🟫🟫🟩🟫
🟫🟩🟩🟩🟩🟩🟫
🟫❎🟫🟫🟫🟫🟫";
        let maze = Maze::parse_emojis(emojis.trim()).unwrap();

        let expected = vec![
            Point { y: 0, x: 1 },
            Point { y: 1, x: 1 },
            Point { y: 1, x: 2 },
            Point { y: 1, x: 3 },
            Point { y: 1, x: 4 },
            Point { y: 1, x: 5 },
            Point { y: 2, x: 1 },
            Point { y: 2, x: 5 },
            Point { y: 3, x: 1 },
            Point { y: 3, x: 2 },
            Point { y: 3, x: 3 },
            Point { y: 3, x: 5 },
            Point { y: 4, x: 1 },
            Point { y: 4, x: 3 },
            Point { y: 4, x: 5 },
            Point { y: 5, x: 1 },
            Point { y: 5, x: 2 },
            Point { y: 5, x: 3 },
            Point { y: 5, x: 5 },
            Point { y: 6, x: 1 },
            Point { y: 6, x: 5 },
            Point { y: 7, x: 1 },
            Point { y: 7, x: 2 },
            Point { y: 7, x: 3 },
            Point { y: 7, x: 4 },
            Point { y: 7, x: 5 },
            Point { y: 8, x: 1 },
        ];
        assert_eq!(maze.nodes, expected);
        assert_eq!(maze.hero_start, Point { y: 0, x: 1 });
        assert_eq!(maze.dragon_start, Point { y: 4, x: 1 });
        assert_eq!(maze.goal, Point { y: 8, x: 1 });
    }
}
