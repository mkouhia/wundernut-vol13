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
//! let mut maze = Maze::parse_emojis(maze_emojis.trim()).unwrap();
//! let solution = maze.solve().unwrap();
//! solution.print_report();
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
//! let mut maze = Maze::parse_emojis(maze_emojis.trim()).unwrap();
//! let solution = maze.solve().unwrap();
//! solution.print_report();
//! ```

use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Context};
use itertools::Itertools;
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, IntoNodeReferences};
use petgraph::{Graph, Undirected};

/// Location in the maze
#[derive(PartialEq, Clone, Debug)]
pub struct Point {
    y: usize,
    x: usize,
}

/// Representation of the hero-dragon maze
pub struct Maze {
    /// Original layout of the problem
    squares: Vec<Vec<char>>,
    /// Location of the final target
    goal: Point,
    /// Node indices
    nodes: Vec<Vec<Option<NodeIndex>>>,
    /// Graph of the vertices between nodes
    ///
    /// Weights of the nodes are the original (y, x) coordinates
    graph: Graph<(usize, usize), (), Undirected>,

    /// Current hero position
    hero_pos: Point,
    /// Current dragon position
    dragon_pos: Point,

    /// Number of steps that the hero has taken
    current_step: usize,

    /// Previous steps on the shortest path (u->v)
    ///
    /// This is initialized by [Self::floyd_warshall]
    prev: Option<Vec<Vec<Option<NodeIndex>>>>,
}

/// Solution to the maze
pub struct MazeSolution {
    /// The steps that the hero took, including start & end
    pub hero_steps: Vec<Point>,
    /// The steps that the dragon took, including start & end
    pub dragon_steps: Vec<Point>,
    /// Game status
    pub ending_condition: EndingCondition,
}

/// How the game ended
pub enum EndingCondition {
    /// Hero reached goal
    GOAL,
    /// Dragon reached hero
    FAIL,
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

        let squares: Vec<Vec<char>> = emojis
            .split('\n')
            .map(|row| row.chars().collect())
            .collect();

        let mut graph = Graph::new_undirected();
        let shape = (squares.len(), squares[0].len());
        let mut nodes: Vec<Vec<Option<NodeIndex>>> = (0..shape.0)
            .map(|_| (0..shape.1).map(|_| None).collect())
            .collect();

        for (y, row) in squares.iter().enumerate() {
            for (x, c) in row.iter().enumerate() {
                if (y < shape.0 - 1) && (x < shape.1 - 1) {
                    Self::add_to_graph(y, x, &mut graph, &mut nodes, &squares)?
                }

                // Find special squares
                match *c {
                    Self::S_HERO => hero_start = Some(Point { y, x }),
                    Self::S_DRAGON => {
                        dragon_start = Some(Point { y, x });
                    }
                    Self::S_GOAL => {
                        goal = Some(Point { y, x });
                    }
                    _ => (),
                }
            }
        }

        let hero_pos = hero_start.ok_or_else(|| anyhow!("Hero is not found in maze"))?;
        let dragon_pos = dragon_start.ok_or_else(|| anyhow!("Dragon is not found in maze"))?;
        Ok(Maze {
            squares,
            hero_pos,
            dragon_pos,
            nodes,
            graph,
            goal: goal.ok_or_else(|| anyhow!("Goal not found in maze"))?,
            // nodes: nodes.into_iter().flatten().collect(),
            current_step: 0,
            prev: None,
        })
    }

    /// Add nodes (x, y) and edges (y, x) <--> (y1, x1) to graph.
    ///
    /// Process only positive delta x and delta y, because graph is undirected.
    ///
    /// ## Arguments
    /// - `y`: Current y position.
    /// - `x`: Current x position.
    /// - `graph`: Graph that we are building.
    /// - `nodes`: Node matrix (y*x), where node indices are inserted.
    /// - `squares`: Original character array.
    fn add_to_graph(
        y: usize,
        x: usize,
        graph: &mut Graph<(usize, usize), (), Undirected>,
        nodes: &mut [Vec<Option<NodeIndex>>],
        squares: &[Vec<char>],
    ) -> anyhow::Result<()> {
        match squares[y][x] {
            Self::S_VALID | Self::S_HERO | Self::S_DRAGON | Self::S_GOAL => {
                let node_a: NodeIndex = Self::get_or_create_node(y, x, nodes, graph);
                for (dy, dx) in [(1, 0), (0, 1)] {
                    let y1 = y + dy;
                    let x1 = x + dx;
                    match squares[y1][x1] {
                        Self::S_VALID | Self::S_HERO | Self::S_DRAGON | Self::S_GOAL => {
                            let node_b = Self::get_or_create_node(y1, x1, nodes, graph);
                            graph.add_edge(node_a, node_b, ());
                        }
                        Self::S_WALL => (), // Cannot access the other square
                        val => {
                            return Err(anyhow!(format!(
                                "Unexpected character `{}` at y={}, x={}",
                                val, x, y
                            )))
                        }
                    }
                }
            }
            Self::S_WALL => (), // Could not access this square
            val => {
                return Err(anyhow!(format!(
                    "Unexpected character `{}` at y={}, x={}",
                    val, x, y
                )))
            }
        }
        Ok(())
    }

    /// Get node index from `nodes` array, or create new node.
    ///
    /// Nodes are created in `graph` and resulting indices inserted into
    /// `nodes`.
    fn get_or_create_node(
        y: usize,
        x: usize,
        nodes: &mut [Vec<Option<NodeIndex>>],
        graph: &mut Graph<(usize, usize), (), Undirected>,
    ) -> NodeIndex {
        if let Some(node) = nodes[y][x] {
            node
        } else {
            let node = graph.add_node((y, x));
            nodes[y][x] = Some(node);
            node
        }
    }

    /// Solve maze
    ///
    /// Find the shortest path that the hero can take to reach the exit,
    /// without being caught by the dragon.
    pub fn solve(&mut self) -> anyhow::Result<MazeSolution> {
        self.init_shortest_paths();

        let mut hero_steps = vec![self.hero_pos.clone()];
        let mut dragon_steps = vec![self.dragon_pos.clone()];
        let ending_condition = loop {
            self.take_step_hero();
            self.current_step += 1;
            hero_steps.push(self.hero_pos.clone());

            if self.hero_pos == self.goal {
                break EndingCondition::GOAL;
            }

            self.take_step_dragon()?;
            dragon_steps.push(self.dragon_pos.clone());

            if self.dragon_pos == self.hero_pos {
                break EndingCondition::FAIL;
            }
        };
        Ok(MazeSolution {
            hero_steps,
            dragon_steps,
            ending_condition,
        })
    }

    /// Run Floyd-Warshall algorithm for determining all shortest paths
    ///
    /// The algorithm will find the shortest path for any node combinations
    /// (u, v). As a result, the method will populate [Self::prev] with
    /// a 2D vec, containing the _penultimate step_ on the path from node
    /// `u` towards node `v`.
    ///
    /// See more: [Wikipedia](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm#Path_reconstruction)
    fn init_shortest_paths(&mut self) {
        let n = self.graph.node_indices().len();

        let mut dist: Vec<Vec<usize>> = (0..n)
            .map(|_| (0..n).map(|_| usize::MAX).collect())
            .collect();
        let mut prev: Vec<Vec<_>> = (0..n).map(|_| (0..n).map(|_| None).collect()).collect();

        for edge in self.graph.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            dist[u][v] = 1;
            dist[v][u] = 1;
            prev[u][v] = Some(edge.source());
            prev[v][u] = Some(edge.target());
        }
        for (v, _) in self.graph.node_references() {
            let v_idx = v.index();
            dist[v_idx][v_idx] = 0;
            prev[v_idx][v_idx] = Some(v);
        }

        for k in 0..n {
            for i in 0..n {
                let v2 = dist[i][k];
                for j in 0..n {
                    if (dist[k][j] == usize::MAX) | (v2 == usize::MAX) {
                    } else if dist[i][j] > v2 + dist[k][j] {
                        dist[i][j] = v2 + dist[k][j];
                        prev[i][j] = prev[k][j]
                    };
                }
            }
        }

        self.prev = Some(prev)
    }

    fn take_step_hero(&mut self) {
        // panic!("Not implemented")
    }

    /// Dragon movement is simple shortest-path algorithm to hero position
    ///
    /// Employ Floyd-Warshall algorithm at the start to find the shortest
    /// paths from each square to another; at runtime dragon will follow
    /// those paths.
    ///
    /// Prior to calling this method, [Self::init_shortest_paths] shall
    /// be performed.
    fn take_step_dragon(&mut self) -> anyhow::Result<()> {
        let u = self.nodes[self.hero_pos.y][self.hero_pos.x]
            .context("Hero position not in node index")?
            .index();
        let v = self.nodes[self.dragon_pos.y][self.dragon_pos.x]
            .context("Dragon position not in node index")?
            .index();

        let prev = self.prev.as_ref().ok_or_else(|| {
            anyhow!("Shortest paths not initialized. Please run Maze::init_shortest_paths")
        })?;

        if let Some(dragon_next) = prev[u][v] {
            let (y, x) = *self
                .graph
                .node_weight(dragon_next)
                .context("Expected node address in graph")?;
            self.dragon_pos = Point { y, x };

            Ok(())
        } else {
            bail!("No connection from dragon position to hero position");
        }
    }

    /// Print solution to console
    ///
    /// ## Arguments
    /// - `solution`: Solution to the maze.
    /// - `step_ms`: Time step for each frame, milliseconds.
    pub fn playback(&self, solution: &MazeSolution, step_ms: usize) {
        let hero_swaps = solution.hero_steps.windows(2).collect::<Vec<_>>();
        let dragon_swaps = solution.dragon_steps.windows(2).collect::<Vec<_>>();
        let all_swaps = itertools::interleave(hero_swaps, dragon_swaps);

        fn print_squares(squares: &[Vec<char>]) {
            print!("\x1B[2J\x1B[1;1H");
            let sq_str = squares.iter().map(|row| row.iter().join("")).join("\n");
            println!("{}", sq_str);
        }

        let mut squares = self.squares.clone();
        print_squares(&squares);

        for swap in all_swaps {
            thread::sleep(Duration::from_millis(step_ms as u64));

            let sq0 = squares[swap[0].y][swap[0].x];
            let sq1 = squares[swap[1].y][swap[1].x];

            // Special cases on the swap, when target is not `S_VALID`
            let (sq0, sq1) = match (sq0, sq1) {
                // If dragon slays hero, dragon replaces hero with S_VALID
                (Self::S_DRAGON, Self::S_HERO) => (sq0, Self::S_VALID),
                // If hero reaches goal, hero replaces goal with S_VALID
                (Self::S_HERO, Self::S_GOAL) => (sq0, Self::S_VALID),
                _ => (sq0, sq1),
            };

            squares[swap[0].y][swap[0].x] = sq1;
            squares[swap[1].y][swap[1].x] = sq0;

            print_squares(&squares);
        }
    }
}

impl MazeSolution {
    /// Print report
    pub fn print_report(&self) {
        match self.ending_condition {
            EndingCondition::GOAL => {
                println!("The shortest path is {} steps.", self.hero_steps.len() - 1)
            }
            EndingCondition::FAIL => {
                println!(
                    "The dragon slayed the hero after {} steps.",
                    self.dragon_steps.len() - 1
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Maze, Point};

    #[test]
    fn parse_maze_input() {
        let emojis = "
🟫🏃🟫🟫🟫🟫🟫
🟫🟩🟩🟩🟩🟩🟫
🟫🟩🟫🟫🟫🟩🟫
🟫🟩🟩🟩🟫🟩🟫
🟫🐉🟫🟩🟫🟩🟫
🟫🟩🟩🟩🟫🟩🟫
🟫🟩🟫🟫🟫🟩🟫
🟫🟩🟩🟩🟩🟩🟫
🟫❎🟫🟫🟫🟫🟫"
            .trim();
        let maze = Maze::parse_emojis(emojis).unwrap();

        assert_eq!(maze.hero_pos, Point { y: 0, x: 1 });
        assert_eq!(maze.dragon_pos, Point { y: 4, x: 1 });
        assert_eq!(maze.goal, Point { y: 8, x: 1 });

        assert_eq!(
            maze.nodes
                .iter()
                .flatten()
                .filter(|n| n.is_some())
                .collect::<Vec<_>>()
                .len(),
            27
        );
        assert_eq!(maze.graph.edge_indices().len(), 28)
    }

    #[ignore = "not implemented"]
    #[test]
    fn hero_take_step() {
        let emojis = "
🟫🟫🟫🟫🟫
🟫🟩🏃🟩❎
🟫🐉🟫🟩🟫
🟫🟩🟩🟩🟫
🟫🟫🟫🟫🟫"
            .trim();
        let mut maze = Maze::parse_emojis(emojis).unwrap();
        println!("{:?}\n{:?}", maze.nodes, maze.graph);

        assert_eq!(maze.hero_pos, Point { y: 1, x: 2 });
        maze.take_step_hero();
        assert_eq!(maze.hero_pos, Point { y: 1, x: 3 });
    }

    #[test]
    fn dragon_goes_directly_towards_hero() {
        let emojis = "
🟫🟫🟫🟫🟫
🟫🟩🟩🏃❎
🟫🐉🟫🟩🟫
🟫🟩🟩🟩🟫
🟫🟫🟫🟫🟫"
            .trim();
        let mut maze = Maze::parse_emojis(emojis).unwrap();
        assert_eq!(maze.dragon_pos, Point { y: 2, x: 1 });

        maze.init_shortest_paths();

        maze.take_step_dragon().unwrap();
        assert_eq!(maze.dragon_pos, Point { y: 1, x: 1 });

        maze.take_step_dragon().unwrap();
        assert_eq!(maze.dragon_pos, Point { y: 1, x: 2 });

        maze.take_step_dragon().unwrap();
        assert_eq!(maze.dragon_pos, Point { y: 1, x: 3 });

        maze.take_step_dragon().unwrap();
        assert_eq!(maze.dragon_pos, Point { y: 1, x: 3 });
    }

    #[test]
    fn dragon_changes_direction_if_necessary() {
        let emojis = "
🟫🟫🟫🟫🟫
🟫🟩🏃🟩❎
🟫🟩🟫🟩🟫
🟫🐉🟩🟩🟫
🟫🟫🟫🟫🟫"
            .trim();
        let mut maze = Maze::parse_emojis(emojis).unwrap();
        assert_eq!(maze.dragon_pos, Point { y: 3, x: 1 });

        maze.init_shortest_paths();

        maze.take_step_dragon().unwrap();
        assert_eq!(maze.dragon_pos, Point { y: 2, x: 1 });

        // Move hero to another position (not according to actual rules); dragon adjusts
        maze.hero_pos = Point { y: 3, x: 3 };

        maze.take_step_dragon().unwrap();
        assert_eq!(maze.dragon_pos, Point { y: 3, x: 1 });
    }

    #[ignore = "not implemented"]
    #[test]
    fn sample() {
        let emojis = "
🟫🏃🟫🟫🟫🟫🟫
🟫🟩🟩🟩🟩🟩🟫
🟫🟩🟫🟫🟫🟩🟫
🟫🟩🟩🟩🟫🟩🟫
🟫🐉🟫🟩🟫🟩🟫
🟫🟩🟩🟩🟫🟩🟫
🟫🟩🟫🟫🟫🟩🟫
🟫🟩🟩🟩🟩🟩🟫
🟫❎🟫🟫🟫🟫🟫"
            .trim();
        let mut maze = Maze::parse_emojis(emojis).unwrap();
        let solution = maze.solve().unwrap();
        assert_eq!(solution.shortest_path, 16);
    }

    #[ignore = "not implemented"]
    #[test]
    fn do_not_go_towards_dragon() {
        let emojis = "
🟫🏃🟫🟫🟫🟫🟫
🟫🟩🟩🟩🟩🟩🟫
🟫🟩🟫🟫🟫🟩🟫
🟫🟩🟩🟩🟫🟩🟫
🟫🟩🟫🟩🟫🟩🟫
🟫🐉🟩🟩🟫🟩🟫
🟫🟩🟫🟫🟫🟩🟫
🟫🟩🟩🟩🟩🟩🟫
🟫❎🟫🟫🟫🟫🟫"
            .trim();
        let mut maze = Maze::parse_emojis(emojis).unwrap();
        let solution = maze.solve().unwrap();
        assert_eq!(solution.shortest_path, 16);
    }
}
