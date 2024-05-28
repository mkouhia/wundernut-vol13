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
//! let solution = maze.solve().unwrap();
//! solution.print_report();
//! assert_eq!(solution.hero_positions.len(), 16 + 1, "Hero traveled 16 steps + original position")
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
//! let solution = maze.solve().unwrap();
//! solution.print_report();
//! ```

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::thread;
use std::time::Duration;

use anyhow::anyhow;
use itertools::Itertools;

/// Location in the maze
#[derive(PartialEq, Clone, Debug)]
pub struct Point {
    y: usize,
    x: usize,
}

/// Representation of the hero-dragon maze
#[derive(Debug)]
pub struct Maze {
    /// Original layout of the problem
    squares: Vec<Vec<char>>,

    /// Hero starting position
    hero_start: Point,
    /// Dragon starting position
    dragon_start: Point,
    /// Location of the final target
    goal: Point,

    /// Graph of the vertices between nodes
    ///
    /// The original layout is deconstructed to a graph, with which
    /// the routing problems are solved.
    graph: Graph,
}

/// Solution to the maze
#[derive(PartialEq, Debug)]
pub struct MazeSolution {
    /// The steps that the hero took, including start & end
    pub hero_positions: Vec<Point>,
    /// The steps that the dragon took, including start & end
    pub dragon_positions: Vec<Point>,
    /// Game status
    pub ending_condition: EndingCondition,
}

/// How the game ended
#[derive(PartialEq, Debug)]
pub enum EndingCondition {
    /// Hero reached goal
    GOAL,
    /// Dragon reached hero
    FAIL,
}

/// Graph representation
#[derive(PartialEq, Debug)]
struct Graph {
    /// The edges are stored as adjacency list representation
    ///
    /// First vec index is node `u` index, (usize), the inner vecs contain
    /// `v` index values
    edges: Vec<Vec<usize>>,
    /// Node indices
    ///
    /// Each node is represented as usize, and the 2D coordinates in the
    /// original grid are stored here.
    nodes: Vec<Point>,
}

impl Graph {
    /// Get node index from `nodes` array, or create new node.
    ///
    /// Nodes are created in `graph` and resulting indices inserted into
    /// `nodes`.
    fn get_or_create_node(&mut self, point: &Point) -> usize {
        if let Some(idx) = self.get_node(point) {
            idx
        } else {
            let j = self.nodes.len();
            self.nodes.push(point.clone());
            j
        }
    }

    /// Get node from graph
    fn get_node(&self, point: &Point) -> Option<usize> {
        self.nodes
            .iter()
            .enumerate()
            .find_map(|(i, p)| if p == point { Some(i) } else { None })
    }

    /// Add undirected edge between nodes u, v
    ///
    /// Actually this is just (u->v) and (v->u)
    fn add_edge_undirected(&mut self, u: usize, v: usize) {
        self.add_edge_directed(u, v);
        self.add_edge_directed(v, u);
    }

    /// Add directed edge between nodes u, v
    fn add_edge_directed(&mut self, u: usize, v: usize) {
        while self.edges.len() <= u {
            self.edges.push(Vec::new());
        }
        self.edges[u].push(v);
    }

    /// Run Floyd-Warshall algorithm for determining all shortest paths
    ///
    /// The algorithm will find the shortest path for any node combinations
    /// (u, v).
    ///
    /// ## Returns
    /// Positions as a 2D vec, containing the _penultimate step_ on the
    /// path from node `u` towards node `v`. The vec values are node
    /// indices on [Self::nodes]
    ///
    /// See more: [Wikipedia](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm#Path_reconstruction)
    fn get_shortest_path_steps(&self) -> Vec<Vec<Option<usize>>> {
        let n = self.nodes.len();

        let mut dist: Vec<Vec<Option<usize>>> =
            (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
        let mut prev: Vec<Vec<_>> = (0..n).map(|_| (0..n).map(|_| None).collect()).collect();

        for (u, edges) in self.edges.iter().enumerate() {
            for &v in edges {
                dist[u][v] = Some(1);
                dist[v][u] = Some(1);
                prev[u][v] = Some(u);
                prev[v][u] = Some(v);
            }
        }
        for v in 0..self.nodes.len() {
            dist[v][v] = Some(0);
            prev[v][v] = Some(v);
        }

        for k in 0..n {
            for i in 0..n {
                if let Some(dist_ik) = dist[i][k] {
                    for j in 0..n {
                        if let Some(dist_kj) = dist[k][j] {
                            if dist[i][j].unwrap_or(usize::MAX) > dist_ik + dist_kj {
                                dist[i][j] = Some(dist_ik + dist_kj);
                                prev[i][j] = prev[k][j]
                            };
                        }
                    }
                }
            }
        }

        prev
    }
}

/// Game state, employed in Dijkstra's algorithm binary heap
///
/// See: [Rust binary heap example](https://doc.rust-lang.org/std/collections/binary_heap/index.html)
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct State {
    /// Current number of steps taken
    steps: usize,
    /// Current position of the hero
    hero_node: usize,
    /// Current position of the dragon
    dragon_node: usize,
}

impl Ord for State {
    /// Min-heap placement comparison
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .steps
            .cmp(&self.steps)
            .then_with(|| self.hero_node.cmp(&other.hero_node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Maze {
    const S_HERO: char = '🏃';
    const S_GOAL: char = '❎';
    const S_VALID: char = '🟩';
    const S_WALL: char = '🟫';
    const S_DRAGON: char = '🐉';

    /// Parse maze representation from string
    ///
    /// ## Arguments
    /// - `emojis`: Emoji representation of the maze, as per Wundernut
    ///   instructions.
    ///
    /// ## Errors
    /// Returns error, if parsing fails:
    /// - maze contains unknown characters
    /// - maze does not contain hero, dragon and goal.
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

        let mut graph = Graph {
            edges: vec![],
            nodes: vec![],
        };
        let shape = (squares.len(), squares[0].len());

        for (y, row) in squares.iter().enumerate() {
            for (x, c) in row.iter().enumerate() {
                let point = Point { y, x };
                Self::add_to_graph(&point, &mut graph, &squares, &shape)?;

                // Find special squares
                match *c {
                    Self::S_HERO => hero_start = Some(point),
                    Self::S_DRAGON => {
                        dragon_start = Some(point);
                    }
                    Self::S_GOAL => {
                        goal = Some(point);
                    }
                    _ => (),
                }
            }
        }

        Ok(Maze {
            squares,
            hero_start: hero_start.ok_or_else(|| anyhow!("Hero is not found in maze"))?,
            dragon_start: dragon_start.ok_or_else(|| anyhow!("Dragon is not found in maze"))?,
            goal: goal.ok_or_else(|| anyhow!("Goal not found in maze"))?,
            graph,
        })
    }

    /// Add nodes (x, y) and edges (y, x) <--> (y1, x1) to graph.
    ///
    /// Process only positive delta x and delta y, because graph is undirected.
    ///
    /// ## Arguments
    /// - `point`: Current y, x position.
    /// - `graph`: Graph that we are building.
    /// - `squares`: Original character array.
    /// - `shape`: Shape of the `squares` array.
    fn add_to_graph(
        point: &Point,
        graph: &mut Graph,
        squares: &[Vec<char>],
        shape: &(usize, usize),
    ) -> anyhow::Result<()> {
        match squares[point.y][point.x] {
            Self::S_VALID | Self::S_HERO | Self::S_DRAGON | Self::S_GOAL => {
                let node_a = graph.get_or_create_node(point);
                for (dy, dx) in [(1, 0), (0, 1)] {
                    let y1 = point.y + dy;
                    let x1 = point.x + dx;
                    if (y1 == shape.0) || (x1 == shape.1) {
                        continue;
                    }

                    let point_b = Point { y: y1, x: x1 };
                    match squares[y1][x1] {
                        Self::S_VALID | Self::S_HERO | Self::S_DRAGON | Self::S_GOAL => {
                            let node_b = graph.get_or_create_node(&point_b);
                            graph.add_edge_undirected(node_a, node_b);
                        }
                        Self::S_WALL => (), // Cannot access the other square
                        val => {
                            return Err(anyhow!(format!(
                                "Unexpected character `{}` at {:?}",
                                val, point
                            )))
                        }
                    }
                }
            }
            Self::S_WALL => (), // Could not access this square
            val => {
                return Err(anyhow!(format!(
                    "Unexpected character `{}` at {:?}",
                    val, point
                )))
            }
        }
        Ok(())
    }

    /// Solve maze
    ///
    /// Find the shortest path that the hero can take to reach the exit,
    /// without being caught by the dragon.
    ///
    /// Employ [Dijkstra's algorithm][1], modified with simultaneous
    /// dragon movement. After each path evaluation, the dragon also
    /// moves. If dragon reaches the hero, the path is discarded.
    ///
    /// # Returns
    /// Solution, with hero positions and dragon positions.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
    pub fn solve(&self) -> anyhow::Result<MazeSolution> {
        let states = self.solve_hero_shortest_path()?;

        let (hero_positions, mut dragon_positions): (Vec<_>, Vec<_>) = states
            .iter()
            .map(|s| {
                (
                    self.graph.nodes[s.hero_node].clone(),
                    self.graph.nodes[s.dragon_node].clone(),
                )
            })
            .unzip();

        let ending_condition = match hero_positions.last() {
            Some(last) if *last == self.goal => EndingCondition::GOAL,
            _ => EndingCondition::FAIL,
        };

        // If hero reached their goal, remove ultimate dragon movement
        if ending_condition == EndingCondition::GOAL {
            dragon_positions.pop();
        }

        Ok(MazeSolution {
            hero_positions,
            dragon_positions,
            ending_condition,
        })
    }

    /// Solve hero path with modified Dijkstra's algorithm
    ///
    /// First, solve optimal dragon steps with Floyd-Warshall.
    /// Then, run Dijkstra's algorithm for hero.
    ///
    /// ## Returns
    /// Vec of [State] objects, which represent what had happened
    /// under the hero`s journey, including the start and end.
    /// Dragon movement after hero reaches the goal is included in the
    /// last state.
    ///
    /// # Errors
    /// If no connection from the dragon position to the hero position
    /// was found.
    fn solve_hero_shortest_path(&self) -> anyhow::Result<Vec<State>> {
        let dragon_prev: Vec<Vec<Option<usize>>> = self.graph.get_shortest_path_steps();

        let mut dist: Vec<Option<usize>> = Vec::new();
        let mut prev = Vec::new();
        let mut heap = BinaryHeap::new();

        for _ in &self.graph.nodes {
            dist.push(None);
            prev.push(None);
        }

        let hero_node = self
            .graph
            .get_node(&self.hero_start)
            .expect("Hero node shall be in graph");
        let dragon_node = self
            .graph
            .get_node(&self.dragon_start)
            .expect("Dragon node shall be in graph");
        let goal_node = self
            .graph
            .get_node(&self.goal)
            .expect("Goal node shall be in graph");
        dist[hero_node] = Some(0);
        let mut outer_state = State {
            steps: 0,
            hero_node,
            dragon_node,
        };
        heap.push(outer_state);

        while let Some(state) = heap.pop() {
            if state.hero_node == goal_node {
                outer_state = state; // Store goal state
                break;
            }

            for &v in &self.graph.edges[state.hero_node] {
                let next = State {
                    hero_node: v,
                    steps: state.steps + 1,
                    dragon_node: self.dragon_step_inner(
                        state.hero_node,
                        state.dragon_node,
                        &dragon_prev,
                    )?,
                };

                // Do not allow paths where dragon meets hero
                if next.dragon_node == state.hero_node {
                    continue;
                }

                if next.steps < dist[v].unwrap_or(usize::MAX) {
                    heap.push(next);
                    dist[v] = Some(next.steps);
                    prev[v] = Some(state);
                }
            }
        }

        // Transfer prev statements recursively to the beginning
        let mut path = vec![outer_state];
        let mut u = goal_node;
        while let Some(state) = prev[u] {
            u = state.hero_node;
            path.push(state)
        }
        path.reverse();

        Ok(path)
    }

    /// Take step on the shortest path from the dragon to the hero
    ///
    /// Actually, find the penultimate position on the path from the hero
    /// position to the dragon position, utilizing Floyd-Warshall path
    /// reconstruction.
    fn dragon_step_inner(
        &self,
        hero_node: usize,
        dragon_node: usize,
        dragon_prev: &[Vec<Option<usize>>],
    ) -> anyhow::Result<usize> {
        dragon_prev[hero_node][dragon_node]
            .ok_or_else(|| anyhow!("No connection from dragon position to hero position"))
    }

    /// Print solution to console
    ///
    /// Clear screen, display animation with the same characters as in
    /// the original puzzle. Swap characters around the map with each
    /// movement.
    ///
    /// ## Arguments
    /// - `solution`: Solution to the maze.
    /// - `step_ms`: Time step for each frame, milliseconds.
    pub fn playback(&self, solution: &MazeSolution, step_ms: usize) {
        let hero_swaps = solution.hero_positions.windows(2).collect::<Vec<_>>();
        let dragon_swaps = solution.dragon_positions.windows(2).collect::<Vec<_>>();
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
                // If hero walks to the dragon, dragon replaces hero with S_VALID
                (Self::S_HERO, Self::S_DRAGON) => (sq1, Self::S_VALID),
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
                println!(
                    "The shortest path is {} steps.",
                    self.hero_positions.len() - 1
                )
            }
            EndingCondition::FAIL => {
                println!(
                    "The dragon slayed the hero after {} steps.",
                    self.dragon_positions.len() - 1
                )
            }
        }
    }
}

/// Unit tests
#[cfg(test)]
mod tests {
    use crate::{EndingCondition, Maze, MazeSolution, Point};

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

        assert_eq!(maze.hero_start, Point { y: 0, x: 1 });
        assert_eq!(maze.dragon_start, Point { y: 4, x: 1 });
        assert_eq!(maze.goal, Point { y: 8, x: 1 });

        let edge_pairs: Vec<(usize, usize)> = maze
            .graph
            .edges
            .iter()
            .enumerate()
            .flat_map(|(u, v_vec)| v_vec.iter().map(|&v| (u, v)).collect::<Vec<_>>())
            .collect();

        assert_eq!(maze.graph.nodes.len(), 27);
        assert_eq!(
            edge_pairs.len(),
            28 * 2,
            "Edges contain all (u->v) and (v->u) pairs"
        )
    }

    #[test]
    fn check_simple_solution() {
        let emojis = "
🟫🟫🟫🟫🟫
🟫🟩🏃🟩❎
🟫🐉🟫🟩🟫
🟫🟩🟩🟩🟫
🟫🟫🟫🟫🟫"
            .trim();
        let maze = Maze::parse_emojis(emojis).unwrap();
        let solution = maze.solve().unwrap();
        let expected = MazeSolution {
            hero_positions: vec![
                Point { y: 1, x: 2 },
                Point { y: 1, x: 3 },
                Point { y: 1, x: 4 },
            ],
            dragon_positions: vec![
                Point { y: 2, x: 1 },
                Point { y: 1, x: 1 },
                // This is one step shorter, because the hero reached the goal.
            ],
            ending_condition: EndingCondition::GOAL,
        };

        assert_eq!(solution, expected);
    }

    #[test]
    fn around_the_edges() {
        let emojis = "
🟩🟩🟩🟩🟩
🟩🟫🟫🟫🏃
🐉🟩🟩🟫🟩
🟩🟫🟩🟫🟩
🟩🟩🟩🟫🟩
🟩🟫🟫🟫🟩
❎🟩🟩🟩🟩"
            .trim();
        let maze = Maze::parse_emojis(emojis).unwrap();
        let solution = maze.solve().unwrap();
        assert_eq!(solution.hero_positions.len() - 1, 9);
    }

    #[test]
    fn example_maze_1() {
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
        let solution = maze.solve().unwrap();
        assert_eq!(solution.hero_positions.len() - 1, 16);
    }
}
