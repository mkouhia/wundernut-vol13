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

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Context};
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

/// Game state, employed in Dijkstra's algorithm binary heap
///
/// See: [Rust binary heap example](https://doc.rust-lang.org/std/collections/binary_heap/index.html)
#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    /// Current number of steps taken
    steps: usize,
    /// Current position of the hero
    hero_idx: NodeIndex,
    /// Current position of the dragon
    dragon_idx: NodeIndex,
}

impl Ord for State {
    /// Min-heap placement comparison
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .steps
            .cmp(&self.steps)
            .then_with(|| self.hero_idx.cmp(&other.hero_idx))
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
                Self::add_to_graph(y, x, &mut graph, &mut nodes, &squares, &shape)?;

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
    /// - `shape`: Shape of the `squares` array.
    fn add_to_graph(
        y: usize,
        x: usize,
        graph: &mut Graph<(usize, usize), (), Undirected>,
        nodes: &mut [Vec<Option<NodeIndex>>],
        squares: &[Vec<char>],
        shape: &(usize, usize),
    ) -> anyhow::Result<()> {
        match squares[y][x] {
            Self::S_VALID | Self::S_HERO | Self::S_DRAGON | Self::S_GOAL => {
                let node_a: NodeIndex = Self::get_or_create_node(y, x, nodes, graph);
                for (dy, dx) in [(1, 0), (0, 1)] {
                    let y1 = y + dy;
                    let x1 = x + dx;
                    if (y1 == shape.0) || (x1 == shape.1) {
                        continue;
                    }

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
            self.take_step_hero()?;
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

        let mut dist: Vec<Vec<Option<usize>>> =
            (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
        let mut prev: Vec<Vec<_>> = (0..n).map(|_| (0..n).map(|_| None).collect()).collect();

        for edge in self.graph.edge_references() {
            let u = edge.source().index();
            let v = edge.target().index();
            dist[u][v] = Some(1);
            dist[v][u] = Some(1);
            prev[u][v] = Some(edge.source());
            prev[v][u] = Some(edge.target());
        }
        for (v, _) in self.graph.node_references() {
            let v_idx = v.index();
            dist[v_idx][v_idx] = Some(0);
            prev[v_idx][v_idx] = Some(v);
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

        self.prev = Some(prev)
    }

    /// WIP: Hero takes the shortest path to the goal
    ///
    /// The path is re-evaluated at every step.
    fn take_step_hero(&mut self) -> anyhow::Result<()> {
        let path = self.hero_dijkstra_naive()?;
        self.hero_pos = path.into_iter().next().unwrap();
        Ok(())
    }

    /// Solve hero path, without taking the dragon into account
    fn hero_dijkstra_naive(&self) -> anyhow::Result<Vec<Point>> {
        let mut dist: Vec<Option<usize>> = Vec::new();
        let mut prev: Vec<Option<NodeIndex>> = Vec::new();
        let mut heap = BinaryHeap::new();

        for _ in self.graph.node_indices() {
            dist.push(None);
            prev.push(None);
        }

        let hero_idx =
            self.nodes[self.hero_pos.y][self.hero_pos.x].context("Hero node not found")?;
        let dragon_idx =
            self.nodes[self.dragon_pos.y][self.dragon_pos.x].context("Dragon node not found")?;
        let goal = self.nodes[self.goal.y][self.goal.x].context("Goal node not found")?;
        dist[hero_idx.index()] = Some(0);
        heap.push(State {
            steps: 0,
            hero_idx,
            dragon_idx,
        });

        while let Some(State {
            steps,
            hero_idx,
            dragon_idx,
        }) = heap.pop()
        {
            if hero_idx == goal {
                break;
            }

            for edge in self.graph.edges(hero_idx) {
                let next = State {
                    hero_idx: if edge.target() == hero_idx {
                        edge.source()
                    } else {
                        edge.target()
                    },
                    steps: steps + 1,
                    dragon_idx: self.dragon_step_inner(hero_idx, dragon_idx)?,
                };
                if steps > dist[hero_idx.index()].unwrap_or(usize::MAX) {
                    continue;
                }

                // Do not allow paths where dragon meets hero
                if next.dragon_idx == hero_idx {
                    continue;
                }

                let v = next.hero_idx.index();
                if next.steps < dist[v].unwrap_or(usize::MAX) {
                    heap.push(next);
                    dist[v] = Some(next.steps);
                    prev[v] = Some(hero_idx);
                }
            }
        }

        let mut path = vec![self.goal.clone()];
        let mut u_idx = goal.index();
        while let Some(u) = prev[u_idx] {
            let (y, x) = *self.graph.node_weight(u).context("Node was not in graph")?;
            u_idx = u.index();
            path.push(Point { y, x })
        }
        path.pop(); // Remove last value (current position)

        Ok(path.into_iter().rev().collect())
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
        let hero_idx = self.nodes[self.hero_pos.y][self.hero_pos.x]
            .context("Hero position not in node index")?;
        let dragon_idx = self.nodes[self.dragon_pos.y][self.dragon_pos.x]
            .context("Dragon position not in node index")?;
        let dragon_next = self.dragon_step_inner(hero_idx, dragon_idx)?;
        let (y, x) = *self
            .graph
            .node_weight(dragon_next)
            .context("Expected node address in graph")?;
        self.dragon_pos = Point { y, x };
        Ok(())
    }

    fn dragon_step_inner(
        &self,
        hero_idx: NodeIndex,
        dragon_idx: NodeIndex,
    ) -> anyhow::Result<NodeIndex> {
        let prev = self.prev.as_ref().ok_or_else(|| {
            anyhow!("Shortest paths not initialized. Please run Maze::init_shortest_paths")
        })?;

        prev[hero_idx.index()][dragon_idx.index()]
            .ok_or_else(|| anyhow!("No connection from dragon position to hero position"))
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
        maze.init_shortest_paths();

        assert_eq!(maze.hero_pos, Point { y: 1, x: 2 });
        maze.take_step_hero().unwrap();
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
        let mut maze = Maze::parse_emojis(emojis).unwrap();
        let solution = maze.solve().unwrap();
        assert_eq!(solution.hero_steps.len() - 1, 9);
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
        let mut maze = Maze::parse_emojis(emojis).unwrap();
        let solution = maze.solve().unwrap();
        assert_eq!(solution.hero_steps.len() - 1, 16);
    }
}
