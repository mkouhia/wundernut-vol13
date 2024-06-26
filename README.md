# The shortest way out of a maze inhabited by a dragon

This program is in solution to Wunderdog [Wundernut](https://www.wunderdog.io/wundernut) vol. 13, whose instructions can be found in [their GitHub](https://github.com/wunderdogsw/wundernut-vol13).

The problem is to find a way out of a maze while avoiding a moving dragon.

## Quickstart

- If Rust is not installed, install it as per instructions in [rust-lang.org](https://www.rust-lang.org/tools/install)
- Clone the repository and enter the cloned directory

      git clone git@github.com:mkouhia/wundernut-vol13.git
      cd wundernut-vol13

- Compile the program with

      cargo build --release

  You will now find the executable in `./target/release/solve-maze`, or if on Windows, `./target/release/solve-maze.exe`. In following examples, replace program name with the correct path to the built executable, or copy the executable to a convenient location.

- With the maze in some local file, solve the maze with

      solve-maze [OPTIONS] <FILE>

- _Fun factor:_ solve and display the hero's journey in the maze:

      solve-maze --playback <FILE>

  [solve-maze-example1.webm](https://github.com/mkouhia/wundernut-vol13/assets/1469093/23a9fed2-088a-4c8b-b3c7-5357f388b910)

If necessary, consult the program help in

    solve-maze --help

```
Solution to Wundernut vol. 13

Usage: solve-maze [OPTIONS] <FILE>

Arguments:
  <FILE>  File, where to read the maze. Use `-` for stdin

Options:
  -p, --playback                     Display solution on the terminal
  -f, --frame-length <FRAME_LENGTH>  Playback frame length in milliseconds [default: 300]
  -h, --help                         Print help
  -V, --version                      Print version
```

### Extras (additional fun factor)
Additional feature `mapgen` will can generate more maps for an increased fun factor. Build with feature `mapgen` to create another binary `generate-maze`:

      cargo build --features mapgen --release

Then you can generate additional mazes with 

    generate-maze [OPTIONS]

To generate a maze and play back results, perform

    generate-maze | solve-maze -p -

  [generate-and-solve-maze.webm](https://github.com/mkouhia/wundernut-vol13/assets/1469093/cfd23422-a921-4882-8bae-b038b80e65e9)


## Implementation

### Hero movement

As per the problem rules, the hero is required to take the _shortest path_ to the exit, while avoiding the dragon.
This problem statement implies, that we need to find the optimal solution.
This problem setup can be rephrased as finding the shortest paths between nodes in a [graph](https://en.wikipedia.org/wiki/Graph_(abstract_data_type)), and one well-known algorithm to solve this proble is the [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm). In our case, there are some excemptions and additions:
- The hero and the dragon can take steps in any cardinal direction, and the step distance from one tile to another is always one. Thus, the graph in _undirected_ and all _edge weights_ are equal to one.
- The dragon is not allowed to catch the hero. Thus, we need additional checks in the algorithm to weed out those branches.

In the implemented algorithm, the tiled map is first structured as graph, where each accessible tile is represented as a node. Each possible hero movement is recorded as `State`, which contains the hero position, dragon position and the number of steps taken until that step.

The search for the optimal path starts at the hero location.
1. Structs representing the steps to the neighbouring nodes are placed in a binary heap, where the structs are ordered by increasing cost.
2. The lowest cost struct is removed from the heap for evaluation. This way, we are always addressing the shortest path up until now.
3. We check if the hero position in the current state is the goal. If it is, we just found the shortest path to the goal.
4. If hero has already visited this node with a lower cost, s.t. dragon is at the same position, we skip this branch.
5. For every neighbour of the hero node in current state, we evaluate possible hero movements, and after hero movement, where the dragon would go.
6. If the dragon would meet the hero, the state is not accepted.
7. All acceptable resulting states are pushed to the heap for evaluation.
8. We loop over the possible states, taking more steps one by one until the goal is found.

Hero back-steps are allowed, to enable solving of such situations, where the hero cannot always go forward:

> ```
> 🟫🟫🟫🟫🟫🟫
> 🟫🏃🟩🟩🟫🟫
> 🟫🟩🟫🟩🟫🟫
> 🟫🟩🟩🐉🟩❎
> 🟫🟫🟫🟫🟫🟫
> ```
> Optimal solution: hero takes either step to the right or down, dragon comes to meet, hero steps back and then continues around the circle in the opposite direction to the first step.

This is implemented in the Dijkstra's algorithm, by having current best distance `dist` and best previous step to the node `prev` to be indexed by the hero position _and_ the dragon position. This increases the number of possible solutions quite heavily, but is required for these edge cases.

### Dragon movement

The dragon always wants to go towards the hero, taking the shortest possible path.
Once the maze has been set up, the shortest paths from one square to another does not change.
Thus, we can pre-calculate the shortest paths from any square `u` to any other square `v` with [Floyd-Warshall algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm#Path_reconstruction).
Upon calculating the shortest paths, the previous steps on the path `u->v`are saved, and queried upon dragon movement.

**Special case**: If there is no viable path from the dragon position to the hero position, the dragon does not awake. It will remain sleeping in its current position.

### General implementation notes
- The algorithm is designed for small to medium sized, quite closed maps. Since back-steps for the hero are allowed, the number of possible solutions increases rapidly with the map size. For open maps, where back-steps for the hero are not required, reqular Dijkstra `dist` and `prev` with only hero position would be better.
- Rust is selected as the implementation language to provide solutions efficiently. Current implementation is still single-threaded. Multi-threading can be implemented to improve solution times.
- If the maze would have a square type with single access direction, that could be handled with directional edges.
- Different terrain types could be handled by introducing edge weights.
- The underlying graph may be simplified by converting long passageways to just two nodes with longer edge in between. This would reduce solution time for the optimal path, but converting to hero/dragon steps requires more attention.
- Non-rectangular grids, or other tile shapes (hexagonal, mixed shapes) can be implemented using the same graph-based approach. New map parsing methods would need to be introduced, and the `Point` struct revised.
- For the problem solving algorithm, meeting a dragon and path to the goal not existing are equivalent: there is no feasible path to the end. The cases are differentiated by running the same algorithm without the dragon constraint. If a solution is found, then the dragon was guilty.
    - Because this is a family friendly implementation, no playback is available for the cases that the dragon would slay the hero.

## Development

### Unit tests

The unit tests can be run with

    cargo test

### Building

Build release version of the program with

    cargo build --release

After this, you may run the compiled binary program with

    ./target/release/solve-maze

When developing, the program may also be run with

    cargo run

### Documentation

You can build the developer documentation with 

    cargo doc

### Code quality

Code must be formatted with `rustfmt`:

    rustfmt src/*.rs

And quality checked with static analyzer `cargo clippy`:

    cargo clippy

An easy way to examine test coverage is to install additional cargo command with `cargo install cargo-llvm-cov` and then run

    cargo llvm-cov --open

## License

This project is licensed under the terms of the MIT license.
