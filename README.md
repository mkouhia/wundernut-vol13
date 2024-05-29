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

### Extras (additional fun factor)
Additional feature `mapgen` will can generate more maps for an increased fun factor. Build with feature `mapgen` to create another binary `generate-maze`:

      cargo build --features mapgen --release

Then you can generate additional mazes with 

    generate-maze [OPTIONS]

To generate a maze and play back results, perform

    generate-maze | solve-maze -p -

  [generate-and-solve-maze.webm](https://github.com/mkouhia/wundernut-vol13/assets/1469093/a65ba354-da9d-424a-88fd-5c66eaaf9ef1)


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


### Dragon movement

The dragon always wants to go towards the hero, taking the shortest possible path.
Once the maze has been set up, the shortest paths from one square to another does not change.
Thus, we can pre-calculate the shortest paths from any square `u` to any other square `v` with [Floyd-Warshall algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm#Path_reconstruction).
Upon calculating the shortest paths, the previous steps on the path `u->v`are saved, and queried upon dragon movement.

**Special case**: If there is no viable path from the dragon position to the hero position, the dragon does not awake. It will remain sleeping in its current position.

### General implementation notices
- If the maze would have a square type with single access direction, that could be handled with directional edges.
- Different terrain types could be handled by introducing edge weights.
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

## License

This project is licensed under the terms of the MIT license.
