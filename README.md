# The shortest way out of a maze inhabited by a dragon

This program is in solution to Wunderdog [Wundernut](https://www.wunderdog.io/wundernut) vol. 13, whose instructions can be found in [their GitHub](https://github.com/wunderdogsw/wundernut-vol13).

The problem is to find a way out of a maze while avoiding a moving dragon.

## Quickstart

- If Rust is not installed, install it as per instructions in [rust-lang.org](https://www.rust-lang.org/tools/install)
- Clone repository: `git clone git@github.com:mkouhia/wundernut-vol13.git`
- With the maze in some local file, solve the maze and play back the solution with

      cargo run -- --playback <FILE>

See program help in

    cargo run -- --help


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
4. For every neighbour of the hero node in current state, we evaluate possible hero movements, and after hero movement, where the dragon would go.
5. If the dragon would meet the hero, the state is not accepted.
6. All acceptable resulting states are pushed to the heap for evaluation.
7. We loop over the possible states, taking more steps one by one until the goal is found.


### Dragon movement

The dragon always wants to go towards the hero, taking the shortest possible path.
Once the maze has been set up, the shortest paths from one square to another does not change.
Thus, we can pre-calculate the shortest paths from any square `u` to any other square `v` with [Floyd-Warshall algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm#Path_reconstruction).
Upon calculating the shortest paths, the previous steps on the path `u->v`are saved, and queried upon dragon movement.


## Development

### Unit tests

The unit tests can be run with

    cargo test

### Building

Build release version of the program with

    cargo build --release

After this, you may run the compiled binary program with

    ./target/release/wundernut-vol13
