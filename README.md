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
