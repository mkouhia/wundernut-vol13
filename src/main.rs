//! CLI for maze solving

use std::{fs, path::PathBuf};

use clap::Parser;
use wundernut_vol13::Maze;

/// The shortest way out of a maze inhabited by a dragon
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Display solution on the terminal
    #[arg(short, long)]
    playback: bool,

    /// Playback frame length in milliseconds
    #[arg(short, long, default_value_t = 300)]
    frame_length: usize,

    /// File, where to read the maze
    file: PathBuf,
}

/// Read maze from file, print output
fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let emojis: String = fs::read_to_string(args.file)?;
    let maze = Maze::parse_emojis(emojis.trim())?;
    let solution = maze.solve()?;
    if args.playback {
        maze.playback(&solution, args.frame_length);
    }
    solution.print_report();
    Ok(())
}
