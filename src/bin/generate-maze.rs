//! CLI for maze solving

use clap::Parser;
use itertools::Itertools;
use wundernut_vol13::maze_generator::MazeGenerator;

/// Map generator for Wundernut vol. 13
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Generated field height
    #[arg(long, default_value_t = 19)]
    height: usize,

    /// Generated field width
    #[arg(long, default_value_t = 15)]
    width: usize,

    /// Random seed
    #[arg(long)]
    seed: Option<u64>,
}

/// Read maze from file, print output
fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut gen = MazeGenerator::new(args.seed);
    let res: Vec<Vec<char>> = gen.generate_maze(args.height, args.width);
    println!("{}", res.iter().map(|row| row.iter().join("")).join("\n"));
    Ok(())
}
