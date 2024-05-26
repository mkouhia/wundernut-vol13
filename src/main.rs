//! CLI for maze solving

use std::{env, fs};

use wundernut_vol13::Maze;

/// Read maze from file, print output
fn main() -> anyhow::Result<()> {
    let mut args = env::args();
    let _program_name = args.next();
    let file_arg = args.next();

    if let Some(emoji_file) = file_arg {
        let emojis: String = fs::read_to_string(emoji_file)?;
        let mut maze = Maze::parse_emojis(emojis.trim())?;
        let solution = maze.solve()?;
        solution.print_report();
    } else {
        println!("Provide emoji file name as argument to the program.")
    }

    Ok(())
}
