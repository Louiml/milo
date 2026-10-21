//! `mpm` — the Milo package manager (command-line entry point).

use std::process::ExitCode;

mod cli;

fn main() -> ExitCode {
    cli::run(std::env::args().skip(1).collect())
}
