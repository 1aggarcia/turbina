use std::io::BufReader;
use std::fs::File;

use clap::Parser;

use turbina::{run_as_file, run_repl};
use turbina::streams::{FileStream, OutputStreams};

#[derive(Parser)]
struct CliArgs {
    #[arg(required = false)]
    path: Option<std::path::PathBuf>,
}

fn main() {
    let args = CliArgs::parse();
    // TODO: end-to-end tests reading and checking stdout results
    let Some(filename) = args.path else {
        run_repl();
        return;
    };
    match File::open(filename) {
        Ok(file) => {
            let reader = BufReader::new(file);
            let file_stream = Box::new(FileStream { reader });
            let result = run_as_file(file_stream, OutputStreams::std_streams());
            if let Err(err) = result {
                eprintln!("{err}");
            }
        },
        Err(err) => eprintln!("{err}"),
    }
}
