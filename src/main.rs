use std::io::BufReader;
use std::fs::File;
use clap::Parser;

use turbina::{run_as_file, run_repl, CliArgs};
use turbina::streams::{FileStream, OutputStreams};

fn main() {
    let args = CliArgs::parse();
    // TODO: end-to-end tests reading and checking stdout results
    let Some(filename) = args.path.clone() else {
        run_repl(args);
        return;
    };
    match File::open(filename) {
        Ok(file) => {
            let reader = BufReader::new(file);
            let file_stream = Box::new(FileStream { reader });
            let result = run_as_file(
                file_stream,
                OutputStreams::std_streams(),
               args 
            );
            if let Err(err) = result {
                eprintln!("{err}");
            }
        },
        Err(err) => eprintln!("{err}"),
    }
}
