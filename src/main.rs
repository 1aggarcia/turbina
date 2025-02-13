use std::io::BufReader;
use std::env;
use std::fs::File;

use turbina::{run_as_file, run_repl};
use turbina::streams::{FileStream, OutputStreams};

fn main() {
    // TODO: end-to-end tests reading and checking stdout results
    let Some(filename) = env::args().nth(1) else {
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
