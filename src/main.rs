use std::io::BufReader;
use std::fs::File;
use clap::Parser;

use turbina::rustyline::RustylineArgs;
use turbina::{run_as_file, run_repl, CliArgs};
use turbina::streams::{FileStream, OutputStreams};

fn main() {
    let args = RustylineArgs::parse();
    let cli_args = CliArgs {
        path: args.path,
        disable_type_checker: args.disable_type_checker,
    };
    let Some(filename) = cli_args.path.clone() else {
        run_repl(cli_args);
        return;
    };
    match File::open(filename) {
        Ok(file) => {
            let reader = BufReader::new(file);
            let file_stream = Box::new(FileStream { reader });
            let result = run_as_file(
                file_stream,
                OutputStreams::std_streams(),
                cli_args 
            );
            if let Err(err) = result {
                eprintln!("{err}");
            }
        },
        Err(err) => eprintln!("{err}"),
    }
}
