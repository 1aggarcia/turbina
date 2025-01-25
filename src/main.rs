use std::io::{stdin, stdout, BufRead, BufReader, Write};
use std::env;
use std::fs::File;

mod errors;
mod models;
mod lexer;
mod parser;
mod validator;
mod evaluator;

use errors::IntepreterError;
use models::{Literal, Program};
use lexer::tokenize;
use parser::parse;
use validator::validate;
use evaluator::evaluate;

/// Abstraction over source code input
enum InputStream {
    File(BufReader<File>),
    Stdin,
}

impl InputStream {
    fn next_line(&mut self) -> Result<String, IntepreterError> {
        let mut buf: String = String::new();

        // keep reading until a non-empty line is found
        while buf.trim().is_empty() {
            let bytes_read = if let Self::File(reader) = self {
                reader.read_line(&mut buf)
            } else {
                print!("> ");
                self::stdout().flush().map_err(IntepreterError::io_err)?;
                stdin().read_line(&mut buf)
            }.map_err(IntepreterError::io_err)?;

            if bytes_read == 0 {
                return Err(IntepreterError::EndOfFile);
            }
        }
        return Ok(buf.trim().to_string());
    }
}

fn main() {
    let arg = env::args().nth(1);
    let mut input_stream = match arg {
        Some(filename) => {
            let file = File::open(filename).expect("Bad filepath");
            let reader = BufReader::new(file);
            InputStream::File(reader)
        },
        None => InputStream::Stdin,
    };

    println!("Starting interpreter");
    let mut program = Program::new();
    loop {
        match process_next_line(&mut program, &mut input_stream) {
            Ok(result) => println!("{result:?}"),
            Err(errors) => {
                if errors.contains(&IntepreterError::EndOfFile) {
                    break;
                }
                errors.iter().for_each(|e| eprintln!("{e}"));
                // the REPL can keep executing, but source files cannot
                if let InputStream::File(_) = input_stream {
                    break;
                }
            }
        };
    }
}

/// Read the next line from the input stream and evaluate it on the program
fn process_next_line(
    program: &mut Program,
    input_stream: &mut InputStream
) -> Result<Literal, Vec<IntepreterError>> {
    let next_line = input_stream.next_line().map_err(|e| vec![e])?;
    let tokens = tokenize(&next_line).map_err(|e| vec![e])?;
    let syntax_tree = parse(tokens).map_err(|e| vec![e])?;
    validate(program, &syntax_tree)?;
    let result = evaluate(program, &syntax_tree);

    return Ok(result);
}
