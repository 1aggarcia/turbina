use std::io::{stdin, stdout, BufRead, BufReader, Write};
use std::env;
use std::fs::File;

mod errors;
mod models;
mod lexer;
mod parser;
mod validator;
mod evaluator;
mod library;

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

        // re-read if the last line is empty or a comment
        while buf.trim().is_empty() || buf.starts_with("//") {
            buf.clear();
            let bytes_read = if let Self::File(reader) = self {
                reader.read_line(&mut buf)?
            } else {
                print!("> ");
                self::stdout().flush()?;
                stdin().read_line(&mut buf)?
            };

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
    let mut program = Program::init();
    loop {
        // TODO: type-check complete file before evaluating
        match process_next_line(&mut program, &mut input_stream) {
            Ok(result) => println!("{result}"),
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
/// Returns the value produced by evaluating the line
fn process_next_line(
    program: &mut Program,
    input_stream: &mut InputStream
) -> Result<String, Vec<IntepreterError>> {
    // TODO: end-to-end tests to show working bindings across multiple evaluations
    let next_line = input_stream.next_line()?;
    let tokens = tokenize(&next_line)?;
    let syntax_tree = parse(tokens)?;

    let tree_type = validate(program, &syntax_tree)?;
    if let Some(name) = &tree_type.name_to_bind {
        program.type_context.insert(name.clone(), tree_type.datatype.clone());
    }

    let output = match evaluate(program, &syntax_tree) {
        // Function types look nicer to print than anything inside the function
        Literal::Func(_) => tree_type.datatype.to_string(),
        literal => literal.to_string(),
    };
    Ok(output)
}
