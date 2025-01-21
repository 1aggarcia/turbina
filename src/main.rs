use std::io::{stdin, stdout, Write};

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

fn main() {
    println!("Starting interpreter");
    let mut program = Program::new();

    // user must enter Ctrl+C to quit
    loop {
        match process_next_line(&mut program) {
            Ok(result) => println!("{result:?}"),
            Err(errors) => errors
                .iter()
                .for_each(|e| eprintln!("{e}")),
        }
    }
}

/// Read the next line from stdin and evaluate it on the program
fn process_next_line(program: &mut Program) -> Result<Literal, Vec<IntepreterError>> {
    let next_line = get_next_line().map_err(|e| vec![e])?;
    let tokens = tokenize(&next_line);
    let syntax_tree = parse(tokens).map_err(|e| vec![e])?;
    validate(program, &syntax_tree)?;
    let result = evaluate(program, &syntax_tree);

    return Ok(result);
}

/// Read the next non-empty line from stdin
fn get_next_line() -> Result<String, IntepreterError> {
    let mut buf = String::new();

    while buf.trim().is_empty() {
        print!("> ");
        self::stdout().flush().map_err(IntepreterError::io_err)?;
        stdin().read_line(&mut buf).map_err(IntepreterError::io_err)?;
    }
    return Ok(buf.trim().to_string());
}
