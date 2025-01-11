use std::io::{stdin, stdout, Error, Write};
mod models;
use models::{Program, TokenV2};

mod lexer;
use lexer::tokenize;

mod parser;
use parser::parse;
use validation::validate;
mod errors;
mod validation;

fn main() {
    println!("Starting interpreter");
    let mut program = Program::new();

    // user must enter Ctrl+C to quit
    loop {
        // read user input
        let next_line = match get_next_line() {
            Ok(line) => line,
            Err(err) => {
                eprintln!("{err}");
                continue;
            }
        };

        // handle input
        let tokens: Vec<TokenV2> = tokenize(next_line.as_str());
        if tokens.is_empty() {
            continue;
        }
        let syntax_tree = match parse(tokens) {
            Ok(tree) => tree,
            Err(err) => {
                eprintln!("{err}");
                continue;
            },
        };

        let validation_result = validate(&program, &syntax_tree);
        match validation_result {
            Err(errors) => {
                for error in errors {
                    eprintln!("Error: {}", error)
                }
                continue;
            }
            Ok(_) => {}
        }

        // TODO: evaluate tree
        println!("{syntax_tree:?}");
    }
}

fn get_next_line() -> Result<String, Error> {
    print!("> ");
    self::stdout().flush()?;

    let mut buf = String::new();
    stdin().read_line(&mut buf)?;

    return Ok(buf.trim().to_string());
}
