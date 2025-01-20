use std::io::{stdin, stdout, Error, Write};
mod errors;

mod models;
use models::{Program, Token};

mod lexer;
use lexer::tokenize;

mod parser;
use parser::parse;

mod validator;
use validator::validate;

mod evaluator;
use evaluator::evaluate;

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
        let tokens: Vec<Token> = tokenize(next_line.as_str());
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
                    eprintln!("{}", error)
                }
                continue;
            }
            Ok(_) => {}
        }

        let eval_result = evaluate(&mut program, &syntax_tree);
        println!("{eval_result:?}");
    }
}

fn get_next_line() -> Result<String, Error> {
    print!("> ");
    self::stdout().flush()?;

    let mut buf = String::new();
    stdin().read_line(&mut buf)?;

    return Ok(buf.trim().to_string());
}
