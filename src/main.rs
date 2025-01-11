use std::io::{stdin, stdout, Write, Error};
mod models;
use models::TokenV2;

mod lexer;
use lexer::tokenize;

mod parser;
use parser::parse;

mod errors;

fn main() {
    println!("Starting interpreter");

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
        let syntax_tree = parse(tokens);
        match syntax_tree {
            Ok(tree) => println!("{:?}", tree),
            Err(err) => println!("{err}"),
        }
        // TODO: typecheck and evaluate
    }
}

fn get_next_line() -> Result<String, Error> {
    print!("> ");
    self::stdout().flush()?;

    let mut buf = String::new();
    stdin().read_line(&mut buf)?;

    return Ok(buf.trim().to_string());
}
