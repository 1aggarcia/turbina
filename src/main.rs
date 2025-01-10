use std::io::{stdin, stdout, Write};
mod models;
use models::TokenV2;

mod lexer;
use lexer::tokenize;

mod parser;
use parser::parse;

fn main() {
    println!("Starting interpreter");

    // user must enter Ctrl+C to quit
    loop {
        // get next line from user
        print!("> ");
        match stdout().flush() {
            Err(err) =>{
                println!("{err}");
                continue;
            },
            _ => {}
        }

        let mut buf = String::new();
        match stdin().read_line(&mut buf) {
            Err(err) =>{
                println!("{err}");
                continue;
            },
            _ => {}
        }
        let next_line = buf.trim();
        
        // parse line
        let tokens: Vec<TokenV2> = tokenize(next_line);
        if tokens.is_empty() {
            continue;
        }
        let syntax_tree = parse(tokens);
        match syntax_tree {
            Ok(tree) => println!("{:?}", tree),
            Err(err) => println!("Syntax Error: {}", err),
        }
    }
}
