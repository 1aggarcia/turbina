use std::io::BufReader;
use std::env;
use std::fs::File;

mod errors;
mod models;
mod lexer;
mod parser;
mod validator;
mod evaluator;
mod library;
mod streams;

use errors::IntepreterError;
use models::{AbstractSyntaxTree, Literal, Program, Type};
use parser::parse_statement;
use streams::{FileStream, StdinStream, TokenStream};
use validator::validate;
use evaluator::evaluate;

/// Executable unit of code.
/// The type is only needed when a function is printed.
type Statement = (AbstractSyntaxTree, Type);

fn main() {
    // TODO: end-to-end tests reading and checking stdout results
    let Some(filename) = env::args().nth(1) else {
        run_repl();
        return;
    };
    match File::open(filename) {
        Ok(file) => run_from_file(file),
        Err(err) => eprintln!("{err}")
    }
}

fn run_from_file(file: File) {
    let reader = BufReader::new(file);
    let file_stream = Box::new(FileStream { reader });
    let mut token_stream = TokenStream::new(file_stream);

    let mut program = Program::init();
    let mut statements = Vec::new();
    let mut errors = Vec::new();

    // type check the file
    loop {
        match validate_next_statement(&mut program, &mut token_stream) {
            Ok(result) => statements.push(result),
            Err(statement_errors) => {
                if statement_errors.contains(&IntepreterError::EndOfFile) {
                    break;
                }
                errors.extend(statement_errors);
            }
        };
    }

    if !errors.is_empty() {
        errors.iter().for_each(|err| eprintln!("{err}"));
        return;
    }

    // evaluate validated statements
    for statement in statements {
        match evaluate_statement(&mut program, &statement) {
            Ok(result) => println!("{result}"),
            Err(err) => {
                eprintln!("{err}");
                break;
            },
        };
    }
}

/// Command line interface for using Turbina.
/// REPL = Read-eval-print loop
fn run_repl() {
    let mut token_stream = TokenStream::new(Box::new(StdinStream {}));
    let mut program = Program::init();

    println!("Welcome to Turbina");

    loop {
        let eval_result =
            validate_next_statement(&mut program, &mut token_stream)
                .and_then(|statement|
                    evaluate_statement(&mut program, &statement)
                        .map_err(|e| vec![e]));

        match eval_result {
            Ok(result) => println!("{result}"),
            Err(errors) => {
                if errors.contains(&IntepreterError::EndOfFile) {
                    break;
                }
                errors.iter().for_each(|e| eprintln!("{e}"));
                continue;
            },
        };
    }
}

/// Read the next statement from the token stream and validate it.
/// Returns a validated statement that is safe to execute.
fn validate_next_statement(
    program: &mut Program,
    token_stream: &mut TokenStream
) -> Result<Statement, Vec<IntepreterError>> {
    let syntax_tree = parse_statement(token_stream)?;

    let tree_type = validate(program, &syntax_tree)?;
    if let Some(name) = &tree_type.name_to_bind {
        program.type_context.insert(name.clone(), tree_type.datatype.clone());
    }
    Ok((syntax_tree, tree_type.datatype))
}

/// Evaluates a statment on the given program.
/// May return a runtime error.
fn evaluate_statement(
    program: &mut Program,
    (syntax_tree, tree_type): &Statement,
) -> Result<String, IntepreterError> {
    let output = match evaluate(program, &syntax_tree) {
        // Function types look nicer to print than any form of the literal struct
        Literal::Func(_) => tree_type.to_string(),
        literal => literal.to_string(),
    };
    Ok(output)
}
