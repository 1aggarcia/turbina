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
use models::{AbstractSyntaxTree, Literal, Program, Type};
use lexer::tokenize;
use parser::parse;
use validator::validate;
use evaluator::evaluate;

/// Executable unit of code.
/// The type is only needed when a function is printed.
type Statement = (AbstractSyntaxTree, Type);

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
    let mut file_stream = InputStream::File(reader);

    let mut program = Program::init();

    let mut statements = Vec::new();
    let mut errors = Vec::new();

    // type check the file
    loop {
        match validate_next_line(&mut program, &mut file_stream) {
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
    let mut stdin = InputStream::Stdin;
    let mut program = Program::init();

    println!("Welcome to Turbina");

    loop {
        let eval_result = validate_next_line(&mut program, &mut stdin)
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

/// Read the next line from the input stream and validate it.
/// Returns a validated statement that is safe to execute.
fn validate_next_line(
    program: &mut Program,
    input_stream: &mut InputStream
) -> Result<Statement, Vec<IntepreterError>> {
    let next_line = input_stream.next_line()?;
    let tokens = tokenize(&next_line)?;
    let syntax_tree = parse(tokens)?;

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
