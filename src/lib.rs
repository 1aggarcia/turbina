use errors::IntepreterError;
use evaluator::evaluate;
use models::{AbstractSyntaxTree, Literal, Program, Type};
use parser::parse_statement;
use streams::{InputStream, StdinStream, StringStream, TokenStream};
use validator::validate;
use wasm_bindgen::prelude::*;

pub mod errors;
pub mod models;
pub mod lexer;
pub mod parser;
pub mod validator;
pub mod evaluator;
pub mod library;
pub mod streams;

/// Executable unit of code.
/// The type is only needed when a function is printed.
type Statement = (AbstractSyntaxTree, Type);

/// Public JavaScript function to access Turbina through Web Assembly.
/// 
/// Starts a program session, runs the source code on it, and returns the
/// output produced after running the program.
#[wasm_bindgen]
pub fn run_turbina_program(source_code: &str) -> Vec<String> {
    let input_stream = Box::new(StringStream::new(source_code));
    run_as_file(input_stream)
}

pub fn run_as_file(input_stream: Box<dyn InputStream>) -> Vec<String> {
    let mut token_stream = TokenStream::new(input_stream);

    let mut program = Program::init();
    let mut statements = vec![];
    let mut errors = vec![];

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
        return errors.iter().map(|e| e.to_string()).collect();
    }

    let mut results = vec![];

    // evaluate validated statements
    for statement in statements {
        match evaluate_statement(&mut program, &statement) {
            Ok(result) => {
                // TODO: replace with a generic writer so that it can be output to JS
                println!("{result}");
                results.push(result);
            },
            Err(err) => {
                eprintln!("{err}");
                break;
            },
        };
    }

    results
}

/// Command line interface for using Turbina.
/// REPL = Read-eval-print loop
pub fn run_repl() {
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

#[cfg(test)]
mod test_wasm {
    use super::*;

    #[test]
    fn test_run_turbina_program() {
        let source_code = "let x = 5; // comment \n";
        let expected = vec!["5".to_string()];

        assert_eq!(run_turbina_program(source_code), expected)
    }
}
