use std::io::{Error, Write};
use clap::Parser;

use errors::InterpreterError;
use evaluator::evaluate;
use models::{AbstractSyntaxTree, Literal, Program, Type};
use parser::parse_statement;
use streams::{InputStream, OutputStreams, RustylineStream, StringStream, TokenStream};
use type_resolver::resolve_type;
use wasm_bindgen::prelude::*;

pub mod errors;
pub mod models;
pub mod lexer;
pub mod parser;
pub mod type_resolver;
pub mod evaluator;
pub mod library;
pub mod streams;
pub mod library_io;

/// Executable unit of code.
/// The type is only needed when a function is printed.
type Statement = (AbstractSyntaxTree, Option<Type>);

#[derive(Parser)]
pub struct CliArgs {
    #[arg(required = false)]
    pub path: Option<std::path::PathBuf>,

    #[arg(long)]
    pub disable_type_checker: bool,
}

/// Writer that passes data to the JavaScript runtime
struct JavaScriptWriter { write_callback: js_sys::Function }

impl Write for JavaScriptWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let string_data = String::from_utf8(buf.to_vec())
            .map_err(|e| Error::other(e))?;

        let js_string = JsValue::from_str(&string_data);
        self.write_callback.call1(&JsValue::NULL, &js_string)
            .map_err(|_| std::io::Error::other("Failed to call JS write callback"))?;

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

/// Function visible in JavaScript to use Turbina through Web Assembly.
/// 
/// Accepts two callback functions that consume all string data that is written
/// to either stdout or stderr. They are called every time a write is made.
/// 
/// ```typescript
/// on_stdout_write: (data: string) => void
/// on_stderr_write: (data: string) => void
/// ```
#[wasm_bindgen]
pub fn run_turbina_program(
    source_code: &str,
    on_stdout_write: js_sys::Function,
    on_stderr_write: js_sys::Function,
) {
    let input_stream = Box::new(StringStream::new(source_code));
    let output_streams = OutputStreams {
        stdout: Box::new(JavaScriptWriter { write_callback: on_stdout_write }),
        stderr: Box::new(JavaScriptWriter { write_callback: on_stderr_write }),
    };

    let result = run_as_file(
        input_stream,
        output_streams,
        CliArgs { path: None, disable_type_checker: false },
    );

    if let Err(e) = result {
        // call `console.error`in JavaScript runtime
        error(&e.to_string());
    }
}

/// Execute from the input stream as a file, writing to the given output streams.
/// Returns `Ok(())` unless there is an error writing to the streams
pub fn run_as_file(
    input_stream: Box<dyn InputStream>,
    out_streams: OutputStreams,
    args: CliArgs
) -> std::io::Result<()> {
    let mut token_stream = TokenStream::new(input_stream);

    let mut program = Program::init(out_streams);
    let mut statements = vec![];
    let mut errors = vec![];

    // type check the file
    loop {
        match validate_next_statement(&mut program, &mut token_stream, &args) {
            Ok(result) => statements.push(result),
            Err(statement_errors) => {
                if statement_errors.contains(&InterpreterError::EndOfFile) {
                    break;
                }
                errors.extend(statement_errors);
            }
        };
    }

    if !errors.is_empty() {
        for err in errors {
            writeln!(program.output.stderr, "{err}")?;
        }
        return Ok(());
    }

    // evaluate validated statements
    for statement in statements {
        if let Err(err) = evaluate_statement(&mut program, &statement) {
            writeln!(program.output.stderr, "{err}")?;
            break;
        }
    }
    Ok(())
}

/// Command line interface for using Turbina.
/// REPL = Read-eval-print loop
pub fn run_repl(args: CliArgs) {
    let input_stream = RustylineStream::new()
        .expect("Failed to open input stream");

    let mut token_stream = TokenStream::new(Box::new(input_stream));
    let mut program = Program::init_with_std_streams();

    println!("Welcome to Turbina");

    loop {
        let eval_result =
            validate_next_statement(&mut program, &mut token_stream, &args)
                .and_then(|statement|
                    evaluate_statement(&mut program, &statement)
                        .map_err(|e| vec![e]));

        match eval_result {
            Ok(result) => println!("{result}"),
            Err(errors) => {
                if errors.contains(&InterpreterError::EndOfFile) {
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
/// Skip type checking if specified by `args`
fn validate_next_statement(
    program: &mut Program,
    token_stream: &mut TokenStream,
    args: &CliArgs
) -> Result<Statement, Vec<InterpreterError>> {
    let syntax_tree = parse_statement(token_stream)?;
    if args.disable_type_checker {
        return Ok((syntax_tree, None));
    }

    let tree_type = resolve_type(program, &syntax_tree)?;
    if let Some(name) = &tree_type.name_to_bind {
        program.type_context.insert(name.clone(), tree_type.datatype.clone());
    }
    Ok((syntax_tree, Some(tree_type.datatype)))
}

/// Evaluates a statement on the given program.
/// May return a runtime error.
fn evaluate_statement(
    program: &mut Program,
    (syntax_tree, tree_type): &Statement,
) -> Result<String, InterpreterError> {
    let output = match (evaluate(program, &syntax_tree)?, tree_type) {
        // Function types look nicer to print than any form of the struct
        (Literal::Closure(_), Some(datatype)) => datatype.to_string(),
        (literal, _) => literal.to_string(),
    };
    Ok(output)
}

// Currently not possible to test without mocking JS functions
// #[cfg(test)]
// mod test_wasm {
//     use super::*;

//     #[test]
//     fn test_run_turbina_program() {
//         let source_code = "let x = 5\nx";
//         let expected = vec!["5".to_string(), "5".to_string()];

//         assert_eq!(run_turbina_program(source_code), expected)
//     }
// }
