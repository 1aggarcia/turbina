//! CLI parser, input stream and all other related code for the CLI Rustyline
//! integration.
//! 
//! Kept separate from the rest of the codebase since Rustyline is
//! not supported in WASM compilation and breaks the build when compiling for
//! WASM. Having this module separate makes it easy to conditionally compile it
//! only for the CLI.

use std::process::exit;

use clap::Parser;
use rustyline::{DefaultEditor, error::ReadlineError, config::Configurer};

use crate::{errors::InterpreterError, streams::InputStream};

#[derive(Parser)]
pub struct RustylineArgs {
    #[arg(required = false)]
    pub path: Option<std::path::PathBuf>,

    #[arg(long)]
    pub disable_type_checker: bool,
}

impl From<ReadlineError> for InterpreterError {
    fn from(value: ReadlineError) -> Self {
        Self::IOError { message: value.to_string() }
    }
}

pub struct RustylineStream {
    editor: DefaultEditor,
}

impl RustylineStream {
    pub fn new() -> rustyline::Result<Self> {
        let mut editor = DefaultEditor::new()?;
        editor.set_auto_add_history(true);

        Ok(Self { editor })
    }
}

impl InputStream for RustylineStream {
    fn next_line(&mut self) -> Result<String, InterpreterError> {
        match self.editor.readline("> ") {
            // rustyline does not include newline character
            Ok(input) => Ok(input + "\n"),

            Err(ReadlineError::Interrupted | ReadlineError::Eof) => exit(0),
            Err(err) => Err(err.into())
        }
    }
}