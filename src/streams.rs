use std::io::{stdin, stdout, BufReader, Write, BufRead};
use std::fs::File;

use crate::errors::{error, IntepreterError};
use crate::models::Token;

/// Abstraction over source code input
pub trait InputStream {
    /// Read the next line containing source code from the stream.
    /// If there is no more source code, returns `EndOfFile`` error.
    fn next_line(&mut self) -> Result<String, IntepreterError>;
}

pub struct FileStream {
    pub reader: BufReader<File>
}

impl InputStream for FileStream {
    fn next_line(&mut self) -> Result<String, IntepreterError> {
        let mut buf: String = String::new();

        // re-read if the last line is empty or a comment
        while buf.trim().is_empty() || buf.starts_with("//") {
            buf.clear();
            if self.reader.read_line(&mut buf)? == 0 {
                return Err(IntepreterError::EndOfFile);
            }
        }
        Ok(buf.to_string())
    }
}

pub struct StdinStream;
impl InputStream for StdinStream {
    fn next_line(&mut self) -> Result<String, IntepreterError> {
        let mut buf: String = String::new();

        // re-read if the last line is empty or a comment
        while buf.trim().is_empty() || buf.starts_with("//") {
            buf.clear();
            print!("> ");
            stdout().flush()?;
            if stdin().read_line(&mut buf)? == 0 {
                return Err(IntepreterError::EndOfFile);
            }
        }
        Ok(buf.to_string())
    }
}

/// Abstract Data Type used internally by the parser to facilitate tracking
/// token position and end-of-stream errors
pub struct TokenStream {
    tokens: Vec<Token>,
    position: usize,
}

impl TokenStream {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, position: 0 }
    }

    pub fn has_next(&self) -> bool {
        self.position < self.tokens.len()
    }

    /// Advances to the next token and returns the current one
    pub fn pop(&mut self) -> Result<Token, IntepreterError> {
        let token = self.peek()?;
        self.position += 1;
        return Ok(token);
    }

    /// Returns the token currently being pointed to
    pub fn peek(&self) -> Result<Token, IntepreterError> {
        self.lookahead(0)
    }

    /// Returns the token at the current position plus `offset`
    pub fn lookahead(&self, offset: usize) -> Result<Token, IntepreterError> {
        match self.tokens.get(self.position + offset) {
            Some(token) => Ok(token.clone()),
            None => Err(error::unexpected_end_of_input()),
        }
    }
}
