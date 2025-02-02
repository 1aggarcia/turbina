use std::io::{stdin, stdout, BufReader, Write, BufRead};
use std::fs::File;

use crate::errors::IntepreterError;

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
