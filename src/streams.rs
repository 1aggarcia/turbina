use std::io::{stdin, stdout, BufReader, Write, BufRead};
use std::fs::File;

use crate::errors::{IntepreterError, error};
use crate::lexer::tokenize;
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
                return Err(error::unexpected_end_of_input());
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
                return Err(error::unexpected_end_of_input());
            }
        }
        Ok(buf.to_string())
    }
}

/// Input stream made out of a plain string
pub struct StringStream {
    lines: Vec<String>,
    line_num: usize,
}

impl StringStream {
    pub fn new(source_code: &str) -> Self {
        StringStream {
            lines: source_code.lines()
                .map(|s| s.to_owned() + "\n")
                .filter(|s| !s.trim().is_empty())
                .filter(|s| !s.starts_with("//"))
                .collect(),
            line_num: 0,
        }
    }
}

impl InputStream for StringStream {
    fn next_line(&mut self) -> Result<String, IntepreterError> {
        if self.line_num >= self.lines.len() {
            return Err(error::unexpected_end_of_input());
        }
        let line = self.lines.get(self.line_num).unwrap().to_string();
        self.line_num += 1;
        Ok(line)
    }
}

/// Abstract Data Type used internally by the parser to facilitate tracking
/// token position and end-of-stream errors
pub struct TokenStream {
    input_stream: Box<dyn InputStream>,
    tokens: Vec<Token>,
    position: usize,
}

impl TokenStream {
    pub fn new(input_stream: Box<dyn InputStream>) -> Self {
        Self { input_stream, tokens: vec![], position: 0 }
    }

    pub fn from_tokens(tokens: Vec<Token>) -> Self {
        Self {
            input_stream: Box::new(StringStream::new("")),
            tokens,
            position: 0,
        }
    }

    /// Advances to the next token and returns the current one
    pub fn pop(&mut self) -> Result<Token, IntepreterError> {
        let token = self.peek()?;
        self.position += 1;
        return Ok(token);
    }

    /// Returns the token currently being pointed to
    pub fn peek(&mut self) -> Result<Token, IntepreterError> {
        self.lookahead(0)
    }

    /// Returns the token at the current position plus `offset`
    pub fn lookahead(&mut self, offset: usize) -> Result<Token, IntepreterError> {
        if let Some(token) = self.tokens.get(self.position + offset) {
            return Ok(token.clone());
        }
        // replenish buffer, then try again
        let next_line = self.input_stream.next_line()?;
        self.tokens = tokenize(&next_line).map_err(|e| e[0].clone())?;
        self.position = 0;
        self.lookahead(offset)
    }
}

#[cfg(test)]
mod test_token_stream {
    use super::*;
    use crate::models::test_utils::id_token;

    #[test]
    fn test_has_next_is_false_for_empty_stream() {
        let mut stream = TokenStream::from_tokens(vec![]);
        assert_eq!(stream.pop(), Err(error::unexpected_end_of_input()));
    }

    #[test]
    fn test_has_next_is_true_for_non_empty_stream() {
        let mut stream = TokenStream::from_tokens(vec![id_token("data")]);
        assert_eq!(stream.pop(), Ok(id_token("data")));
    }

    #[test]
    fn test_peek_doesnt_consume_data() {
        let mut stream = TokenStream::from_tokens(vec![id_token("data")]);
        assert_eq!(stream.peek(), Ok(id_token("data")));
        assert_eq!(stream.pop(), Ok(id_token("data")));
        assert_eq!(stream.peek(), Err(error::unexpected_end_of_input()));
    }
}

#[cfg(test)]
mod test_string_stream {
    use super::*;

    #[test]
    fn it_preserves_newlines_at_the_end_of_each_line() {
        let mut stream = StringStream::new("1\n2\n3");
        assert_eq!(stream.next_line(), Ok("1\n".into()));
        assert_eq!(stream.next_line(), Ok("2\n".into()));
        assert_eq!(stream.next_line(), Ok("3\n".into()));
    }

    #[test]
    fn it_skips_blank_lines() {
        let mut stream = StringStream::new("1\n\n\n\n2\n\n\n3");
        assert_eq!(stream.next_line(), Ok("1\n".into()));
        assert_eq!(stream.next_line(), Ok("2\n".into()));
        assert_eq!(stream.next_line(), Ok("3\n".into()));
        assert_eq!(stream.next_line(), Err(error::unexpected_end_of_input()));
    }

    #[test]
    fn it_skips_comments() {
        let mut stream = StringStream::new("// a comment\n3");
        assert_eq!(stream.next_line(), Ok("3\n".into()));
    }
}
