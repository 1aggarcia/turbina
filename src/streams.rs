use std::fmt::Debug;
use std::io::{stderr, stdin, stdout, BufRead, BufReader, Write};
use std::fs::File;

use crate::errors::{InterpreterError, error};
use crate::lexer::tokenize;
use crate::models::Token;

/// Where the virtual machine should write to
pub struct OutputStreams {
    pub stdout: Box<dyn Write>,
    pub stderr: Box<dyn Write>,
}

impl OutputStreams {
    /// Create a new struct using stdout and stderr
    pub fn std_streams() -> Self {
        Self { stdout: Box::new(stdout()), stderr: Box::new(stderr()) }
    }
}

impl Debug for OutputStreams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "OutputStreams")
    }
}

/// Abstraction over source code input
pub trait InputStream {
    /// Read the next line containing source code from the stream.
    /// If there is no more source code (EOF), returns empty string.
    fn next_line(&mut self) -> Result<String, InterpreterError>;
}

pub struct FileStream {
    pub reader: BufReader<File>
}

impl InputStream for FileStream {
    fn next_line(&mut self) -> Result<String, InterpreterError> {
        let mut buf: String = String::new();

        // re-read if the last line is empty or a comment
        while buf.trim().is_empty() || buf.starts_with("//") {
            buf.clear();
            if self.reader.read_line(&mut buf)? == 0 {
                // 0 bytes read indicates end of file
                return Ok("".to_string());
            }
        }
        Ok(buf.to_string())
    }
}

// Currently not in use, replaced by RustylineStream
pub struct StdinStream;
impl InputStream for StdinStream {
    fn next_line(&mut self) -> Result<String, InterpreterError> {
        let mut buf: String = String::new();

        // re-read if the last line is empty or a comment
        while buf.trim().is_empty() || buf.starts_with("//") {
            buf.clear();
            print!("> ");
            stdout().flush()?;
            if stdin().read_line(&mut buf)? == 0 {
                // 0 bytes read indicates end of file
                return Ok("".to_string());
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
    fn next_line(&mut self) -> Result<String, InterpreterError> {
        // Use empty string to indicate EOF
        let line = match self.lines.get(self.line_num) {
            Some(l) => l,
            None => return Ok("".to_string()),
        };

        self.line_num += 1;
        Ok(line.to_string())
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
    pub fn pop(&mut self) -> Result<Token, InterpreterError> {
        let token = self.peek()?;
        self.position += 1;
        return Ok(token);
    }

    /// Returns the token currently being pointed to
    pub fn peek(&mut self) -> Result<Token, InterpreterError> {
        self.lookahead(0)
    }

    /// Returns the token at the current position plus `offset`
    pub fn lookahead(&mut self, offset: usize) -> Result<Token, InterpreterError> {
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
mod test_file_stream {
    use super::*;

    static TEST_FILE_PATH: &str = "./src/test_data/fileStreamTest.tb";

    #[test]
    fn it_returns_empty_string_for_eof() {
        let file = File::open(TEST_FILE_PATH).expect(
            "Test file should exist"
        );
        let reader = BufReader::new(file);
        let mut file_stream = FileStream { reader };

        assert_eq!(file_stream.next_line(), Ok("Text before blank line\n".to_string()));
        assert_eq!(file_stream.next_line(), Ok("Text after blank line\n".to_string()));
        assert_eq!(file_stream.next_line(), Ok("".to_string()));
    }
}

#[cfg(test)]
mod test_token_stream {
    use super::*;
    use crate::models::test_utils::id_token;

    #[test]
    fn test_empty_stream_returns_eof_token() {
        let mut stream = TokenStream::from_tokens(vec![]);
        assert_eq!(stream.pop(), Ok(Token::EndOfFile));
    }

    #[test]
    fn test_non_empty_stream_returns_tokens() {
        let mut stream = TokenStream::from_tokens(vec![id_token("data")]);
        assert_eq!(stream.pop(), Ok(id_token("data")));
    }

    #[test]
    fn test_peek_doesnt_consume_data() {
        let mut stream = TokenStream::from_tokens(vec![id_token("data")]);
        assert_eq!(stream.peek(), Ok(id_token("data")));
        assert_eq!(stream.pop(), Ok(id_token("data")));
        assert_eq!(stream.peek(), Ok(Token::EndOfFile));
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
        let mut stream = StringStream::new("1\n\n  \n\n2\n\n\n3");
        assert_eq!(stream.next_line(), Ok("1\n".into()));
        assert_eq!(stream.next_line(), Ok("2\n".into()));
        assert_eq!(stream.next_line(), Ok("3\n".into()));
        assert_eq!(stream.next_line(), Ok("".into())); // EOF
    }

    #[test]
    fn it_skips_comments() {
        let mut stream = StringStream::new("// a comment\n3");
        assert_eq!(stream.next_line(), Ok("3\n".into()));
    }
}
