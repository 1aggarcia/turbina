use crate::models::Token;

use crate::errors::{error, Result};
use crate::streams::TokenStream;

/// Test if the next token in the stream matches the given token without
/// consuming it. If the stream is empty, this returns false
pub fn next_token_matches(stream: &mut TokenStream, token: Token) -> bool {
    stream.peek() == Ok(token)
}

/// Take the next token from the stream and compare it to the expected token.
/// If they do not match, return a syntax error 
pub fn match_next(stream: &mut TokenStream, expected: Token) -> Result<()> {
    let next = stream.pop()?;
    if next != expected {
        // TODO: improve debug string
        Err(error::unexpected_token(format!("{expected:?}").as_str(), next))
    } else {
        Ok(())
    }
}

/// Skip over any newlines at the front of the stream.
/// Will never error.
pub fn skip_newlines(tokens: &mut TokenStream) {
    while let Ok(Token::Newline) = tokens.peek() {
        // safe to unwrap since the peeked value is Ok
        let _ = tokens.pop();
    }
}
