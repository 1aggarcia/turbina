extern crate custom_error;
use custom_error::custom_error;

use crate::models::Token;

use IntepreterError::*;

custom_error!{#[derive(PartialEq)] pub IntepreterError
    SyntaxError { message: String } = "Syntax Error: {message}",
    TypeError { message: String } = "Type Error: {message}",
}

pub fn unexpected_end_of_input() -> IntepreterError {
    SyntaxError { message: "Unexpected end of input".into() }
}

pub fn unexpected_token(expected: &str, got: Token) -> IntepreterError {
    SyntaxError { message: format!("Expected {}, got {:?}", expected, got) }
}

pub fn not_a_type(token: Token) -> IntepreterError {
    TypeError { message: format!("'{:?}' is not a valid type", token) }
}
