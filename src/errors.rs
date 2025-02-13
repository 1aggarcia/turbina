extern crate custom_error;
use custom_error::custom_error;

use crate::models::{Term, Token, Type};

pub type Result<T> = std::result::Result<T, Vec<IntepreterError>>;

custom_error!{#[derive(PartialEq, Clone)] pub IntepreterError
    SyntaxError { message: String } = "Syntax Error: {message}",
    TypeError { message: String } = "Type Error: {message}",
    MismatchedTypes { type1: Type, type2: Type } = "Mismatched Types: got {type1} and {type2}",
    InvalidType { datatype: Type } = "Expression of type '{datatype}' not allowed in this position",
    UnexpectedType { got: Type, expected: Type } = "Expression of type '{got}' cannot be assigned to '{expected}'",
    InvalidNullable { inner_type: Type } = "Type '{inner_type}' cannot be made nullable",
    IOError { message: String } = "IO Error: {message}",
    UndefinedError { id: String } = "Undefined Error: Identifier '{id}' is undefined",
    ReassignError { id: String } = "Reassign Error: Idenfitier '{id}' cannot be redefined",
    UnrecognizedToken { payload: String } = "Unrecognized Token: {payload}",
    ArgCount { got: usize, expected: usize } = "Passed {got} args to function but expected {expected}",

    EndOfFile = "End of File: THIS SHOULD NOT BE SHOWN TO USERS",
}

impl IntepreterError {
    pub fn not_a_function(term: &Term) -> Self {
        Self::TypeError {
            message: format!("Tried to call '{term:?}', but it is not a function")
        }
    }

    pub fn end_of_statement(token: Token) -> Self {
        Self::SyntaxError {
            message: format!("Expected newline or semicolon, got '{:?}'", token)
        }
    }

    pub fn bad_return_type(declared: &Type, body: &Type) -> Self {
        let message = format!("Function should return {declared}, but evaluates to {body}");
        Self::TypeError { message }
    }
}

// allows implicit conversion using ? operator
impl From<std::io::Error> for IntepreterError {
    fn from(value: std::io::Error) -> Self {
        Self::IOError { message: value.to_string() }
    }
}

impl From<IntepreterError> for Vec<IntepreterError> {
    fn from(value: IntepreterError) -> Self {
        vec![value]
    }
}

/// Utility functions to format common error types.
/// 
/// This was more necessary when I had plain string errors, but over the
/// functions should be removed in favor of the enum impl above. 
pub mod error {
    use super::IntepreterError;
    use super::IntepreterError::*;
    use crate::models::{BinaryOp, Token, Type};

    pub fn unexpected_end_of_input() -> IntepreterError {
        SyntaxError { message: "Unexpected end of input".into() }
    }

    pub fn unexpected_token(expected: &str, got: Token) -> IntepreterError {
        SyntaxError { message: format!("Expected {}, got {:?}", expected, got) }
    }

    pub fn not_a_type(token: Token) -> IntepreterError {
        TypeError { message: format!("'{:?}' is not a valid type", token) }
    }

    pub fn binary_op_types(
        operator: BinaryOp,
        left_type: &Type,
        right_type: &Type
    ) -> IntepreterError {
        let message = format!(
            "Illegal types for '{:?}' operator: {:?}, {:?}",
            operator,
            left_type,
            right_type
        );
        return TypeError { message };
    }

    pub fn unary_op_type(operator: &str, datatype: Type) -> IntepreterError {
        let message = format!("Cannot apply {} to token of type {:?}", operator, datatype);
        return TypeError { message };
    }

    pub fn undefined_id(id: &str) -> IntepreterError {
        UndefinedError { id: id.into() }
    }

    pub fn already_defined(id: &str) -> IntepreterError {
        ReassignError { id: id.into() }
    }
}