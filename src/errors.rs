extern crate custom_error;
use custom_error::custom_error;

custom_error!{#[derive(PartialEq, Clone)] pub IntepreterError
    SyntaxError { message: String } = "Syntax Error: {message}",
    TypeError { message: String } = "Type Error: {message}",
    IoError { message: String } = "IO Error: {message}",
    UndefinedError { id: String } = "Undefined Error: Identifier '{id}' is undefined",
    ReassignError { id: String } = "Reassign Error: Idenfitier '{id} cannot be redefined",
}

impl IntepreterError {
    pub fn io_err(err: std::io::Error) -> Self {
        Self::IoError { message: err.to_string() }
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
        left_type: Type,
        right_type: Type
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

    pub fn declared_type(id: &str, declared: Type, expression: Type) -> IntepreterError {
        let message = format!(
            "Declared type {:?} for '{}' does not match expression type {:?}",
            declared,
            id,
            expression
        );
        return TypeError { message };
    }

    pub fn undefined_id(id: &str) -> IntepreterError {
        UndefinedError { id: id.into() }
    }

    pub fn already_defined(id: &str) -> IntepreterError {
        ReassignError { id: id.into() }
    }
}