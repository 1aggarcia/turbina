extern crate custom_error;
use custom_error::custom_error;

use crate::models::{Term, Token, Type};

pub type Result<T> = std::result::Result<T, InterpreterError>;
pub type MultiResult<T> = std::result::Result<T, Vec<InterpreterError>>;

custom_error!{#[derive(PartialEq, Clone)] pub InterpreterError
    SyntaxError { message: String } = "Syntax Error: {message}",

    TypeError { message: String } = "Type Error: {message}",
    MismatchedTypes { type1: Type, type2: Type } = "Mismatched Types: got {type1} and {type2}",
    InvalidType { datatype: Type } = "Expression of type '{datatype}' not allowed in this position",
    UnexpectedType { got: Type, expected: Type } = "Expression of type '{got}' cannot be assigned to '{expected}'",
    EmptyTypeList = "Cannot define a function with an empty type parameter list",
    UndeclaredGeneric { generic: String } =
        "Cannot use generic type '{generic}' without declaring it in the function definition",

    InvalidNullable { inner_type: Type } = "Type '{inner_type}' cannot be made nullable",
    IOError { message: String } = "IO Error: {message}",
    UndefinedError { id: String } = "Undefined Error: Identifier '{id}' is undefined",
    ReassignError { id: String } = "Reassign Error: Idenfitier '{id}' cannot be redefined",
    UnrecognizedToken { payload: String } = "Unrecognized Token: {payload}",
    ArgCount { got: usize, expected: usize } = "Passed {got} args to function but expected {expected}",

    EndOfFile = "End of File: THIS SHOULD NOT BE SHOWN TO USERS",
}

impl InterpreterError {
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
impl From<std::io::Error> for InterpreterError {
    fn from(value: std::io::Error) -> Self {
        Self::IOError { message: value.to_string() }
    }
}

impl From<InterpreterError> for Vec<InterpreterError> {
    fn from(value: InterpreterError) -> Self {
        vec![value]
    }
}

/// Utility functions to format common error types.
/// 
/// This was more necessary when I had plain string errors, but over the
/// functions should be removed in favor of the enum impl above. 
pub mod error {
    use super::InterpreterError;
    use super::InterpreterError::*;
    use crate::models::{BinaryOp, Token, Type};

    pub fn unexpected_end_of_input() -> InterpreterError {
        SyntaxError { message: "Unexpected end of input".into() }
    }

    pub fn unexpected_token(expected: &str, got: Token) -> InterpreterError {
        SyntaxError { message: format!("Expected {}, got {:?}", expected, got) }
    }

    pub fn not_a_type(token: Token) -> InterpreterError {
        TypeError { message: format!("'{:?}' is not a valid type", token) }
    }

    pub fn binary_op_types(
        operator: BinaryOp,
        left_type: &Type,
        right_type: &Type
    ) -> InterpreterError {
        let message = format!(
            "Illegal types for '{:?}' operator: {:?}, {:?}",
            operator,
            left_type,
            right_type
        );
        return TypeError { message };
    }

    pub fn unary_op_type(operator: &str, datatype: Type) -> InterpreterError {
        let message = format!("Cannot apply {} to token of type '{}'", operator, datatype);
        return TypeError { message };
    }

    pub fn undefined_id(id: &str) -> InterpreterError {
        UndefinedError { id: id.into() }
    }

    pub fn already_defined(id: &str) -> InterpreterError {
        ReassignError { id: id.into() }
    }
}