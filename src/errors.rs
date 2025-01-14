/// Collection of formatter functions for common errors

use crate::models::{Operator, Token, Type};

static SYNTAX_ERROR: &str = "Syntax Error: ";
static TYPE_ERROR: &str = "Type Error: ";
static VALUE_ERROR: &str = "Value Error: ";

pub fn unexpected_end_of_input() -> String {
    format!("{SYNTAX_ERROR}Unexpected end of input")
}

pub fn unexpected_token(expected: &str, got: Token) -> String {
    format!("{SYNTAX_ERROR}Expected {}, got {:?}", expected, got)
}

pub fn token_not_allowed(token: Token) -> String {
    format!("{SYNTAX_ERROR}Token '{:?}' not allowed in this position", token)
}

pub fn undefined_id(id: &str) -> String {
    format!("{VALUE_ERROR}Idenfitier '{id}' is undefined")
}

pub fn already_defined(id: &str) -> String {
    format!("{VALUE_ERROR}Symbol '{}' is already defined", id)
}

pub fn binary_op_types(
    operator: Operator,
    left_type: Type,
    right_type: Type
) -> String {
    format!(
        "{TYPE_ERROR}Illegal types for '{:?}' operator: {:?}, {:?}",
        operator,
        left_type,
        right_type
    )
}

pub fn declared_type(id: &str, declared: Type, expression: Type) -> String {
    format!(
        "{}Declared type {:?} for '{}' does not match expression type {:?}",
        TYPE_ERROR,
        declared,
        id,
        expression
    )
}

pub fn not_a_type(token: Token) -> String {
    format!("{TYPE_ERROR}'{:?}' is not a valid type", token)
}
