/// Collection of formatter functions for common errors

use crate::models::{BinaryOp, Type};

static TYPE_ERROR: &str = "Type Error: ";
static VALUE_ERROR: &str = "Value Error: ";

pub fn undefined_id(id: &str) -> String {
    format!("{VALUE_ERROR}Idenfitier '{id}' is undefined")
}

pub fn already_defined(id: &str) -> String {
    format!("{VALUE_ERROR}Symbol '{}' is already defined", id)
}

pub fn binary_op_types(
    operator: BinaryOp,
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

pub fn unary_op_type(operator: &str, datatype: Type) -> String {
    format!("{TYPE_ERROR}Cannot apply {} to token of type {:?}", operator, datatype)
}
