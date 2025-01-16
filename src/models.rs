use core::fmt;
use std::collections::HashMap;

/// State of the running program
pub struct Program {
    pub vars: HashMap<String, Variable>,
}

impl Program {
    pub fn new() -> Program {
        Self { vars: HashMap::new ()}
    }
}

/// Raw data with an assigned type
#[derive(PartialEq, Debug)]
pub struct Variable {
    pub datatype: Type,
    pub value: Literal,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Token {
    Literal(Literal),
    Operator(Operator),
    Id(String),
    Formatter(String),

    // keywords
    Let,
    Type(Type),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Literal {
    Int(i32),
    String(String),
    Bool(bool),
}

// will be extended beyond literal types (e.g. functions, arrays, structs)
// so should not be merged with enum `Literal`
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Type {
    Int,
    String,
    Bool,
}

pub fn get_literal_type(literal: &Literal) -> Type {
    match literal {
        Literal::Bool(_) => Type::Bool,
        Literal::Int(_) => Type::Int,
        Literal::String(_) => Type::String,
    }
}

#[derive(PartialEq, Clone, Copy)]
pub enum Operator {
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    OneEq,
    TwoEq,
    NotEq,
}

impl fmt::Debug for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = match self {
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Star => "*",
            Self::Slash => "/",
            Self::Percent => "%",
            Self::OneEq => "=",
            Self::TwoEq => "==",
            Self::NotEq => "!=",
        };
        write!(f, "{string}")?;
        Ok(())
    }
}

#[derive(PartialEq)]
pub enum AbstractSyntaxTree {
    Let(LetNode),
    Operator(OperatorNode),

    // leaf nodes
    Literal(Literal),
    Id(String),
}

impl fmt::Debug for AbstractSyntaxTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "")?;
        pretty_print(f, self, 0);
        Ok(())
    }
}

/// recursively print out children with increasing indent
fn pretty_print(
    f: &mut std::fmt::Formatter<'_>,
    node: &AbstractSyntaxTree,
    indent: usize
) {
    let indent_str = " ".repeat(indent * 2);
    match node {
        AbstractSyntaxTree::Literal(lit) => {
            writeln!(f, "{}{:?}", indent_str, lit).unwrap()
        },
        AbstractSyntaxTree::Id(id) => {
            writeln!(f, "{}{}", indent_str, id).unwrap()
        },
        AbstractSyntaxTree::Operator(node) => {
            writeln!(f, "{}{:?}", indent_str, node.operator).unwrap();
            pretty_print(f, &node.left, indent + 1);
            pretty_print(f, &node.right, indent + 1);
        },
        AbstractSyntaxTree::Let(node) => {
            let type_str = match node.datatype {
                Some(t) => format!("{t:?}"),
                None => String::from("unknown"),
            };
            writeln!(f, "{}let {}: {} =", indent_str, node.id, type_str).unwrap();
            pretty_print(f, &node.value, indent + 1);
        }
    }
}

/// For binary operators
#[derive(Debug, PartialEq)]
pub struct OperatorNode {
    pub operator: Operator,
    pub left: Box<AbstractSyntaxTree>,
    pub right: Box<AbstractSyntaxTree>,
}

#[derive(Debug, PartialEq)]
pub struct LetNode {
    pub id: String,
    pub datatype: Option<Type>,
    pub value: Box<AbstractSyntaxTree>,
}

#[cfg(test)]
pub mod test_utils {
    use super::*;

    pub fn bool_token(data: bool) -> Token {
        Token::Literal(Literal::Bool(data))
    }

    pub fn string_token(data: &str) -> Token {
        Token::Literal(Literal::String(data.to_string()))
    }

    pub fn int_token(data: i32) -> Token {
        Token::Literal(Literal::Int(data))
    }

    pub fn op_token(operator: Operator) -> Token {
        Token::Operator(operator)
    }

    pub fn id_token(data: &str) -> Token {
        Token::Id(data.to_string())
    }

    pub fn formatter_token(data: &str) -> Token {
        Token::Formatter(data.to_string())
    }

    pub fn type_token(datatype: Type) -> Token {
        Token::Type(datatype)
    }
}
