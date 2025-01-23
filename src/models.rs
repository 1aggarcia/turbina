use core::fmt;
use std::collections::HashMap;

/// State of the running program
#[derive(Debug)]
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
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    Id(String),
    Formatter(String),

    // keywords
    Let,
    If,
    Else,
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

impl fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = match self {
            Type::Int => "int",
            Type::String => "string",
            Type::Bool => "bool",
        };
        write!(f, "{}", string)
    }
}

pub fn get_literal_type(literal: &Literal) -> Type {
    match literal {
        Literal::Bool(_) => Type::Bool,
        Literal::Int(_) => Type::Int,
        Literal::String(_) => Type::String,
    }
}

#[derive(PartialEq, Clone, Copy)]
pub enum BinaryOp {
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Equals,
    NotEq,
}

impl fmt::Debug for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = match self {
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Star => "*",
            Self::Slash => "/",
            Self::Percent => "%",
            Self::Equals => "==",
            Self::NotEq => "!=",
        };
        write!(f, "{string}")?;
        Ok(())
    }
}

#[derive(PartialEq, Clone)]
pub enum UnaryOp {
    Equals,
    Not,
}

impl fmt::Debug for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = match self {
            Self::Equals => "=",
            Self::Not => "!",
        };
        write!(f, "{string}")?;
        Ok(())
    }
}

#[derive(PartialEq, Debug)]
pub enum AbstractSyntaxTree {
    Let(LetNode),
    Expr(Expr),
}

#[derive(PartialEq, Debug, Clone)]
pub enum Term {
    Literal(Literal),
    Id(String),
    Not(Box<Term>),  // negate boolean terms
    Minus(Box<Term>),  // negate int terms
    Expr(Box<Expr>),
}

// to make construction easier
impl Term {
    pub fn negative_int(term: Term) -> Self {
        Self::Minus(Box::new(term))
    }

    pub fn negated_bool(term: Term) -> Self {
        Self::Not(Box::new(term))
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Binary(BinaryExpr),
    Cond(CondExpr),
}

/// For variable length expressions on binary operators
#[derive(Debug, PartialEq, Clone)]
pub struct BinaryExpr {
    pub first: Term,
    pub rest: Vec<(BinaryOp, Term)>
}

/// if/else expressions
#[derive(Debug, PartialEq, Clone)]
pub struct CondExpr {
    pub cond: Box<Expr>,
    pub if_true: Box<Expr>,
    pub if_false: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct LetNode {
    pub id: String,
    pub datatype: Option<Type>,
    pub value: Expr,
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

    pub fn op_token(operator: BinaryOp) -> Token {
        Token::BinaryOp(operator)
    }

    pub fn unary_op_token(operator: UnaryOp) -> Token {
        Token::UnaryOp(operator)
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

    pub fn int_term(int: i32) -> Term {
        Term::Literal(Literal::Int(int))
    }

    pub fn str_term(string: &str) -> Term {
        Term::Literal(Literal::String(string.into()))
    }

    pub fn term_tree(term: Term) -> AbstractSyntaxTree {
        AbstractSyntaxTree::Expr(term_expr(term))
    }

    pub fn term_expr(term: Term) -> Expr {
        bin_expr(term, vec![])
    }

    pub fn bin_expr(first: Term, rest: Vec<(BinaryOp, Term)>) -> Expr {
        Expr::Binary(BinaryExpr { first, rest })
    }
}
