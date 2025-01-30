use core::fmt;
use std::collections::HashMap;

use crate::library::LIBRARY;

/// State of the running program
#[derive(Debug)]
pub struct Program {
    pub bindings: HashMap<String, Literal>,
    pub type_context: HashMap<String, Type>,
}

impl Program {
    /// Initialize a `Program` with library functions imported into the enviorment
    pub fn init() -> Program {
        let mut bindings = HashMap::<String, Literal>::new();
        let mut type_context = HashMap::<String, Type>::new();

        // cloning here is needed to insert into the hashmap
        for (name, func) in LIBRARY.clone() {
            bindings.insert(name.to_owned(), Literal::Func(func.clone()));
            type_context.insert(name.to_owned(), Type::Func {
                input: func.params.iter().map(|(_, t)| t.clone()).collect(),
                output: Box::new(func.return_type),
            });
        }

        Self { bindings, type_context }
    }
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

    // null token for lexing needed since type vs literal "null" is ambiguous
    Null,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Literal {
    Int(i32),
    String(String),
    Bool(bool),
    Func(Func),
    Null,
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(i) => write!(f, "{i}"),
            Self::String(s) => write!(f, "\"{s}\""),
            Self::Bool(s) => write!(f, "{s}"),
            Self::Func(_) => write!(f, "{}", get_literal_type(self)),
            Self::Null => write!(f, "null"),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Func {
   pub params: Vec<(String, Type)>,
   pub return_type: Type,
   pub body: FuncBody,
}

#[derive(PartialEq, Debug, Clone)]
pub enum FuncBody {
    Expr(Box<Expr>),
    Native(fn(Vec<Literal>) -> Literal)
}

// will be extended beyond literal types (e.g. functions, arrays, structs)
// so should not be merged with enum `Literal`
#[derive(PartialEq, Debug, Clone)]
pub enum Type {
    Int,
    String,
    Bool,
    Func { input: Vec<Type>, output: Box<Type> },
    Null,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::String => write!(f, "string"),
            Type::Bool => write!(f, "bool"),
            Type::Null => write!(f, "null"),
            Type::Func { input, output } => {
                // don't show parentheses for functions with one argument
                if input.len() == 1 {
                    return write!(f, "{} -> {}", input[0], output)
                }
                let args = input.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");

                write!(f, "({}) -> {}", args, output)
            }
        }
    }
}

// TODO: remove and replace with type-context aware function
pub fn get_literal_type(literal: &Literal) -> Type {
    match literal {
        Literal::Bool(_) => Type::Bool,
        Literal::Int(_) => Type::Int,
        Literal::String(_) => Type::String,
        Literal::Func(func) => Type::Func {
            input: func.params.iter().map(|(_, t)| t.clone()).collect(),
            output: Box::new(func.return_type.clone()),
        },
        Literal::Null => Type::Null,
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
    FuncCall(FuncCall),
    Expr(Box<Expr>),
    Not(Box<Term>),  // negate boolean terms
    Minus(Box<Term>),  // negate int terms
}

#[derive(Debug, PartialEq, Clone)]
pub struct FuncCall {
    // a function might be a variety of expressions, so we use the blanket type `Term`
    pub func: Box<Term>,
    pub args: Vec<Expr>,
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
    //FuncCall(FuncCall),
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
