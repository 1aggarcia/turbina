use core::fmt;
use std::collections::HashMap;

use crate::{library::LIBRARY, streams::OutputStreams};

/// State of the running program
#[derive(Debug)]
pub struct Program {
    pub bindings: HashMap<String, Literal>,
    pub type_context: HashMap<String, Type>,
    pub output: OutputStreams,
}

impl Program {
    /// Initialize a `Program` with library functions imported into the enviorment
    pub fn init(output: OutputStreams) -> Program {
        let mut bindings = HashMap::<String, Literal>::new();
        let mut type_context = HashMap::<String, Type>::new();

        // cloning here is needed to insert into the hashmap
        for (name, func) in LIBRARY.clone() {
            let Some(return_type) = &func.return_type else {
                eprintln!("WARNING: Cannot resolve return type for library function {}", name);
                continue;
            };
            bindings.insert(name.to_owned(), Literal::Closure(Closure {
                function: func.clone(),
                parent_scope: HashMap::new()
            }));

            type_context.insert(name.to_owned(), Type::Func {
                input: func.params.iter().map(|(_, t)| t.clone()).collect(),
                output: Box::new(return_type.clone())
            });
        }

        Self { bindings, type_context, output }
    }

    pub fn init_with_std_streams() -> Self {
        Self::init(OutputStreams::std_streams())
    }
}

/// A reference to the programs' output streams and current evaluation scope
pub struct EvalContext<'a> {
    pub output: &'a mut OutputStreams,
    pub scope: Scope<'a>,
}

/// Linked-list like structure to model all bindings that a can be accessed
/// in a scope.
/// 
/// Stores references to data (instead of copying) for memory efficiency.
#[derive(Debug)]
pub struct Scope<'a> {
    pub bindings: &'a mut HashMap<String, Literal>,
    pub parent: Option<&'a Scope<'a>>,
}

impl Scope<'_> {
    /// Search for a binding in the local scope. If it is not found,
    /// recursively search the parent scopes until reaching the global scope.
    pub fn lookup(&self, id: &str) -> Option<Literal> {
        match self.bindings.get(id) {
            // TODO: try to make this return a reference 
            Some(value) => Some(value.clone()),
            None => self.parent
                .and_then(|parent_scope| parent_scope.lookup(id)),
        }
    }

    /// Is this scope temporary, i.e. not the global scope?
    pub fn is_local_scope(&self) -> bool {
        self.parent.is_some()
    }
}


impl std::fmt::Display for Scope<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_scope(f, self, 0)
    }
}

static INDENT_SPACES: usize = 2;

/// For the Display trait of Scope.
/// Recursively format the scope and parent scopes as string
fn format_scope(
    f: &mut std::fmt::Formatter<'_>,
    scope: &Scope,
    indent_level: usize
) -> std::fmt::Result {
    let outer_indent = " ".repeat(indent_level * INDENT_SPACES);
    let inner_indent = " ".repeat((indent_level + 1) * INDENT_SPACES);

    writeln!(f, "{outer_indent}{{")?;
    for (k, v) in scope.bindings.clone() {
        writeln!(f, "{inner_indent}{}: {}", k, v)?;
    }
    if let Some(parent) = scope.parent {
        format_scope(f, parent, indent_level + 1)?;
        writeln!(f)?;
    }
    write!(f, "{outer_indent}{}", "}")
}

#[derive(PartialEq, Debug, Clone)]
pub enum Token {
    Literal(Literal),
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    Id(String),

    // formatters
    Newline,
    Colon,
    Semicolon,
    OpenParens,
    CloseParens,
    Comma,
    OpenSquareBracket,
    CloseSquareBracket,
    Arrow,

    // keywords
    Let,
    If,
    Else,
    Type(Type),

    // null token for lexing needed since type vs literal "null" is ambiguous
    Null,
}

impl Token {
    pub fn open_type_list() -> Self {
        Token::BinaryOp(BinaryOp::LessThan)
    }

    pub fn close_type_list() -> Self {
        Token::BinaryOp(BinaryOp::GreaterThan)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Literal {
    Int(i32),
    String(String),
    Bool(bool),
    Closure(Closure),
    List(Vec<Literal>),
    Null,
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(i) => write!(f, "{i}"),
            Self::String(s) => write!(f, "\"{s}\""),
            Self::Bool(s) => write!(f, "{s}"),
            Self::Closure(_) => write!(f, "<function>"),
            Self::List(list) => {
                let string_representation = list
                    .iter()
                    .map(|elem| elem.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "[{}]", string_representation)
            },
            Self::Null => write!(f, "null"),
        }
    }
}

/// A function with a copy of the scope it was created in
#[derive(PartialEq, Debug, Clone)]
pub struct Closure {
   pub function: Function,
   pub parent_scope: HashMap<String, Literal>,
}

#[derive(PartialEq, Debug, Clone)]
pub enum FuncBody {
    Expr(Box<Expr>),
    Native(fn(Vec<Literal>, &mut EvalContext) -> Literal)
}

#[derive(PartialEq, Debug, Clone, Hash, Eq)]
pub enum Type {
    // primitives
    Int,
    String,
    Bool,
    Null,
    EmptyList,

    // compound types
    Nullable(Box<Type>),
    Func { input: Vec<Type>, output: Box<Type> },
    List(Box<Type>),

    // type variable
    Generic(String),

    // type that includes all values
    Unknown,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::String => write!(f, "string"),
            Type::Bool => write!(f, "bool"),
            Type::Unknown => write!(f, "unknown"),
            Type::Null => write!(f, "null"),
            Type::EmptyList => write!(f, "[]"),
            Type::List(element_type) => {
                if matches!(**element_type, Type::Func { .. }) {
                    write!(f, "({})[]", element_type)
                } else {
                    write!(f, "{}[]", element_type)
                }
            },
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
            },
            Type::Generic(id) => write!(f, "{}", id),
            Type::Nullable(datatype) => {
                // nullable functions are ambiguous without parentheses
                if matches!(**datatype, Type::Func {..}) {
                    write!(f, "({})?", datatype)
                } else {
                    write!(f, "{}?", datatype)
                }
            },
        }
    }
}

impl Type {
    pub fn func(input: &[Type], output: Type) -> Self {
        Self::Func { input: input.to_vec(), output: Box::new(output) }
    }

    /// Create a list type from some base type
    pub fn as_list(self) -> Self {
        Self::List(Box::new(self))
    }

    /// Wrap a type in the nullable type.
    pub fn as_nullable(self) -> Self {
        if let Type::Nullable(_) = self {
            return self;
        }
        if Type::Null == self {
            return self;
        }
        Type::Nullable(Box::new(self))
    }

    /// Returns true if and only if this type is a valid member of the supertype
    pub fn is_assignable_to(&self, supertype: &Type) -> bool {
        // TODO: this essentially bypasses type checking, use a type context
        // and perform proper type checking with type parameters
        if matches!(self, Self::Generic(_)) {
            return true;
        }
        match supertype {
            Type::Unknown => true,
            Type::Generic(_) => true,
            Type::Nullable(inner_type) =>
                *self == Type::Null || self.is_assignable_to(inner_type),
            Type::Func { input: super_ins, output: super_out } => {
                let Type::Func { input: sub_ins, output: sub_out } = self else {
                    return false;
                };
                if !sub_out.is_assignable_to(&super_out) {
                    return false;
                }
                if super_ins.len() < sub_ins.len() {
                    return false;
                }
                // all inputs that the supertype accepts must be accepted by the subtype
                super_ins
                    .iter()
                    .zip(sub_ins.iter())
                    .all(|(super_in, sub_in)| super_in.is_assignable_to(sub_in))
            },
            Type::List(supertype_list) => match self {
                Type::EmptyList => true,
                Type::List(subtype_list) =>
                    subtype_list.is_assignable_to(&supertype_list),
                _ => false,
            },
            _ => supertype == self,
        }
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
    And,
    Or,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
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
            Self::And => "&&",
            Self::Or => "||",
            Self::LessThan => "<",
            Self::LessThanOrEqual => "<=",
            Self::GreaterThan => ">",
            Self::GreaterThanOrEqual => ">=",
        };
        write!(f, "{string}")?;
        Ok(())
    }
}

#[derive(PartialEq, Clone)]
pub enum UnaryOp {
    Equals,
    Not,
    Nullable,
}

impl fmt::Debug for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = match self {
            Self::Equals => "=",
            Self::Not => "!",
            Self::Nullable => "?",
        };
        write!(f, "{string}")?;
        Ok(())
    }
}

#[derive(PartialEq, Debug, Clone)]
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
    NotNull(Box<Term>),
    List(Vec<Expr>),
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

    /// Returns the term wrapped as `NotNull`
    pub fn as_not_null(self) -> Self {
        Self::NotNull(Box::new(self))
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Binary(BinaryExpr),
    Cond(CondExpr),
    Function(Function),
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

#[derive(PartialEq, Debug, Clone)]
pub struct Function {
    pub type_params: Vec<String>,
    pub params: Vec<(String, Type)>,
    pub return_type: Option<Type>,
    pub body: FuncBody,
}

#[derive(Debug, PartialEq, Clone)]
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

    pub fn type_token(datatype: Type) -> Token {
        Token::Type(datatype)
    }

    pub fn int_term(int: i32) -> Term {
        Term::Literal(Literal::Int(int))
    }

    pub fn str_term(string: &str) -> Term {
        Term::Literal(Literal::String(string.into()))
    }

    pub fn bool_term(boolean: bool) -> Term {
        Term::Literal(Literal::Bool(boolean))
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

#[cfg(test)]
mod test_type {
    use super::*;

    #[test]
    fn test_display_adds_parentheses_for_nullable_function() {
        let nullable_func = Type::Func {
            input: vec![],
            output: Box::new(Type::Int)
        }.as_nullable();

        assert_eq!(format!("{}", nullable_func), "(() -> int)?");
    }

    #[test]
    fn test_display_function_with_nullable_return_type() {
        let func = Type::Func {
            input: vec![],
            output: Box::new(Type::Int.as_nullable())
        };
        assert_eq!(format!("{}", func), "() -> int?");
    }
}
