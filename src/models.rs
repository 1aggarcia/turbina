use core::fmt;

#[derive(PartialEq, Debug, Clone)]
pub enum TokenV2 {
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
#[derive(PartialEq, Debug, Clone)]
pub enum Type {
    Int,
    String,
    Bool,
}

#[derive(PartialEq, Clone, Copy)]
pub enum Operator {
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Equals,
}

impl fmt::Debug for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = match self {
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Star => "*",
            Self::Slash => "/",
            Self::Percent => "%",
            Self::Equals => "=",
        };
        write!(f, "{string}")?;
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub enum AbstractSyntaxTreeV2 {
    Let(LetNode),
    Operator(OperatorNode),

    // leaf nodes
    Literal(Literal),
    Id(String),
}

/// For binary operators
#[derive(Debug, PartialEq)]
pub struct OperatorNode {
    pub operator: Operator,
    pub left: Box<AbstractSyntaxTreeV2>,
    pub right: Box<AbstractSyntaxTreeV2>,
}

#[derive(Debug, PartialEq)]
pub struct LetNode {
    pub id: String,
    pub datatype: Option<Type>,
    pub value: Box<AbstractSyntaxTreeV2>,
}

#[cfg(test)]
pub mod test_utils {
    use super::*;

    pub fn bool_token(data: bool) -> TokenV2 {
        TokenV2::Literal(Literal::Bool(data))
    }

    pub fn string_token(data: &str) -> TokenV2 {
        TokenV2::Literal(Literal::String(data.to_string()))
    }

    pub fn int_token(data: i32) -> TokenV2 {
        TokenV2::Literal(Literal::Int(data))
    }

    pub fn op_token(operator: Operator) -> TokenV2 {
        TokenV2::Operator(operator)
    }

    pub fn id_token(data: &str) -> TokenV2 {
        TokenV2::Id(data.to_string())
    }

    pub fn formatter_token(data: &str) -> TokenV2 {
        TokenV2::Formatter(data.to_string())
    }

    pub fn type_token(datatype: Type) -> TokenV2 {
        TokenV2::Type(datatype)
    }
}


// ----------------------------------------------------------------------------
// UNUSED
// ----------------------------------------------------------------------------

#[derive(PartialEq)]
struct AbstractSyntaxTree {
    token: TokenV2,
    children: Vec<AbstractSyntaxTree>,
}


impl AbstractSyntaxTree {
    fn leaf(token: TokenV2) -> Self {
        Self { token, children: vec![] }
    }

    /// returns true if this node can accept another node as the next
    /// child, false otherwise
    fn can_accept(&self, node: &AbstractSyntaxTree) -> bool {
        match self.token {
            TokenV2::Operator(_) => self.children.len() < 2,
            _ => false,
        }
    }
}

impl fmt::Debug for AbstractSyntaxTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // recursively print out children
        fn pretty_print(
            f: &mut std::fmt::Formatter<'_>,
            node: &AbstractSyntaxTree,
            indent: usize
        ) {
            writeln!(
                f,
                "{}{:?}",
                " ".repeat(indent * 4),
                node.token
            ).expect("failed to write to formatter");
            for child in &node.children {
                pretty_print(f, &child, indent + 1);
            }
        }
        writeln!(f, "")?;
        pretty_print(f, self, 0);
        Ok(())
    }
}