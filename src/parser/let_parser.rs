use crate::models::{Expr, LetNode, Token, UnaryOp};

use crate::errors::Result;
use crate::parser::expr_parser::{parse_expr, parse_function};
use crate::parser::shared_parsers::parse_id;
use crate::parser::type_declaration_parser::parse_type_declaration;
use crate::parser::utils::{match_next, skip_newlines};
use crate::streams::TokenStream;

/// Create an AST for the "let" keyword given the remaining tokens
/// ```text
/// <let> ::= Let Id [<type_declaration>] Equals <expr> | Let Id <function>
/// ```
pub fn parse_let(tokens: &mut TokenStream) -> Result<LetNode> {
    match_next(tokens, Token::Let)?;
    let id = parse_id(tokens)?;

    let includes_type_declaration = tokens.peek()? == Token::Colon;
    let is_shorthand_function = [
        Token::OpenParens,
        Token::open_type_list()
    ].contains(&tokens.peek()?);

    let datatype = if includes_type_declaration {
        Some(parse_type_declaration(tokens)?)
    } else {
        None
    };

    let expr = if is_shorthand_function {
        let function = parse_function(tokens)?;
        Expr::Function(function)
    } else {
        match_next(tokens, Token::UnaryOp(UnaryOp::Equals))?;
        skip_newlines(tokens);
        parse_expr(tokens)?
    };

    return Ok(LetNode { id, datatype, value: expr });
}
