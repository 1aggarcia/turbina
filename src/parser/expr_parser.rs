use crate::models::{
    AbstractSyntaxTree, BinaryExpr, BinaryOp, CodeBlock, CondExpr, Expr, FuncBody, FuncCall, Function, Literal, Term, Token, Type, UnaryOp
};

use crate::errors::{error, InterpreterError, Result};
use crate::parser::let_parser::parse_let;
use crate::parser::shared_parsers::{ListParserConfig, parse_id, parse_list};
use crate::parser::type_declaration_parser::parse_type_declaration;
use crate::parser::utils::{match_next, next_token_matches, skip_newlines};
use crate::streams::TokenStream;

/// ```text
/// <expr> ::= <code_block> | <cond_expr> | <function> | <binary_expr>
/// ```
pub fn parse_expr(tokens: &mut TokenStream) -> Result<Expr> {
    if tokens.peek()? == Token::If {
        let cond_expr = parse_cond_expr(tokens)?;
        return Ok(Expr::Cond(cond_expr));
    }
    if tokens.peek()? == Token::OpenCurlyBracket {
        let code_block = parse_code_block(tokens)?;
        return Ok(Expr::CodeBlock(code_block));
    }

    let next_is_function =
        if tokens.lookahead(0)? == Token::open_type_list() {
            true
        } else if tokens.lookahead(0)? != Token::OpenParens {
            false
        } else if tokens.lookahead(1)? == Token::CloseParens {
            true
        } else {
            // function with at least one argument
            matches!(tokens.lookahead(1)?, Token::Id(_))
            && [
                Token::CloseParens,
                Token::Colon,
                Token::Comma,
            ].contains(&tokens.lookahead(2)?)
        };

    if next_is_function {
        let function = parse_function(tokens)?;
        let expr = Expr::Function(function);
        return Ok(expr);
    }

    let binary_expr = parse_binary_expr(tokens, 0)?;
    Ok(Expr::Binary(binary_expr))
}

/// ```text
/// <cond_expr> ::= If "(" <expr> ")" <expr> Else <expr>
/// ```
fn parse_cond_expr(tokens: &mut TokenStream) -> Result<CondExpr> {
    match_next(tokens, Token::If)?;
    match_next(tokens, Token::OpenParens)?;
    let condition = Box::new(parse_expr(tokens)?);
    match_next(tokens, Token::CloseParens)?;

    skip_newlines(tokens);
    let if_true = Box::new(parse_expr(tokens)?);

    skip_newlines(tokens);
    match_next(tokens, Token::Else)?;

    skip_newlines(tokens);
    let if_false = Box::new(parse_expr(tokens)?);

    return Ok(CondExpr { cond: condition, if_true, if_false });
}

/// <code_block> ::= 
///     | "{" (<statement> | {<statement>}) [<expr>] "}"
///     | "{" <expr> "}"
fn parse_code_block(tokens: &mut TokenStream) -> Result<CodeBlock> {
    let mut statements = vec![];

    match_next(tokens, Token::OpenCurlyBracket)?;
    skip_newlines(tokens);
    while !next_token_matches(tokens, Token::CloseCurlyBracket) {
        let statement = match tokens.peek()? {
            Token::Let => AbstractSyntaxTree::Let(parse_let(tokens)?),
            _ => AbstractSyntaxTree::Expr(parse_expr(tokens)?),
        };
        statements.push(statement);
        let statement_end = tokens.peek()?;
        match statement_end {
            Token::Semicolon | Token::Newline => {
                tokens.pop()?;
            },
            // final expression without semicolon allowed, but no more
            // statements are allowed afterwards
            _ => break,
        }
        skip_newlines(tokens);
    }
    match_next(tokens, Token::CloseCurlyBracket)?;

    if statements.is_empty() {
        return Err(InterpreterError::EmptyCodeBlock);
    }
    let code_block = CodeBlock { statements };
    Ok(code_block)
}

/// ```text
/// <function> ::= [type_param_list] "(" <param_list> ")" [<type_declaration>] <function_body>
/// <function_body> ::= "->" <expr> | <code_block>
/// 
/// <type_param_list> ::= "<" [Id] {"," Id} ">"
/// <param_list> ::= [<param> {"," <param>}]
/// ```
pub fn parse_function(tokens: &mut TokenStream) -> Result<Function> {
    let mut type_params = Vec::<String>::new();

    if tokens.peek()? == Token::open_type_list() {
        tokens.pop()?;
        if tokens.peek()? == Token::close_type_list() {
            return Err(InterpreterError::EmptyTypeList);
        }
        type_params = parse_list(tokens, ListParserConfig {
            item_parser: parse_id,
            opening_token: None,
            closing_token: Token::close_type_list(),
        })?;
    }

    let params = parse_list(tokens, ListParserConfig {
        item_parser: parse_param,
        opening_token: Some(Token::OpenParens),
        closing_token: Token::CloseParens,
    })?;

    let return_type = if tokens.peek()? == Token::Colon {
        Some(parse_type_declaration(tokens)?)
    } else {
        None
    };

    let body_expr = if tokens.peek()? == Token::OpenCurlyBracket {
        Expr::CodeBlock(parse_code_block(tokens)?)
    } else {
        match_next(tokens, Token::Arrow)?;
        skip_newlines(tokens);
        parse_expr(tokens)?
    };

    Ok(Function {
        type_params,
        params,
        return_type,
        body: FuncBody::Expr(Box::new(body_expr))
    })
}

static MAX_EXPRESSION_PRECEDENCE: u8 = 4;

/// ```text
/// <binary_expr> ::= <expr_lvl_0> {Pipe <expr_lvl_0>}
/// <expr_lvl_0> ::= <expr_lvl_1> {(And | Or) <expr_lvl_1>}
/// <expr_lvl_1> :: = <expr_lvl_2> {(== | != | < | <= | > | >=) <expr_lvl_2>}
/// <expr_lvl_2> :: = <expr_lvl_3> {(+ | -) <expr_lvl_3>}
/// <expr_lvl_3> :: = <term> {(* | / | %) <term>}
/// ```
/// Operator precedence is expressed by giving levels to each expression; e.g.
/// `<expr_lvl_2>` has precedence above `<expr_lvl_3>` but not `<expr_lvl_1>`.
/// Naming things is hard, so why bother? Let's just use numbers.
fn parse_binary_expr(
    tokens: &mut TokenStream,
    precedence: u8
) -> Result<BinaryExpr> {
    if precedence > MAX_EXPRESSION_PRECEDENCE {
        let expr = BinaryExpr { first: parse_term(tokens)?, rest: vec![] };
        return Ok(expr);
    }

    let first = parse_binary_expr(tokens, precedence + 1)?;
    let mut rest = vec![];

    while next_is_operator(tokens, precedence) {
        let op_token = tokens.pop()?;
        let operator = match op_token {
            Token::BinaryOp(op) => op,
            _ => return Err(error::unexpected_token("binary op", op_token))
        };
        skip_newlines(tokens);
        let expr = parse_binary_expr(tokens, precedence + 1)?;
        rest.push((operator, expr_as_term(expr)));
    }

    // avoid re-wrapping expressions as another expression if possible
    if rest.is_empty() {
        return Ok(first);
    }
    Ok(BinaryExpr { first: expr_as_term(first), rest })
}

///```text
/// <param> ::= Id [<type_declaration>]
/// ```
fn parse_param(tokens: &mut TokenStream) -> Result<(String, Type)> {
    let id = parse_id(tokens)?;
    let datatype = if next_token_matches(tokens, Token::Colon) {
        parse_type_declaration(tokens)?
    } else {
        Type:: Unknown
    };

    Ok((id, datatype))
}

/// ```text
/// <term> ::= "!" <term> | ["-"] <base_term>
/// ```
fn parse_term(tokens: &mut TokenStream) -> Result<Term> {
    let first = tokens.peek()?;

    if first == Token::UnaryOp(UnaryOp::Not) {
        tokens.pop()?;
        let inner_term = parse_term(tokens)?;
        return Ok(Term::negated_bool(inner_term));
    }
    if first == Token::BinaryOp(BinaryOp::Minus) {
        tokens.pop()?;
        let base_term = parse_base_term(tokens)?;
        return Ok(Term::negative_int(base_term));
    }
    parse_base_term(tokens)
}

/// ```text
/// <base_term> ::=
///     | Null
///     | Literal
///     | <list>
///     | (Id | "(" <expr> ")") ["!"] {<arg_list> ["!"]}
/// 
/// <list> ::= "[" [<expr> {"," <expr>}] "]"
/// ```
fn parse_base_term(tokens: &mut TokenStream) -> Result<Term> {
    let first = tokens.pop()?;
    if Token::Null == first {
        return Ok(Term::Literal(Literal::Null));
    }
    if let Token::Literal(lit) = first {
        return Ok(Term::Literal(lit));
    }
    if Token::OpenSquareBracket == first {
        let elements = parse_list(tokens, ListParserConfig {
            item_parser: parse_expr,
            opening_token: None,
            closing_token: Token::CloseSquareBracket,
        })?;
        return Ok(Term::List(elements));
    }
    let callable = if let Token::Id(id) = first {
        Term::Id(id)
    } else if first == Token::OpenParens {
        skip_newlines(tokens);
        let expr = parse_expr(tokens)?;
        skip_newlines(tokens);
        match_next(tokens, Token::CloseParens)?;
        Term::Expr(Box::new(expr))
    } else {
        return Err(error::unexpected_token("identifier or expression", first));
    };

    if next_token_matches(tokens, Token::UnaryOp(UnaryOp::Not)) {
        tokens.pop()?;
        complete_term_with_arg_list(callable.as_not_null(), tokens)
    } else {
        complete_term_with_arg_list(callable, tokens)
    }
}

/// Try to construct a function call term with the function supplied as a
/// callable term. If a function call cannot be constructed, the callable term is returned.
/// 
/// The `<arg_list>` below is preceded by the callable term and may be
/// succeeded by the not-null assertion operator "!".
/// ```text
/// <arg_list> ::= "(" [<expr> {"," <expr>}] ")"
/// ```
fn complete_term_with_arg_list(
    callable: Term,
    tokens: &mut TokenStream
) -> Result<Term> {
    // we might be at the end of the stream, but that's allowed since
    // callable is a valid term
    if !next_token_matches(tokens, Token::OpenParens) {
        return Ok(callable);
    }
    let args = parse_list(tokens, ListParserConfig {
        item_parser: parse_expr,
        opening_token: Some(Token::OpenParens),
        closing_token: Token::CloseParens,
    })?;

    let func_call = FuncCall { func: Box::new(callable), args };
    let callable = if next_token_matches(tokens, Token::UnaryOp(UnaryOp::Not)) {
        tokens.pop()?;
        Term::FuncCall(func_call).as_not_null()
    } else {
        Term::FuncCall(func_call)
    };

    complete_term_with_arg_list(callable, tokens)
}

/// If the expression contains one term, unwrap the term.
/// Otherwise, wrap the expression into a term
fn expr_as_term(expr: BinaryExpr) -> Term {
    if expr.rest.is_empty() {
        expr.first
    } else {
        Term::Expr(Box::new(Expr::Binary(expr)))
    }
}

/// Decide if the next token is an operator with the given precedence without
/// consuming it. If there is no next token, returns false.
/// 
/// Consumes any newlines at the front of the stream.
fn next_is_operator(tokens: &mut TokenStream, precedence: u8) -> bool {
    skip_newlines(tokens);
    let Ok(Token::BinaryOp(operator)) = tokens.peek() else {
        return false;
    };
    return get_binary_operator_precedence(operator) == precedence
}

/// Returns precedence of operator, larger numbers being higher precedence
fn get_binary_operator_precedence(operator: BinaryOp) -> u8 {
    use BinaryOp::*;

    // pattern matching guarantees that all operators have exactly one precedence level
    match operator {
        Pipe => 0,

        And | Or => 1,

        Equals | NotEq | LessThan | LessThanOrEqual
        | GreaterThan | GreaterThanOrEqual => 2,

        Plus | Minus => 3,

        Star | Slash | Percent => 4,
    }
}
