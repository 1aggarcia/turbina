use crate::errors::{Result, error};
use crate::models::{Token, Type, UnaryOp};
use crate::parser::shared_parsers::{ListParserConfig, parse_list};
use crate::parser::utils::{match_next, next_token_matches};
use crate::streams::TokenStream;

/// Type declarations cannot include function types without parentheses
/// since return types and function bodies are ambiguous 
///
/// e.g. `(): (bool -> bool) -> (x: bool) -> true` is a function that returns a
/// function returning bool, but without parentheses it is unclear. (Even with
/// parentheses I find it confusing, I suggest just using inferred return types).
/// ```text
/// <type_declaration> :: = ":" <base_type>
/// ```
pub fn parse_type_declaration(tokens: &mut TokenStream) -> Result<Type> {
    match_next(tokens, Token::Colon)?;
    parse_base_type(tokens)
}

/// ```text
/// <base_type> ::= Type ["?"] | "(" <type> ")" ["?"] | <base-type> "[]"
/// ```
fn parse_base_type(tokens: &mut TokenStream) -> Result<Type> {
    if tokens.peek()? == Token::OpenParens {
        tokens.pop()?;
        let datatype = parse_type(tokens)?;
        match_next(tokens, Token::CloseParens)?;
        return complete_base_type(datatype, tokens);
    }

    let type_token = tokens.pop()?;
    if let Token::Null = type_token {
        return Ok(Type::Null);
    }

    let datatype = match type_token {
        Token::Type(literal_type) => literal_type,
        Token::Id(type_name) => Type::Generic(type_name),
        _ => return Err(error::not_a_type(type_token)), 
    };
    complete_base_type(datatype, tokens)
}

/// ```text
/// <type> ::= <base_type> | <function_type>
/// ```
///
/// Note the `arg_types` non-terminal cannot be a list of one type, since that
/// conflicts with a variant of `base_type` 
/// ```text
/// <function_type> ::= <arg_types> "->" <type>
/// <arg_types> ::= <base_type> | "(" ") | "(" <type> "," <type> {"," <type>} ")"
/// ```
fn parse_type(tokens: &mut TokenStream) -> Result<Type> {
    /// Decide between leaving a base type alone or parsing it as a function
    fn complete_type(
        base_type: Type, tokens: &mut TokenStream
    ) -> Result<Type> {
        if next_token_matches(tokens, Token::Arrow) {
            complete_function_type(vec![base_type], tokens)
        } else {
            Ok(base_type)
        }
    }

    fn complete_function_type(
        arg_types: Vec<Type>,
        tokens: &mut TokenStream
    ) -> Result<Type> {
        match_next(tokens, Token::Arrow)?;
        let return_type = parse_type(tokens)?;

        let func_type = Type::Func {
            input: arg_types,
            output: Box::new(return_type)
        };
        Ok(func_type)
    }

    // start of parsing
    if tokens.peek()? != Token::OpenParens {
        let base_type = parse_base_type(tokens)?;
        return complete_type(base_type, tokens);
    }

    let types_in_parens = parse_list(tokens, ListParserConfig {
        item_parser: parse_type,
        opening_token: Some(Token::OpenParens),
        closing_token: Token::CloseParens,
    })?;

    if types_in_parens.len() == 1 {
        let base_type = complete_base_type(types_in_parens[0].clone(), tokens)?;
        // one type in parentheses is allowed to not be a function
        return complete_type(base_type, tokens);
    }
    complete_function_type(types_in_parens, tokens)

}

/// Given a base type, recursively wrap it as a list type if the next tokens
/// are "[]". If the next token is "?", make it a nullable type. Nullable
/// types cannot be stacked the same way list types can.
fn complete_base_type(base_type: Type, tokens: &mut TokenStream) -> Result<Type> {
    if !matches!(base_type, Type::Nullable(..))
        && next_token_matches(tokens, Token::UnaryOp(UnaryOp::Nullable))
    {
        tokens.pop()?;
        complete_base_type(base_type.as_nullable(), tokens)
    } else if next_token_matches(tokens, Token::OpenSquareBracket) {
        tokens.pop()?;
        match_next(tokens, Token::CloseSquareBracket)?;
        complete_base_type(base_type.as_list(), tokens)
    } else {
        Ok(base_type.clone())
    }
}
