use crate::parser::utils::{match_next, skip_newlines};

use crate::{models::Token, streams::TokenStream};
use crate::errors::{Result, error};

/// Match the next token as an Id and extract the String name from it
pub fn parse_id(tokens: &mut TokenStream) -> Result<String> {
    let id_token = tokens.pop()?;
    match id_token {
        Token::Id(s) => Ok(s.to_string()),
        _ => Err(error::unexpected_token("identifier", id_token)),
    }
}

pub struct ListParserConfig<T> {
    pub item_parser: fn(&mut TokenStream) -> Result<T>,
    /// Set to `None` if the first token is already consumed
    pub opening_token: Option<Token>,
    pub closing_token: Token,
}

/// Utility to parse lists of any type of element and with any open/closing
/// brackets (e.g. [], (), <>)
/// ```text
/// <generic_list> ::= <opening_token> [<item> {"," <item>} [","]] <closing_token>
/// ```
pub fn parse_list<T>(
    tokens: &mut TokenStream,
    config: ListParserConfig<T>
) -> Result<Vec<T>> {
    let mut items = Vec::<T>::new();

    if let Some(token) = config.opening_token {
        match_next(tokens, token)?;
    }
    while tokens.peek()? != config.closing_token {
        if !items.is_empty() {
            match_next(tokens, Token::Comma)?;
        }
        skip_newlines(tokens);
        if tokens.peek()? != config.closing_token {
            let next_item = (config.item_parser)(tokens)?;
            items.push(next_item);
        }
    }
    skip_newlines(tokens);
    match_next(tokens, config.closing_token)?;
    Ok(items)
}
