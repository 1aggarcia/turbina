use crate::errors::Result;
use crate::models::{Import, Token};
use crate::parser::shared_parsers::parse_id;
use crate::parser::utils::{match_next, next_token_matches};
use crate::streams::TokenStream;

/// ```text
/// <import> ::= Import {<id> Dot} <id>
/// ```
pub fn parse_import(tokens: &mut TokenStream) -> Result<Import> {
    match_next(tokens, Token::Import)?;

    let mut import = Import { path_elements: vec![] };
    import.path_elements.push(parse_id(tokens)?);
    while next_token_matches(tokens, Token::Dot) {
        tokens.pop()?; 
        import.path_elements.push(parse_id(tokens)?);
    }

    Ok(import)
}
