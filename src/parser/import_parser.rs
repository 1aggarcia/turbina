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

#[cfg(test)]
mod test {
    use super::*;
    use crate::errors::error;
    use crate::parser::test_utils::*;
    use crate::models::AbstractSyntaxTree;

    mod import_statement {

        use super::*;

        #[test]
        fn it_returns_error_for_missing_path() {
            let input = force_tokenize("import;");
            let error =
                error::unexpected_token("identifier", Token::Semicolon);
            assert_eq!(parse_tokens(input), Err(error));
        }

        #[test]
        fn it_returns_correct_path_for_one_path_element() {
            let input = force_tokenize("import someLibrary;");
            let expected = Import { path_elements: vec!["someLibrary".into()] };
            assert_eq!(
                parse_tokens(input),
                Ok(AbstractSyntaxTree::Import(expected))
            );
        }

        #[test]
        fn it_returns_correct_path_for_many_path_elements() {
            let input = force_tokenize("import src.directory.utils;");
            let expected = Import {
                path_elements: vec![
                    "src".into(),
                    "directory".into(),
                    "utils".into(),
                ]
            };
            assert_eq!(
                parse_tokens(input),
                Ok(AbstractSyntaxTree::Import(expected))
            );
        }
    }
}
