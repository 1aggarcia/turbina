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

#[cfg(test)]
mod test {
    use rstest::rstest;
    use crate::parser::test_utils::*;
    use crate::models::test_utils::*;
    use crate::models::{BinaryOp, Term, Expr, FuncBody, Function};

    mod list {
        use super::*;

        #[rstest]
        #[case::empty("[];", &[])]
        #[case::one_element("[1];", &[1])]
        #[case::many_elements("[1, 6, 2, 4543, 5];", &[1, 6, 2, 4543, 5])]
        fn it_parses_list_of_literals(#[case] input: &str, #[case] expected: &[i32]) {
            let tokens = force_tokenize(input);
            let list_term = Term::List(
                expected.iter().map(|num| term_expr(int_term(*num))).collect()
            );
            assert_eq!(parse_tokens(tokens), Ok(term_tree(list_term)));
        }

        #[test]
        fn it_parses_list_of_expressions() {
            let tokens = force_tokenize("[false && true, () -> 15];");
            let expected_list = vec![
                bin_expr(
                    bool_term(false),
                    vec![(BinaryOp::And, bool_term(true))]
                ),
                Expr::Function(Function {
                    type_params: vec![],
                    params: vec![],
                    return_type: None,
                    body: FuncBody::Expr(Box::new(term_expr(int_term(15))))
                })
            ];
            let expected_tree = term_tree(Term::List(expected_list));
            assert_eq!(parse_tokens(tokens), Ok(expected_tree));
        }
    }
}