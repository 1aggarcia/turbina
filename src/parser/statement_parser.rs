use crate::models::{AbstractSyntaxTree, Token};
use crate::errors::{InterpreterError, Result};
use crate::parser::expr_parser::parse_expr;
use crate::parser::import_parser::parse_import;
use crate::parser::let_parser::parse_let;
use crate::parser::utils::skip_newlines;
use crate::streams::TokenStream;

/// Consumes the next statement from the token stream and returns a syntax tree
/// representing the statement using recursive descent parsing.
/// 
/// A syntax error is returned for any syntactical errors in the token sequence.
/// 
/// ```text
/// <statement> ::=  (<let> | <expr>) [";" | Newline]
/// ```
pub fn parse_statement(token_stream: &mut TokenStream) -> Result<AbstractSyntaxTree> {
    skip_newlines(token_stream);

    let Ok(first) = token_stream.peek() else {
        return Err(InterpreterError::EndOfFile);
    };
    let statement = match first {
        Token::Let => AbstractSyntaxTree::Let(parse_let(token_stream)?),
        Token::Import => AbstractSyntaxTree::Import(
            parse_import(token_stream)?
        ),
        _ => AbstractSyntaxTree::Expr(parse_expr(token_stream)?),
    };

    let statement_end = token_stream.pop()?;
    match statement_end {
        Token::Semicolon | Token::Newline => {},
        _ => return Err(InterpreterError::end_of_statement(statement_end)),
    }
    return Ok(statement);
}

#[cfg(test)]
pub mod test_utils {
    use crate::{lexer::tokenize, streams::StringStream};

    use super::*;

    /// Bypass errors and convert a string statement into a syntax tree
    pub fn make_tree(statement: &str) -> AbstractSyntaxTree {
        let input_stream = StringStream::new(statement);
        let mut token_stream = TokenStream::new(Box::new(input_stream));
        parse_statement(&mut token_stream).unwrap()
    }

    pub fn parse_tokens(tokens: Vec<Token>) -> Result<AbstractSyntaxTree> {
        let mut tokens = TokenStream::from_tokens(tokens);
        parse_statement(&mut tokens)
    }

    pub fn force_tokenize(text: &str) -> Vec<Token> {
        text
            .split("\n")
            .flat_map(|line| tokenize(line).unwrap())
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::models::*;
    use crate::parser::test_utils::{force_tokenize, parse_tokens};
    use crate::errors::error;
    use rstest::rstest;

    #[rstest]
    #[case(force_tokenize("let"), error::unexpected_end_of_input())]
    #[case(force_tokenize("let x"), error::unexpected_end_of_input())]
    #[case(force_tokenize("3 +"), error::unexpected_end_of_input())]
    fn it_returns_error_for_incomplete_statements(
        #[case] tokens: Vec<Token>,
        #[case] error: InterpreterError
    ) {
        assert_eq!(parse_tokens(tokens), Err(error));
    }

    #[test]
    fn it_returns_error_for_statement_without_newline_or_semicolon() {
        let input = force_tokenize("2 + 2");
        let expected = error::unexpected_end_of_input();
        assert_eq!(parse_tokens(input), Err(expected));
    }

    #[test]
    fn it_ignores_tokens_after_semicolon() {
        let multiple_statements =
        parse_tokens(force_tokenize("2 + 2; 5 + 5; let = badsyntax ->"));
        let one_statement = parse_tokens(force_tokenize("2+2;"));
    
        assert!(matches!(multiple_statements, Ok(_)));
        assert_eq!(multiple_statements, one_statement);
    }

    #[rstest]
    #[case::shorthand_function("let not(b: bool) -> !b;", "let not = (b: bool) -> !b;")]
    #[case::shorthand_function_with_generic_types(
        "let identity <T> (x: T) -> x;",
        "let identity = <T> (x: T) -> x;",
    )]
    #[case::if_else_on_separate_lines("if (true) 1 else 0;", "
        if (true)
            1
        else
            0;
    ")]
    #[case::expression_in_parens_on_separate_lines("(5 + 4);", "
        (
            5 + 4
        );
    ")]
    #[case::function_on_separate_lines(
        r#"let toString(x: int) -> if (x == 0) "0" else if (x == 1) "1" else "unknown";"#,
        r#"
        let toString(x: int) ->
            if (x == 0) "0"
            else if (x == 1) "1"
            else "unknown";
        "#
    )]
    #[case::function_args_on_separate_lines("randInt(1, 2);", "
        randInt(
            1,
            2  // TODO: this should work with trailing comma
        );
    ")]
    #[case::function_args_split_unconventionally("randInt(1, 2);", "
        randInt(


            1
            ,

            2);
    ")]
    #[case::function_args_with_trailing_comma("randInt(1, 2);", "
        randInt(
            1,
            2,
        );
    ")]
    #[case::function_body_without_arrow(
        "let fn(x: int) -> {
            let square = x * 2;
            square
        };",
        "let fn(x: int) {
            let square = x * 2;
            square
        };"
    )]
    #[case::list_with_trailing_comma("[1,2,3,4,5];", "[
        1, 2,
        3, 4, 5,
    ];")]
    fn it_parses_equivalent_statements(#[case] vers1: &str, #[case] vers2: &str) {
        let res1 = parse_tokens(force_tokenize(vers1));
        let res2 = parse_tokens(force_tokenize(vers2));

        assert_eq!(res1, res2);
        assert!(matches!(res1, Ok(_)));
    }

    #[test]
    fn it_returns_end_of_file_error_for_empty_stream() {
        assert_eq!(parse_tokens(vec![]), Err(InterpreterError::EndOfFile));
    }
}
