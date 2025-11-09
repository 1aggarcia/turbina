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

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;
    use crate::errors::{error, InterpreterError};
    use crate::parser::test_utils::*;
    use crate::models::test_utils::*;
    use crate::models::{AbstractSyntaxTree, BinaryOp, Literal, Term, Type};
    
    mod let_binding {
        use super::*;

        fn test_parse_let(tokens: Vec<Token>) -> Result<LetNode> {
            let ast = parse_tokens(tokens)?;
            match ast {
                AbstractSyntaxTree::Let(let_node) => Ok(let_node),
                other => panic!("Not a binding: {:?}", other),
            }
        }

        #[test]
        fn it_returns_error_for_bad_var_id() {
            let input = force_tokenize("let 3 = 3;");
            let error = error::unexpected_token("identifier", int_token(3));
            assert_eq!(test_parse_let(input), Err(error));
        }

        #[test]
        fn it_returns_error_for_equals_in_let_expr() {
            let input = force_tokenize("let x = 1 = 0;");
            let error = InterpreterError::end_of_statement(
                Token::UnaryOp(UnaryOp::Equals)
            );
            assert_eq!(test_parse_let(input), Err(error)); 
        }

        #[rstest]
        #[case::int_literal("let x = 4;", LetNode {
            id: "x".to_string(),
            datatype: None,
            value: term_expr(int_term(4)),
        })]
        #[case::simple_expression("let two = 6 / 3;", LetNode {
            id: "two".to_string(),
            datatype: None,
            value: bin_expr(
                int_term(6),
                vec![(BinaryOp::Slash, int_term(3))]
            ),
        })]
        #[case::declared_type("let twoStrings: string = \"a\" + \"b\";", LetNode {
            id: "twoStrings".to_string(),
            datatype: Some(Type::String),
            value: bin_expr(
            str_term("a"),
                vec![(BinaryOp::Plus, str_term("b"))]
            ),
        })]
        #[case::nullable_type("let nullableData: string? = null;", LetNode {
            id: "nullableData".into(),
            datatype: Some(Type::String.as_nullable()),
            value: bin_expr(Term::Literal(Literal::Null), vec![]),
        })]
        #[case::list_type("let nums: int[] = [1, 2];", LetNode {
            id: "nums".into(),
            datatype: Some(Type::Int.as_list()),
            value: term_expr(Term::List(vec![
                term_expr(int_term(1)),
                term_expr(int_term(2)),
            ]))
        })]
        #[case::nested_list_type("let x: string[][][] = [];", LetNode {
            id: "x".into(),
            datatype: Some(Type::String.as_list().as_list().as_list()),
            value: term_expr(Term::List(vec![])),
        })]
        #[case::nested_list_and_nullable_types("let x: int?[][]?[] = [];", LetNode {
            id: "x".into(),
            datatype: Some(
                Type::Int.as_nullable()
                    .as_list()
                    .as_list().as_nullable()
                    .as_list()
            ),
            value: term_expr(Term::List(vec![])),
        })]
        #[case::multiple_lines(r#"
            let x: string =
                "some expression too long for one line";
        "#, LetNode {
            id: "x".into(),
            datatype: Some(Type::String), 
            value: term_expr(str_term("some expression too long for one line"))
        })]
        fn it_parses_var_binding(#[case] input: &str, #[case] expected: LetNode) {
            let input = force_tokenize(input);
            assert_eq!(test_parse_let(input), Ok(expected));
        }

        #[rstest]
        #[case::simple("(int -> bool)", Type::func(&[Type::Int], Type::Bool))]
        #[case::no_args("(() -> bool)", Type::func(&[], Type::Bool))]
        #[case::nullable(
            "(int -> bool)?",
            Type::func(&[Type::Int], Type::Bool).as_nullable()
        )]
        #[case::many_args(
            "((int, string) -> bool)",
            Type::func(&[Type::Int, Type::String], Type::Bool))
        ]
        #[case::func_retruning_func(
            "(int -> bool -> string)",
            Type::func(&[Type::Int], Type::func(&[Type::Bool], Type::String))
        )]
        #[case::func_retruning_func_in_parens(
            "(int -> (bool -> string))",
            Type::func(&[Type::Int], Type::func(&[Type::Bool], Type::String))
        )]
        #[case::func_with_func_input(
            "((int -> int) -> unknown)",
            Type::func(&[Type::func(&[Type::Int], Type::Int)], Type::Unknown)
        )]
        fn it_parses_correct_function_type(
            #[case] type_string: &str,
            #[case] function_type: Type
        ) {
            let tokens = force_tokenize(&format!("let f: {} = null;", type_string));
            let expected = LetNode {
                id: "f".into(),
                datatype: Some(function_type),
                value: bin_expr(Term::Literal(Literal::Null), vec![]),
            };

            assert_eq!(test_parse_let(tokens), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_nullable_null() {
            let input = force_tokenize("let nullableData: null? = null;");
            let expected = error::unexpected_token(
                &format!("{:?}", Token::UnaryOp(UnaryOp::Equals)),
                Token::UnaryOp(UnaryOp::Nullable)
            );
            assert_eq!(test_parse_let(input), Err(expected));
        }

        #[test]
        fn it_returns_error_for_multiple_nullable_operators() {
            let input = force_tokenize("let nullable: int????? = 2;");
            let expected = error::unexpected_token(
                &format!("{:?}", Token::UnaryOp(UnaryOp::Equals)),
                Token::UnaryOp(UnaryOp::Nullable)
            );
            assert_eq!(test_parse_let(input), Err(expected));
        }

        #[test]
        fn it_returns_error_for_invalid_let_type() {
            let input = force_tokenize("let x: 5 = z;");
            let error = error::not_a_type(int_token(5));
            assert_eq!(test_parse_let(input), Err(error)); 
        }

        #[test]
        fn it_returns_error_for_unexpected_let() {
            let input = force_tokenize("let x = let y = 2;");
            let error = error::unexpected_token("identifier or expression", Token::Let);
            assert_eq!(test_parse_let(input), Err(error));
        }
    }
}
