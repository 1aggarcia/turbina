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
    use crate::streams::StringStream;

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
}

#[cfg(test)]
mod test_parse {
    use super::*;
    use crate::lexer::tokenize;
    use crate::models::test_utils::*;
    use crate::models::*;
    use crate::parser::test_utils::parse_tokens;
    use crate::errors::error;
    use rstest::rstest;

    mod term {
        use super::*;

        #[rstest]
        #[case(int_token(2), term_tree(int_term(2)))]
        #[case(string_token("prueba test"), term_tree(str_term("prueba test")))]
        #[case(id_token("name"), term_tree(Term::Id("name".into())))]
        fn it_parses_one_token_to_one_node(
            #[case] token: Token,
            #[case] node: AbstractSyntaxTree
        ) {
            let input = vec![token.clone(), Token::Semicolon];
            assert_eq!(parse_tokens(input), Ok(node));
        }

        #[rstest]
        #[case(op_token(BinaryOp::Plus))]
        #[case(op_token(BinaryOp::Percent))]
        #[case(unary_op_token(UnaryOp::Equals))]
        fn it_returns_error_for_one_operator(#[case] op: Token) {
            let error = error::unexpected_token("identifier or expression", op.clone());
            assert_eq!(parse_tokens(vec![op, Token::Newline]), Err(error));
        }

        #[rstest]
        #[case(force_tokenize("-9;"), -9)]
        #[case(force_tokenize("- 123;"), -123)]
        fn it_parses_negative_numbers(
            #[case] input: Vec<Token>,
            #[case] negative_num: i32
        ) {
            let inner_term = int_term(negative_num * -1);
            let term = Term::negative_int(inner_term);
            let expected = term_tree(term);
            assert_eq!(parse_tokens(input), Ok(expected));
        }

        #[test]
        fn it_parses_negative_num_in_parens() {
            let input = force_tokenize("-(9);");

            // the parentheses creates this unfortunate indirection in the AST
            let nine = term_expr(int_term(9));
            let nine_as_term = Term::Expr(Box::new(nine));
            let negative_nine = Term::negative_int(nine_as_term);
            let expected_tree = term_tree(negative_nine);

            assert_eq!(parse_tokens(input), Ok(expected_tree)); 
        }

        #[test]
        fn it_parses_negated_boolean() {
            let term = Term::negated_bool(Term::Id("someVariable".into()));
            let expected = term_tree(term);
            assert_eq!(parse_tokens(force_tokenize("!someVariable;")), Ok(expected));
        }

        #[test]
        fn it_parses_many_negations() {
            let term = Term::negated_bool(
                Term::negated_bool(
                    Term::negated_bool(Term::Id("x".into()))
                )
            );
            let expected = term_tree(term);
            assert_eq!(parse_tokens(force_tokenize("!!!x;")), Ok(expected));
        }

        #[rstest]
        #[case::id("x!;", Term::Id("x".into()))]
        #[case::expression(
            "(3 + 4)!;",
            Term::Expr(
                Box::new(
                    bin_expr(int_term(3), vec![(BinaryOp::Plus, int_term(4))])
                )
            )
        )]
        #[case::function_call(
            "f!()!;",
            Term::FuncCall(FuncCall {
                func: Box::new(Term::Id("f".into()).as_not_null()),
                args: vec![]
            })
        )]
        fn it_parses_not_null_assertion(
            #[case] input: &str, #[case] not_null: Term
        ) {
            let tokens = force_tokenize(input);
            let expected = term_tree(not_null.as_not_null());

            assert_eq!(parse_tokens(tokens), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_not_null_literal() {
            let tokens = force_tokenize("null!;");
            let expected =
                InterpreterError::end_of_statement(Token::UnaryOp(UnaryOp::Not));
            assert_eq!(parse_tokens(tokens), Err(expected));
        }

        #[test]
        fn it_returns_error_for_many_negative_symbols() {
            let error = error::unexpected_token(
                "identifier or expression",
                Token::BinaryOp(BinaryOp::Minus)
            );
            assert_eq!(parse_tokens(force_tokenize("---9;")), Err(error));
        }

        #[test]
        fn it_returns_error_for_multiple_ints() {
            let input = force_tokenize("3 2;");
            let error = InterpreterError::end_of_statement(int_token(2));
            assert_eq!(parse_tokens(input), Err(error));
        }

        #[rstest]
        #[case::empty("doSomething();", FuncCall {
            func: Box::new(Term::Id("doSomething".into())), args: vec![]
        })]
        #[case::one_arg("print(3);", FuncCall {
            func: Box::new(Term::Id("print".into())), args: vec![term_expr(int_term(3))]
        })]
        #[case::many_args("list(1, 2, 3);", FuncCall {
            func: Box::new(Term::Id("list".into())),
            args: [1, 2, 3].iter().map(|i| term_expr(int_term(*i))).collect()
        })]
        #[case::curried_function("x(1)(2);", FuncCall {
            func: Box::new(Term::FuncCall(FuncCall {
                func: Box::new(Term::Id("x".into())),
                args: vec![term_expr(int_term(1))],
            })),
            args: vec![term_expr(int_term(2))],
        })]
        fn it_parses_function_call(#[case] input: &str, #[case] call: FuncCall) {
            let tree = force_tokenize(input);
            let expr = term_expr(Term::FuncCall(call));
            let expected = Ok(AbstractSyntaxTree::Expr(expr));
            assert_eq!(parse_tokens(tree), expected);
        }
    }

    mod expr {
        use super::*;

        #[test]
        fn it_parses_string_plus_string() {
            let input = force_tokenize("\"a\" + \"b\";");

            let expr = bin_expr(
                str_term("a"),
                vec![(BinaryOp::Plus, str_term("b"))]
            );
            let expected = AbstractSyntaxTree::Expr(expr);
            assert_eq!(parse_tokens(input), Ok(expected));
        }

        #[test]
        fn it_parses_expression_in_parens() {
            let input = force_tokenize("3 * (2 - 5);");

            let inner_expr = bin_expr(
                int_term(2),
                vec![(BinaryOp::Minus, int_term(5))]
            );
            let expr = bin_expr(
                int_term(3),
                vec![(BinaryOp::Star, Term::Expr(Box::new(inner_expr)))]
            );
            let expected = AbstractSyntaxTree::Expr(expr);
            assert_eq!(parse_tokens(input), Ok(expected));
        }

        #[test]
        fn it_respects_boolean_operator_precedence() {
            let input = force_tokenize("1 == 0 || 1 != 0;");

            let left = bin_expr(int_term(1), vec![(BinaryOp::Equals, int_term(0))]); 
            let right = bin_expr(int_term(1), vec![(BinaryOp::NotEq, int_term(0))]);
            
            let expr = bin_expr(
                Term::Expr(Box::new(left)),
                vec![(BinaryOp::Or, Term::Expr(Box::new(right)))]
            );
            let expected = AbstractSyntaxTree::Expr(expr);
            assert_eq!(parse_tokens(input), Ok(expected));
        }

        #[test]
        fn it_parses_if_else_expression() {
            let input = force_tokenize("if (cond) \"abc\" else \"def\";");

            let expr = Expr::Cond(CondExpr {
                cond: Box::new(term_expr(Term::Id("cond".into()))),
                if_true: Box::new(term_expr(str_term("abc"))),
                if_false: Box::new(term_expr(str_term("def"))),
            });
            let expected = Ok(AbstractSyntaxTree::Expr(expr));

            assert_eq!(parse_tokens(input), expected);
        }

        #[test]
        fn it_parses_expression_with_multiple_functions() {
            let input = force_tokenize("x() + y();");

            let x_call = FuncCall {
                func: Box::new(Term::Id("x".into())),
                args: vec![],
            };
            let y_call = FuncCall {
                func: Box::new(Term::Id("y".into())),
                args: vec![],
            }; 
            let expr = BinaryExpr {
                first: Term::FuncCall(x_call),
                rest: vec![(BinaryOp::Plus, Term::FuncCall(y_call))]
            };
            let expected = AbstractSyntaxTree::Expr(Expr::Binary(expr));

            assert_eq!(parse_tokens(input), Ok(expected));
        }

        #[rstest]
        #[case(force_tokenize("3 + 2;"), BinaryOp::Plus, 3, 2)]
        #[case(force_tokenize("1 % 4;"), BinaryOp::Percent, 1, 4)]
        #[case(force_tokenize("1 - 8;"), BinaryOp::Minus, 1, 8)]
        #[case(force_tokenize("0 == 1;"), BinaryOp::Equals, 0, 1)]
        #[case(force_tokenize("2 != 3;"), BinaryOp::NotEq, 2, 3)]
        fn it_parses_binary_expressions(
            #[case] input: Vec<Token>,
            #[case] operator: BinaryOp,
            #[case] left_val: i32,
            #[case] right_val: i32,
        ) {
            let left = int_term(left_val);
            let right = int_term(right_val);
            let expected = AbstractSyntaxTree::Expr(
                bin_expr(left, vec![(operator, right)])
            );
            assert_eq!(parse_tokens(input), Ok(expected));
        }

        #[test]
        fn it_parses_code_block_with_no_result_value() {
            let tokens = force_tokenize("{
                println(1);
                println(2);
            };");
            let expected_block = CodeBlock {
                statements: vec![
                    parse_tokens(force_tokenize("println(1);")).unwrap(),
                    parse_tokens(force_tokenize("println(2);")).unwrap(),
                ]
            };
            let expected =
                AbstractSyntaxTree::Expr(Expr::CodeBlock(expected_block));
            assert_eq!(parse_tokens(tokens), Ok(expected));
        }

        #[test]
        fn it_parses_code_block_with_binding() {
            let tokens = force_tokenize("{
                let two = 2;
                two
            };");
            let expected_block = CodeBlock {
                statements: vec![
                    parse_tokens(force_tokenize("let two = 2;")).unwrap(),
                    parse_tokens(force_tokenize("two;")).unwrap(),
                ],
            };
            let expected =
                AbstractSyntaxTree::Expr(Expr::CodeBlock(expected_block));
            assert_eq!(parse_tokens(tokens), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_empty_code_block() {
            let tokens = force_tokenize("{};");
            assert_eq!(parse_tokens(tokens), Err(InterpreterError::EmptyCodeBlock))
        }
    }

    mod function {
        use super::*;

        #[rstest]
        #[case::thunk("() -> 3;", vec![], vec![], None, int_term(3))]
        #[case::many_params("(x: int, y: string) -> 3;",
            vec![],
            vec![("x", Type::Int), ("y".into(), Type::String)],
            None,
            int_term(3)
        )]
        #[case::declared_return_type("(): bool -> true;",
            vec![],
            vec![],
            Some(Type::Bool),
            Term::Literal(Literal::Bool(true))
        )]
        #[case::one_generic_type("<T>(x: T) -> x;",
            vec!["T"],
            vec![("x", Type::Generic("T".into()))],
            None,
            Term::Id("x".into())
        )]
        #[case::many_generic_types("<A, B, C> (a: A, b: B, c: C): C -> null;",
            vec!["A", "B", "C"],
            vec![
                ("a", Type::Generic("A".into())),
                ("b", Type::Generic("B".into())),
                ("c", Type::Generic("C".into())),
            ],
            Some(Type::Generic("C".into())),
            Term::Literal(Literal::Null)
        )]
        #[case::implicit_param_types("(x, y) -> false;",
            vec![],
            vec![("x", Type::Unknown), ("y", Type::Unknown)],
            None,
            bool_term(false)
        )]
        fn it_parses_function(
            #[case] input: &str,
            #[case] type_params: Vec<&str>,
            #[case] params: Vec<(&str, Type)>,
            #[case] return_type: Option<Type>,
            #[case] body_term: Term,
        ) {
            let tokens = force_tokenize(input);
            let function = Function {
                type_params: type_params
                    .into_iter()
                    .map(|t| t.to_owned())
                    .collect(),
                params: params
                    .into_iter()
                    .map(|(p, t)|(p.to_owned(), t))
                    .collect(),
                return_type,
                body: FuncBody::Expr(Box::new(term_expr(body_term))),
            };
            let expr = Expr::Function(function);
            let expected = AbstractSyntaxTree::Expr(expr);
            assert_eq!(parse_tokens(tokens), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_non_id_param() {
            let input = force_tokenize("(x: int, 3: int) => x;");
            let err = error::unexpected_token("identifier", int_token(3));
            assert_eq!(parse_tokens(input), Err(err));
        }

        #[test]
        fn it_returns_error_for_empty_type_param_list() {
            let input = force_tokenize("<>() -> null;");
            let err = InterpreterError::EmptyTypeList;
            assert_eq!(parse_tokens(input), Err(err));
        }
    }

    mod let_binding {
        use super::*;

        #[test]
        fn it_returns_error_for_bad_var_id() {
            let input = force_tokenize("let 3 = 3;");
            let error = error::unexpected_token("identifier", int_token(3));
            assert_eq!(parse_tokens(input), Err(error));
        }

        #[test]
        fn it_returns_error_for_equals_in_let_expr() {
            let input = force_tokenize("let x = 1 = 0;");
            let error = InterpreterError::end_of_statement(
                Token::UnaryOp(UnaryOp::Equals)
            );
            assert_eq!(parse_tokens(input), Err(error)); 
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
            let syntax_tree = AbstractSyntaxTree::Let(expected);
            assert_eq!(parse_tokens(input), Ok(syntax_tree));
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
            let syntax_tree = AbstractSyntaxTree::Let(LetNode {
                id: "f".into(),
                datatype: Some(function_type),
                value: bin_expr(Term::Literal(Literal::Null), vec![]),
            });

            assert_eq!(parse_tokens(tokens), Ok(syntax_tree));
        }

        #[test]
        fn it_returns_error_for_nullable_null() {
            let input = force_tokenize("let nullableData: null? = null;");
            let expected = error::unexpected_token(
                &format!("{:?}", Token::UnaryOp(UnaryOp::Equals)),
                Token::UnaryOp(UnaryOp::Nullable)
            );
            assert_eq!(parse_tokens(input), Err(expected));
        }

        #[test]
        fn it_returns_error_for_multiple_nullable_operators() {
            let input = force_tokenize("let nullable: int????? = 2;");
            let expected = error::unexpected_token(
                &format!("{:?}", Token::UnaryOp(UnaryOp::Equals)),
                Token::UnaryOp(UnaryOp::Nullable)
            );
            assert_eq!(parse_tokens(input), Err(expected));
        }

        #[test]
        fn it_returns_error_for_invalid_let_type() {
            let input = force_tokenize("let x: 5 = z;");
            let error = error::not_a_type(int_token(5));
            assert_eq!(parse_tokens(input), Err(error)); 
        }

        #[test]
        fn it_returns_error_for_unexpected_let() {
            let input = force_tokenize("let x = let y = 2;");
            let error = error::unexpected_token("identifier or expression", Token::Let);
            assert_eq!(parse_tokens(input), Err(error));
        }
    }

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

    fn force_tokenize(text: &str) -> Vec<Token> {
        text
            .split("\n")
            .flat_map(|line| tokenize(line).unwrap())
            .collect()
    }
}
