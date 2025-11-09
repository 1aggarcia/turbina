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

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;
    use crate::models::test_utils::*;
    use crate::parser::test_utils::{force_tokenize, parse_tokens};

    fn test_parse_expr(tokens: Vec<Token>) -> Result<Expr> {
        let ast = parse_tokens(tokens)?;
        match ast {
            AbstractSyntaxTree::Expr(expr) => Ok(expr),
            other => panic!("Not an expression: {:?}", other),
        }
    }

    mod expr {
        use super::*;

        #[test]
        fn it_parses_string_plus_string() {
            let input = force_tokenize("\"a\" + \"b\";");

            let expected = bin_expr(
                str_term("a"),
                vec![(BinaryOp::Plus, str_term("b"))]
            );
            assert_eq!(test_parse_expr(input), Ok(expected));
        }

        #[test]
        fn it_parses_expression_in_parens() {
            let input = force_tokenize("3 * (2 - 5);");

            let inner_expr = bin_expr(
                int_term(2),
                vec![(BinaryOp::Minus, int_term(5))]
            );
            let expected = bin_expr(
                int_term(3),
                vec![(BinaryOp::Star, Term::Expr(Box::new(inner_expr)))]
            );
            assert_eq!(test_parse_expr(input), Ok(expected));
        }

        #[test]
        fn it_respects_boolean_operator_precedence() {
            let input = force_tokenize("1 == 0 || 1 != 0;");

            let left = bin_expr(int_term(1), vec![(BinaryOp::Equals, int_term(0))]); 
            let right = bin_expr(int_term(1), vec![(BinaryOp::NotEq, int_term(0))]);
            
            let expected = bin_expr(
                Term::Expr(Box::new(left)),
                vec![(BinaryOp::Or, Term::Expr(Box::new(right)))]
            );
            assert_eq!(test_parse_expr(input), Ok(expected));
        }

        #[test]
        fn it_parses_if_else_expression() {
            let input = force_tokenize("if (cond) \"abc\" else \"def\";");

            let expected = Expr::Cond(CondExpr {
                cond: Box::new(term_expr(Term::Id("cond".into()))),
                if_true: Box::new(term_expr(str_term("abc"))),
                if_false: Box::new(term_expr(str_term("def"))),
            });

            assert_eq!(test_parse_expr(input), Ok(expected));
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
            let expected = Expr::Binary(expr);

            assert_eq!(test_parse_expr(input), Ok(expected));
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
            let expected = bin_expr(left, vec![(operator, right)]);
            assert_eq!(test_parse_expr(input), Ok(expected));
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
            let expected = Expr::CodeBlock(expected_block);
            assert_eq!(test_parse_expr(tokens), Ok(expected));
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
            let expected = Expr::CodeBlock(expected_block);
            assert_eq!(test_parse_expr(tokens), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_empty_code_block() {
            let tokens = force_tokenize("{};");
            assert_eq!(test_parse_expr(tokens), Err(InterpreterError::EmptyCodeBlock))
        }
    }

    mod function {
        use super::*;

        pub fn test_parse_function(tokens: Vec<Token>) -> Result<Function> {
            let expr = test_parse_expr(tokens)?;
            match expr {
                Expr::Function(func) => Ok(func),
                other => panic!("Not a function: {:?}", other),
            }
        }

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
            let expected = Function {
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
            assert_eq!(test_parse_function(tokens), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_non_id_param() {
            let input = force_tokenize("(x: int, 3: int) => x;");
            let err = error::unexpected_token("identifier", int_token(3));
            assert_eq!(test_parse_function(input), Err(err));
        }

        #[test]
        fn it_returns_error_for_empty_type_param_list() {
            let input = force_tokenize("<>() -> null;");
            let err = InterpreterError::EmptyTypeList;
            assert_eq!(test_parse_function(input), Err(err));
        }
    }

    mod term {
        use crate::parser::test_utils::parse_tokens;

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
}
