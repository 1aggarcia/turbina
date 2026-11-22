use std::collections::HashMap;

use crate::models::{
    AbstractSyntaxTree, BinaryExpr, BinaryOp, CodeBlock, CondExpr, Expr,
    FieldAccess, FuncBody, FuncCall, Function, Literal, StructPrimaryExpr,
    Token, Type, UnaryExpr, UnaryOp,
};

use crate::errors::{error, InterpreterError, Result};
use crate::parser::let_parser::parse_let;
use crate::parser::shared_parsers::{parse_id, parse_list, ListParserConfig};
use crate::parser::struct_definition_parser::parse_struct_definition;
use crate::parser::type_alias_parser::parse_type_alias;
use crate::parser::type_declaration_parser::parse_type_declaration;
use crate::parser::utils::{
    match_next, next_token_matches, next_token_matches_any, skip_newlines,
};
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

    let next_is_function = if tokens.lookahead(0)? == Token::open_type_list() {
        true
    } else if tokens.lookahead(0)? != Token::OpenParens {
        false
    } else if tokens.lookahead(1)? == Token::CloseParens {
        true
    } else {
        // function with at least one argument
        matches!(tokens.lookahead(1)?, Token::Id(_))
            && [Token::CloseParens, Token::Colon, Token::Comma]
                .contains(&tokens.lookahead(2)?)
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

    return Ok(CondExpr {
        cond: condition,
        if_true,
        if_false,
    });
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
            Token::TypeKeyword => {
                AbstractSyntaxTree::TypeAlias(parse_type_alias(tokens)?)
            }
            Token::Struct => AbstractSyntaxTree::StructDefinition(
                parse_struct_definition(tokens)?,
            ),
            _ => AbstractSyntaxTree::Expr(parse_expr(tokens)?),
        };
        statements.push(statement);
        let statement_end = tokens.peek()?;
        match statement_end {
            Token::Semicolon | Token::Newline => {
                tokens.pop()?;
            }
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
        type_params = parse_list(
            tokens,
            ListParserConfig {
                item_parser: parse_id,
                opening_token: None,
                closing_token: Token::close_type_list(),
            },
        )?;
    }

    let params = parse_list(
        tokens,
        ListParserConfig {
            item_parser: parse_param,
            opening_token: Some(Token::OpenParens),
            closing_token: Token::CloseParens,
        },
    )?;

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
        body: FuncBody::Expr(Box::new(body_expr)),
    })
}

static MAX_EXPRESSION_PRECEDENCE: u8 = 4;

/// ```text
/// <binary_expr> ::= <expr_lvl_0> {Pipe <expr_lvl_0>}
/// <expr_lvl_0> ::= <expr_lvl_1> {(And | Or) <expr_lvl_1>}
/// <expr_lvl_1> :: = <expr_lvl_2> {(== | != | < | <= | > | >=) <expr_lvl_2>}
/// <expr_lvl_2> :: = <expr_lvl_3> {(+ | - | "|" | ^) <expr_lvl_3>}
/// <expr_lvl_3> :: = <unary_expr> {(* | / | % | & | >> | <<) <unary_expr>}
/// ```
/// Operator precedence is expressed by giving levels to each expression; e.g.
/// `<expr_lvl_2>` has precedence above `<expr_lvl_3>` but not `<expr_lvl_1>`.
/// Naming things is hard, so why bother? Let's just use numbers.
fn parse_binary_expr(
    tokens: &mut TokenStream,
    precedence: u8,
) -> Result<BinaryExpr> {
    if precedence > MAX_EXPRESSION_PRECEDENCE {
        let expr = BinaryExpr {
            first: parse_unary_expr(tokens)?,
            rest: vec![],
        };
        return Ok(expr);
    }

    let first = parse_binary_expr(tokens, precedence + 1)?;
    let mut rest = vec![];

    while next_is_operator(tokens, precedence) {
        let op_token = tokens.pop()?;
        let operator = match op_token {
            Token::BinaryOp(op) => op,
            _ => return Err(error::unexpected_token("binary op", op_token)),
        };
        skip_newlines(tokens);
        let expr = parse_binary_expr(tokens, precedence + 1)?;
        rest.push((operator, expr_as_unary_expr(expr)));
    }

    // avoid re-wrapping expressions as another expression if possible
    if rest.is_empty() {
        return Ok(first);
    }
    Ok(BinaryExpr {
        first: expr_as_unary_expr(first),
        rest,
    })
}

///```text
/// <param> ::= Id [<type_declaration>]
/// ```
fn parse_param(tokens: &mut TokenStream) -> Result<(String, Type)> {
    let id = parse_id(tokens)?;
    let datatype = if next_token_matches(tokens, Token::Colon) {
        parse_type_declaration(tokens)?
    } else {
        Type::Unknown
    };

    Ok((id, datatype))
}

/// ```text
/// <unary_expr> ::= "!" <unary_expr> | ["-"] <postfix_expr>
/// ```
fn parse_unary_expr(tokens: &mut TokenStream) -> Result<UnaryExpr> {
    let first = tokens.peek()?;

    if first == Token::UnaryOp(UnaryOp::Not) {
        tokens.pop()?;
        let inner_unary_expr = parse_unary_expr(tokens)?;
        return Ok(UnaryExpr::negated_bool(inner_unary_expr));
    }
    if first == Token::BinaryOp(BinaryOp::Minus) {
        tokens.pop()?;
        let postfix_expr = parse_postfix_expr(tokens)?;
        return Ok(UnaryExpr::negative_int(postfix_expr));
    }
    parse_postfix_expr(tokens)
}

/// ```text
/// <postfix_expr> ::= <primary_expr> ["!"] { <postfix_suffix> ["!"] }
/// ```
fn parse_postfix_expr(tokens: &mut TokenStream) -> Result<UnaryExpr> {
    let mut result = parse_primary_expr(tokens)?;

    if next_token_matches(tokens, Token::UnaryOp(UnaryOp::Not)) {
        tokens.pop()?;
        result = result.as_not_null();
    }

    skip_newlines(tokens);
    while has_postfix_suffix(tokens) {
        result = append_postfix_suffix(tokens, result)?;

        skip_newlines(tokens);
        if next_token_matches(tokens, Token::UnaryOp(UnaryOp::Not)) {
            tokens.pop()?;
            result = result.as_not_null();
        }
        skip_newlines(tokens);
    }

    Ok(result)
}

fn has_postfix_suffix(tokens: &mut TokenStream) -> bool {
    next_token_matches_any(tokens, vec![Token::OpenParens, Token::Dot])
}

/// ```text
/// <postfix_suffix> ::= <func_call> | <field_access>
/// ```
fn append_postfix_suffix(
    tokens: &mut TokenStream,
    current: UnaryExpr,
) -> Result<UnaryExpr> {
    let next = tokens.peek()?;
    match next {
        Token::OpenParens => append_func_call(tokens, current),
        Token::Dot => append_field_access(tokens, current),
        other => Err(error::unexpected_token(
            "Function call or field access",
            other,
        )),
    }
}

/// ```text
/// <func_call> ::= "(" [<expr> {"," <expr>}] ")"
/// ```
fn append_func_call(
    tokens: &mut TokenStream,
    current: UnaryExpr,
) -> Result<UnaryExpr> {
    let args = parse_list(
        tokens,
        ListParserConfig {
            item_parser: parse_expr,
            opening_token: Some(Token::OpenParens),
            closing_token: Token::CloseParens,
        },
    )?;
    let func_call = FuncCall {
        func: Box::new(current),
        args,
    };
    Ok(UnaryExpr::FuncCall(func_call))
}

/// ```text
/// <field_access> ::= "." Id
/// ```
fn append_field_access(
    tokens: &mut TokenStream,
    current: UnaryExpr,
) -> Result<UnaryExpr> {
    match_next(tokens, Token::Dot)?;
    skip_newlines(tokens);

    let field = parse_id(tokens)?;
    let field_access = FieldAccess {
        r#struct: Box::new(current),
        field,
    };

    Ok(UnaryExpr::FieldAccess(field_access))
}

/// ```text
/// <primary_expr> ::=
///     | Null
///     | Literal
///     | <list>
///     | Id [<struct_body>]
///     | "(" <expr> ")"
///
/// <list> ::= "[" [<expr> {"," <expr>}] "]"
/// ```
fn parse_primary_expr(tokens: &mut TokenStream) -> Result<UnaryExpr> {
    let first = tokens.pop()?;
    match first {
        Token::Null => Ok(UnaryExpr::Literal(Literal::Null)),
        Token::Literal(lit) => Ok(UnaryExpr::Literal(lit)),
        Token::OpenSquareBracket => {
            let elements = parse_list(
                tokens,
                ListParserConfig {
                    item_parser: parse_expr,
                    opening_token: None,
                    closing_token: Token::CloseSquareBracket,
                },
            )?;
            Ok(UnaryExpr::List(elements))
        }
        Token::Id(id) => {
            if next_token_matches(tokens, Token::OpenCurlyBracket) {
                let struct_body = parse_struct_body(tokens)?;
                return Ok(UnaryExpr::Struct(StructPrimaryExpr {
                    name: id,
                    fields: struct_body,
                }));
            }

            Ok(UnaryExpr::Id(id))
        }
        Token::OpenParens => {
            skip_newlines(tokens);
            let expr: Expr = parse_expr(tokens)?;
            skip_newlines(tokens);
            match_next(tokens, Token::CloseParens)?;
            let result = UnaryExpr::Expr(Box::new(expr));
            Ok(result)
        }
        _ => Err(error::unexpected_token("identifier or expression", first)),
    }
}

/// ```text
/// <struct_body> ::= "{" struct_field> {"," <struct_field>} [","] "}"
/// ```
fn parse_struct_body(
    tokens: &mut TokenStream,
) -> Result<HashMap<String, Expr>> {
    let field_list = parse_list(
        tokens,
        ListParserConfig {
            item_parser: parse_struct_field,
            opening_token: Some(Token::OpenCurlyBracket),
            closing_token: Token::CloseCurlyBracket,
        },
    )?;

    Ok(field_list.into_iter().collect())
}

fn parse_struct_field(tokens: &mut TokenStream) -> Result<(String, Expr)> {
    let key = parse_id(tokens)?;
    match_next(tokens, Token::Colon)?;
    skip_newlines(tokens);
    let value = parse_expr(tokens)?;

    Ok((key, value))
}

/// If the expression contains one unary expr, unwrap it.
/// Otherwise, wrap the expression into a unary expr
fn expr_as_unary_expr(expr: BinaryExpr) -> UnaryExpr {
    if expr.rest.is_empty() {
        expr.first
    } else {
        UnaryExpr::Expr(Box::new(Expr::Binary(expr)))
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
    return get_binary_operator_precedence(operator) == precedence;
}

/// Returns precedence of operator, larger numbers being higher precedence
fn get_binary_operator_precedence(operator: BinaryOp) -> u8 {
    use BinaryOp::*;

    // pattern matching guarantees that all operators have exactly one precedence level
    match operator {
        Pipe => 0,

        And | Or => 1,

        Equals | NotEq | LessThan | LessThanOrEqual | GreaterThan
        | GreaterThanOrEqual => 2,

        Plus | Minus | BitwiseOr | BitwiseXor => 3,

        Star | Slash | Percent | BitwiseAnd | RightShift | LeftShift => 4,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::models::test_utils::*;
    use crate::parser::test_utils::{force_tokenize, parse_tokens};
    use rstest::rstest;

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
                str_unary_expr("a"),
                vec![(BinaryOp::Plus, str_unary_expr("b"))],
            );
            assert_eq!(test_parse_expr(input), Ok(expected));
        }

        #[test]
        fn it_parses_expression_in_parens() {
            let input = force_tokenize("3 * (2 - 5);");

            let inner_expr = bin_expr(
                int_unary_expr(2),
                vec![(BinaryOp::Minus, int_unary_expr(5))],
            );
            let expected = bin_expr(
                int_unary_expr(3),
                vec![(BinaryOp::Star, UnaryExpr::Expr(Box::new(inner_expr)))],
            );
            assert_eq!(test_parse_expr(input), Ok(expected));
        }

        #[test]
        fn it_respects_boolean_operator_precedence() {
            let input = force_tokenize("1 == 0 || 1 != 0;");

            let left = bin_expr(
                int_unary_expr(1),
                vec![(BinaryOp::Equals, int_unary_expr(0))],
            );
            let right = bin_expr(
                int_unary_expr(1),
                vec![(BinaryOp::NotEq, int_unary_expr(0))],
            );

            let expected = bin_expr(
                UnaryExpr::Expr(Box::new(left)),
                vec![(BinaryOp::Or, UnaryExpr::Expr(Box::new(right)))],
            );
            assert_eq!(test_parse_expr(input), Ok(expected));
        }

        #[test]
        fn it_parses_if_else_expression() {
            let input = force_tokenize("if (cond) \"abc\" else \"def\";");

            let expected = Expr::Cond(CondExpr {
                cond: Box::new(unary_expr(UnaryExpr::Id("cond".into()))),
                if_true: Box::new(unary_expr(str_unary_expr("abc"))),
                if_false: Box::new(unary_expr(str_unary_expr("def"))),
            });

            assert_eq!(test_parse_expr(input), Ok(expected));
        }

        #[test]
        fn it_parses_expression_with_multiple_functions() {
            let input = force_tokenize("x() + y();");

            let x_call = FuncCall {
                func: Box::new(UnaryExpr::Id("x".into())),
                args: vec![],
            };
            let y_call = FuncCall {
                func: Box::new(UnaryExpr::Id("y".into())),
                args: vec![],
            };
            let expr = BinaryExpr {
                first: UnaryExpr::FuncCall(x_call),
                rest: vec![(BinaryOp::Plus, UnaryExpr::FuncCall(y_call))],
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
            let left = int_unary_expr(left_val);
            let right = int_unary_expr(right_val);
            let expected = bin_expr(left, vec![(operator, right)]);
            assert_eq!(test_parse_expr(input), Ok(expected));
        }

        #[test]
        fn it_parses_code_block_with_no_result_value() {
            let tokens = force_tokenize(
                "{
                println(1);
                println(2);
            };",
            );
            let expected_block = CodeBlock {
                statements: vec![
                    parse_tokens(force_tokenize("println(1);")).unwrap(),
                    parse_tokens(force_tokenize("println(2);")).unwrap(),
                ],
            };
            let expected = Expr::CodeBlock(expected_block);
            assert_eq!(test_parse_expr(tokens), Ok(expected));
        }

        #[test]
        fn it_parses_code_block_with_binding() {
            let tokens = force_tokenize(
                "{
                let two = 2;
                two
            };",
            );
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
        fn it_parses_code_block_with_type_alias_declaration() {
            let tokens = force_tokenize(
                "{
                type Predicate = int -> bool;
                let isEven: Predicate = (x: int) -> x % 2 == 0;
                isEven(0)
            };",
            );
            let expected_block = CodeBlock {
                statements: vec![
                    "type Predicate = int -> bool;",
                    "let isEven: Predicate = (x: int) -> x % 2 == 0;",
                    "isEven(0);",
                ]
                .into_iter()
                .map(force_tokenize)
                .map(parse_tokens)
                .map(|result| result.unwrap())
                .collect(),
            };
            let expected = Expr::CodeBlock(expected_block);
            assert_eq!(test_parse_expr(tokens), Ok(expected));
        }

        #[test]
        fn it_parses_code_block_with_struct_definition() {
            let tokens = force_tokenize(
                "{
                struct User {
                    username: string,
                    isAdmin: bool,
                };
            };",
            );
            let expected_block = CodeBlock {
                statements: vec![parse_tokens(force_tokenize(
                    "struct User { username: string, isAdmin: bool };",
                ))
                .unwrap()],
            };
            let expected = Expr::CodeBlock(expected_block);
            assert_eq!(test_parse_expr(tokens), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_empty_code_block() {
            let tokens = force_tokenize("{};");
            assert_eq!(
                test_parse_expr(tokens),
                Err(InterpreterError::EmptyCodeBlock)
            )
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
        #[case::thunk("() -> 3;", vec![], vec![], None, int_unary_expr(3))]
        #[case::many_params("(x: int, y: string) -> 3;",
            vec![],
            vec![("x", Type::Int), ("y".into(), Type::String)],
            None,
            int_unary_expr(3)
        )]
        #[case::declared_return_type("(): bool -> true;",
            vec![],
            vec![],
            Some(Type::Bool),
            UnaryExpr::Literal(Literal::Bool(true))
        )]
        #[case::one_generic_type("<T>(x: T) -> x;",
            vec!["T"],
            vec![("x", Type::Generic("T".into()))],
            None,
            UnaryExpr::Id("x".into())
        )]
        #[case::many_generic_types("<A, B, C> (a: A, b: B, c: C): C -> null;",
            vec!["A", "B", "C"],
            vec![
                ("a", Type::Generic("A".into())),
                ("b", Type::Generic("B".into())),
                ("c", Type::Generic("C".into())),
            ],
            Some(Type::Generic("C".into())),
            UnaryExpr::Literal(Literal::Null)
        )]
        #[case::implicit_param_types("(x, y) -> false;",
            vec![],
            vec![("x", Type::Unknown), ("y", Type::Unknown)],
            None,
            bool_unary_expr(false)
        )]
        fn it_parses_function(
            #[case] input: &str,
            #[case] type_params: Vec<&str>,
            #[case] params: Vec<(&str, Type)>,
            #[case] return_type: Option<Type>,
            #[case] body_expr: UnaryExpr,
        ) {
            let tokens = force_tokenize(input);
            let expected = Function {
                type_params: type_params
                    .into_iter()
                    .map(|t| t.to_owned())
                    .collect(),
                params: params
                    .into_iter()
                    .map(|(p, t)| (p.to_owned(), t))
                    .collect(),
                return_type,
                body: FuncBody::Expr(Box::new(unary_expr(body_expr))),
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

    mod postfix_expr {
        use super::*;

        #[rstest]
        #[case::id_struct(
            "user.name;",
            UnaryExpr::Id("user".into())
        )]
        #[case::literal_struct(
            r#"User { name: "Bob" }.name;"#,
            struct_primary_expr(
                "User",
                [("name", unary_expr(str_unary_expr("Bob")))]
            ),
        )]
        #[case::expr_in_parenthesis(
            "(15).name;",
            UnaryExpr::Expr(Box::new(unary_expr(int_unary_expr(15))))
        )]
        fn it_parses_one_field_access(
            #[case] input: &str,
            #[case] expected_struct: UnaryExpr,
        ) {
            let input = force_tokenize(input);

            let expected_field_access = FieldAccess {
                r#struct: Box::new(expected_struct),
                field: "name".into(),
            };
            let expected =
                unary_expr_tree(UnaryExpr::FieldAccess(expected_field_access));
            assert_eq!(parse_tokens(input), Ok(expected));
        }

        #[rstest]
        #[case::oneline("globalState.admins.metadata.isRegistered;")]
        #[case::with_spaces("globalState . admins . metadata . isRegistered;")]
        #[case::with_newlines(
            r#"
            globalState
                .admins
                
                .
                
                metadata.
                isRegistered;
            "#
        )]
        fn it_parses_multiple_field_accesses(#[case] input_str: &str) {
            let input = force_tokenize(input_str);

            let admins_access = FieldAccess {
                r#struct: Box::new(UnaryExpr::Id("globalState".into())),
                field: "admins".into(),
            };

            let metadata_access = FieldAccess {
                r#struct: Box::new(UnaryExpr::FieldAccess(admins_access)),
                field: "metadata".into(),
            };

            let is_registered_access = FieldAccess {
                r#struct: Box::new(UnaryExpr::FieldAccess(metadata_access)),
                field: "isRegistered".into(),
            };

            let expected =
                unary_expr_tree(UnaryExpr::FieldAccess(is_registered_access));
            assert_eq!(parse_tokens(input), Ok(expected));
        }

        #[test]
        fn it_parses_field_accesses_with_non_null_assertions() {
            let input = force_tokenize("globalState!.admins!.metadata!;");

            let admins_access = FieldAccess {
                r#struct: Box::new(
                    UnaryExpr::Id("globalState".into()).as_not_null(),
                ),
                field: "admins".into(),
            };

            let metadata_access = FieldAccess {
                r#struct: Box::new(
                    UnaryExpr::FieldAccess(admins_access).as_not_null(),
                ),
                field: "metadata".into(),
            };

            let expected = unary_expr_tree(
                UnaryExpr::FieldAccess(metadata_access).as_not_null(),
            );
            assert_eq!(parse_tokens(input), Ok(expected));
        }

        #[test]
        fn it_parses_function_call_on_field_access() {
            let input = force_tokenize("user.getName();");

            let expected_access = FieldAccess {
                r#struct: Box::new(UnaryExpr::Id("user".into())),
                field: "getName".into(),
            };
            let expected_func_call = FuncCall {
                func: Box::new(UnaryExpr::FieldAccess(expected_access)),
                args: vec![],
            };
            let expected =
                unary_expr_tree(UnaryExpr::FuncCall(expected_func_call));
            assert_eq!(parse_tokens(input), Ok(expected));
        }
    }

    mod unary_expr {
        use crate::parser::test_utils::parse_tokens;

        use super::*;

        #[rstest]
        #[case(int_token(2), unary_expr_tree(int_unary_expr(2)))]
        #[case(
            string_token("prueba test"),
            unary_expr_tree(str_unary_expr("prueba test"))
        )]
        #[case(id_token("name"), unary_expr_tree(UnaryExpr::Id("name".into())))]
        fn it_parses_one_token_to_one_node(
            #[case] token: Token,
            #[case] node: AbstractSyntaxTree,
        ) {
            let input = vec![token.clone(), Token::Semicolon];
            assert_eq!(parse_tokens(input), Ok(node));
        }

        #[rstest]
        #[case(op_token(BinaryOp::Plus))]
        #[case(op_token(BinaryOp::Percent))]
        #[case(unary_op_token(UnaryOp::Equals))]
        fn it_returns_error_for_one_operator(#[case] op: Token) {
            let error =
                error::unexpected_token("identifier or expression", op.clone());
            assert_eq!(parse_tokens(vec![op, Token::Newline]), Err(error));
        }

        #[rstest]
        #[case(force_tokenize("-9;"), -9)]
        #[case(force_tokenize("- 123;"), -123)]
        fn it_parses_negative_numbers(
            #[case] input: Vec<Token>,
            #[case] negative_num: i32,
        ) {
            let inner_expr = int_unary_expr(negative_num * -1);
            let outer_expr = UnaryExpr::negative_int(inner_expr);
            let expected = unary_expr_tree(outer_expr);
            assert_eq!(parse_tokens(input), Ok(expected));
        }

        #[test]
        fn it_parses_negative_num_in_parens() {
            let input = force_tokenize("-(9);");

            // the parentheses creates this unfortunate indirection in the AST
            let nine = unary_expr(int_unary_expr(9));
            let nine_as_unary_expr = UnaryExpr::Expr(Box::new(nine));
            let negative_nine = UnaryExpr::negative_int(nine_as_unary_expr);
            let expected_tree = unary_expr_tree(negative_nine);

            assert_eq!(parse_tokens(input), Ok(expected_tree));
        }

        #[test]
        fn it_parses_negated_boolean() {
            let expr =
                UnaryExpr::negated_bool(UnaryExpr::Id("someVariable".into()));
            let expected = unary_expr_tree(expr);
            assert_eq!(
                parse_tokens(force_tokenize("!someVariable;")),
                Ok(expected)
            );
        }

        #[test]
        fn it_parses_many_negations() {
            let expr = UnaryExpr::negated_bool(UnaryExpr::negated_bool(
                UnaryExpr::negated_bool(UnaryExpr::Id("x".into())),
            ));
            let expected = unary_expr_tree(expr);
            assert_eq!(parse_tokens(force_tokenize("!!!x;")), Ok(expected));
        }

        #[rstest]
        #[case::id("x!;", UnaryExpr::Id("x".into()))]
        #[case::expression(
            "(3 + 4)!;",
            UnaryExpr::Expr(
                Box::new(
                    bin_expr(int_unary_expr(3), vec![(BinaryOp::Plus, int_unary_expr(4))])
                )
            )
        )]
        #[case::function_call(
            "f!()!;",
            UnaryExpr::FuncCall(FuncCall {
                func: Box::new(UnaryExpr::Id("f".into()).as_not_null()),
                args: vec![]
            })
        )]
        #[case::literal("null!;", UnaryExpr::Literal(Literal::Null))]
        fn it_parses_not_null_assertion(
            #[case] input: &str,
            #[case] not_null: UnaryExpr,
        ) {
            let tokens = force_tokenize(input);
            let expected = unary_expr_tree(not_null.as_not_null());

            assert_eq!(parse_tokens(tokens), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_many_negative_symbols() {
            let error = error::unexpected_token(
                "identifier or expression",
                Token::BinaryOp(BinaryOp::Minus),
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
            func: Box::new(UnaryExpr::Id("doSomething".into())), args: vec![]
        })]
        #[case::one_arg("print(3);", FuncCall {
            func: Box::new(UnaryExpr::Id("print".into())), args: vec![unary_expr(int_unary_expr(3))]
        })]
        #[case::many_args("list(1, 2, 3);", FuncCall {
            func: Box::new(UnaryExpr::Id("list".into())),
            args: [1, 2, 3].iter().map(|i| unary_expr(int_unary_expr(*i))).collect()
        })]
        #[case::curried_function("x(1)(2);", FuncCall {
            func: Box::new(UnaryExpr::FuncCall(FuncCall {
                func: Box::new(UnaryExpr::Id("x".into())),
                args: vec![unary_expr(int_unary_expr(1))],
            })),
            args: vec![unary_expr(int_unary_expr(2))],
        })]
        fn it_parses_function_call(
            #[case] input: &str,
            #[case] call: FuncCall,
        ) {
            let tree = force_tokenize(input);
            let expr = unary_expr(UnaryExpr::FuncCall(call));
            let expected = Ok(AbstractSyntaxTree::Expr(expr));
            assert_eq!(parse_tokens(tree), expected);
        }

        #[rstest]
        #[case::one_line(r#"User { isAdmin: true, id: "12345" };"#)]
        #[case::spaces_in_fields(r#"User { isAdmin : true, id  :"12345" };"#)]
        #[case::fields_on_multiple_lines(
            r#"User {
                isAdmin: true,
                id: "12345"
            };"#
        )]
        #[case::fields_values_on_multiple_lines(
            r#"User {
                isAdmin:
                
                    true,

                id:
                
                    "12345",
            };"#
        )]
        fn it_parses_struct_unary_expr(#[case] input_str: &str) {
            let input = force_tokenize(input_str);

            let expected_unary_expr = struct_primary_expr(
                "User",
                [
                    ("isAdmin", unary_expr(bool_unary_expr(true))),
                    ("id", unary_expr(str_unary_expr("12345"))),
                ],
            );
            let expected = unary_expr_tree(expected_unary_expr);

            assert_eq!(parse_tokens(input), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_struct_with_incomplete_field() {
            let input = force_tokenize("Test { field };");
            let expected =
                error::unexpected_token("Colon", Token::CloseCurlyBracket);
            assert_eq!(parse_tokens(input), Err(expected));
        }

        #[test]
        fn it_returns_error_for_struct_body_on_separate_line() {
            let input = force_tokenize(
                r#"
                User

                    { isAdmin: true, id: "12345" };
            "#,
            );
            let expected =
                InterpreterError::end_of_statement(Token::OpenCurlyBracket);
            assert_eq!(parse_tokens(input), Err(expected));
        }
    }
}
