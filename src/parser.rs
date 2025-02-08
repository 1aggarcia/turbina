use crate::models::{
    AbstractSyntaxTree, BinaryExpr, BinaryOp, CondExpr, Expr, Function, FuncBody, FuncCall, LetNode, Literal, Term, Token, Type, UnaryOp
};

use crate::errors::{IntepreterError, error};
use crate::streams::TokenStream;

type ParseResult<T> = Result<T, IntepreterError>;

/// Consumes the next statement from the token stream and returns a syntax tree
/// representing the statement using recursive descent parsing.
/// 
/// A syntax error is returned for any syntactical errors in the token sequence.
/// 
/// ```text
/// <statement> ::=  (<let> | <expr>) [";" | Newline]
/// ```
pub fn parse_statement(token_stream: &mut TokenStream) -> ParseResult<AbstractSyntaxTree> {
    skip_newlines(token_stream);

    let Ok(first) = token_stream.peek() else {
        return Err(IntepreterError::EndOfFile);
    };
    let statement = match first {
        Token::Let => AbstractSyntaxTree::Let(parse_let(token_stream)?),
        _ => AbstractSyntaxTree::Expr(parse_expr(token_stream)?),
    };

    let statement_end = token_stream.pop()?;
    match statement_end {
        Token::Formatter(ref f) if f == ";" => {},
        Token::Newline => {},
        _ => return Err(IntepreterError::end_of_statement(statement_end)),
    }
    return Ok(statement);
}

/// ```test
/// <expr> ::= <cond_expr> | <function> | <term> {BinaryOp <term>}
/// ```
fn parse_expr(tokens: &mut TokenStream) -> ParseResult<Expr> {
    if tokens.peek()? == Token::If {
        let cond_expr = parse_cond_expr(tokens)?;
        return Ok(Expr::Cond(cond_expr));
    }

    /// to decide when to stop parsing, since expr is variable length
    fn next_is_operator(tokens: &mut TokenStream) -> bool {
        match tokens.peek() {
            Ok(Token::BinaryOp(_)) => true,
            _ => false,
        }
    }

    let next_is_function =
        token_matches_formatter(&tokens.lookahead(0)?, "(")
        && (
            token_matches_formatter(&tokens.lookahead(1)?, ")")
            || token_matches_formatter(&tokens.lookahead(2)?, ":")
        );

    if next_is_function {
        let function = parse_function(tokens)?;
        let expr = Expr::Function(function);
        return Ok(expr);
    }

    let first = parse_term(tokens)?;
    let mut rest = Vec::<(BinaryOp, Term)>::new();

    while next_is_operator(tokens) {
        let op_token = tokens.pop()?;
        let operator = match op_token {
            Token::BinaryOp(op) => op,
            _ => return Err(error::unexpected_token("binary op", op_token))
        };
        let term = parse_term(tokens)?;
        rest.push((operator, term));
    }
    Ok(Expr::Binary(BinaryExpr { first, rest }))
}

/// ```text
/// <cond_expr> ::= If "(" <expr> ")" <expr> Else <expr>
/// ```
fn parse_cond_expr(tokens: &mut TokenStream) -> ParseResult<CondExpr> {
    match_next(tokens, Token::If)?;
    match_next(tokens, Token::Formatter("(".into()))?;
    let condition = Box::new(parse_expr(tokens)?);
    match_next(tokens, Token::Formatter(")".into()))?;

    skip_newlines(tokens);
    let if_true = Box::new(parse_expr(tokens)?);

    skip_newlines(tokens);
    match_next(tokens, Token::Else)?;

    skip_newlines(tokens);
    let if_false = Box::new(parse_expr(tokens)?);

    return Ok(CondExpr { cond: condition, if_true, if_false });
}

/// ```text
/// <function> ::= "(" <param_list> ")" [<type_declaration>] "->" <expr>
/// 
/// <param_list> ::= [Id <type_declaration> {"," Id <type_declaration>}]
/// ```
fn parse_function(tokens: &mut TokenStream) -> ParseResult<Function> {
    let mut params = Vec::<(String, Type)>::new();

    match_next(tokens, Token::Formatter("(".into()))?;
    while !token_matches_formatter(&tokens.peek()?, ")") {
        // skip comma for first parameter
        if !params.is_empty() {
            match_next(tokens, Token::Formatter(",".into()))?;
        }
        let id = parse_id(tokens)?; 
        let datatype = parse_type_declaration(tokens)?;
        params.push((id, datatype));
    }
    match_next(tokens, Token::Formatter(")".into()))?;
    let return_type = if token_matches_formatter(&tokens.peek()?, ":") {
        Some(parse_type_declaration(tokens)?)
    } else {
        None
    };

    match_next(tokens, Token::Formatter("->".into()))?;
    skip_newlines(tokens);
    let body_expr = parse_expr(tokens)?;

    Ok(Function { params, return_type, body: FuncBody::Expr(Box::new(body_expr)) })
}

/// ```text
/// <term> ::= "!" <term> | ["-"] <base_term>
/// ```
fn parse_term(tokens: &mut TokenStream) -> ParseResult<Term> {
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
/// <base_term> ::= Literal | Null | (Id | "(" <expr> ")") ["!"] {<arg_list> ["!"]}
/// ```
fn parse_base_term(tokens: &mut TokenStream) -> ParseResult<Term> {
    let first = tokens.pop()?;
    if Token::Null == first {
        return Ok(Term::Literal(Literal::Null));
    }
    if let Token::Literal(lit) = first {
        return Ok(Term::Literal(lit));
    }

    let callable = if let Token::Id(id) = first {
        Term::Id(id)
    } else if token_matches_formatter(&first, "(") {
        let expr = parse_expr(tokens)?;
        match_next(tokens, Token::Formatter(")".into()))?;
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
/// The `<arg_list>` below is preceeded by the callable term and may be
/// succeeded by the not-null assertion operator "!".
/// ```text
/// <arg_list> ::= "(" [<expr> {"," <expr>}] ")"
/// ```
fn complete_term_with_arg_list(
    callable: Term,
    tokens: &mut TokenStream
) -> ParseResult<Term> {
    // we might be at the end of the stream, but that's allowed since
    // callable is a valid term
    match tokens.peek() {
        Ok(Token::Formatter(ref f)) if f == "(" => {},
        _ => return Ok(callable),
    }
    tokens.pop()?;
    let mut args = Vec::<Expr>::new();

    if !token_matches_formatter(&tokens.peek()?, ")") {
        args.push(parse_expr(tokens)?);
        while token_matches_formatter(&tokens.peek()?, ",") {
            tokens.pop()?;
            args.push(parse_expr(tokens)?);
        }
    } 
    match_next(tokens, Token::Formatter(")".into()))?;

    let func_call = FuncCall { func: Box::new(callable), args };
    let callable = if next_token_matches(tokens, Token::UnaryOp(UnaryOp::Not)) {
        tokens.pop()?;
        Term::FuncCall(func_call).as_not_null()
    } else {
        Term::FuncCall(func_call)
    };

    complete_term_with_arg_list(callable, tokens)
}

/// Create an AST for the "let" keyword given the remaining tokens
/// ```text
/// <let> ::= Let Id [<type_declaration>] Equals <expr> | Let Id <function>
/// ```
fn parse_let(tokens: &mut TokenStream) -> ParseResult<LetNode> {
    match_next(tokens, Token::Let)?;
    let id = parse_id(tokens)?;

    let includes_type_declaration =
        token_matches_formatter(&tokens.peek()?, ":");

    let is_shorthand_function = token_matches_formatter(&tokens.peek()?, "(");

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
        parse_expr(tokens)?
    };

    return Ok(LetNode { id, datatype, value: expr });
}

/// Match the next token as an Id and extract the String name from it
fn parse_id(tokens: &mut TokenStream) -> ParseResult<String> {
    let id_token = tokens.pop()?;
    match id_token {
        Token::Id(s) => Ok(s.to_string()),
        _ => Err(error::unexpected_token("identifier", id_token)),
    }
}

/// ```text
/// <type_declaration> :: = ":" Type ["?"] | ":" Null
/// ```
fn parse_type_declaration(tokens: &mut TokenStream) -> ParseResult<Type> {
    match_next(tokens, Token::Formatter(":".into()))?;
    let type_token = tokens.pop()?;
    if let Token::Null = type_token {
        return Ok(Type::Null);
    }

    let Token::Type(datatype) = type_token else {
        return Err(error::not_a_type(type_token));
    };

    if next_token_matches(tokens, Token::UnaryOp(UnaryOp::Nullable)) {
        tokens.pop()?;
        Ok(datatype.to_nullable())
    } else {
        Ok(datatype.clone())
    }
}

/// Skip over any newlines at the front of the stream.
/// Will never error.
fn skip_newlines(tokens: &mut TokenStream) {
    while let Ok(Token::Newline) = tokens.peek() {
        // safe to unwrap since the peeked value is Ok
        let _ = tokens.pop();
    }
}

fn token_matches_formatter(token: &Token, formatter: &str) -> bool {
    match token {
        Token::Formatter(ref f) if f == formatter => true,
        _ => false,
    }
}

/// Test if the next token in the stream matches the given token without
/// consuming it. If the stream is empty, this returns false
fn next_token_matches(stream: &mut TokenStream, token: Token) -> bool {
    stream.peek() == Ok(token)
}

/// Take the next token from the stream and compare it to the expected token.
/// If they do not match, return a syntax error 
fn match_next(stream: &mut TokenStream, expected: Token) -> ParseResult<()> {
    let next = stream.pop()?;
    if next != expected {
        // TODO: improve debug string
        Err(error::unexpected_token(format!("{expected:?}").as_str(), next))
    } else {
        Ok(())
    }
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

    pub fn parse_tokens(tokens: Vec<Token>) -> ParseResult<AbstractSyntaxTree> {
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
            let input = vec![token.clone(), Token::Newline];
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
            "(x)!;",
            Term::Expr(
                Box::new(
                    term_expr(Term::Id("x".into()))
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
                IntepreterError::end_of_statement(Token::UnaryOp(UnaryOp::Not));
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
            let error = IntepreterError::end_of_statement(int_token(2));
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

    }

    mod function {
        use super::*;

        #[rstest]
        #[case::thunk("() -> 3;", vec![], None, int_term(3))]
        #[case::many_params("(x: int, y: string) -> 3;",
            vec![("x".into(), Type::Int), ("y".into(), Type::String)],
            None,
            int_term(3)
        )]
        #[case::declared_return_type("(): bool -> true;",
            vec![],
            Some(Type::Bool),
            Term::Literal(Literal::Bool(true))
        )]
        fn it_parses_function(
            #[case] input: &str,
            #[case] params: Vec<(String, Type)>,
            #[case] return_type: Option<Type>,
            #[case] body_term: Term,
        ) {
            let tokens = force_tokenize(input);
            let function = Function {
                params,
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
            let error = IntepreterError::end_of_statement(
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
            datatype: Some(Type::String.to_nullable()),
            value: bin_expr(Term::Literal(Literal::Null), vec![]),
        })]
        fn it_parses_var_binding(#[case] input: &str, #[case] expected: LetNode) {
            let input: Vec<Token> = force_tokenize(input);
            let syntax_tree = AbstractSyntaxTree::Let(expected);
            assert_eq!(parse_tokens(input), Ok(syntax_tree));
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
            let input = force_tokenize("let x: y = z;");
            let error = error::not_a_type(Token::Id("y".to_string()));
            assert_eq!(parse_tokens(input), Err(error)); 
        }

        #[test]
        fn it_returns_error_for_unexpected_let() {
            let input = force_tokenize("let x = let y = 2;");
            let error = error::unexpected_token("identifier or expression", Token::Let);
            assert_eq!(parse_tokens(input), Err(error));
        }
    }

    #[rstest]
    #[case(force_tokenize("let"), error::unexpected_end_of_input())]
    #[case(force_tokenize("let x"), error::unexpected_end_of_input())]
    #[case(force_tokenize("3 +"), error::unexpected_end_of_input())]
    fn it_returns_error_for_incomplete_statements(
        #[case] tokens: Vec<Token>,
        #[case] error: IntepreterError
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
    fn it_returns_same_tree_for_statement_ending_in_newlines_and_semicolon() {
        let newline_statement = parse_tokens(force_tokenize("2 + 2\n\n\n"));
        let semicolon_statement = parse_tokens(force_tokenize("2 + 2;"));

        assert!(matches!(newline_statement, Ok(_)));
        assert_eq!(newline_statement, semicolon_statement);
    }

    #[test]
    fn it_ignores_tokens_after_semicolon() {
        let multiple_statements =
        parse_tokens(force_tokenize("2 + 2; 5 + 5; let = badsyntax ->"));
        let one_statement = parse_tokens(force_tokenize("2+2\n"));
    
        assert!(matches!(multiple_statements, Ok(_)));
        assert_eq!(multiple_statements, one_statement);
    }

    #[rstest]
    #[case::shorthand_function("let not(b: bool) -> !b;", "let not = (b: bool) -> !b;")]
    #[case::if_else_on_seperate_lines("if (true) 1 else 0;", "
        if (true)
            1
        else
            0;
    ")]
    #[case::function_on_seperate_lines(
        r#"let toString(x: int) -> if (x == 0) "0" else if (x == 1) "1" else "unknown";"#,
        r#"
        let toString(x: int) ->
            if (x == 0) "0"
            else if (x == 1) "1"
            else "unknown";
        "#
    )]
    fn it_parses_equivalent_statements(#[case] vers1: &str, #[case] vers2: &str) {
        let res1 = parse_tokens(force_tokenize(vers1));
        let res2 = parse_tokens(force_tokenize(vers2));

        assert!(matches!(res1, Ok(_)));
        assert_eq!(res1, res2)
    }

    #[test]
    fn it_returns_end_of_file_error_for_empty_stream() {
        assert_eq!(parse_tokens(vec![]), Err(IntepreterError::EndOfFile));
    }

    fn force_tokenize(line: &str) -> Vec<Token> {
        return tokenize(line).unwrap();
    }
}
