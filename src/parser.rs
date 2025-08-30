use crate::models::{
    AbstractSyntaxTree, BinaryExpr, BinaryOp, CodeBlock, CondExpr, Expr, FuncBody, FuncCall, Function, LetNode, Literal, Term, Token, Type, UnaryOp
};

use crate::errors::{error, InterpreterError, Result};
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
        _ => AbstractSyntaxTree::Expr(parse_expr(token_stream)?),
    };

    let statement_end = token_stream.pop()?;
    match statement_end {
        Token::Semicolon | Token::Newline => {},
        _ => return Err(InterpreterError::end_of_statement(statement_end)),
    }
    return Ok(statement);
}

/// ```text
/// <expr> ::= <code_block> | <cond_expr> | <function> | <binary_expr>
/// ```
fn parse_expr(tokens: &mut TokenStream) -> Result<Expr> {
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
        } else {
            tokens.lookahead(1)? == Token::CloseParens
            || tokens.lookahead(2)? == Token::Colon 
        };

    if next_is_function {
        let function = parse_function(tokens)?;
        let expr = Expr::Function(function);
        return Ok(expr);
    }

    let binary_expr = parse_binary_expr(tokens, 0)?;
    Ok(Expr::Binary(binary_expr))
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

/// If the expression contains one term, unwrap the term.
/// Otherwise, wrap the expression into a term
fn expr_as_term(expr: BinaryExpr) -> Term {
    if expr.rest.is_empty() {
        expr.first
    } else {
        Term::Expr(Box::new(Expr::Binary(expr)))
    }
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

/// ```text
/// <function> ::= [type_param_list] "(" <param_list> ")" [<type_declaration>] <function_body>
/// <function_body> ::= "->" <expr> | <code_block>
/// 
/// <type_param_list> ::= "<" [Id] {"," Id} ">"
/// <param_list> ::= [Id <type_declaration> {"," Id <type_declaration>}]
/// ```
fn parse_function(tokens: &mut TokenStream) -> Result<Function> {
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
        item_parser: |tokens|
            Ok((parse_id(tokens)?, parse_type_declaration(tokens)?)),
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
/// The `<arg_list>` below is preceeded by the callable term and may be
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

/// Create an AST for the "let" keyword given the remaining tokens
/// ```text
/// <let> ::= Let Id [<type_declaration>] Equals <expr> | Let Id <function>
/// ```
fn parse_let(tokens: &mut TokenStream) -> Result<LetNode> {
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

/// Match the next token as an Id and extract the String name from it
fn parse_id(tokens: &mut TokenStream) -> Result<String> {
    let id_token = tokens.pop()?;
    match id_token {
        Token::Id(s) => Ok(s.to_string()),
        _ => Err(error::unexpected_token("identifier", id_token)),
    }
}

/// Type declarations cannot include function types without parentheses
/// since return types and function bodies are ambiguous 
///
/// e.g. `(): (bool -> bool) -> (x: bool) -> true` is a function that returns a
/// function returning bool, but without parentheses it is unclear. (Even with
/// parentheses I find it confusing, I suggest just using inferred return types).
/// ```text
/// <type_declaration> :: = ":" <base_type>
/// ```
fn parse_type_declaration(tokens: &mut TokenStream) -> Result<Type> {
    match_next(tokens, Token::Colon)?;
    parse_base_type(tokens)
}

/// ```text
/// <type> ::= <base_type> | <function_type>
/// ```
///
/// Note the `arg_types` non-terminal cannot be a list of one type, since that
/// conflicts with a variant of `base_type` 
/// ```text
/// <function_type> ::= <arg_types> "->" <type>
/// <arg_types> ::= <base_type> | "(" ") | "(" <type> "," <type> {"," <type>} ")"
/// ```
fn parse_type(tokens: &mut TokenStream) -> Result<Type> {
    /// Decide between leaving a base type alone or parsing it as a function
    fn complete_type(
        base_type: Type, tokens: &mut TokenStream
    ) -> Result<Type> {
        if next_token_matches(tokens, Token::Arrow) {
            complete_function_type(vec![base_type], tokens)
        } else {
            Ok(base_type)
        }
    }

    fn complete_function_type(
        arg_types: Vec<Type>,
        tokens: &mut TokenStream
    ) -> Result<Type> {
        match_next(tokens, Token::Arrow)?;
        let return_type = parse_type(tokens)?;

        let func_type = Type::Func {
            input: arg_types,
            output: Box::new(return_type)
        };
        Ok(func_type)
    }

    // start of parsing
    if tokens.peek()? != Token::OpenParens {
        let base_type = parse_base_type(tokens)?;
        return complete_type(base_type, tokens);
    }

    let types_in_parens = parse_list(tokens, ListParserConfig {
        item_parser: parse_type,
        opening_token: Some(Token::OpenParens),
        closing_token: Token::CloseParens,
    })?;

    if types_in_parens.len() == 1 {
        let base_type = complete_base_type(types_in_parens[0].clone(), tokens)?;
        // one type in parentheses is allowed to not be a function
        return complete_type(base_type, tokens);
    }
    complete_function_type(types_in_parens, tokens)

}

/// ```text
/// <base_type> ::= Type ["?"] | "(" <type> ")" ["?"] | <base-type> "[]"
/// ```
fn parse_base_type(tokens: &mut TokenStream) -> Result<Type> {
    if tokens.peek()? == Token::OpenParens {
        tokens.pop()?;
        let datatype = parse_type(tokens)?;
        match_next(tokens, Token::CloseParens)?;
        return complete_base_type(datatype, tokens);
    }

    let type_token = tokens.pop()?;
    if let Token::Null = type_token {
        return Ok(Type::Null);
    }

    let datatype = match type_token {
        Token::Type(literal_type) => literal_type,
        Token::Id(type_name) => Type::Generic(type_name),
        _ => return Err(error::not_a_type(type_token)), 
    };
    complete_base_type(datatype, tokens)
}

/// Given a base type, recursively wrap it as a list type if the next tokens
/// are "[]". If the next token is "?", make it a nullable type. Nullable
/// types cannot be stacked the same way list types can.
fn complete_base_type(base_type: Type, tokens: &mut TokenStream) -> Result<Type> {
    if !matches!(base_type, Type::Nullable(..))
        && next_token_matches(tokens, Token::UnaryOp(UnaryOp::Nullable))
    {
        tokens.pop()?;
        complete_base_type(base_type.as_nullable(), tokens)
    } else if next_token_matches(tokens, Token::OpenSquareBracket) {
        tokens.pop()?;
        match_next(tokens, Token::CloseSquareBracket)?;
        complete_base_type(base_type.as_list(), tokens)
    } else {
        Ok(base_type.clone())
    }
}

struct ListParserConfig<T> {
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
fn parse_list<T>(
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

/// Skip over any newlines at the front of the stream.
/// Will never error.
fn skip_newlines(tokens: &mut TokenStream) {
    while let Ok(Token::Newline) = tokens.peek() {
        // safe to unwrap since the peeked value is Ok
        let _ = tokens.pop();
    }
}

/// Test if the next token in the stream matches the given token without
/// consuming it. If the stream is empty, this returns false
fn next_token_matches(stream: &mut TokenStream, token: Token) -> bool {
    stream.peek() == Ok(token)
}

/// Take the next token from the stream and compare it to the expected token.
/// If they do not match, return a syntax error 
fn match_next(stream: &mut TokenStream, expected: Token) -> Result<()> {
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
            Term::Literal(Literal::Null),
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
        let toString(
            x: int
        ) ->
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

    fn force_tokenize(line: &str) -> Vec<Token> {
        return tokenize(line).unwrap();
    }
}
