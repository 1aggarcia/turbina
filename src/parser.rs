use crate::models::{
    AbstractSyntaxTree, BinaryExpr, BinaryOp, CondExpr, Expr, FuncCall, LetNode, Literal, Term, Token, Type, UnaryOp
};

use crate::errors::{IntepreterError, error};

type ParseResult<T> = Result<T, IntepreterError>;

/// Abstract Data Type used internally by the parser to facilitate tracking
/// token position and end-of-stream errors
struct TokenStream {
    tokens: Vec<Token>,
    position: usize,
}

impl TokenStream {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, position: 0 }
    }

    fn has_next(&self) -> bool {
        self.position < self.tokens.len()
    }

    /// Advances to the next token and returns the current one
    fn pop(&mut self) -> Result<Token, IntepreterError> {
        let token = self.peek()?;
        self.position += 1;
        return Ok(token);
    }

    /// Returns the token currently being pointed to
    fn peek(&self) -> Result<Token, IntepreterError> {
        match self.tokens.get(self.position) {
            Some(token) => Ok(token.clone()),
            None => Err(error::unexpected_end_of_input()),
        }
    }
}

/// Convert a sequence of tokens into an abstract syntax tree using recursive decent.
/// 
/// The tokens should represent exactly one statement and therefore one syntax
/// tree. Otherwise, an error is returned.
/// 
/// A syntax error is returned for any syntactical errors in the token sequence.
/// 
/// ```
/// <statement> ::=  <let> | <expr>
/// ```
pub fn parse(tokens: Vec<Token>) -> ParseResult<AbstractSyntaxTree> {
    let mut token_stream = TokenStream::new(tokens);

    let statement = match token_stream.peek()? {
        Token::Let => AbstractSyntaxTree::Let(parse_let(&mut token_stream)?),
        _ => AbstractSyntaxTree::Expr(parse_expr(&mut token_stream)?),
    };

    if token_stream.has_next() {
        let err_token = token_stream.peek().unwrap();
        return Err(error::unexpected_token("end of line", err_token));
    }
    return Ok(statement);
}

/// ```
/// <expr> ::= <cond_expr> | <term> {BinaryOp <term>}
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

/// ```
/// <cond_expr> ::= If "(" <expr> ")" <expr> Else <expr>
/// ```
fn parse_cond_expr(tokens: &mut TokenStream) -> ParseResult<CondExpr> {
    match_next(tokens, Token::If)?;
    match_next(tokens, Token::Formatter("(".into()))?;
    let condition = Box::new(parse_expr(tokens)?);
    match_next(tokens, Token::Formatter(")".into()))?;
    let if_true = Box::new(parse_expr(tokens)?);
    match_next(tokens, Token::Else)?;
    let if_false = Box::new(parse_expr(tokens)?);

    return Ok(CondExpr { cond: condition, if_true, if_false });
}

/// ```
/// <term> = "!" <term> | ["-"] <base-term>
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

/// ```
/// <base-term> = Literal | (Id | "(" <expr> ")") {<arg-list>} | Null
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
    complete_term_with_arg_list(callable, tokens)
}

/// Try to construct a function call term with the function supplied as a
/// callable term. If a function call cannot be constructed, the callable term is returned.
/// 
/// The `<arg_list>` below is preceeded by the callable term.
/// ```
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
    complete_term_with_arg_list(Term::FuncCall(func_call), tokens)
}

/// Create an AST for the "let" keyword given the remaining tokens
/// ```
/// <let> = Let Id [":" Type] Equals <expression>
/// ```
fn parse_let(tokens: &mut TokenStream) -> ParseResult<LetNode> {
    match_next(tokens, Token::Let)?;

    let id_token = tokens.pop()?;
    let id = match id_token {
        Token::Id(s) => s.to_string(),
        _ => return Err(error::unexpected_token("identifier", id_token))
    };

    // look for optional declared type (e.g. ": int")
    let is_token_colon = token_matches_formatter(&tokens.peek()?, ":");
    let datatype = if is_token_colon {
        tokens.pop()?;
        let type_token = tokens.pop()?;
        match type_token {
            Token::Type(t) => Some(t.to_owned()),
            Token::Null => Some(Type::Null),
            _ => return Err(error::not_a_type(type_token)) 
        }
    } else {
        None
    };

    match_next(tokens, Token::UnaryOp(UnaryOp::Equals))?;
    let expr = parse_expr(tokens)?;

    return Ok(LetNode { id, datatype, value: expr });
}

fn token_matches_formatter(token: &Token, formatter: &str) -> bool {
    match token {
        Token::Formatter(ref f) if f == formatter => true,
        _ => false,
    }
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
mod test_token_stream {
    use super::*;
    use crate::models::test_utils::id_token;

    #[test]
    fn test_has_next_is_false_for_empty_stream() {
        let mut stream = TokenStream::new(vec![]);
        assert_eq!(stream.has_next(), false);
        assert_eq!(stream.pop(), Err(error::unexpected_end_of_input()));
    }

    #[test]
    fn test_has_next_is_true_for_non_empty_stream() {
        let mut stream = TokenStream::new(vec![id_token("data")]);
        assert_eq!(stream.has_next(), true);
        assert_eq!(stream.pop(), Ok(id_token("data")));
    }

    #[test]
    fn test_peek_doesnt_consume_data() {
        let mut stream = TokenStream::new(vec![id_token("data")]);
        assert_eq!(stream.peek(), Ok(id_token("data")));
        assert_eq!(stream.has_next(), true);
        assert_eq!(stream.pop(), Ok(id_token("data")));
        assert_eq!(stream.has_next(), false);
        assert_eq!(stream.peek(), Err(error::unexpected_end_of_input()));
    }
}

#[cfg(test)]
mod test_parse {
    use super::*;
    use crate::lexer::tokenize;
    use crate::models::test_utils::*;
    use crate::models::*;
    use rstest::rstest;

    #[test]
    fn it_returns_error_for_empty_list()  {
        assert_eq!(parse(vec![]), Err(error::unexpected_end_of_input()));
    }

    #[rstest]
    #[case(int_token(2), term_tree(int_term(2)))]
    #[case(string_token("prueba test"), term_tree(str_term("prueba test")))]
    #[case(id_token("name"), term_tree(Term::Id("name".into())))]
    fn it_parses_one_token_to_one_node(
        #[case] token: Token,
        #[case] node: AbstractSyntaxTree
    ) {
        let input = vec![token.clone()];
        assert_eq!(parse(input), Ok(node));
    }

    #[rstest]
    #[case(op_token(BinaryOp::Plus))]
    #[case(op_token(BinaryOp::Percent))]
    #[case(unary_op_token(UnaryOp::Equals))]
    fn it_returns_error_for_one_operator(#[case] op: Token) {
        let error = error::unexpected_token("identifier or expression", op.clone());
        assert_eq!(parse(vec![op]), Err(error));
    }

    #[rstest]
    #[case(force_tokenize("3 + 2"), BinaryOp::Plus, 3, 2)]
    #[case(force_tokenize("1 % 4"), BinaryOp::Percent, 1, 4)]
    #[case(force_tokenize("1 - 8"), BinaryOp::Minus, 1, 8)]
    #[case(force_tokenize("0 == 1"), BinaryOp::Equals, 0, 1)]
    #[case(force_tokenize("2 != 3"), BinaryOp::NotEq, 2, 3)]
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
        assert_eq!(parse(input), Ok(expected));
    }

    #[rstest]
    #[case(force_tokenize("-9"), -9)]
    #[case(force_tokenize("- 123"), -123)]
    fn it_parses_negative_numbers(
        #[case] input: Vec<Token>,
        #[case] negative_num: i32
    ) {
        let inner_term = int_term(negative_num * -1);
        let term = Term::negative_int(inner_term);
        let expected = term_tree(term);
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_parses_negative_num_in_parens() {
        let input = force_tokenize("-(9)");

        // the parentheses creates this unfortunate indirection in the AST
        let nine = term_expr(int_term(9));
        let nine_as_term = Term::Expr(Box::new(nine));
        let negative_nine = Term::negative_int(nine_as_term);
        let expected_tree = term_tree(negative_nine);

        assert_eq!(parse(input), Ok(expected_tree)); 
    }

    #[test]
    fn it_parses_negated_boolean() {
        let term = Term::negated_bool(Term::Id("someVariable".into()));
        let expected = term_tree(term);
        assert_eq!(parse(force_tokenize("!someVariable")), Ok(expected));
    }

    #[test]
    fn it_parses_many_negations() {
        let term = Term::negated_bool(
            Term::negated_bool(
                Term::negated_bool(Term::Id("x".into()))
            )
        );
        let expected = term_tree(term);
        assert_eq!(parse(force_tokenize("!!!x")), Ok(expected));
    }

    #[test]
    fn it_returns_error_for_many_negative_symbols() {
        let error = error::unexpected_token(
            "identifier or expression",
            Token::BinaryOp(BinaryOp::Minus)
        );
        assert_eq!(parse(force_tokenize("---9")), Err(error));
    }

    #[test]
    fn it_returns_error_for_multiple_ints() {
        let input = force_tokenize("3 2");
        let error = error::unexpected_token("end of line", int_token(2));
        assert_eq!(parse(input), Err(error));
    }

    #[test]
    fn it_parses_string_plus_string() {
        let input = force_tokenize("\"a\" + \"b\"");

        let expr = bin_expr(
            str_term("a"),
            vec![(BinaryOp::Plus, str_term("b"))]
        );
        let expected = AbstractSyntaxTree::Expr(expr);
        assert_eq!(parse(input), Ok(expected)); 
    }

    #[test]
    fn it_parses_var_binding_to_literal() {
        let input = force_tokenize("let x = 4");
        let let_node = LetNode {
            id: "x".to_string(),
            datatype: None,
            value: term_expr(int_term(4)),
        };
        let expected = AbstractSyntaxTree::Let(let_node);
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_parses_expression_in_parens() {
        let input = force_tokenize("3 * (2 - 5)");

        let inner_expr = bin_expr(
            int_term(2),
            vec![(BinaryOp::Minus, int_term(5))]
        );
        let expr = bin_expr(
            int_term(3),
            vec![(BinaryOp::Star, Term::Expr(Box::new(inner_expr)))]
        );
        let expected = AbstractSyntaxTree::Expr(expr);
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_parses_if_else_expression() {
        let input = force_tokenize("if (cond) \"abc\" else \"def\"");

        let expr = Expr::Cond(CondExpr {
            cond: Box::new(term_expr(Term::Id("cond".into()))),
            if_true: Box::new(term_expr(str_term("abc"))),
            if_false: Box::new(term_expr(str_term("def"))),
        });
        let expected = Ok(AbstractSyntaxTree::Expr(expr));

        assert_eq!(parse(input), expected);
    }

    #[rstest]
    #[case::empty("doSomething()", FuncCall {
        func: Box::new(Term::Id("doSomething".into())), args: vec![]
    })]
    #[case::one_arg("print(3)", FuncCall {
        func: Box::new(Term::Id("print".into())), args: vec![term_expr(int_term(3))]
    })]
    #[case::many_args("list(1, 2, 3)", FuncCall {
        func: Box::new(Term::Id("list".into())),
        args: [1, 2, 3].iter().map(|i| term_expr(int_term(*i))).collect()
    })]
    #[case::curried_function("x(1)(2)", FuncCall {
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
        assert_eq!(parse(tree), expected);
    }

    #[test]
    fn it_parses_expression_with_multiple_functions() {
        let input = force_tokenize("x() + y()");

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

        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_returns_error_for_bad_var_id() {
        let input = force_tokenize("let 3 = 3");
        let error = error::unexpected_token("identifier", int_token(3));
        assert_eq!(parse(input), Err(error));
    }

    #[test]
    fn it_returns_error_for_equals_in_let_expr() {
        let input = force_tokenize("let x = 1 = 0");
        let error = error::unexpected_token(
            "end of line",
            Token::UnaryOp(UnaryOp::Equals)
        );
        assert_eq!(parse(input), Err(error)); 
    }

    #[test]
    fn it_parses_var_binding_to_expr() {
        let input = force_tokenize("let two = 6 / 3");
        let let_node = LetNode {
            id: "two".to_string(),
            datatype: None,
            value: bin_expr(
                int_term(6),
                vec![(BinaryOp::Slash, int_term(3))]
            ),
        };
        let expected = AbstractSyntaxTree::Let(let_node);
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_parses_var_binding_with_declared_type() {
        let input = force_tokenize("let two_strings: string = \"a\" + \"b\"");
        let let_node = LetNode {
            id: "two_strings".to_string(),
            datatype: Some(Type::String),
            value: bin_expr(
            str_term("a"),
                vec![(BinaryOp::Plus, str_term("b"))]
            ),
        };
        let expected = AbstractSyntaxTree::Let(let_node);
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_returns_error_for_invalid_let_type() {
        let input = force_tokenize("let x: y = z");
        let error = error::not_a_type(Token::Id("y".to_string()));
        assert_eq!(parse(input), Err(error)); 
    }

    #[test]
    fn it_returns_error_for_unexpected_let() {
        let input = force_tokenize("let x = let y = 2");
        let error = error::unexpected_token("identifier or expression", Token::Let);
        assert_eq!(parse(input), Err(error));
    }

    #[rstest]
    #[case(force_tokenize("let"), error::unexpected_end_of_input())]
    #[case(force_tokenize("let x"),error::unexpected_end_of_input())]
    #[case(force_tokenize("3 +"), error::unexpected_end_of_input())]
    fn it_returns_error_for_incomplete_statements(
        #[case] tokens: Vec<Token>,
        #[case] error: IntepreterError
    ) {
        assert_eq!(parse(tokens), Err(error));
    }

    fn force_tokenize(line: &str) -> Vec<Token> {
        return tokenize(line).unwrap();
    }
}
