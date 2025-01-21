use crate::models::{
    AbstractSyntaxTree, BinaryOp, ExprNode, LetNode, Term, Token, UnaryOp
};

use crate::interpreter_error::IntepreterError;
use crate::interpreter_error as errors;

type ParseResult = Result<AbstractSyntaxTree, IntepreterError>;

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
            None => Err(errors::unexpected_end_of_input()),
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
pub fn parse(tokens: Vec<Token>) -> ParseResult {
    let mut token_stream = TokenStream::new(tokens);

    let statement = match token_stream.peek()? {
        Token::Let => parse_let(&mut token_stream),
        _ => parse_expr(&mut token_stream),
    }?;

    if token_stream.has_next() {
        let err_token = token_stream.peek().unwrap();
        return Err(errors::unexpected_token("end of line", err_token));
    }
    return Ok(statement);
}

/// ```
/// <expr> ::= <term> {<binary_op> <term>}
/// ```
fn parse_expr(tokens: &mut TokenStream) -> ParseResult {
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
            _ => return Err(errors::unexpected_token("binary op", op_token))
        };
        let term = parse_term(tokens)?;
        rest.push((operator, term));
    }
    let node = ExprNode { first, rest };
    return Ok(AbstractSyntaxTree::Expr(node));
}

/// Build an AST node for a Term, which follows the below rule:
/// ```
/// <term> ::= ["-"] (Literal | Id) | "!" <term>
/// ```
fn parse_term(tokens: &mut TokenStream) -> Result<Term, IntepreterError> {
    let mut starts_with_minus = false;
    
    let first = tokens.peek()?;
    if first == Token::UnaryOp(UnaryOp::Not) {
        tokens.pop()?;
        let inner_term = parse_term(tokens)?;
        return Ok(Term::negated_bool(inner_term));
    } else if first == Token::BinaryOp(BinaryOp::Minus) {
        tokens.pop()?;
        starts_with_minus = true;
    }

    let value_token = tokens.pop()?;
    let value = match value_token {
        Token::Literal(lit) => Term::Literal(lit),
        Token::Id(id) => Term::Id(id),
        _ => return Err(
            errors::unexpected_token("identifier or literal", value_token)
        ),
    };

    if starts_with_minus {
        return Ok(Term::negative_int(value))
    }
    return Ok(value);
}

/// Create an AST for the "let" keyword given the remaining tokens
/// ```
/// <let> = Let Id [":" Type] Equals <expression>
/// ```
fn parse_let(tokens: &mut TokenStream) -> ParseResult {
    if tokens.pop()? != Token::Let {
        panic!("THIS SHOULD NOT HAPPEN: check that stream starts with 'let' token");
    }

    let id_token = tokens.pop()?;
    let id = match id_token {
        Token::Id(s) => s.to_string(),
        _ => return Err(errors::unexpected_token("identifier", id_token))
    };

    // look for optional declared type (e.g. ": int")
    let is_token_colon = match tokens.peek()? {
        Token::Formatter(ref f) if f == ":" => true,
        _ => false,
    };
    let datatype = if is_token_colon {
        tokens.pop()?;
        let type_token = tokens.pop()?;
        match type_token {
            Token::Type(t) => Some(t.to_owned()),
            _ => return Err(errors::not_a_type(type_token)) 
        }
    } else {
        None
    };

    let equals_token = tokens.pop()?;
    match equals_token {
        Token::UnaryOp(UnaryOp::Equals) => {},
        _ => return Err(errors::unexpected_token("'='", equals_token))
    };

    let value_node = parse_expr(tokens)?;

    let node = LetNode { id, datatype, value: Box::new(value_node) };
    return Ok(AbstractSyntaxTree::Let(node));
}

#[cfg(test)]
mod test_token_stream {
    use super::*;
    use crate::models::test_utils::id_token;

    #[test]
    fn test_has_next_is_false_for_empty_stream() {
        let mut stream = TokenStream::new(vec![]);
        assert_eq!(stream.has_next(), false);
        assert_eq!(stream.pop(), Err(errors::unexpected_end_of_input()));
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
        assert_eq!(stream.peek(), Err(errors::unexpected_end_of_input()));
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
        assert_eq!(parse(vec![]), Err(errors::unexpected_end_of_input()));
    }

    #[rstest]
    #[case(int_token(2), term_tree(Term::Literal(Literal::Int(2))))]
    #[case(string_token("prueba test"), term_tree(
        Term::Literal(Literal::String("prueba test".into()))
    ))]
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
        let error = errors::unexpected_token("identifier or literal", op.clone());
        assert_eq!(parse(vec![op]), Err(error));
    }

    #[rstest]
    #[case(tokenize("3 + 2"), BinaryOp::Plus, 3, 2)]
    #[case(tokenize("1 % 4"), BinaryOp::Percent, 1, 4)]
    #[case(tokenize("1 - 8"), BinaryOp::Minus, 1, 8)]
    #[case(tokenize("0 == 1"), BinaryOp::Equals, 0, 1)]
    #[case(tokenize("2 != 3"), BinaryOp::NotEq, 2, 3)]
    fn it_parses_binary_expressions(
        #[case] input: Vec<Token>,
        #[case] operator: BinaryOp,
        #[case] left_val: i32,
        #[case] right_val: i32,
    ) {
        let left = Term::Literal(Literal::Int(left_val));
        let right = Term::Literal(Literal::Int(right_val));
        let expected = AbstractSyntaxTree::Expr(
            ExprNode { first: left, rest: vec![(operator, right)] }
        );
        assert_eq!(parse(input), Ok(expected));
    }

    #[rstest]
    #[case(tokenize("-9"), -9)]
    #[case(tokenize("- 123"), -123)]
    fn it_parses_negavite_numbers(
        #[case] input: Vec<Token>,
        #[case] negative_num: i32
    ) {
        let inner_term = Term::Literal(Literal::Int(negative_num * -1));
        let term = Term::negative_int(inner_term);
        let expected = term_tree(term);
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_parses_negated_boolean() {
        let term = Term::negated_bool(Term::Id("someVariable".into()));
        let expected = term_tree(term);
        assert_eq!(parse(tokenize("!someVariable")), Ok(expected));
    }

    #[test]
    fn it_parses_many_negations() {
        let term = Term::negated_bool(
            Term::negated_bool(
                Term::negated_bool(Term::Id("x".into()))
            )
        );
        let expected = term_tree(term);
        assert_eq!(parse(tokenize("!!!x")), Ok(expected));
    }

    #[test]
    fn it_returns_error_for_many_negative_symbols() {
        let error = errors::unexpected_token(
            "identifier or literal",
            Token::BinaryOp(BinaryOp::Minus)
        );
        assert_eq!(parse(tokenize("---9")), Err(error));
    }

    #[test]
    fn it_returns_error_for_multiple_ints() {
        let input = tokenize("3 2");
        let error = errors::unexpected_token("end of line", Token::Literal(Literal::Int(2)));
        assert_eq!(parse(input), Err(error));
    }

    #[test]
    fn it_parses_string_plus_string() {
        let input = tokenize("\"a\" + \"b\"");

        let left = Term::Literal(Literal::String("a".into()));
        let operator = BinaryOp::Plus;
        let right = Term::Literal(Literal::String("b".into()));
        let expected = AbstractSyntaxTree::Expr(
            ExprNode { first: left, rest: vec![(operator, right)] }
        );
        assert_eq!(parse(input), Ok(expected)); 
    }

    #[test]
    fn it_parses_var_binding_to_literal() {
        let input = tokenize("let x = 4");
        let let_node = LetNode {
            id: "x".to_string(),
            datatype: None,
            value: Box::new(term_tree(Term::Literal(Literal::Int(4)))),
        };
        let expected = AbstractSyntaxTree::Let(let_node);
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_returns_error_for_bad_var_id() {
        let input = tokenize("let 3 = 3");
        let error = errors::unexpected_token(
            "identifier",
            Token::Literal(Literal::Int(3))
        );
        assert_eq!(parse(input), Err(error));
    }

    #[test]
    fn it_returns_error_for_equals_in_let_expr() {
        let input = tokenize("let x = 1 = 0");
        let error = errors::unexpected_token(
            "end of line",
            Token::UnaryOp(UnaryOp::Equals)
        );
        assert_eq!(parse(input), Err(error)); 
    }

    #[test]
    fn it_parses_var_binding_to_expr() {
        let input = tokenize("let two = 6 / 3");
        let let_node = LetNode {
            id: "two".to_string(),
            datatype: None,
            value: Box::new(parse(tokenize("6 / 3")).unwrap()),
        };
        let expected = AbstractSyntaxTree::Let(let_node);
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_parses_var_binding_with_declared_type() {
        let input = tokenize("let two_strings: string = \"a\" + \"b\"");
        let let_node = LetNode {
            id: "two_strings".to_string(),
            datatype: Some(Type::String),
            value: Box::new(parse(vec![
                string_token("a"),
                op_token(BinaryOp::Plus),
                string_token("b")
            ]).unwrap()),
        };
        let expected = AbstractSyntaxTree::Let(let_node);
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_returns_error_for_invalid_let_type() {
        let input = tokenize("let x: y = z");
        let error = errors::not_a_type(Token::Id("y".to_string()));
        assert_eq!(parse(input), Err(error)); 
    }

    #[test]
    fn it_returns_error_for_unexpected_let() {
        let input = tokenize("let x = let y = 2");
        let error = errors::unexpected_token("identifier or literal", Token::Let);
        assert_eq!(parse(input), Err(error));
    }

    #[rstest]
    #[case(tokenize("let"), errors::unexpected_end_of_input())]
    #[case(tokenize("let x"),errors::unexpected_end_of_input())]
    #[case(tokenize("3 +"), errors::unexpected_end_of_input())]
    fn it_returns_error_for_incomplete_statements(
        #[case] tokens: Vec<Token>,
        #[case] error: IntepreterError
    ) {
        assert_eq!(parse(tokens), Err(error));
    }
}
