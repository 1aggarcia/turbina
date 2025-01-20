use crate::models::{
    AbstractSyntaxTree, BinaryOp, LetNode, Literal, OperatorNode, Token, UnaryOp
};

use crate::errors::{self, token_not_allowed};

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
    fn pop(&mut self) -> Result<Token, String> {
        let token = self.peek()?;
        self.position += 1;
        return Ok(token);
    }

    /// Returns the token currently being pointed to
    fn peek(&self) -> Result<Token, String> {
        match self.tokens.get(self.position) {
            Some(token) => Ok(token.clone()),
            None => Err(errors::unexpected_end_of_input()),
        }
    }
}

/// Convert a sequence of tokens into an abstract syntax tree.
/// 
/// The tokens should represent exactly one statement and therefore one syntax
/// tree. Otherwise, an error is returned.
/// A single literal value is a valid statement (e.g. "3").
/// 
/// A syntax error is returned for any syntactical errors in the token sequence
pub fn parse(tokens: Vec<Token>) -> Result<AbstractSyntaxTree, String> {
    let mut stream = TokenStream::new(tokens);
    return parse_stream(&mut stream);
}

/// Parent function to parse a token stream
fn parse_stream(tokens: &mut TokenStream) -> Result<AbstractSyntaxTree, String> {
    // first token determines the type of the root node
    let head = tokens.pop()?;

    let head_as_leaf = token_to_leaf(&head);
    if !tokens.has_next() {
        return head_as_leaf.ok_or(errors::unexpected_end_of_input());
    }

    match head {
        Token::Literal(_) =>
            build_binary_node(head_as_leaf.unwrap(), tokens),
        Token::Id(_) =>
            build_binary_node(head_as_leaf.unwrap(), tokens),
        Token::Let => build_let_node(tokens),
        // add more keyword cases...

        Token::BinaryOp(o) => parse_leading_operator(&o, tokens),
        _ => return Err(token_not_allowed(head)),
    }
}

/// Try to create an AST with a leading operator.
/// This only works for tokens representing negative numbers (e.g. -1),
/// will fail in all other cases.
fn parse_leading_operator(
    operator: &BinaryOp,
    tokens: &mut TokenStream
) -> Result<AbstractSyntaxTree, String> {
    match operator {
        BinaryOp::Minus => {},
        _ => return Err(token_not_allowed(Token::BinaryOp(*operator)))
    };

    // TODO: support variable length expressions /w parenthesis
    let value_token = &tokens.pop()?;
    let value = match value_token {
        Token::Literal(Literal::Int(i)) => i,
        _ => return Err(errors::unexpected_token("int", value_token.clone())),
    };
    let node = AbstractSyntaxTree::Literal(Literal::Int(value * -1));
    return Ok(node);
}

/// Create a AST for a binary operator given the left argument and
/// the remaining tokens
fn build_binary_node(
    left_arg: AbstractSyntaxTree,
    tokens: &mut TokenStream
) -> Result<AbstractSyntaxTree, String> {
    let operator_token = tokens.pop()?;
    let operator = match operator_token {
        Token::BinaryOp(o) => o,
        _ => return Err(
            errors::unexpected_token("binary operator", operator_token)
        ),
    };

    // TODO: handle infinite args, not just one
    let right_token = tokens.pop()?;
    let right_arg = match token_to_leaf(&right_token) {
        Some(t) => t,
        None => return Err(errors::token_not_allowed(right_token))
    };

    let node = OperatorNode {
        operator,
        left: Box::new(left_arg),
        right: Box::new(right_arg),
    };
    return Ok(AbstractSyntaxTree::Operator(node));
}

/// Create an AST for the "let" keyword given the remaining tokens
fn build_let_node(tokens: &mut TokenStream) -> Result<AbstractSyntaxTree, String> {
    // TODO: eliminate this explicit `position` check by
    // parsing statements and expressions seperatly
    if tokens.position != 1 {
        return Err(errors::token_not_allowed(Token::Let));
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

    let value_node = parse_stream(tokens)?;

    let node = LetNode { id, datatype, value: Box::new(value_node) };
    return Ok(AbstractSyntaxTree::Let(node));
}

fn token_to_leaf(token: &Token) -> Option<AbstractSyntaxTree> {
    match token {
        Token::Id(i) => Some(AbstractSyntaxTree::Id(i.clone())),
        Token::Literal(l) => Some(AbstractSyntaxTree::Literal(l.clone())),
        _ => None,
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
    #[case(int_token(2), AbstractSyntaxTree::Literal(Literal::Int(2)))]
    #[case(string_token("prueba test"), AbstractSyntaxTree::Literal(Literal::String("prueba test".to_string())))]
    #[case(id_token("name"), AbstractSyntaxTree::Id("name".to_string()))]
    fn it_parses_one_token_to_one_node(
        #[case] token: Token,
        #[case] node: AbstractSyntaxTree
    ) {
        let input = vec![token.clone()];
        assert_eq!(parse(input), Ok(node));
    }

    #[rstest]
    #[case(op_token(BinaryOp::Plus))]
    #[case(op_token(BinaryOp::Minus))]
    #[case(unary_op_token(UnaryOp::Equals))]
    fn it_returns_error_for_one_operator(#[case] op: Token) {
        assert_eq!(parse(vec![op]), Err(errors::unexpected_end_of_input()));
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
        let left = Box::new(
            AbstractSyntaxTree::Literal(Literal::Int(left_val))
        );
        let right = Box::new(
            AbstractSyntaxTree::Literal(Literal::Int(right_val))
        );
        let expected = AbstractSyntaxTree::Operator(
            OperatorNode { operator, left, right }
        );
        assert_eq!(parse(input), Ok(expected));
    }

    #[rstest]
    #[case(tokenize("-9"), -9)]
    #[case(tokenize("- 123"), -123)]
    fn it_parses_negavite_numbers(
        #[case] input: Vec<Token>,
        #[case] expected_val: i32
    ) {
        let expected =
            AbstractSyntaxTree::Literal(Literal::Int(expected_val));
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_returns_error_for_multiple_ints() {
        let input = tokenize("3 2");
        let error = errors::unexpected_token(
            "binary operator",
            Token::Literal(Literal::Int(2))
        );
        assert_eq!(parse(input), Err(error));
    }

    #[test]
    fn it_parses_string_plus_string() {
        let input = tokenize("\"a\" + \"b\"");

        let operator = BinaryOp::Plus;
        let left = Box::new(
            AbstractSyntaxTree::Literal(Literal::String("a".to_string()))
        );
        let right = Box::new(
            AbstractSyntaxTree::Literal(Literal::String("b".to_string()))
        );
        let expected = AbstractSyntaxTree::Operator(
            OperatorNode { operator, left, right }
        );
        assert_eq!(parse(input), Ok(expected)); 
    }

    #[test]
    fn it_parses_var_binding_to_literal() {
        let input = tokenize("let x = 4");
        let let_node = LetNode {
            id: "x".to_string(),
            datatype: None,
            value: Box::new(AbstractSyntaxTree::Literal(Literal::Int(4))),
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
            "binary operator",
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
        let error = errors::token_not_allowed(Token::Let);
        assert_eq!(parse(input), Err(error));
    }

    #[rstest]
    #[case(tokenize("let"))]
    #[case(tokenize("let x"))]
    #[case(tokenize("3 +"))]
    fn it_returns_error_for_incomplete_statements(#[case] tokens: Vec<Token>) {
        assert_eq!(parse(tokens), Err(errors::unexpected_end_of_input())); 
    }
}
