use crate::models::{
    AbstractSyntaxTree,
    LetNode,
    Literal,
    Operator,
    OperatorNode,
    Token,
};

use crate::errors::{self, token_not_allowed};

/// Convert a sequence of tokens into an abstract syntax tree.
/// 
/// The tokens should represent exactly one statement and therefore one syntax
/// tree. Otherwise, an error is returned.
/// A single literal value is a valid statement (e.g. "3").
/// 
/// A syntax error is returned for any syntactical errors in the token sequence
pub fn parse(tokens: Vec<Token>) -> Result<AbstractSyntaxTree, String> {
    return parse_remaining(&tokens, 0);
}

/// Parse tokens starting at position `position`. Some keywords (like "let")
/// are only allowed in the first position
fn parse_remaining(
    tokens: &Vec<Token>,
    position: usize
) -> Result<AbstractSyntaxTree, String> {
    // first token determines the type of the root node
    let head = get_token(tokens, position)?;

    let head_as_leaf = token_to_leaf(&head);
    if tokens.len() == position + 1 {
        return match head_as_leaf {
            Some(leaf) => Ok(leaf),
            None => Err(errors::unexpected_end_of_input()) 
        };
    }

    match head {
        Token::Literal(_) =>
            build_binary_node(head_as_leaf.unwrap(), &tokens, position + 1),
        Token::Id(_) =>
            build_binary_node(head_as_leaf.unwrap(), &tokens, position + 1),
        Token::Let => build_let_node(&tokens, position + 1),
        // add more keyword cases...

        Token::Operator(o) =>
            parse_leading_operator(&o, &tokens, position + 1),
        _ => return Err(token_not_allowed(head)),
    }
}

/// Try to create an AST with a leading operator.
/// This only works for tokens representing negative numbers (e.g. -1),
/// will fail in all other cases.
/// 
/// first token after operator indicated by `position`
fn parse_leading_operator(
    operator: &Operator,
    tokens: &Vec<Token>,
    position: usize
) -> Result<AbstractSyntaxTree, String> {
    match operator {
        Operator::Minus => {},
        _ => return Err(token_not_allowed(Token::Operator(*operator)))
    };

    // TODO: support variable length expressions /w parenthesis
    let value_token = &tokens[position];
    let value = match value_token {
        Token::Literal(Literal::Int(i)) => i,
        _ => return Err(errors::unexpected_token("int", value_token.clone())),
    };
    let node = AbstractSyntaxTree::Literal(Literal::Int(value * -1));
    return Ok(node);
}

/// Create a AST for a binary operator given the left argument and
/// the remaining tokens (indicated by `position`).
/// 
/// tokens[position] should be the first token after `left_arg`
fn build_binary_node(
    left_arg: AbstractSyntaxTree,
    tokens: &Vec<Token>,
    position: usize
) -> Result<AbstractSyntaxTree, String> {
    let operator_token = get_token(tokens, position)?;
    let operator = match operator_token {
        Token::Operator(Operator::OneEq) =>
            return Err(errors::token_not_allowed(operator_token)),
        Token::Operator(o) => o,
        _ => return Err(
            errors::unexpected_token("binary operator", operator_token)
        ),
    };

    // TODO: handle infinite args, not just one
    let right_token = get_token(tokens, position + 1)?;
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
/// (first token indicated by `position`)
fn build_let_node(
    tokens: &Vec<Token>,
    position: usize
) -> Result<AbstractSyntaxTree, String> {
    if position != 1 {
        return Err(errors::token_not_allowed(Token::Let));
    }
    let mut idx = position;

    let id_token = get_token(tokens, idx)?;
    let id = match id_token {
        Token::Id(s) => s.to_string(),
        _ => return Err(errors::unexpected_token("identifier", id_token))
    };

    idx += 1;
    // look for optional declared type (e.g. ": int")
    let is_token_colon = match get_token(tokens, idx)? {
        Token::Formatter(ref f) if f == ":" => true,
        _ => false,
    };
    let datatype = if is_token_colon {
        idx += 1;
        let type_token = get_token(tokens, idx)?;
        idx += 1;
        match type_token {
            Token::Type(t) => Some(t.to_owned()),
            _ => return Err(errors::not_a_type(type_token)) 
        }
    } else {
        None
    };

    let equals_token = get_token(tokens, idx)?;
    match equals_token {
        Token::Operator(Operator::OneEq) => {},
        _ => return Err(errors::unexpected_token("'='", equals_token))
    };

    idx += 1;
    let value_node = parse_remaining(tokens, idx)?;

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

/// Safe way to retreive a token at a specific location.
/// Makes it easy to propagate errors by adding the `?` operator at the end of
/// invocations.
fn get_token(tokens: &Vec<Token>, position: usize) -> Result<Token, String> {
    match tokens.get(position) {
        Some(token) => Ok(token.clone()),
        None => Err(errors::unexpected_end_of_input()),
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
    #[case(op_token(Operator::Plus))]
    #[case(op_token(Operator::Minus))]
    #[case(op_token(Operator::OneEq))]
    fn it_returns_error_for_one_operator(#[case] op: Token) {
        assert_eq!(parse(vec![op]), Err(errors::unexpected_end_of_input()));
    }

    #[rstest]
    #[case(tokenize("3 + 2"), Operator::Plus, 3, 2)]
    #[case(tokenize("1 % 4"), Operator::Percent, 1, 4)]
    #[case(tokenize("1 - 8"), Operator::Minus, 1, 8)]
    #[case(tokenize("0 == 1"), Operator::TwoEq, 0, 1)]
    #[case(tokenize("2 != 3"), Operator::NotEq, 2, 3)]
    fn it_parses_binary_expressions(
        #[case] input: Vec<Token>,
        #[case] operator: Operator,
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

        let operator = Operator::Plus;
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
        let error = errors::token_not_allowed(Token::Operator(Operator::OneEq));
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
                op_token(Operator::Plus),
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
