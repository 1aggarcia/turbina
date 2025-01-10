use crate::models::{
    AbstractSyntaxTreeV2,
    LetNode,
    Literal,
    Operator,
    OperatorNode,
    TokenV2,
};

/// Convert a sequence of tokens into an abstract syntax tree.
/// 
/// The tokens should represent exactly one statement and therefore one syntax
/// tree. Otherwise, an error is returned.
/// A single literal value is a valid statement (e.g. "3").
/// 
/// A syntax error is returned for any syntactical errors in the token sequence
pub fn parse(tokens: Vec<TokenV2>) -> Result<AbstractSyntaxTreeV2, String> {
    if tokens.is_empty() {
        return Err("Cannot parse empty tokens".to_string());
    }
    // first token determines the type of the root node
    let head = &tokens[0];
    let head_as_leaf = token_to_leaf(head);
    if tokens.len() == 1 {
        return match head_as_leaf {
            Some(leaf) => Ok(leaf),
            None => Err(format!("cannot parse single token: {:?}", head))
        };
    }

    match head {
        TokenV2::Literal(_) =>
            build_binary_node(head_as_leaf.unwrap(), &tokens, 1),
        TokenV2::Id(_) =>
            build_binary_node(head_as_leaf.unwrap(), &tokens, 1),
        TokenV2::Let => build_let_node(&tokens, 1),
        // add more keyword cases...

        TokenV2::Operator(o) =>
            parse_leading_operator(o, &tokens, 1),
        TokenV2::Formatter(f) =>
            return Err(format!("cannot parse {} as first token", f)),
    }
}

/// Try to create an AST with a leading operator.
/// This only works for tokens representing negative numbers (e.g. -1),
/// will fail in all other cases.
/// 
/// first token after operator indicated by `position`
fn parse_leading_operator(
    operator: &Operator,
    tokens: &Vec<TokenV2>,
    position: usize
) -> Result<AbstractSyntaxTreeV2, String> {
    match operator {
        Operator::Minus => {},
        _ => return Err("Cannot parse operator as first token".to_string())
    };

    // TODO: support variable length expressions
    let value_token = &tokens[position];
    let value = match value_token {
        TokenV2::Literal(Literal::Int(i)) => i,
        _ => return Err(format!("Invalid token following '-': {:?}", value_token)),
    };
    let node = AbstractSyntaxTreeV2::Literal(Literal::Int(value * -1));
    return Ok(node);
}

/// Create a AST for a binary operator given the left argument and
/// the remaining tokens (indicated by `position`).
/// 
/// tokens[position] should be the first token after `left_arg`
fn build_binary_node(
    left_arg: AbstractSyntaxTreeV2,
    tokens: &Vec<TokenV2>,
    position: usize
) -> Result<AbstractSyntaxTreeV2, String> {
    let operator_token = &tokens[position];
    let operator = match operator_token {
        TokenV2::Operator(o) => o,
        _ => return Err(format!("Invalid operator: {:?}", operator_token)),
    };

    // TODO: handle infinite args, not just one
    // TODO: range check
    let right_token = &tokens[position + 1];
    let right_arg = match token_to_leaf(right_token) {
        Some(t) => t,
        None => return Err(format!("invalid right token: {:?}", right_token))
    };

    let node = OperatorNode {
        operator: *operator,
        left: Box::new(left_arg),
        right: Box::new(right_arg),
    };
    return Ok(AbstractSyntaxTreeV2::Operator(node));
}

/// Create an AST for the "let" keyword given the remaining tokens
/// (first token indicated by `position`)
fn build_let_node(
    tokens: &Vec<TokenV2>,
    position: usize
) -> Result<AbstractSyntaxTreeV2, String> {
    let mut idx = position;

    let id_token = &tokens[idx];
    let id = match id_token {
        TokenV2::Id(s) => s.to_string(),
        _ => return Err(format!("invalid variable name: {:?}", id_token))
    };
    idx += 1;

    // TODO: support declared types

    let equals_token = &tokens[idx];
    match equals_token {
        TokenV2::Operator(Operator::Equals) => {},
        _ => return Err(format!("expected '=', got token {:?}", equals_token))
    };
    idx += 1;

    // TODO: add support for infinitly long expressions
    let value_token = &tokens[idx];
    let value = match token_to_leaf(value_token) {
        Some(v) => v,
        None => return Err(format!("expected leaf, got token {:?}", value_token))
    };

    let node = LetNode { id, value: Box::new(value) };
    return Ok(AbstractSyntaxTreeV2::Let(node));
}

fn token_to_leaf(token: &TokenV2) -> Option<AbstractSyntaxTreeV2> {
    match token {
        TokenV2::Id(i) => Some(AbstractSyntaxTreeV2::Id(i.clone())),
        TokenV2::Literal(l) => Some(AbstractSyntaxTreeV2::Literal(l.clone())),
        _ => None,
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
        assert!(matches!(
            parse(vec![]),
            Err { .. }
        ));
    }

    #[rstest]
    #[case(int_token(2), AbstractSyntaxTreeV2::Literal(Literal::Int(2)))]
    #[case(string_token("prueba test"), AbstractSyntaxTreeV2::Literal(Literal::String("prueba test".to_string())))]
    #[case(id_token("name"), AbstractSyntaxTreeV2::Id("name".to_string()))]
    fn it_parses_one_token_to_one_node(
        #[case] token: TokenV2,
        #[case] node: AbstractSyntaxTreeV2
    ) {
        let input = vec![token.clone()];
        assert_eq!(parse(input), Ok(node));
    }

    #[rstest]
    #[case(op_token(Operator::Plus))]
    #[case(op_token(Operator::Minus))]
    #[case(op_token(Operator::Equals))]
    fn it_returns_error_for_one_operator(#[case] op: TokenV2) {
        assert!(matches!(parse(vec![op]), Err { .. }));
    }

    #[rstest]
    #[case(tokenize("3 + 2"), Operator::Plus, 3, 2)]
    #[case(tokenize("1 % 4"), Operator::Percent, 1, 4)]
    #[case(tokenize("1 - 8"), Operator::Minus, 1, 8)]
    fn it_parses_binary_expressions(
        #[case] input: Vec<TokenV2>,
        #[case] operator: Operator,
        #[case] left_val: i32,
        #[case] right_val: i32,
    ) {
        let left = Box::new(
            AbstractSyntaxTreeV2::Literal(Literal::Int(left_val))
        );
        let right = Box::new(
            AbstractSyntaxTreeV2::Literal(Literal::Int(right_val))
        );
        let expected = AbstractSyntaxTreeV2::Operator(
            OperatorNode { operator, left, right }
        );
        assert_eq!(parse(input), Ok(expected));
    }

    #[rstest]
    #[case(tokenize("-9"), -9)]
    #[case(tokenize("- 123"), -123)]
    fn it_parses_negavite_numbers(
        #[case] input: Vec<TokenV2>,
        #[case] expected_val: i32
    ) {
        let expected =
            AbstractSyntaxTreeV2::Literal(Literal::Int(expected_val));
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_returns_error_for_multiple_ints() {
        let input = tokenize("3 2");
        assert!(matches!(parse(input), Err { .. }));
    }

    #[test]
    fn it_returns_error_for_int_plus_string() {
        let input = tokenize("1 + \"string\"");
        assert!(matches!(parse(input), Err { .. }));
    }

    #[test]
    fn it_parses_string_plus_string() {
        let input = tokenize("\"a\" + \"b\"");

        let operator = Operator::Plus;
        let left = Box::new(
            AbstractSyntaxTreeV2::Literal(Literal::String("a".to_string()))
        );
        let right = Box::new(
            AbstractSyntaxTreeV2::Literal(Literal::String("b".to_string()))
        );
        let expected = AbstractSyntaxTreeV2::Operator(
            OperatorNode { operator, left, right }
        );
        assert_eq!(parse(input), Ok(expected)); 
    }

    #[test]
    fn it_parses_var_binding_to_literal() {
        let input = tokenize("let x = 4;");
        let let_node = LetNode {
            id: "x".to_string(),
            value: Box::new(AbstractSyntaxTreeV2::Literal(Literal::Int(4))),
        };
        let expected = AbstractSyntaxTreeV2::Let(let_node);
        assert_eq!(parse(input), Ok(expected));
    }

    #[test]
    fn it_returns_error_for_bad_var_id() {
        let input = tokenize("let 3 = 3");
        assert!(matches!(parse(input), Err { .. }));
    }

    #[test]
    fn it_parses_var_binding_to_expr() {
        let input = tokenize("let two = 6 / 3;");
        let let_node = LetNode {
            id: "two".to_string(),
            value: Box::new(parse(tokenize("6 / 3")).unwrap()),
        };
        let expected = AbstractSyntaxTreeV2::Let(let_node);
        assert_eq!(parse(input), Ok(expected));
    }
}
