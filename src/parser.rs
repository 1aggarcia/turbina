use crate::models::{
    AbstractSyntaxTreeV2,
    TokenV2,
    OperatorNode,
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
        // TokenV2::Let => build_let(tokens[1..]),
        // more keyword cases
        TokenV2::Operator(_) => Err("Cannot parse operator as first token".to_string()),
        _ => Err(format!("unsupported token: {:?}", head))
    }
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
    fn it_parses_var_binding() {
        let input = tokenize("let x = 4;");
        let let_node = LetNode {
            id: "x".to_string(),
            value: Box::new(AbstractSyntaxTreeV2::Literal(Literal::Int(4))),
        };
        let expected = AbstractSyntaxTreeV2::Let(let_node);
        assert_eq!(parse(input), Ok(expected));
    }
}