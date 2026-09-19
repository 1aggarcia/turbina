use crate::{errors::Result, models::{Token, TypeAlias, UnaryOp}, parser::{shared_parsers::parse_id, type_declaration_parser::parse_type, utils::match_next}, streams::TokenStream};

/// Create an AST for the "type" keyword given the remaining tokens
/// ```text
/// <type_alias> ::= TypeKeyword Id Equals <type>
/// ```
pub fn parse_type_alias(tokens: &mut TokenStream) -> Result<TypeAlias> {
    match_next(tokens, Token::TypeKeyword)?;
    let type_alias = parse_id(tokens)?;
    match_next(tokens, Token::UnaryOp(UnaryOp::Equals))?;
    let datatype = parse_type(tokens)?;

    Ok(TypeAlias { type_alias, datatype })
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;
    use crate::{errors::error, models::{AbstractSyntaxTree, Type, test_utils::int_token}, parser::test_utils::{force_tokenize, parse_tokens}};


    fn test_parse_type_alias(tokens: Vec<Token>) -> Result<TypeAlias> {
        let ast = parse_tokens(tokens)?;
        match ast {
            AbstractSyntaxTree::TypeAlias(type_alias) => Ok(type_alias),
            other => panic!("Not a binding: {:?}", other),
        }
    }

    #[rstest]
    #[case::simple_type("int", Type::Int)]
    #[case::list_type("string[]", Type::String.as_list())]
    #[case::function_type("int -> bool", Type::func(&[Type::Int], Type::Bool))]
    #[case::type_in_parenthesis(
        "((string, int[]) -> null)",
        Type::func(&[Type::String, Type::Int.as_list()], Type::Null)
    )]
    fn it_returns_correct_ast_node_for_valid_type_alias(
        #[case] input_type: &str, #[case] expected_type: Type
    ) {
        let input = force_tokenize(&format!("type X = {};", input_type));
        let expected =
            TypeAlias { type_alias: "X".into(), datatype: expected_type };
        let actual = test_parse_type_alias(input);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn it_returns_correct_error_for_invalid_type() {
        let input = force_tokenize("type X = 5;");
        let expected = error::not_a_type(int_token(5));
        let actual = test_parse_type_alias(input);

        assert_eq!(actual, Err(expected));
    }
}
