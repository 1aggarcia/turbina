use crate::{
    errors::{InterpreterError, Result},
    models::{StructDefinition, Token, Type},
    parser::{
        shared_parsers::{parse_id, parse_list, ListParserConfig},
        type_declaration_parser::parse_type_declaration,
        utils::match_next,
    },
    streams::TokenStream,
};

/// ```text
/// <struct_definition> ::=
///     Struct "{" <struct_field> {"," <struct_field>} [","] "}"
/// ```
pub fn parse_struct_definition(
    tokens: &mut TokenStream,
) -> Result<StructDefinition> {
    match_next(tokens, Token::Struct)?;
    let struct_name = parse_id(tokens)?;

    let field_list = parse_list(
        tokens,
        ListParserConfig {
            item_parser: parse_struct_field,
            opening_token: Some(Token::OpenCurlyBracket),
            closing_token: Token::CloseCurlyBracket,
        },
    )?;

    if field_list.is_empty() {
        return Err(InterpreterError::EmptyStruct { name: struct_name });
    }

    Ok(StructDefinition {
        name: struct_name,
        fields: field_list.into_iter().collect(),
    })
}

///```text
/// <struct_field> ::= Id <type_declaration>
/// ```
fn parse_struct_field(tokens: &mut TokenStream) -> Result<(String, Type)> {
    let field_name = parse_id(tokens)?;
    let field_type = parse_type_declaration(tokens)?;

    Ok((field_name, field_type))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::models::AbstractSyntaxTree;
    use crate::parser::test_utils::*;
    use rstest::rstest;

    mod struct_definition {
        use std::collections::HashMap;

        use crate::errors::{error, InterpreterError};

        use super::*;

        fn test_parse_struct_definition(
            tokens: Vec<Token>,
        ) -> Result<StructDefinition> {
            let ast = parse_tokens(tokens)?;
            match ast {
                AbstractSyntaxTree::StructDefinition(struct_definition) => {
                    Ok(struct_definition)
                }
                other => panic!("Not a struct declaration: {:?}", other),
            }
        }

        #[rstest]
        #[case::one_line("struct User { username: string, isAdmin: bool };")]
        #[case::multi_line(
            "
            struct User {
                username: string,
                isAdmin: bool 
            };"
        )]
        #[case::multi_line_trailing_comma(
            "
            struct User {
                username: string,
                isAdmin: bool,
            };"
        )]
        fn it_parses_valid_struct_correctly(#[case] input: &str) {
            let tokens = force_tokenize(input);
            let expected = StructDefinition {
                name: "User".into(),
                fields: HashMap::from([
                    ("username".into(), Type::String),
                    ("isAdmin".into(), Type::Bool),
                ]),
            };
            assert_eq!(test_parse_struct_definition(tokens), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_struct_with_no_fields() {
            let tokens = force_tokenize("struct Empty {};");
            let expected = InterpreterError::EmptyStruct {
                name: "Empty".into(),
            };
            assert_eq!(test_parse_struct_definition(tokens), Err(expected));
        }

        #[test]
        fn it_returns_error_for_newline_after_struct_keyword() {
            let tokens = force_tokenize("struct\nNew { n: int };");
            let expected =
                error::unexpected_token("identifier", Token::Newline);

            assert_eq!(test_parse_struct_definition(tokens), Err(expected));
        }
    }
}
