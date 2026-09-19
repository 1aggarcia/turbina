use crate::errors::{InterpreterError, MultiResult};
use crate::models::{Type, TypeAlias};
use crate::type_resolver::TreeType;
use crate::type_resolver::shared::TypeContext;

/// Check that the type alias name has not already been declared.
pub fn resolve_type_alias_type(
    context: &TypeContext, node: &TypeAlias
) -> MultiResult<TreeType> {
    if let Some(_) = context.lookup_type_alias(&node.type_alias) {
        return Err(vec![InterpreterError::ReassignTypeError {
            type_alias: node.type_alias.clone()
        }]);
    }

    let result = TreeType {
        datatype: Type::Null,  // type alias is not usable runtime data
        name_to_bind: None,
        type_alias_to_bind: Some((node.type_alias.clone(), node.datatype.clone()))
    };
    Ok(result)
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;
    use crate::models::{Program, Type};
    use crate::parser::test_utils::make_tree;
    use crate::type_resolver::resolve_type;
    use crate::type_resolver::shared::test_utils::*;

    #[rstest]
    #[case::simple_type("int", Type::Int)]
    #[case::list_type("string[]", Type::String.as_list())]
    fn it_returns_correct_tree_for_new_type_alias(
        #[case] input_type: &str, #[case] expected_type: Type
    ) {
        let input = make_tree(&format!("type X = {};", input_type));
        let expected = TreeType {
            datatype: Type::Null,
            name_to_bind: None,
            type_alias_to_bind: Some(("X".into(), expected_type)),
        };
        let actual = resolve_type_fresh(input);
        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn it_returns_error_for_type_alias_already_taken() {
        let mut program = Program::init_with_std_streams();
        program.type_aliases.insert("X".into(), Type::Byte);
        let input = make_tree("type X = string;");

        let expected =
            InterpreterError::ReassignTypeError { type_alias: "X".into() };

        let actual = resolve_type(&program, &input);

        assert_eq!(actual, Err(vec![expected]));
    }
}