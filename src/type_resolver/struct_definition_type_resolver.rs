use crate::{
    errors::{InterpreterError, MultiResult},
    models::{StructDefinition, Type},
    type_resolver::{shared::TypeContext, TreeType},
};

pub fn resolve_struct_definition_type(
    context: &TypeContext,
    struct_definition: &StructDefinition,
) -> MultiResult<TreeType> {
    let struct_name = &struct_definition.name;
    if let Some(_) = context.lookup_struct(struct_name) {
        return Err(vec![InterpreterError::ReassignStructError {
            struct_name: struct_name.to_owned(),
        }]);
    }

    let result = TreeType {
        // struct declaration is not usable runtime data
        datatype: Type::Null,
        name_to_bind: None,
        type_alias_to_bind: None,
        // Must be cloned since the struct definition lives for the entire
        // lifetime of the program, i.e. longer than the AST that created it
        struct_to_bind: Some(struct_definition.clone()),
    };
    Ok(result)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::parser::test_utils::make_tree;
    use crate::type_resolver::resolve_type;
    use crate::type_resolver::shared::test_utils::*;

    mod struct_definition {
        use super::*;
        use crate::{errors::InterpreterError, models::Program};
        use std::collections::HashMap;

        #[test]
        fn it_should_return_tree_type_with_struct_definition() {
            let input = "struct User { username: string, isAdmin: bool };";
            let tree = make_tree(input);

            let result = resolve_type_fresh(tree);
            let expected_declaration = StructDefinition {
                name: "User".into(),
                fields: HashMap::from([
                    ("username".into(), Type::String),
                    ("isAdmin".into(), Type::Bool),
                ]),
            };
            let expected = TreeType {
                datatype: Type::Null,
                name_to_bind: None,
                type_alias_to_bind: None,
                struct_to_bind: Some(expected_declaration),
            };
            assert_eq!(result, Ok(expected));
        }

        #[test]
        fn it_should_return_error_for_struct_already_declared() {
            let mut program = Program::init_with_std_streams();
            program.struct_definitions.insert(
                "User".into(),
                HashMap::from([
                    ("id".into(), Type::String),
                    ("email".into(), Type::String),
                ]),
            );

            let input = "struct User { username: string, isAdmin: bool };";
            let tree = make_tree(input);

            let expected = InterpreterError::ReassignStructError {
                struct_name: "User".into(),
            };
            let actual = resolve_type(&program, &tree);

            assert_eq!(actual, Err(vec![expected]));
        }
    }
}
