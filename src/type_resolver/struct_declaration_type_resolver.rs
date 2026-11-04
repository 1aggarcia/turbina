use crate::{
    errors::MultiResult,
    models::{StructDeclaration, Type},
    type_resolver::{shared::TypeContext, TreeType},
};

pub fn resolve_struct_declaration_type(
    _context: &TypeContext,
    struct_declaration: &StructDeclaration,
) -> MultiResult<TreeType> {
    let result = TreeType {
        // struct declaration is not usable runtime data
        datatype: Type::Null,
        name_to_bind: None,
        type_alias_to_bind: None,
        // Must be cloned since the struct definition lives for the entire
        // lifetime of the program, i.e. longer than the AST that created it
        struct_to_bind: Some(struct_declaration.clone()),
    };
    Ok(result)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::parser::test_utils::make_tree;
    use crate::type_resolver::resolve_type;
    use crate::type_resolver::shared::test_utils::*;

    mod struct_declaration {
        use super::*;
        use crate::{errors::InterpreterError, models::Program};
        use std::collections::HashMap;

        #[test]
        fn it_should_return_tree_type_with_struct_declaration() {
            let input = "struct User { username: string, isAdmin: bool };";
            let tree = make_tree(input);

            let result = resolve_type_fresh(tree);
            let expected_declaration = StructDeclaration {
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
        #[ignore = "Not implemented"]
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
