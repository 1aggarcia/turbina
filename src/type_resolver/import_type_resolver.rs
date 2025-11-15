use std::{fs::File, io::BufReader, path::PathBuf};

static TURBINA_FILE_EXTENSION: &str = "tb";

use crate::errors::{InterpreterError, MultiResult, error};
use crate::models::{Import, Program, Type};
use crate::parser::parse_statement;
use crate::streams::{FileStream, TokenStream};
use crate::type_resolver::{TreeType, resolve_type};
use crate::type_resolver::shared::TypeContext;

pub fn resolve_import_type(
    _: &TypeContext, import: &Import
) -> MultiResult<TreeType> {
    // TODO: protect against circular imports with backtracking
    let Some(module_name) = import.path_elements.last() else {
        return Err(InterpreterError::IOError {
            message: "Attempted to import empty filepath".into()
        }.into());
    };

    let mut path = PathBuf::from_iter(import.path_elements.clone());
    path.set_extension(TURBINA_FILE_EXTENSION);

    match File::open(path.clone()) {
        Ok(file) => {
            // TODO: refactor duplicated logic from main
            let reader = BufReader::new(file);
            let file_stream = Box::new(FileStream { reader });

            // TODO: refactor duplicated logic from run_as_file
            let mut token_stream = TokenStream::new(file_stream);
            let mut program = Program::init_with_std_streams();
            program.type_context.clear(); // remove library functions

            let mut errors = vec![];
            loop {
                match validate_next_statement(&mut program, &mut token_stream) {
                    Ok(_) => {},
                    Err(statement_errors) => {
                        if statement_errors.contains(&InterpreterError::EndOfFile) {
                            break;
                        }
                        errors.extend(statement_errors);
                    }
                };
            }

            if !errors.is_empty() {
                return Err(
                    errors.into_iter().map(|err|
                        error::module_error(&module_name, err)
                    )
                    .collect()
                )
            }

            let tree_type = TreeType {
                datatype: Type::Struct(program.type_context),
                name_to_bind: Some(module_name.to_owned()),
            };

            Ok(tree_type)
        },
        Err(_) => Err(vec![
            InterpreterError::ImportError {
                filepath: path.to_str().unwrap_or("unknown filepath").into()
            }
        ])
    }
}

/// Read the next statement from the token stream and assert that the type
/// can be resolved. Insert the type into the type context of the program
/// passed in.
fn validate_next_statement(
    program: &mut Program,
    token_stream: &mut TokenStream
) -> MultiResult<()> {
    let syntax_tree = parse_statement(token_stream)?;
    let tree_type = resolve_type(program, &syntax_tree)?;
    if let Some(name) = &tree_type.name_to_bind {
        program.type_context.insert(name.clone(), tree_type.datatype.clone());
    }
    Ok(())
}

#[cfg(test)]
mod test_resolve_import_type {
    use super::*;
    use std::collections::HashMap;
    use std::path::Path;

    use crate::models::Program;
    use crate::parser::test_utils::*;
    use crate::errors::{InterpreterError, error};
    use crate::type_resolver::resolve_type;
    use crate::type_resolver::shared::test_utils::resolve_type_fresh;
 
    static TEST_MODULES_IMPORT: &str = "src.type_resolver.test_data";
    static MODULE_WITH_TYPE_ERROR: &str = "moduleWithTypeError";
    static VALID_MODULE: &str = "validModule";

    #[test]
    fn it_returns_error_if_module_does_not_exist() {
        let input = make_tree("import nonexistentModule;");
        let expected = InterpreterError::ImportError{
            filepath: "nonexistentModule.tb".into(),
        };
        assert_eq!(resolve_type_fresh(input), Err(vec![expected]))
    }

    #[test]
    fn it_returns_error_if_module_in_directory_does_not_exist() {
        let input = make_tree("import src.fake.path;");

        // Path::new should handle the file separators correctly on all OSes
        let expected_path = Path::new("src/fake/path.tb").to_str().unwrap();
        let expected =
            InterpreterError::ImportError{ filepath: expected_path.into() };
        assert_eq!(resolve_type_fresh(input), Err(vec![expected]))
    }

    #[test]
    fn it_returns_error_if_module_has_type_errors() {
        let input = make_tree(format!(
            "import {}.{}",
            TEST_MODULES_IMPORT,
            MODULE_WITH_TYPE_ERROR,
        ).as_str());
        
        let module_error = InterpreterError::UnexpectedType {
            got: Type::String,
            expected: Type::Int,
        };
        let expected =
            error::module_error(MODULE_WITH_TYPE_ERROR, module_error);
        assert_eq!(resolve_type_fresh(input), Err(vec![expected]))
    }

    #[test]
    fn it_returns_imported_members_as_a_struct_if_module_has_no_type_errors() {
        let input = make_tree(&format!(
            "import {}.{};",
            TEST_MODULES_IMPORT,
            VALID_MODULE
        ));
        let program = Program::init_with_std_streams();
        assert_eq!(program.type_context.get(VALID_MODULE), None);

        // Types come from the test data file
        let expected_struct = Type::Struct(HashMap::from([
            ("validConstant".into(),
                Type::Int),
            ("validFunction".into(),
                Type::func(&[Type::Bool], Type::Bool)),
            ("functionReferencingValidConstant".into(),
                Type::func(&[Type::Int], Type::Int)),
        ]));
        let expected_tree = TreeType {
            datatype: expected_struct,
            name_to_bind: Some(VALID_MODULE.into())
        };
        assert_eq!(resolve_type(&program, &input), Ok(expected_tree));
    }
}
