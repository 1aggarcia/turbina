use crate::errors::{error, InterpreterError};
use crate::models::LetNode;
use crate::type_resolver::expr_type_resolver::resolve_expr_type;
use crate::type_resolver::shared::{SubResult, TypeContext};

/// Check that the expression type does not conflict with the declared type
/// and that the variable name is unique
pub fn resolve_let_type(context: &TypeContext, node: &LetNode) -> SubResult {
    if node.id == "_" {
        // reserved for function piping
        return Err(InterpreterError::ReservedId { id: "_".into() }.into())
    }
    if let Some(_) = context.lookup(&node.id) {
        return Err(vec![error::already_defined(&node.id)]);
    }

    let expr_type = resolve_expr_type(context, &node.value)?;
    let declared_type = match node.datatype.clone() {
        Some(t) => t,
        None => return Ok(expr_type),
    };

    if !expr_type.is_assignable_to(&declared_type) {
        let err = InterpreterError::UnexpectedType {
            got: expr_type,
            expected: declared_type
        };
        return Err(vec![err]);
    }
    // declared takes precedence if present
    return Ok(declared_type);
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use crate::errors::{error, InterpreterError};
    use crate::models::{Program, Type};
    use crate::parser::test_utils::make_tree;
    use crate::type_resolver::resolve_type;
    use crate::type_resolver::shared::test_utils::*;

    mod bindings {
        use super::*;

        #[rstest]
        #[case::literal_with_declared_type("let x: int = 3;", "x", Type::Int)]
        #[case::math_expr("let something = 5 + 2;", "something", Type::Int)]
        #[case::string_expr(
            "let something = \"a\" + \"b\";", "something", Type::String)]

        #[case::int_as_unknown(
            "let x: unknown = 5;", "x", Type::Unknown)]
        #[case::string_as_unknown(
            "let y: unknown = \"\";", "y", Type::Unknown)]
        #[case::function_as_unknown(
            "let f: unknown = () -> 3;", "f", Type::Unknown)]

        #[case::int_as_nullable_type(
            "let n: int? = 3;", "n", Type::Int.as_nullable())]
        #[case::null_as_nullable_type(
                "let n: int? = null;", "n", Type::Int.as_nullable())]
        #[case::int_as_nullable_unknown(
            "let x: unknown? = 5;", "x", Type::Unknown.as_nullable())]
        #[case::null_as_nullable_unknown(
            "let x: unknown? = null;", "x", Type::Unknown.as_nullable())]

        #[case::func_with_explicit_type(
            "let f: (int -> unknown) = (x: unknown): int -> 0;",
            "f",
            Type::func(&[Type::Int], Type::Unknown)
        )]
        #[case::list_with_explicit_type(
            "let x: int[] = [1];",
            "x",
            Type::Int.as_list(),
        )]
        #[case::empty_list(
            "let x: string[] = [];",
            "x",
            Type::String.as_list(),
        )]
        #[case::nullable_list_with_only_null(
            "let x: string?[] = [null, null];",
            "x",
            Type::String.as_nullable().as_list(),
        )]
        fn it_returns_correct_type(
            #[case] input: &str,
            #[case] symbol: &str,
            #[case] datatype: Type,
        ) {
            let tree = make_tree(input);
            assert_eq!(resolve_type_fresh(tree), ok_with_binding(symbol, datatype));
        }

        #[test]
        fn it_allows_casted_nullable_value_to_be_assigned_as_not_null() {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("nullString".into(), Type::String.as_nullable());
    
            let input = make_tree("let validString: string = nullString!;");
            let expected = ok_with_binding("validString", Type::String);
            assert_eq!(resolve_type(&mut program, &input), expected);
        }

        #[rstest]
        #[case("let x: int = \"string\";", Type::Int, Type::String)]
        #[case(
            "let f: (unknown -> int) = (x: int): unknown -> null;",
            Type::func(&[Type::Unknown], Type::Int),
            Type::func(&[Type::Int], Type::Unknown),
        )]
        #[case::empty_list(
            "let x: string = [];",
            Type::String,
            Type::EmptyList,
        )]
        fn it_returns_type_error_for_conflicting_types(
            #[case] input: &str,
            #[case] declared: Type,
            #[case] actual: Type,
        ) {
            let tree = make_tree(input);
            let error = InterpreterError::UnexpectedType {
                got: actual,
                expected: declared,
            };
            assert_eq!(resolve_type_fresh(tree), Err(vec![error]));
        }

        #[test]
        fn it_returns_error_for_assigning_unknown_to_int() {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("a".to_string(), Type::Unknown);

            let tree = make_tree("let b: int = a;");
            let error = InterpreterError::UnexpectedType {
                got: Type::Unknown,
                expected: Type::Int
            };
            assert_eq!(resolve_type(&program, &tree), Err(vec![error]));
        }

        #[test]
        fn it_propagates_error_in_expression() {
            let tree = make_tree("let y: string = undefined;");
            let error = error::undefined_id("undefined");
            assert_eq!(resolve_type_fresh(tree), Err(vec![error]));
        }

        #[test]
        fn it_returns_err_for_duplicate_id() {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("b".to_string(), Type::Bool);
            let tree = make_tree("let b = true;");
            let error = error::already_defined("b");
            assert_eq!(resolve_type(&program, &tree), Err(vec![error]));
        }

        #[test]
        fn it_returns_error_for_single_underscore_id() {
            let tree = make_tree("let _ = 0;");
            let error = InterpreterError::ReservedId { id: "_".into() };
            assert_eq!(resolve_type_fresh(tree), Err(vec![error]));
        }
    }
}
