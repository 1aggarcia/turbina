use crate::errors;
use crate::models::{
    get_literal_type, AbstractSyntaxTree, LetNode, BinaryOp, OperatorNode, Program, Type
};

type ValidationResult = Result<Type, Vec<String>>;

/// Find syntax errors not caught while parsing
/// - Check that all symbols exist in the program
/// - Type-check all nodes
pub fn validate(
    program: &Program, tree: &AbstractSyntaxTree
) -> ValidationResult {
    match tree {
        AbstractSyntaxTree::Literal(literal) => Ok(get_literal_type(literal)),
        AbstractSyntaxTree::Id(id) => validate_id(program, id),
        AbstractSyntaxTree::Operator(node) => validate_operator(program, node),
        AbstractSyntaxTree::Let(node) => validate_let(program, node),
    }
}

/// Check that the id exists in the program
fn validate_id(program: &Program, id: &String) -> ValidationResult {
    if !program.vars.contains_key(id) {
        let error = errors::undefined_id(id);
        return Err(vec![error]);
    }
    let var = program.vars.get(id).unwrap();
    return Ok(var.datatype);
}

/// Check that the types of both arguments of the operator are legal
fn validate_operator(
    program: &Program, node: &OperatorNode
) -> ValidationResult {
    // validate children invididually first
    let left_result = validate(program, &node.left);
    let right_result = validate(program, &node.right);

    let child_errors = combine_errors(&left_result, &right_result);
    if !child_errors.is_empty() {
        return Err(child_errors);
    }

    // validate types together
    let left_type = left_result.unwrap();
    let right_type = right_result.unwrap();

    if left_type != right_type {
        let err = errors::binary_op_types(node.operator, left_type, right_type);
        return Err(vec![err]);
    }

    let output_type = binary_op_return_type(node.operator, left_type)?;
    return Ok(output_type);
}

/// Get the return type of a binary operator if `input_type` is valid,
/// otherwise return a validation error
fn binary_op_return_type(operator: BinaryOp, input_type: Type) -> ValidationResult {
    let type_error = errors::binary_op_types(operator, input_type, input_type);
    match operator {
        // equality operators
        BinaryOp::NotEq | BinaryOp::Equals => Ok(Type::Bool),

        // math operators
        BinaryOp::Plus => match input_type {
            Type::String | Type::Int => Ok(input_type),
            _ => Err(vec![type_error])
        },
        BinaryOp::Minus
        | BinaryOp::Percent
        | BinaryOp::Slash
        | BinaryOp::Star => if input_type == Type::Int {
            Ok(Type::Int)
        } else {
            Err(vec![type_error])
        }
    }
}

/// Check that the expression type does not conflic with the declared type
/// and that the variable name is unique
fn validate_let(program: &Program, node: &LetNode) -> ValidationResult {
    if program.vars.contains_key(&node.id) {
        return Err(vec![errors::already_defined(&node.id)]);
    }

    let value_type = validate(program, &node.value)?;
    let declared_type = match node.datatype {
        Some(t) => t,
        None => return Ok(value_type),
    };

    if value_type != declared_type {
        let err = errors::declared_type(&node.id, declared_type, value_type);
        return Err(vec![err]);
    }
    return Ok(value_type);
}

/// Join any errors in either of the two results into a single list of errors
fn combine_errors(res1: &ValidationResult, res2: &ValidationResult) -> Vec<String> {
    let mut errors1 = res1
        .clone()
        .map_or_else(|err| err, |_| Vec::new());

    let errors2 = res2
        .clone()
        .map_or_else(|err| err, |_| Vec::new());

    errors1.extend(errors2);
    return errors1;

}

#[cfg(test)]
mod test_validate {
    use rstest::rstest;
    use crate::{lexer::*, models::*, parser::*, validation::*};

    #[rstest]
    #[case(Literal::Int(3))]
    #[case(Literal::String("asdf".to_string()))]
    #[case(Literal::Bool(false))]
    fn returns_ok_for_literals(#[case] literal: Literal) {
        let tree = AbstractSyntaxTree::Literal(literal.clone());
        let expected = get_literal_type(&literal);
        assert_eq!(validate(&Program::new(), &tree), Ok(expected));
    }

    #[test]
    fn it_returns_ok_for_valid_symbol() {
        let tree = make_tree("x");
        let mut program = Program::new();
        program.vars.insert("x".to_string(), Variable {
            datatype: Type::Int,
            value: Literal::Int(3),
        });

        assert_eq!(validate(&program, &tree), Ok(Type::Int));
    }

    #[test]
    fn it_returns_error_for_non_existent_symbol() {
        let tree = make_tree("x");
        let expected = vec![errors::undefined_id("x")];
        assert_eq!(validate(&Program::new(), &tree), Err(expected));
    }

    mod operator {
        use super::*;

        #[rstest]
        #[case("3 + \"\"", BinaryOp::Plus, Type::Int, Type::String)]
        #[case("\"\" - \"\"", BinaryOp::Minus, Type::String, Type::String)]
        #[case("true % false", BinaryOp::Percent, Type::Bool, Type::Bool)]
        #[case("0 == false", BinaryOp::Equals, Type::Int, Type::Bool)]
        #[case("\"\" != 1", BinaryOp::NotEq, Type::String, Type::Int)]
        fn it_returns_error_for_illegal_types(
            #[case] input: &str,
            #[case] op: BinaryOp,
            #[case] left_type: Type,
            #[case] right_type: Type
        ) {
            let tree = make_tree(input);
            let expected = errors::binary_op_types(op, left_type, right_type);
            assert_eq!(validate(&Program::new(), &tree), Err(vec![expected]));
        }

        #[rstest]
        // right arg undefined
        #[case("a + 3", vec![errors::undefined_id("a")])]

        // left arg undefined
        #[case("1 + c", vec![errors::undefined_id("c")])]

        // both args undefined
        #[case("x + y", ["x", "y"].map(errors::undefined_id).to_vec())]
        fn it_returns_error_for_child_errors(
            #[case] input: &str, #[case] errors: Vec<String>
        ) {
            // symbol does not exist
            let tree = make_tree(input);
            assert_eq!(validate(&Program::new(), &tree), Err(errors));
        }

        #[test]
        fn it_returns_ok_for_int_addition() {
            let tree = make_tree("2 + 2");
            assert_eq!(validate(&Program::new(), &tree), Ok(Type::Int));
        }

        #[test]
        fn it_returns_ok_for_int_division() {
            let tree = make_tree("2 / 2");
            assert_eq!(validate(&Program::new(), &tree), Ok(Type::Int));
        }

        #[test]
        fn it_returns_ok_for_string_concatenation() {
            let tree = make_tree("\"a\" + \"b\"");
            assert_eq!(validate(&Program::new(), &tree), Ok(Type::String));
        }

        #[rstest]
        #[case(make_tree("0 == 1"))]
        #[case(make_tree("true != false"))]
        #[case(make_tree("\"a\" == \"b\""))]
        fn it_returns_ok_for_boolean_operator_on_same_type(#[case] tree: AbstractSyntaxTree) {
            let expected = Ok(Type::Bool);
            assert_eq!(validate(&Program::new(), &tree), expected);
        }
    }

    mod let_node {
        use super::*;

        #[test]
        fn it_infers_correct_type_for_math_expr() {
            let tree = make_tree("let something = 5 + 2");
            assert_eq!(validate(&Program::new(), &tree), Ok(Type::Int));
        }

        #[test]
        fn it_infers_correct_type_for_string_expr() {
            let tree = make_tree("let something = \"a\" + \"b\"");
            assert_eq!(validate(&Program::new(), &tree), Ok(Type::String));
        }

        #[test]
        fn it_returns_ok_for_declared_type() {
            let tree = make_tree("let x: int = 2 + 3");
            assert_eq!(validate(&Program::new(), &tree), Ok(Type::Int));
        }

        #[test]
        fn it_returns_type_error_for_conflicting_types() {
            let tree = make_tree("let x: int = \"string\"");
            let error = errors::declared_type("x", Type::Int, Type::String);
            assert_eq!(validate(&Program::new(), &tree), Err(vec![error]));
        }

        #[test]
        fn it_propagates_errors_in_expression() {
            let tree = make_tree("let y: string = undefined");
            let error = errors::undefined_id("undefined");
            assert_eq!(validate(&Program::new(), &tree), Err(vec![error]));
        }

        #[test]
        fn it_returns_err_for_duplicate_id() {
            let mut program = Program::new();
            program.vars.insert("b".to_string(), Variable {
                datatype: Type::Bool,
                value: Literal::Bool(false),
            });
            let tree = make_tree("let b = true");
            let error = errors::already_defined("b");
            assert_eq!(validate(&program, &tree), Err(vec![error])); 
        }
    }

    fn make_tree(statement: &str) -> AbstractSyntaxTree {
        return parse(tokenize(statement)).unwrap();
    }
}
