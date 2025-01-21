use crate::errors;
use crate::models::{
    get_literal_type, AbstractSyntaxTree, BinaryOp, ExprNode, LetNode, OperatorNode, Program, Term, Type
};

type ValidationResult = Result<Type, Vec<String>>;

/// Find syntax errors not caught while parsing
/// - Check that all symbols exist in the program
/// - Type-check all nodes
pub fn validate(
    program: &Program, tree: &AbstractSyntaxTree
) -> ValidationResult {
    match tree {
        AbstractSyntaxTree::Let(node) => validate_let(program, node),
        AbstractSyntaxTree::Expr(node) => validate_expr(program, node),
    }
}

/// Check that the types for every term in the expression are valid
fn validate_expr(program: &Program, expr: &ExprNode) -> ValidationResult {
    // TODO: dont escape an error on the first token, collect it into the errors vector
    let mut result_type = validate_term(program, &expr.first)?;
    let mut errors = Vec::<String>::new();

    for (op, term) in &expr.rest {
        let right_result = validate_term(program, &term);
        if right_result.is_err() {
            errors.extend(right_result.err().unwrap());
            continue;
        }
        let right_type = right_result.unwrap();

        if result_type != right_type {
            let err = errors::binary_op_types(*op, result_type, right_type);
            return Err(vec![err]);
        }
        result_type = binary_op_return_type(*op, result_type)?;
    }

    if errors.is_empty() {
        return Ok(result_type);
    } else {
        return Err(errors);
    }
}

/// For symbols (ID tokens), check that they exist and their type matches any
/// unary operators applied (! and -).
/// 
/// For literals, check that the type matches any unary operators.
fn validate_term(program: &Program, term: &Term) -> ValidationResult {
    match term {
        Term::Literal(lit) => Ok(get_literal_type(&lit)),
        Term::Id(id) => validate_id(program, &id),
        Term::Not(term) => validate_negated_bool(program, term),
        Term::Minus(term) => validated_negated_int(program, term),
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

/// Check that the passed in term is a boolean
fn validate_negated_bool(program: &Program, inner_term: &Term) -> ValidationResult {
    let datatype = validate_term(program, inner_term)?;
    match datatype {
        Type::Bool => Ok(datatype),
        _ => Err(vec![errors::unary_op_type("!", datatype)])
    }
}

/// Check that the passed in term is an int
fn validated_negated_int(program: &Program, inner_term: &Term) -> ValidationResult {
    let datatype = validate_term(program, inner_term)?;
    match datatype {
        Type::Int=> Ok(datatype),
        _ => Err(vec![errors::unary_op_type("-", datatype)])
    }
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
    use test_utils::term_tree;
    use crate::{lexer::*, models::*, parser::*, validator::*};

    mod term {
        use super::*;

        #[rstest]
        #[case(Literal::Int(3))]
        #[case(Literal::String("asdf".to_string()))]
        #[case(Literal::Bool(false))]
        fn returns_ok_for_literals(#[case] literal: Literal) {
            let tree = term_tree(Term::Literal(literal.clone()));
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

        #[rstest]
        #[case("!3", errors::unary_op_type("!", Type::Int))]
        #[case("!\"str\"", errors::unary_op_type("!", Type::String))]
        #[case("-false", errors::unary_op_type("-", Type::Bool))]
        #[case("-\"str\"", errors::unary_op_type("-", Type::String))]
        fn it_returns_error_for_bad_negated_types(
            #[case] input: &str,
            #[case] error: String,
        ) {
            let tree = make_tree(input);
            assert_eq!(validate(&Program::new(), &tree), Err(vec![error]));
        }
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
        #[case("x + y - z", ["x", "y", "z"].map(errors::undefined_id).to_vec())]
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
