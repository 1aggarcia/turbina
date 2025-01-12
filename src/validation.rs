use crate::models::{AbstractSyntaxTreeV2, Literal, Operator, OperatorNode, Program, Type};

type ValidationResult = Result<Type, Vec<String>>;

/// Find syntax errors not caught while parsing
/// - Check that all symbols exist in the program
/// - Type-check all nodes
pub fn validate(
    program: &Program, tree: &AbstractSyntaxTreeV2
) -> ValidationResult {
    match tree {
        AbstractSyntaxTreeV2::Literal(literal) => Ok(get_literal_type(literal)),
        AbstractSyntaxTreeV2::Id(id) => validate_id(program, id),
        AbstractSyntaxTreeV2::Operator(node) => validate_operator(program, node),
        
        // TODO: validate 'let' 
        _ => panic!("unimplemented node type: {:?}", tree),
    }
}

/// Check that the id exists in the program
fn validate_id(program: &Program, id: &String) -> ValidationResult {
    if !program.vars.contains_key(id) {
        let error = errors::undefined_id(id);
        return Err(vec![error]);
    }
    let var= program.vars.get(id).unwrap();
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

    let legal_types = match node.operator {
        // + operator can be used for addition or string concat
        Operator::Plus => vec![Type::Int, Type::String],
        _ => vec![Type::Int],
    };

    if left_type != right_type || !legal_types.contains(&left_type) {
        let err = errors::binary_op_types(node.operator, left_type, right_type);
        return Err(vec![err]);
    }

    return Ok(left_type);
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

fn get_literal_type(literal: &Literal) -> Type {
    match literal {
        Literal::Bool(_) => Type::Bool,
        Literal::Int(_) => Type::Int,
        Literal::String(_) => Type::String,
    }
}

/// Collection of error message templates
mod errors {
    // TODO: move to 'errors' file & combine with syntax_error macro
    use crate::models::{Operator, Type};

    static TYPE_ERROR: &str = "Type Error: ";
    static VALUE_ERROR: &str = "Value Error: ";

    pub fn undefined_id(id: &str) -> String {
        format!("{VALUE_ERROR}Idenfitier '{id}' is undefined")
    }

    pub fn binary_op_types(
        operator: Operator,
        left_type: Type,
        right_type: Type
    ) -> String {
        format!(
            "{TYPE_ERROR}Illegal types for '{:?}' operator: {:?}, {:?}",
            operator,
            left_type,
            right_type
        )
    }
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
        let tree = AbstractSyntaxTreeV2::Literal(literal.clone());
        let expected = get_literal_type(&literal);
        assert_eq!(validate(&Program::new(), &tree), Ok(expected));
    }

    #[test]
    fn it_returns_ok_for_valid_symbol() {
        let tree = parse(tokenize("x")).unwrap();
        let mut program = Program::new();
        program.vars.insert("x".to_string(), Variable {
            datatype: Type::Int,
            value: Literal::Int(3),
        });

        assert_eq!(validate(&program, &tree), Ok(Type::Int));
    }

    #[test]
    fn it_returns_error_for_non_existent_symbol() {
        let tree = parse(tokenize("x")).unwrap();
        let expected = vec![errors::undefined_id("x")];
        assert_eq!(validate(&Program::new(), &tree), Err(expected));
    }

    mod operator {
        use super::*;

        #[rstest]
        #[case("3 + \"\"", Operator::Plus, Type::Int, Type::String)]
        #[case("\"\" - \"\"", Operator::Minus, Type::String, Type::String)]
        #[case("true % false", Operator::Percent, Type::Bool, Type::Bool)]
        fn it_returns_error_for_illegal_types(
            #[case] input: &str,
            #[case] op: Operator,
            #[case] left_type: Type,
            #[case] right_type: Type
        ) {
            let tree = parse(tokenize(input)).unwrap();
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
            let tree = parse(tokenize(input)).unwrap();
            assert_eq!(validate(&Program::new(), &tree), Err(errors));
        }

        #[test]
        fn it_returns_ok_for_int_addition() {
            let tree = parse(tokenize("2 + 2")).unwrap();
            assert_eq!(validate(&Program::new(), &tree), Ok(Type::Int));
        }

        #[test]
        fn it_returns_ok_for_int_division() {
            let tree = parse(tokenize("2 / 2")).unwrap();
            assert_eq!(validate(&Program::new(), &tree), Ok(Type::Int));
        }

        #[test]
        fn it_returns_ok_for_string_concatenation() {
            let tree = parse(tokenize("\"a\" + \"b\"")).unwrap();
            assert_eq!(validate(&Program::new(), &tree), Ok(Type::String));
        }
    }
}
