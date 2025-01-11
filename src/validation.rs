use crate::models::{AbstractSyntaxTreeV2, Literal, Program, Type};

static UNDEFINED_ID: &str = "Undefined Identifier: ";

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

        // TODO: recursive node cases
        _ => panic!("unimplemented node type: {:?}", tree),
    }
}

fn validate_id(program: &Program, id: &String) -> ValidationResult {
    if !program.vars.contains_key(id) {
        let error = format!("{}{}", UNDEFINED_ID, id);
        return Err(vec![error]);
    }
    let var= program.vars.get(id).unwrap();
    return Ok(var.datatype);
}

fn get_literal_type(literal: &Literal) -> Type {
    match literal {
        Literal::Bool(_) => Type::Bool,
        Literal::Int(_) => Type::Int,
        Literal::String(_) => Type::String,
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
        let expected = vec![format!("{}x", UNDEFINED_ID)];
        assert_eq!(validate(&Program::new(), &tree), Err(expected));
    }
}
